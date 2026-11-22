use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use super::*;

#[derive(Parser, Debug, Clone)]
pub(crate) struct UpgradePriorityArgs {
    /// Path to Cargo.toml or directory.
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Include dev-dependencies in graph analysis.
    #[arg(long)]
    pub dev: bool,

    /// Include build-dependencies.
    #[arg(long)]
    pub build: bool,

    /// Rank an existing cargo-outdated JSON report instead of querying for updates again.
    #[arg(long)]
    pub outdated_report: Option<PathBuf>,

    /// Enable Cargo features in both the upgrade report and the resolved graph.
    #[arg(long, value_delimiter = ',')]
    pub features: Vec<String>,

    /// Restrict graph to workspace members (external dependencies are normally ranked).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub workspace_only: bool,

    /// Top-N results.
    #[arg(short = 'n', long, default_value_t = 20)]
    pub top: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
    pub format: OutputFormat,

    /// Cache graph analysis results.
    #[arg(long, default_value_t = false)]
    pub cache: bool,

    /// Force-refresh cached results.
    #[arg(long, default_value_t = false)]
    pub cache_refresh: bool,
}

#[derive(Debug, Deserialize)]
struct CargoOutdatedOutput {
    crate_name: Option<String>,
    dependencies: Vec<OutdatedDep>,
}

#[derive(Debug, Deserialize)]
struct OutdatedDep {
    name: String,
    project: String,
    compat: String,
    latest: String,
    kind: Option<String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GraphMatch {
    Matched,
    OutsideGraph,
    Ambiguous,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Serialize)]
struct ReportedVia {
    workspace_member: Option<String>,
    dependency: String,
    compat_version: String,
    latest_version: String,
}

#[derive(Ord, PartialOrd, Eq, PartialEq)]
struct Upgrade {
    name: String,
    current: String,
    target: String,
}

fn target_version(current: &str, compat: &str, latest: &str) -> Option<String> {
    [latest, compat]
        .into_iter()
        .find(|v| *v != current && !matches!(*v, "---" | "Removed"))
        .map(str::to_owned)
}

#[derive(Debug, Serialize)]
pub(crate) struct UpgradePriorityRow {
    pub name: String,
    reported_via: BTreeSet<ReportedVia>,
    target_version: String,
    graph_match: GraphMatch,
    pub current_version: String,
    pub compat_version: String,
    pub latest_version: String,
    pub pagerank: f64,
    pub transitive_dependents: usize,
    pub priority_score: f64,
    pub upgrade_urgency: String,
}

/// Compute upgrade priority rows. Returns (outdated_count, ranked_rows).
pub(crate) fn compute_upgrade_priority(
    args: &UpgradePriorityArgs,
) -> Result<(usize, Vec<UpgradePriorityRow>)> {
    let outdated = match &args.outdated_report {
        Some(path) => {
            let file =
                std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            parse_outdated(std::io::BufReader::new(file))?
        }
        None => run_cargo_outdated(&args.path, &args.features)?,
    };

    // 2. Build graph analysis.
    let analyze = AnalyzeArgs {
        ecosystem: None,
        path: args.path.clone(),
        metric: Metric::Pagerank,
        top: 0,
        dev: args.dev,
        build: args.build,
        workspace_only: args.workspace_only,
        all_features: false,
        no_default_features: false,
        features: Some(args.features.join(",")).filter(|s| !s.is_empty()),
        format: OutputFormat::Json,
        stats: false,
        json_limit: None,
        cache: args.cache,
        cache_refresh: args.cache_refresh,
    };
    let (rows, _convergence) = analyze_rows_with_convergence(&analyze)?;

    // Cargo can resolve several versions or sources of one crate in a graph.
    let mut versions: HashMap<(&str, &str), Vec<&Row>> = HashMap::new();
    for row in &rows {
        versions
            .entry((&row.name, &row.version))
            .or_default()
            .push(row);
    }

    // Compute reachability for dependents count.
    let mpath = manifest_path(&analyze.path)?;
    let metadata = metadata_for(&mpath, &analyze)?;
    let (graph, _) = build_graph(&metadata, &analyze)?;

    let edge_pairs: Vec<(usize, usize)> = graph
        .edge_references()
        .map(|e| (e.source().index(), e.target().index()))
        .collect();
    let (dependents, _deps) = graphops::reachability_counts_edges(graph.node_count(), &edge_pairs);

    let by_id: HashMap<_, _> = graph
        .node_indices()
        .map(|idx| (graph[idx].repr.as_str(), dependents[idx.index()]))
        .collect();

    let outdated_count = outdated
        .iter()
        .map(|report| report.dependencies.len())
        .sum();
    let mut upgrades: BTreeMap<Upgrade, BTreeSet<ReportedVia>> = BTreeMap::new();
    for report in outdated {
        for dep in report.dependencies {
            if (!args.dev && dep.kind.as_deref() == Some("Development"))
                || (!args.build && dep.kind.as_deref() == Some("Build"))
            {
                continue;
            }
            // cargo-outdated labels a transitive dependency as parent->crate.
            let name = dep.name.rsplit("->").next().unwrap_or(&dep.name).to_owned();
            let Some(target) = target_version(&dep.project, &dep.compat, &dep.latest) else {
                continue;
            };
            let upgrade = Upgrade {
                name,
                current: dep.project,
                target,
            };
            upgrades.entry(upgrade).or_default().insert(ReportedVia {
                workspace_member: report.crate_name.clone(),
                dependency: dep.name,
                compat_version: dep.compat,
                latest_version: dep.latest,
            });
        }
    }
    let mut priority_rows = Vec::new();
    for (upgrade, reported_via) in upgrades {
        let candidates = versions
            .get(&(upgrade.name.as_str(), upgrade.current.as_str()))
            .map(Vec::as_slice);
        let (graph_match, pr, trans_deps) = match candidates {
            Some([row]) => match by_id.get(row.id.as_str()) {
                Some(&reach) => (GraphMatch::Matched, row.pagerank, reach),
                None => (GraphMatch::OutsideGraph, 0.0, 0),
            },
            Some(_) => (GraphMatch::Ambiguous, 0.0, 0),
            None => (GraphMatch::OutsideGraph, 0.0, 0),
        };
        let urgency = version_urgency(&upgrade.current, &upgrade.target);
        let urgency_bonus = match urgency.as_str() {
            "major" => 10.0,
            "minor" => 5.0,
            _ => 1.0,
        };

        let score = 10.0 * ((trans_deps as f64) + 1.0).ln() + 1000.0 * pr + urgency_bonus;

        // Preserve the legacy version fields from one actual path. Prefer a path
        // where the target is the latest version; every observation is in reported_via.
        let example = reported_via
            .iter()
            .max_by_key(|path| {
                (
                    path.latest_version == upgrade.target,
                    path.compat_version == upgrade.target,
                )
            })
            .expect("each upgrade has a reporting path");
        priority_rows.push(UpgradePriorityRow {
            name: upgrade.name,
            target_version: upgrade.target,
            graph_match,
            current_version: upgrade.current,
            compat_version: example.compat_version.clone(),
            latest_version: example.latest_version.clone(),
            reported_via,
            pagerank: pr,
            transitive_dependents: trans_deps,
            priority_score: score,
            upgrade_urgency: urgency,
        });
    }
    priority_rows.sort_by(|a, b| {
        (b.graph_match == GraphMatch::Matched)
            .cmp(&(a.graph_match == GraphMatch::Matched))
            .then_with(|| b.priority_score.total_cmp(&a.priority_score))
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.current_version.cmp(&b.current_version))
            .then_with(|| a.target_version.cmp(&b.target_version))
    });
    if args.top > 0 {
        priority_rows.truncate(args.top);
    }

    Ok((outdated_count, priority_rows))
}

pub(crate) fn run_upgrade_priority(args: &UpgradePriorityArgs) -> Result<()> {
    let (outdated_count, priority_rows) = compute_upgrade_priority(args)?;

    let fmt = effective_format(args.format);
    match fmt {
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct Out {
                schema_version: u32,
                ok: bool,
                command: &'static str,
                outdated_count: usize,
                rows: Vec<UpgradePriorityRow>,
            }
            let out = Out {
                schema_version: 1,
                ok: true,
                command: "upgrade-priority",
                outdated_count,
                rows: priority_rows,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Text | OutputFormat::Auto => {
            println!("upgrade-priority  ({} outdated)\n", outdated_count);
            println!(
                "{:>4}  {:<20} {:>10} {:>10} {:>10} {:>7} {:>8} {:>5}  pr",
                "rank", "name", "current", "compat", "target", "urgency", "priority", "deps"
            );
            println!("{:\u{2500}<105}", "");
            for (i, r) in priority_rows.iter().enumerate() {
                println!(
                    "{:>4}. {:<20} {:>10} {:>10} {:>10} {:>7} {:>8.1} {:>5}  {:.4}",
                    i + 1,
                    match r.graph_match {
                        GraphMatch::Matched => r.name.clone(),
                        GraphMatch::OutsideGraph => format!("{} (outside graph)", r.name),
                        GraphMatch::Ambiguous => format!("{} (ambiguous source)", r.name),
                    },
                    r.current_version,
                    r.compat_version,
                    r.target_version,
                    r.upgrade_urgency,
                    r.priority_score,
                    r.transitive_dependents,
                    r.pagerank
                );
            }
        }
    }
    Ok(())
}

fn run_cargo_outdated(path: &Path, features: &[String]) -> Result<Vec<CargoOutdatedOutput>> {
    let manifest = if path.is_dir() {
        path.join("Cargo.toml")
    } else {
        path.to_path_buf()
    };

    // cargo-outdated compares workspace root paths literally for some feature sets.
    let manifest = manifest
        .canonicalize()
        .with_context(|| format!("opening {}", manifest.display()))?;
    let mut cmd = ProcessCommand::new("cargo");
    cmd.args(["outdated", "--format", "json", "--manifest-path"])
        .arg(manifest);
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(","));
    }

    match cmd.output() {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow!("cargo outdated failed: {}", stderr.trim()));
            }
            parse_outdated(output.stdout.as_slice())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(anyhow!(
            "cargo-outdated not found. Install with: cargo install cargo-outdated"
        )),
        Err(e) => Err(anyhow!("failed to run cargo outdated: {}", e)),
    }
}

fn parse_outdated(reader: impl std::io::Read) -> Result<Vec<CargoOutdatedOutput>> {
    let reports = serde_json::Deserializer::from_reader(reader)
        .into_iter::<CargoOutdatedOutput>()
        .collect::<Result<Vec<_>, _>>()
        .context("failed to parse cargo outdated JSON output")?;
    anyhow::ensure!(!reports.is_empty(), "cargo outdated returned no report");
    Ok(reports)
}

fn version_urgency(current: &str, latest: &str) -> String {
    let cur_parts: Vec<&str> = current.split('.').collect();
    let lat_parts: Vec<&str> = latest.split('.').collect();

    if cur_parts.first() != lat_parts.first() {
        "major".to_string()
    } else if cur_parts.get(1) != lat_parts.get(1) {
        "minor".to_string()
    } else {
        "patch".to_string()
    }
}
