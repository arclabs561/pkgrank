use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

    /// Restrict graph to workspace members.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub workspace_only: bool,

    /// Top-N results.
    #[arg(short = 'n', long, default_value_t = 20)]
    pub top: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
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
    dependencies: Vec<OutdatedDep>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OutdatedDep {
    name: String,
    project: String,
    compat: String,
    latest: String,
    kind: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct UpgradePriorityRow {
    pub name: String,
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
    let outdated = run_cargo_outdated(&args.path)?;

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
        features: None,
        format: OutputFormat::Json,
        stats: false,
        json_limit: None,
        cache: args.cache,
        cache_refresh: args.cache_refresh,
    };
    let (rows, _convergence) = analyze_rows_with_convergence(&analyze)?;

    // Build lookup by crate name.
    let row_by_name: HashMap<&str, &Row> = rows.iter().map(|r| (r.name.as_str(), r)).collect();

    // Compute reachability for dependents count.
    let mpath = manifest_path(&analyze.path)?;
    let metadata = metadata_for(&mpath, &analyze)?;
    let (graph, node_map) = build_graph(&metadata, &analyze)?;
    let pkg_by_id: HashMap<&cargo_metadata::PackageId, &cargo_metadata::Package> =
        metadata.packages.iter().map(|p| (&p.id, p)).collect();

    let edge_pairs: Vec<(usize, usize)> = graph
        .edge_references()
        .map(|e| (e.source().index(), e.target().index()))
        .collect();
    let (dependents, _deps) = graphops::reachability_counts_edges(graph.node_count(), &edge_pairs);

    // Build name -> dependents lookup.
    let mut dependents_by_name: HashMap<String, usize> = HashMap::new();
    for (id, &idx) in &node_map {
        if let Some(pkg) = pkg_by_id.get(id) {
            dependents_by_name.insert(pkg.name.to_string(), dependents[idx.index()]);
        }
    }

    // 3. Join and score.
    let mut priority_rows: Vec<UpgradePriorityRow> = Vec::new();
    for dep in &outdated {
        if dep.latest == dep.project || dep.latest == "---" {
            continue;
        }
        let pr = row_by_name
            .get(dep.name.as_str())
            .map(|r| r.pagerank)
            .unwrap_or(0.0);
        let trans_deps = dependents_by_name.get(&dep.name).copied().unwrap_or(0);

        let urgency = version_urgency(&dep.project, &dep.latest);
        let urgency_bonus = match urgency.as_str() {
            "major" => 10.0,
            "minor" => 5.0,
            _ => 1.0,
        };

        let score = 10.0 * ((trans_deps as f64) + 1.0).ln() + 1000.0 * pr + urgency_bonus;

        priority_rows.push(UpgradePriorityRow {
            name: dep.name.clone(),
            current_version: dep.project.clone(),
            compat_version: dep.compat.clone(),
            latest_version: dep.latest.clone(),
            pagerank: pr,
            transitive_dependents: trans_deps,
            priority_score: score,
            upgrade_urgency: urgency,
        });
    }

    priority_rows.sort_by(|a, b| b.priority_score.total_cmp(&a.priority_score));
    let outdated_count = outdated.len();
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
        OutputFormat::Text => {
            println!("upgrade-priority  ({} outdated)\n", outdated_count);
            println!(
                "{:>4}  {:<20} {:>10} {:>10} {:>10} {:>7} {:>8} {:>5}  pr",
                "rank", "name", "current", "compat", "latest", "urgency", "priority", "deps"
            );
            println!("{:\u{2500}<105}", "");
            for (i, r) in priority_rows.iter().enumerate() {
                println!(
                    "{:>4}. {:<20} {:>10} {:>10} {:>10} {:>7} {:>8.1} {:>5}  {:.4}",
                    i + 1,
                    r.name,
                    r.current_version,
                    r.compat_version,
                    r.latest_version,
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

fn run_cargo_outdated(path: &Path) -> Result<Vec<OutdatedDep>> {
    let manifest = if path.is_dir() {
        path.join("Cargo.toml")
    } else {
        path.to_path_buf()
    };

    let out = ProcessCommand::new("cargo")
        .args([
            "outdated",
            "--format",
            "json",
            "--manifest-path",
            &manifest.to_string_lossy(),
        ])
        .output();

    match out {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow!("cargo outdated failed: {}", stderr.trim()));
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            let parsed: CargoOutdatedOutput = serde_json::from_str(&stdout)
                .with_context(|| "failed to parse cargo outdated JSON output")?;
            Ok(parsed.dependencies)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(anyhow!(
            "cargo-outdated not found. Install with: cargo install cargo-outdated"
        )),
        Err(e) => Err(anyhow!("failed to run cargo outdated: {}", e)),
    }
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
