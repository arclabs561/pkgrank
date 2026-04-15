use anyhow::{Context, Result};
use clap::Parser;
use petgraph::prelude::*;
use serde::Serialize;
use std::collections::VecDeque;
use std::path::PathBuf;

use super::*;
use crate::dep_graph::Ecosystem;

#[derive(Parser, Debug, Clone)]
pub(crate) struct BlastRadiusArgs {
    /// Package name to analyze (e.g. "serde").
    pub package: String,

    /// Path to Cargo.toml, lock file, or directory.
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Ecosystem (auto-detected from directory if omitted).
    #[arg(long, value_enum)]
    pub ecosystem: Option<Ecosystem>,

    /// Include dev-dependencies (Cargo only).
    #[arg(long)]
    pub dev: bool,

    /// Include build-dependencies (Cargo only).
    #[arg(long)]
    pub build: bool,

    /// Restrict to workspace members only (Cargo only).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub workspace_only: bool,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
    pub format: OutputFormat,

    /// Top-N dependents to show (0 = all).
    #[arg(short = 'n', long, default_value_t = 50)]
    pub top: usize,

    /// Cache graph analysis results.
    #[arg(long, default_value_t = false)]
    pub cache: bool,

    /// Force-refresh cached results.
    #[arg(long, default_value_t = false)]
    pub cache_refresh: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct BlastRadiusRow {
    pub name: String,
    pub version: String,
    pub bfs_depth: usize,
    pub pagerank: f64,
    pub in_degree: usize,
    pub out_degree: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct BlastRadiusResult {
    pub target: String,
    pub found: bool,
    pub total_transitive_dependents: usize,
    pub rows: Vec<BlastRadiusRow>,
}

/// Generic blast-radius BFS on any DiGraph. Returns BlastRadiusResult.
fn blast_radius_bfs<N>(
    graph: &DiGraph<N, f64>,
    target_idx: NodeIndex,
    pr: &[f64],
    name_fn: impl Fn(NodeIndex) -> (String, String),
    top: usize,
) -> BlastRadiusResult
where
    N: Clone,
{
    let rev = reverse_graph(graph);
    let mut visited = vec![false; graph.node_count()];
    let mut depth = vec![0usize; graph.node_count()];
    let mut queue = VecDeque::new();
    visited[target_idx.index()] = true;
    queue.push_back(target_idx);

    while let Some(node) = queue.pop_front() {
        for neighbor in rev.neighbors(node) {
            if !visited[neighbor.index()] {
                visited[neighbor.index()] = true;
                depth[neighbor.index()] = depth[node.index()] + 1;
                queue.push_back(neighbor);
            }
        }
    }

    let mut rows: Vec<BlastRadiusRow> = graph
        .node_indices()
        .filter(|&n| visited[n.index()] && n != target_idx)
        .map(|n| {
            let (name, version) = name_fn(n);
            BlastRadiusRow {
                name,
                version,
                bfs_depth: depth[n.index()],
                pagerank: pr[n.index()],
                in_degree: graph.neighbors_directed(n, Direction::Incoming).count(),
                out_degree: graph.neighbors_directed(n, Direction::Outgoing).count(),
            }
        })
        .collect();

    rows.sort_by(|a, b| {
        a.bfs_depth
            .cmp(&b.bfs_depth)
            .then_with(|| b.pagerank.total_cmp(&a.pagerank))
    });

    let total = rows.len();
    if top > 0 {
        rows.truncate(top);
    }

    BlastRadiusResult {
        target: String::new(), // caller fills this
        found: true,
        total_transitive_dependents: total,
        rows,
    }
}

/// Compute blast radius for any ecosystem. Returns structured result.
pub(crate) fn compute_blast_radius(args: &BlastRadiusArgs) -> Result<BlastRadiusResult> {
    let ecosystem = args
        .ecosystem
        .or_else(|| detect_ecosystem(&args.path))
        .unwrap_or(Ecosystem::Rust);
    match ecosystem {
        Ecosystem::Rust => compute_blast_radius_cargo(args),
        _ => compute_blast_radius_polyglot(args, ecosystem),
    }
}

fn compute_blast_radius_cargo(args: &BlastRadiusArgs) -> Result<BlastRadiusResult> {
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

    let mpath = manifest_path(&analyze.path)?;
    let metadata = metadata_for(&mpath, &analyze)
        .with_context(|| format!("cargo metadata failed for {}", mpath.display()))?;
    let (graph, node_map) = build_graph(&metadata, &analyze)?;

    let pkg_by_id: std::collections::HashMap<&cargo_metadata::PackageId, &cargo_metadata::Package> =
        metadata.packages.iter().map(|p| (&p.id, p)).collect();

    let target_idx = node_map
        .iter()
        .find(|(id, _)| {
            pkg_by_id
                .get(id)
                .map(|p| p.name.as_str() == args.package)
                .unwrap_or(false)
        })
        .map(|(_, &idx)| idx);

    let Some(target_idx) = target_idx else {
        return Ok(BlastRadiusResult {
            target: args.package.clone(),
            found: false,
            total_transitive_dependents: 0,
            rows: Vec::new(),
        });
    };

    let pr = pagerank_auto(&graph);
    let mut result = blast_radius_bfs(
        &graph,
        target_idx,
        &pr,
        |n| {
            let id = graph.node_weight(n).expect("valid");
            let pkg = pkg_by_id.get(id);
            (
                pkg.map(|p| p.name.as_str()).unwrap_or("?").to_string(),
                pkg.map(|p| p.version.to_string()).unwrap_or_default(),
            )
        },
        args.top,
    );
    result.target = args.package.clone();
    Ok(result)
}

fn compute_blast_radius_polyglot(
    args: &BlastRadiusArgs,
    ecosystem: Ecosystem,
) -> Result<BlastRadiusResult> {
    let (packages, edges) = match ecosystem {
        Ecosystem::Js => crate::polyglot::parse_npm(&args.path)?,
        Ecosystem::Python => crate::polyglot::parse_python(&args.path)?,
        Ecosystem::Go => crate::polyglot::parse_go_mod_graph(&args.path)?,
        Ecosystem::Rust => unreachable!(),
    };

    let (graph, map) = dep_graph::build_dep_graph(&packages, &edges);

    let target_idx = map.get(&args.package).copied();
    let Some(target_idx) = target_idx else {
        return Ok(BlastRadiusResult {
            target: args.package.clone(),
            found: false,
            total_transitive_dependents: 0,
            rows: Vec::new(),
        });
    };

    let pr = pagerank_auto(&graph);
    let mut result = blast_radius_bfs(
        &graph,
        target_idx,
        &pr,
        |n| {
            let node = graph.node_weight(n).expect("valid");
            (node.name.clone(), node.version.clone().unwrap_or_default())
        },
        args.top,
    );
    result.target = args.package.clone();
    Ok(result)
}

pub(crate) fn run_blast_radius(args: &BlastRadiusArgs) -> Result<()> {
    let result = compute_blast_radius(args)?;
    print_blast_radius(&result, args)
}

fn print_blast_radius(result: &BlastRadiusResult, args: &BlastRadiusArgs) -> Result<()> {
    let fmt = effective_format(args.format);
    match fmt {
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct Out<'a> {
                schema_version: u32,
                ok: bool,
                command: &'a str,
                #[serde(flatten)]
                inner: &'a BlastRadiusResult,
            }
            let out = Out {
                schema_version: 1,
                ok: true,
                command: "blast-radius",
                inner: result,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Text | OutputFormat::Auto => {
            if !result.found {
                println!(
                    "blast-radius: package '{}' not found in the dependency graph.",
                    result.target
                );
                return Ok(());
            }
            println!(
                "blast-radius: {}  ({} transitive dependents)\n",
                result.target, result.total_transitive_dependents
            );
            println!(
                "{:>5}  {:>10}  {:>3}  {:>3}  {:<24} version",
                "depth", "pagerank", "in", "out", "name"
            );
            println!("{:\u{2500}<70}", "");
            for r in &result.rows {
                println!(
                    "{:>5}  {:>10.6}  {:>3}  {:>3}  {:<24} {}",
                    r.bfs_depth, r.pagerank, r.in_degree, r.out_degree, r.name, r.version
                );
            }
        }
    }
    Ok(())
}
