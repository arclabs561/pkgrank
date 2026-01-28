//! `pkgrank`: centrality scores over a Cargo dependency graph.
//!
//! This is a workspace tool intended to answer a simple question:
//! “Which crates are structurally central in the dependency DAG?”
//!
//! We treat crates as nodes and direct dependency edges as directed edges:
//! \[
//!   A \to B \quad \text{iff crate A depends on crate B}
//! \]

use anyhow::{anyhow, Context, Result};
use cargo_metadata::{
    CargoOpt, DepKindInfo, DependencyKind, Metadata, MetadataCommand, Node, PackageId,
};
use clap::{Parser, Subcommand, ValueEnum};
use petgraph::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::Instant;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use graphops::{
    betweenness_centrality, pagerank, pagerank_checked, pagerank_checked_run, pagerank_run,
    pagerank_weighted, pagerank_weighted_checked, pagerank_weighted_checked_run,
    pagerank_weighted_run, personalized_pagerank, reachability_counts_edges, PageRankConfig,
    PageRankRun,
};

#[cfg(feature = "stdio")]
use rmcp::{
    handler::server::router::tool::ToolRouter as RmcpToolRouter,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData as McpError, ServiceExt,
};
#[cfg(feature = "stdio")]
use schemars::JsonSchema;
#[cfg(feature = "stdio")]
use tokio::process::Command as TokioCommand;
#[cfg(feature = "stdio")]
use tokio::time::timeout;

#[derive(Parser, Debug)]
#[command(name = "pkgrank")]
#[command(about = "Cargo dependency graph centrality analysis")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    analyze: AnalyzeArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run pkgrank across a local multi-repo workspace and write artifacts.
    SweepLocal(SweepLocalArgs),
    /// Build a crates.io dependency graph and rank it.
    CratesIo(CratesIoArgs),
    /// Rank internal module/item centrality using `cargo modules dependencies`.
    ///
    /// This shells out to the `cargo-modules` tool and parses its DOT output.
    Modules(ModulesArgs),
    /// Run `pkgrank modules` across multiple packages (workspace sweep).
    ModulesSweep(ModulesSweepArgs),
    /// One-shot view: local sweep + crates.io crawl, written as HTML + JSON.
    View(ViewArgs),
    /// Triage bundle: top TLC crates/repos + invariants + PPR top-k (artifact-backed).
    ///
    /// This is the same “triage” payload as the MCP tool `pkgrank_triage`, but usable from CLI.
    Triage(TriageCliArgs),
    /// Serve as an MCP stdio server (for Cursor).
    McpStdio,
}

fn pagerank_auto<N>(graph: &DiGraph<N, f64>) -> Vec<f64> {
    // In pkgrank, many graphs use "all weights == 1.0" to mean unweighted. Prefer the unweighted
    // implementation in that case, but fall back to weighted PageRank when any edge has a non-unit
    // weight.
    let is_unweighted = graph.edge_weights().all(|w| (*w - 1.0).abs() < 1e-12);
    let cfg = PageRankConfig::default();
    if is_unweighted {
        pagerank_checked(graph, cfg).unwrap_or_else(|_| pagerank(graph, cfg))
    } else {
        pagerank_weighted_checked(graph, cfg).unwrap_or_else(|_| pagerank_weighted(graph, cfg))
    }
}

#[derive(Debug, Clone, Serialize)]
struct ConvergenceReport {
    iterations: usize,
    diff_l1: f64,
    converged: bool,
}

fn convergence_report(run: &PageRankRun) -> ConvergenceReport {
    ConvergenceReport {
        iterations: run.iterations,
        diff_l1: run.diff_l1,
        converged: run.converged,
    }
}

fn pagerank_auto_run<N>(graph: &DiGraph<N, f64>) -> PageRankRun {
    let is_unweighted = graph.edge_weights().all(|w| (*w - 1.0).abs() < 1e-12);
    let cfg = PageRankConfig::default();
    if is_unweighted {
        pagerank_checked_run(graph, cfg).unwrap_or_else(|_| pagerank_run(graph, cfg))
    } else {
        pagerank_weighted_checked_run(graph, cfg)
            .unwrap_or_else(|_| pagerank_weighted_run(graph, cfg))
    }
}

fn reverse_graph<N: Clone>(graph: &DiGraph<N, f64>) -> DiGraph<N, f64> {
    // Reverse a graph while preserving node order (so NodeIndex::index() aligns).
    let mut rev: DiGraph<N, f64> = DiGraph::new();
    let mut idx_map: Vec<NodeIndex> = Vec::with_capacity(graph.node_count());
    for n in graph.node_indices() {
        idx_map.push(rev.add_node(graph.node_weight(n).expect("node weight").clone()));
    }
    for e in graph.edge_references() {
        let u = e.source().index();
        let v = e.target().index();
        let w = *e.weight();
        rev.update_edge(idx_map[v], idx_map[u], w);
    }
    rev
}

#[derive(Parser, Debug, Clone)]
struct ModulesArgs {
    /// Path to a Cargo manifest to analyze.
    #[arg(long, default_value = "Cargo.toml")]
    manifest_path: PathBuf,

    /// Package to analyze (same meaning as `cargo modules -p ...`).
    #[arg(short = 'p', long)]
    package: Option<String>,

    /// Analyze the package library target.
    #[arg(long)]
    lib: bool,

    /// Analyze the named binary target.
    #[arg(long)]
    bin: Option<String>,

    /// Analyze with `#[cfg(test)]` enabled (as if built via `cargo test`).
    #[arg(long)]
    cfg_test: bool,

    /// Centrality metric to compute.
    #[arg(short, long, value_enum, default_value_t = Metric::Pagerank)]
    metric: Metric,

    /// Preset of common analysis settings.
    ///
    /// This exists to avoid long flag strings while still keeping every knob overrideable.
    ///
    /// Use `--preset none` to disable preset application and rely only on explicit flags.
    #[arg(long, value_enum, default_value_t = ModulesPreset::None)]
    preset: ModulesPreset,

    /// Number of top rows to show (text output only).
    #[arg(short = 'n', long, default_value_t = 25)]
    top: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    /// Which edge kinds from cargo-modules to include.
    #[arg(long, value_enum, default_value_t = ModuleEdgeKind::Uses)]
    edge_kind: ModuleEdgeKind,

    /// Aggregate nodes before scoring.
    ///
    /// Why this exists: developers often organize work by *files* or *modules*, not by individual
    /// items. Aggregation provides a more “actionable” hotspot list.
    #[arg(long, value_enum, default_value_t = ModuleAggregate::File)]
    aggregate: ModuleAggregate,

    /// When `--edge-kind both`, weight for `uses` edges.
    ///
    /// Default: 1.0
    #[arg(long, default_value_t = 1.0)]
    uses_weight: f64,

    /// When `--edge-kind both`, weight for `owns` edges.
    ///
    /// Default: 0.2 (structural containment is informative but can dominate if unweighted).
    #[arg(long, default_value_t = 0.2)]
    owns_weight: f64,

    /// Show top inter-group edges (by total weight) after the table.
    #[arg(long, default_value_t = 3)]
    edges_top: usize,

    /// When aggregating, show the top-N member items by node-level PageRank.
    #[arg(long, default_value_t = 2)]
    members_top: usize,

    /// Cache `cargo modules dependencies` DOT output on disk (default: true).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cache: bool,
    /// Force refresh cache (ignore existing cached DOT).
    #[arg(long, default_value_t = false)]
    cache_refresh: bool,

    /// Hide extern crates in the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_externs: bool,
    /// Include extern items (override for `--no-externs`).
    #[arg(long, default_value_t = false)]
    include_externs: bool,

    /// Hide sysroot crates (`std`, `core`, etc.) in the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_sysroot: bool,
    /// Include sysroot crates (override for `--no-sysroot`).
    #[arg(long, default_value_t = false)]
    include_sysroot: bool,

    /// Hide functions from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_fns: bool,
    /// Include functions (override for `--no-fns`).
    #[arg(long, default_value_t = false)]
    include_fns: bool,

    /// Hide traits from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_traits: bool,
    /// Include traits (override for `--no-traits`).
    #[arg(long, default_value_t = true)]
    include_traits: bool,

    /// Hide types (struct/enum/union) from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_types: bool,
    /// Include types (override for `--no-types`).
    #[arg(long, default_value_t = true)]
    include_types: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ModuleEdgeKind {
    Uses,
    Owns,
    Both,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModulesPreset {
    /// Do not apply a preset (use explicit flags only).
    None,
    /// File-level hotspots, include fns/types/traits; show strongest edges.
    FileFull,
    /// File-level hotspots, types+traits only (no fns), smaller output.
    FileApi,
    /// Node-level (items), uses-only, include fns/types/traits.
    NodeFull,
    /// Node-level API surface (types+traits only), uses-only.
    NodeApi,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModuleAggregate {
    /// Score each cargo-modules node (module/type/trait/function/etc).
    Node,
    /// Collapse items into their owning module path, then score that graph.
    Module,
    /// Best-effort collapse into source files (Rust module path → `src/...rs`), then score.
    ///
    /// This is heuristic: it will be wrong for `#[path = ...]` and nonstandard layouts.
    File,
}

#[derive(Parser, Debug, Clone)]
struct ModulesSweepArgs {
    /// Path to a Cargo manifest (workspace root or a single crate).
    #[arg(long, default_value = "Cargo.toml")]
    manifest_path: PathBuf,

    /// Package(s) to analyze (repeatable). If omitted, use `--all-packages`.
    #[arg(short = 'p', long)]
    package: Vec<String>,

    /// Analyze all workspace member packages under the manifest.
    #[arg(long, default_value_t = false)]
    all_packages: bool,

    /// Analyze the package library target.
    #[arg(long)]
    lib: bool,

    /// Analyze the named binary target.
    ///
    /// Note: for a sweep, specifying a single `--bin` applies to all packages (often not what you want).
    #[arg(long)]
    bin: Option<String>,

    /// Analyze with `#[cfg(test)]` enabled (as if built via `cargo test`).
    #[arg(long)]
    cfg_test: bool,

    /// Centrality metric to compute.
    #[arg(short, long, value_enum, default_value_t = Metric::Pagerank)]
    metric: Metric,

    /// Preset of common analysis settings.
    ///
    /// Use `--preset none` to disable preset application and rely only on explicit flags.
    #[arg(long, value_enum, default_value_t = ModulesPreset::None)]
    preset: ModulesPreset,

    /// Only print the compact summary table (no per-package tables).
    ///
    /// This is the “fast scan” mode for large workspaces.
    #[arg(long, default_value_t = true)]
    summary_only: bool,

    /// Continue the sweep if a package analysis fails.
    ///
    /// When enabled, failures are recorded in the summary table and (unless `--summary-only`)
    /// printed with their full error under a per-package section.
    ///
    /// Note: this is a boolean *value* (not a pure flag), because the default is `true`.
    /// Example: `--continue-on-error=false`
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    continue_on_error: bool,

    /// Convenience: stop on first package error (`--continue-on-error=false`).
    #[arg(long, default_value_t = false)]
    fail_fast: bool,

    /// Number of top rows per package to show (text output only).
    #[arg(short = 'n', long, default_value_t = 12)]
    top: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    /// Which edge kinds from cargo-modules to include.
    #[arg(long, value_enum, default_value_t = ModuleEdgeKind::Uses)]
    edge_kind: ModuleEdgeKind,

    /// Aggregate nodes before scoring.
    #[arg(long, value_enum, default_value_t = ModuleAggregate::File)]
    aggregate: ModuleAggregate,

    /// When `--edge-kind both`, weight for `uses` edges.
    #[arg(long, default_value_t = 1.0)]
    uses_weight: f64,

    /// When `--edge-kind both`, weight for `owns` edges.
    #[arg(long, default_value_t = 0.2)]
    owns_weight: f64,

    /// Show top inter-group edges (by total weight) after each package table.
    #[arg(long, default_value_t = 3)]
    edges_top: usize,

    /// When aggregating, show the top-N member items by node-level PageRank.
    #[arg(long, default_value_t = 2)]
    members_top: usize,

    /// Cache `cargo modules dependencies` DOT output on disk (default: true).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cache: bool,
    /// Force refresh cache (ignore existing cached DOT).
    #[arg(long, default_value_t = false)]
    cache_refresh: bool,

    /// Hide extern crates in the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_externs: bool,
    /// Include extern items (override for `--no-externs`).
    #[arg(long, default_value_t = false)]
    include_externs: bool,

    /// Hide sysroot crates (`std`, `core`, etc.) in the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_sysroot: bool,
    /// Include sysroot crates (override for `--no-sysroot`).
    #[arg(long, default_value_t = false)]
    include_sysroot: bool,

    /// Hide functions from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_fns: bool,
    /// Include functions (override for `--no-fns`).
    #[arg(long, default_value_t = false)]
    include_fns: bool,

    /// Hide traits from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_traits: bool,
    /// Include traits (override for `--no-traits`).
    #[arg(long, default_value_t = true)]
    include_traits: bool,

    /// Hide types (struct/enum/union) from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    no_types: bool,
    /// Include types (override for `--no-types`).
    #[arg(long, default_value_t = true)]
    include_types: bool,
}

#[derive(Parser, Debug, Clone)]
struct AnalyzeArgs {
    /// Path to a `Cargo.toml`, or a directory containing one.
    #[arg(default_value = ".")]
    path: PathBuf,

    /// Centrality metric to compute.
    #[arg(short, long, value_enum, default_value_t = Metric::Pagerank)]
    metric: Metric,

    /// Number of top rows to show (text output only).
    #[arg(short = 'n', long, default_value_t = 25)]
    top: usize,

    /// Include dev-dependencies.
    #[arg(long)]
    dev: bool,

    /// Include build-dependencies.
    #[arg(long)]
    build: bool,

    /// Restrict nodes/edges to workspace members only.
    ///
    /// This matches the common “internal crate graph” question.
    ///
    /// Note: this is a boolean *value* (not a pure flag), because the default is `true`.
    /// Examples:
    /// - `--workspace-only=false` (include third-party nodes too)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    workspace_only: bool,

    /// Pass `--all-features` to cargo metadata.
    ///
    /// Note: turning this on can fail in repos where some feature combos require
    /// extra system deps (CUDA/Metal/ONNX toolchains, etc).
    #[arg(long)]
    all_features: bool,

    /// Pass `--no-default-features` to cargo metadata.
    #[arg(long)]
    no_default_features: bool,

    /// Pass `--features <...>` to cargo metadata (comma- or space-separated).
    ///
    /// Example: `--features "cli eval"`
    #[arg(long)]
    features: Option<String>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    /// Print timing + size stats to stderr.
    ///
    /// This is modeled after ripgrep's `--stats`: it does not change the primary
    /// stdout output, it only adds a summary to stderr.
    #[arg(long, default_value_t = false)]
    stats: bool,

    /// Limit rows in JSON output (default: unlimited).
    ///
    /// This is intentionally separate from `-n/--top` (which is text-only) to avoid
    /// surprising existing scripts that pass `-n` while consuming JSON.
    #[arg(long)]
    json_limit: Option<usize>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Metric {
    Pagerank,
    /// PageRank on the reversed graph (“who is an orchestrator / top-level consumer?”).
    ConsumersPagerank,
    Indegree,
    Outdegree,
    Betweenness,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
enum SweepMode {
    /// Run once on the root workspace and slice results per repo using manifest_path.
    ///
    /// Fast, and covers repos that don’t have a `Cargo.toml` at repo root.
    WorkspaceSlice,
    /// Attempt repo-local `cargo metadata` in each repo root.
    ///
    /// Slower and will fail if individual repos have broken workspace metadata.
    RepoRoots,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
enum ViewMode {
    /// Only local workspace slice.
    Local,
    /// Only crates.io crawl.
    CratesIo,
    /// Both local slice and crates.io crawl.
    Both,
}

#[derive(Parser, Debug, Clone)]
struct SweepLocalArgs {
    /// Root directory that contains repos (your dev super-workspace).
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Output directory for artifacts.
    #[arg(long, default_value = "evals/pkgrank")]
    out: PathBuf,

    /// Optional JSON file describing your repo set.
    ///
    /// Defaults to `evals/arch/dev_repos_overview.json` if it exists under `--root`.
    #[arg(long)]
    overview: Option<PathBuf>,

    /// How to compute “per-repo” results.
    #[arg(long, value_enum, default_value_t = SweepMode::WorkspaceSlice)]
    mode: SweepMode,

    /// Top-N rows per repo text artifact.
    #[arg(short = 'n', long, default_value_t = 15)]
    top: usize,

    /// Include dev-dependencies (cargo graph edges).
    #[arg(long)]
    dev: bool,

    /// Include build-dependencies (cargo graph edges).
    #[arg(long)]
    build: bool,

    /// Also write “recently modified files” artifacts (mtime-based; bounded scan).
    ///
    /// This is meant to answer: “what changed recently, and is it central?”
    /// without requiring Git history (the dev super-workspace is not a single repo).
    #[arg(long, default_value_t = false)]
    recent: bool,

    /// Window for recent file scan (days).
    #[arg(long, default_value_t = 14)]
    recent_days: u64,

    /// Max rows to write in `recent.files.json`.
    #[arg(long, default_value_t = 200)]
    recent_max: usize,
}

#[derive(Parser, Debug, Clone)]
struct CratesIoArgs {
    /// Root directory containing your Cargo workspace (used to discover seed crates by default).
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Seed crate names (repeatable). If omitted, we use all workspace crates that exist on crates.io.
    #[arg(long)]
    seed: Vec<String>,

    /// Max BFS depth from seed crates.
    #[arg(long, default_value_t = 2)]
    depth: usize,

    /// Include dev-dependencies from crates.io dependency listings.
    #[arg(long)]
    dev: bool,

    /// Include build-dependencies from crates.io dependency listings.
    #[arg(long)]
    build: bool,

    /// Include optional dependencies.
    #[arg(long)]
    optional: bool,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,

    /// Output directory for artifacts (when using JSON output).
    #[arg(long, default_value = "evals/pkgrank")]
    out: PathBuf,

    /// Do not print results to stdout (still writes artifacts).
    #[arg(long)]
    quiet: bool,
}

#[derive(Parser, Debug, Clone)]
struct ViewArgs {
    /// Root directory containing your dev super-workspace.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Output directory for artifacts (relative to root if not absolute).
    #[arg(long, default_value = "evals/pkgrank")]
    out: PathBuf,

    /// How to run the view.
    #[arg(long, value_enum, default_value_t = ViewMode::Both)]
    mode: ViewMode,

    /// Top-N rows per repo (local view).
    #[arg(long, default_value_t = 10)]
    local_top: usize,

    /// crates.io BFS depth from seed crates.
    #[arg(long, default_value_t = 2)]
    cratesio_depth: usize,

    /// Include dev-dependencies in crates.io crawl.
    #[arg(long)]
    cratesio_dev: bool,

    /// Include build-dependencies in crates.io crawl.
    #[arg(long)]
    cratesio_build: bool,

    /// Include optional dependencies in crates.io crawl.
    #[arg(long)]
    cratesio_optional: bool,

    /// Do not print crates.io rows to stdout during view.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    quiet: bool,
}

#[derive(Parser, Debug, Clone)]
struct TriageCliArgs {
    /// Root directory containing the dev super-workspace.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Artifact directory (relative to root if not absolute).
    #[arg(long, default_value = "evals/pkgrank")]
    out: PathBuf,

    /// If required artifacts are missing, re-run `pkgrank view` first.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    refresh_if_missing: bool,

    /// View mode used for refresh.
    #[arg(long, value_enum, default_value_t = ViewMode::Local)]
    mode: ViewMode,

    /// Mark artifacts stale if older than this many minutes. Defaults to 60.
    #[arg(long, default_value_t = 60)]
    stale_minutes: u64,

    /// Limit returned rows.
    #[arg(long, default_value_t = 15)]
    limit: usize,

    /// Optional axis filter for TLC tables.
    #[arg(long)]
    axis: Option<String>,

    /// Include PPR aggregate top-k.
    #[arg(long, default_value_t = 12)]
    ppr_top: usize,

    /// Summarize READMEs (bounded by the *_top_* limits below). Default: false.
    #[arg(long, default_value_t = false)]
    summarize_readmes: bool,

    /// Number of top repos (post-filter) to summarize. Default: 0.
    #[arg(long, default_value_t = 0)]
    summarize_repos_top: usize,

    /// Number of top crates (post-filter) to summarize. Default: 0.
    #[arg(long, default_value_t = 0)]
    summarize_crates_top: usize,

    /// Max chars of README fed into LLM (default: 12000).
    #[arg(long, default_value_t = 12_000)]
    llm_input_max_chars: usize,

    /// Timeout seconds for the LLM command (default: 30).
    #[arg(long, default_value_t = 30)]
    llm_timeout_secs: u64,

    /// Cache summaries under `<out>/readme_ai_cache/` (default: true).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    llm_cache: bool,

    /// Include raw LLM output in triage results (default: false).
    #[arg(long, default_value_t = false)]
    llm_include_raw: bool,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PackageOrigin {
    /// A workspace member (first-party in the current Cargo workspace).
    WorkspaceMember,
    /// A non-workspace local path dependency (first-party-ish, but outside the workspace set).
    Path,
    /// A crates.io registry dependency.
    Registry,
    /// A git dependency.
    Git,
    /// Unknown / other.
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Row {
    id: String,
    name: String,
    version: String,
    manifest_path: String,
    origin: PackageOrigin,
    in_degree: usize,
    out_degree: usize,
    pagerank: f64,
    consumers_pagerank: f64,
    betweenness: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TlcCrateRow {
    repo: String,
    axis: String,
    name: String,
    manifest_path: String,
    origin: PackageOrigin,
    pagerank: f64,
    betweenness: f64,
    transitive_dependents: usize,
    transitive_dependencies: usize,
    third_party_deps: usize,
    score: f64,
    why: String,
    repo_git_commits_30d: Option<u64>,
    repo_git_days_since_last_commit: Option<u64>,
}

#[cfg(feature = "stdio")]
fn view_mode_str(m: ViewMode) -> &'static str {
    match m {
        ViewMode::Local => "local",
        ViewMode::CratesIo => "cratesio",
        ViewMode::Both => "both",
    }
}

#[cfg(feature = "stdio")]
async fn triage_payload_from_cli(args: &TriageCliArgs) -> Result<(serde_json::Value, String)> {
    let root = args.root.clone();
    let out = args.out.clone();
    let refresh_if_missing = args.refresh_if_missing;
    let stale_minutes = args.stale_minutes.min(60 * 24 * 30);
    let limit = args.limit.min(100);
    let axis = args.axis.clone();
    let ppr_top = args.ppr_top.min(50);
    let summarize_readmes = args.summarize_readmes;
    let summarize_repos_top = args.summarize_repos_top.min(25);
    let summarize_crates_top = args.summarize_crates_top.min(25);
    let llm_input_max_chars = args.llm_input_max_chars.min(80_000);
    let llm_timeout_secs = args.llm_timeout_secs.min(600);
    let llm_cache = args.llm_cache;
    let llm_include_raw = args.llm_include_raw;
    let mode = args.mode;

    let tlc_crates_path = PkgrankStdioMcpFull::artifact_path(&root, &out, "tlc.crates.json");
    let tlc_repos_path = PkgrankStdioMcpFull::artifact_path(&root, &out, "tlc.repos.json");
    let inv_path =
        PkgrankStdioMcpFull::artifact_path(&root, &out, "ecosystem.invariants.violations.json");
    let ppr_agg_path = PkgrankStdioMcpFull::artifact_path(&root, &out, "ppr.aggregate.json");

    let required = [
        ("tlc.crates.json", &tlc_crates_path),
        ("tlc.repos.json", &tlc_repos_path),
        ("ecosystem.invariants.violations.json", &inv_path),
        ("ppr.aggregate.json", &ppr_agg_path),
    ];
    let mut missing = Vec::new();
    for (name, p) in required {
        if !p.exists() {
            missing.push(name.to_string());
        }
    }
    let missing_before_refresh = missing.clone();

    let mut refreshed = false;
    if refresh_if_missing && !missing.is_empty() {
        let args = ViewArgs {
            root: root.clone(),
            out: out.clone(),
            mode,
            local_top: 10,
            cratesio_depth: 2,
            cratesio_dev: false,
            cratesio_build: false,
            cratesio_optional: false,
            quiet: true,
        };
        run_view(&args)?;
        refreshed = true;
        missing.clear();
        for (name, p) in [
            ("tlc.crates.json", &tlc_crates_path),
            ("tlc.repos.json", &tlc_repos_path),
            ("ecosystem.invariants.violations.json", &inv_path),
            ("ppr.aggregate.json", &ppr_agg_path),
        ] {
            if !p.exists() {
                missing.push(name.to_string());
            }
        }
    }

    let mut crates: Vec<TlcCrateRow> =
        PkgrankStdioMcpFull::read_json_file(&tlc_crates_path).map_err(|e| anyhow!(e.message))?;
    let mut repos: Vec<TlcRepoRow> =
        PkgrankStdioMcpFull::read_json_file(&tlc_repos_path).map_err(|e| anyhow!(e.message))?;
    let violations: Vec<RepoInvariantViolation> =
        PkgrankStdioMcpFull::read_json_file(&inv_path).map_err(|e| anyhow!(e.message))?;
    let ppr_agg: Vec<(String, f64)> =
        PkgrankStdioMcpFull::read_json_file(&ppr_agg_path).map_err(|e| anyhow!(e.message))?;

    if let Some(ax) = axis.as_ref() {
        crates.retain(|r| &r.axis == ax);
        repos.retain(|r| &r.axis == ax);
    }

    let violations_sample = violations.iter().take(10).collect::<Vec<_>>();
    let violation_rules_top = summarize_violation_rules(&violations, 5);
    let artifacts = serde_json::json!({
        "tlc_crates": {
            "path": tlc_crates_path.display().to_string(),
            "age_minutes": file_age_minutes(&tlc_crates_path),
            "stale": file_age_minutes(&tlc_crates_path).map(|m| m >= stale_minutes),
        },
        "tlc_repos": {
            "path": tlc_repos_path.display().to_string(),
            "age_minutes": file_age_minutes(&tlc_repos_path),
            "stale": file_age_minutes(&tlc_repos_path).map(|m| m >= stale_minutes),
        },
        "invariants": {
            "path": inv_path.display().to_string(),
            "age_minutes": file_age_minutes(&inv_path),
            "stale": file_age_minutes(&inv_path).map(|m| m >= stale_minutes),
        },
        "ppr_aggregate": {
            "path": ppr_agg_path.display().to_string(),
            "age_minutes": file_age_minutes(&ppr_agg_path),
            "stale": file_age_minutes(&ppr_agg_path).map(|m| m >= stale_minutes),
        },
    });
    let artifact_ages = [
        file_age_minutes(&tlc_crates_path),
        file_age_minutes(&tlc_repos_path),
        file_age_minutes(&inv_path),
        file_age_minutes(&ppr_agg_path),
    ];
    let max_age = artifact_ages.iter().copied().flatten().max();
    let stale_any = artifact_ages
        .iter()
        .copied()
        .flatten()
        .any(|m| m >= stale_minutes);

    let mut readme_ai_repos: Vec<serde_json::Value> = Vec::new();
    let mut readme_ai_crates: Vec<serde_json::Value> = Vec::new();
    if summarize_readmes && (summarize_repos_top > 0 || summarize_crates_top > 0) {
        for r in repos.iter().take(summarize_repos_top) {
            let readme_path = find_readme_for_repo(&root, &r.repo);
            let ai = maybe_add_readme_llm_summary(
                &root,
                &out,
                "repo",
                &r.repo,
                readme_path.as_ref(),
                true,
                llm_input_max_chars,
                llm_timeout_secs,
                llm_cache,
            )
            .await?;
            let parsed = ai.get("parsed").cloned().unwrap_or(serde_json::Value::Null);
            let raw = ai.get("raw").cloned().unwrap_or(serde_json::Value::Null);
            readme_ai_repos.push(serde_json::json!({
                "repo": r.repo,
                "axis": r.axis,
                "tlc_score": r.score,
                "available": ai.get("available").cloned().unwrap_or(serde_json::Value::Null),
                "cached": ai.get("cached").cloned().unwrap_or(serde_json::Value::Null),
                "reason": ai.get("reason").cloned().unwrap_or(serde_json::Value::Null),
                "parsed": parsed,
                "raw": if llm_include_raw { raw } else { serde_json::Value::Null },
            }));
        }
        for c in crates.iter().take(summarize_crates_top) {
            let readme_path = find_readme_for_manifest(&root, &c.manifest_path);
            let ai = maybe_add_readme_llm_summary(
                &root,
                &out,
                "crate",
                &c.name,
                readme_path.as_ref(),
                true,
                llm_input_max_chars,
                llm_timeout_secs,
                llm_cache,
            )
            .await?;
            let parsed = ai.get("parsed").cloned().unwrap_or(serde_json::Value::Null);
            let raw = ai.get("raw").cloned().unwrap_or(serde_json::Value::Null);
            readme_ai_crates.push(serde_json::json!({
                "crate": c.name,
                "repo": c.repo,
                "axis": c.axis,
                "tlc_score": c.score,
                "available": ai.get("available").cloned().unwrap_or(serde_json::Value::Null),
                "cached": ai.get("cached").cloned().unwrap_or(serde_json::Value::Null),
                "reason": ai.get("reason").cloned().unwrap_or(serde_json::Value::Null),
                "parsed": parsed,
                "raw": if llm_include_raw { raw } else { serde_json::Value::Null },
            }));
        }
    }

    let payload = serde_json::json!({
        "ok": true,
        "root": root.display().to_string(),
        "out_dir": if out.is_absolute() { out.display().to_string() } else { root.join(&out).display().to_string() },
        "artifacts": artifacts,
        "filters": {
            "axis": axis,
            "limit": limit,
            "ppr_top": ppr_top,
            "stale_minutes": stale_minutes,
            "refresh_if_missing": refresh_if_missing,
            "mode": view_mode_str(mode),
            "summarize_readmes": summarize_readmes,
            "summarize_repos_top": summarize_repos_top,
            "summarize_crates_top": summarize_crates_top,
            "llm_input_max_chars": llm_input_max_chars,
            "llm_timeout_secs": llm_timeout_secs,
            "llm_cache": llm_cache,
            "llm_include_raw": llm_include_raw,
        },
        "summary": {
            "text": {
                "top_repos": format_top_tlc_repos(&repos, 8),
                "top_crates": format_top_tlc_crates(&crates, 8),
            },
            "staleness": {
                "stale_any": stale_any,
                "max_age_minutes": max_age,
            },
            "violations": violations.len(),
            "violations_sample": violations_sample,
            "violation_rules_top": violation_rules_top,
            "top_ppr": ppr_agg.iter().take(ppr_top).collect::<Vec<_>>(),
            "refreshed": refreshed,
            "missing_before_refresh": missing_before_refresh,
            "missing_after_refresh": missing,
        },
        "readme_ai": {
            "repos": readme_ai_repos,
            "crates": readme_ai_crates,
        },
        "tlc": {
            "crates": crates.into_iter().take(limit).collect::<Vec<_>>(),
            "repos": repos.into_iter().take(limit).collect::<Vec<_>>(),
        }
    });

    let mut summary = String::new();
    let _ = writeln!(
        &mut summary,
        "pkgrank triage (axis={:?}) stale_any={} max_age_minutes={:?} refreshed={} missing_after_refresh={:?}",
        args.axis,
        stale_any,
        max_age,
        refreshed,
        payload
            .get("summary")
            .and_then(|s| s.get("missing_after_refresh"))
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    );
    let _ = writeln!(&mut summary);
    let _ = writeln!(&mut summary, "Top repos:");
    let _ = writeln!(
        &mut summary,
        "{}",
        payload
            .get("summary")
            .and_then(|s| s.get("text"))
            .and_then(|t| t.get("top_repos"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
    );
    let _ = writeln!(&mut summary);
    let _ = writeln!(&mut summary, "Top crates:");
    let _ = writeln!(
        &mut summary,
        "{}",
        payload
            .get("summary")
            .and_then(|s| s.get("text"))
            .and_then(|t| t.get("top_crates"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
    );
    let _ = writeln!(&mut summary);
    let _ = writeln!(
        &mut summary,
        "Invariant rule counts (top): {:?}",
        violation_rules_top
    );

    Ok((payload, summary))
}

fn run_triage(args: &TriageCliArgs) -> Result<()> {
    #[cfg(feature = "stdio")]
    {
        let rt = tokio::runtime::Runtime::new().context("failed to build tokio runtime")?;
        let (payload, summary_text) = rt.block_on(triage_payload_from_cli(args))?;
        match args.format {
            OutputFormat::Json => {
                let out = serde_json::json!({
                    "schema_version": 1,
                    "ok": true,
                    "command": "triage",
                    "summary_text": summary_text,
                    "result": payload,
                });
                println!("{}", serde_json::to_string_pretty(&out)?);
            }
            OutputFormat::Text => {
                print!("{summary_text}");
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "stdio"))]
    {
        let _ = args;
        anyhow::bail!("triage requires feature `stdio` (rebuild pkgrank with default features)");
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let (command_name, result): (&str, Result<()>) = match cli.command {
        None => ("analyze", run_analyze(&cli.analyze)),
        Some(Command::SweepLocal(args)) => ("sweep-local", run_sweep_local(&args)),
        Some(Command::CratesIo(args)) => ("cratesio", run_cratesio(&args)),
        Some(Command::Modules(args)) => ("modules", run_modules(&args)),
        Some(Command::ModulesSweep(args)) => ("modules-sweep", run_modules_sweep(&args)),
        Some(Command::View(args)) => ("view", run_view(&args)),
        Some(Command::Triage(args)) => ("triage", run_triage(&args)),
        Some(Command::McpStdio) => ("mcp-stdio", run_mcp_stdio()),
    };

    let _ = command_name; // reserved for future CLI stats/logging
    result?;

    Ok(())
}

fn run_modules_sweep(args: &ModulesSweepArgs) -> Result<()> {
    let continue_on_error = if args.fail_fast {
        false
    } else {
        args.continue_on_error
    };

    // Resolve package list:
    let packages: Vec<String> = if !args.package.is_empty() {
        args.package.clone()
    } else if args.all_packages {
        let mut cmd = MetadataCommand::new();
        cmd.manifest_path(&args.manifest_path);
        let md = cmd
            .exec()
            .map_err(|e| anyhow!(e))
            .with_context(|| "cargo metadata failed for modules-sweep")?;
        let mut names: Vec<String> = md
            .workspace_members
            .iter()
            .filter_map(|id| {
                md.packages
                    .iter()
                    .find(|p| &p.id == id)
                    .map(|p| p.name.to_string())
            })
            .collect();
        names.sort();
        names
    } else {
        return Err(anyhow!(
            "modules-sweep requires at least one -p/--package, or --all-packages"
        ));
    };

    if packages.is_empty() {
        return Err(anyhow!("no packages selected"));
    }

    // Run once per package; reuse outputs for both summary + per-package details.
    #[derive(Debug)]
    struct SweepOneOk {
        pkg: String,
        rows: Vec<ModuleRow>,
        nodes: usize,
        edges: usize,
        aggregate_label: String,
        top_edges: Vec<(String, String, f64)>,
        args: ModulesArgs,
    }

    #[derive(Debug)]
    struct SweepOneErr {
        pkg: String,
        err: String,
    }

    #[derive(Debug)]
    enum SweepOne {
        Ok(SweepOneOk),
        Err(SweepOneErr),
    }

    // Apply preset to sweep defaults by translating to a per-package ModulesArgs template.
    let template = ModulesArgs {
        manifest_path: args.manifest_path.clone(),
        package: None,
        lib: args.lib,
        bin: args.bin.clone(),
        cfg_test: args.cfg_test,
        metric: args.metric,
        preset: args.preset,
        top: args.top,
        format: args.format,
        edge_kind: args.edge_kind,
        aggregate: args.aggregate,
        uses_weight: args.uses_weight,
        owns_weight: args.owns_weight,
        edges_top: args.edges_top,
        members_top: args.members_top,
        cache: args.cache,
        cache_refresh: args.cache_refresh,
        no_externs: args.no_externs,
        include_externs: args.include_externs,
        no_sysroot: args.no_sysroot,
        include_sysroot: args.include_sysroot,
        no_fns: args.no_fns,
        include_fns: args.include_fns,
        no_traits: args.no_traits,
        include_traits: args.include_traits,
        no_types: args.no_types,
        include_types: args.include_types,
    };
    let template = apply_modules_preset(&template);

    let mut results: Vec<SweepOne> = Vec::new();
    for pkg in &packages {
        let mut one_args = template.clone();
        one_args.package = Some(pkg.clone());
        match run_modules_core(&one_args) {
            Ok((rows, nodes, edges, aggregate_label, top_edges)) => {
                results.push(SweepOne::Ok(SweepOneOk {
                    pkg: pkg.clone(),
                    rows,
                    nodes,
                    edges,
                    aggregate_label,
                    top_edges,
                    args: one_args,
                }));
            }
            Err(e) => {
                if !continue_on_error {
                    return Err(e)
                        .with_context(|| format!("modules-sweep failed for package `{pkg}`"));
                }
                results.push(SweepOne::Err(SweepOneErr {
                    pkg: pkg.clone(),
                    err: format!("{:#}", e),
                }));
            }
        }
    }

    match args.format {
        OutputFormat::Json => {
            #[derive(Debug, Serialize)]
            struct ModulesSweepPackageOut {
                ok: bool,
                error: Option<String>,
                nodes: Option<usize>,
                edges: Option<usize>,
                aggregate_label: Option<String>,
                rows: Option<Vec<ModuleRow>>,
            }

            #[derive(Debug, Serialize)]
            struct ModulesSweepOut {
                schema_version: u32,
                ok: bool,
                command: &'static str,
                manifest_path: String,
                preset: Option<String>,
                effective: ModulesSweepEffective,
                packages: HashMap<String, ModulesSweepPackageOut>,
            }

            #[derive(Debug, Serialize)]
            struct ModulesSweepEffective {
                aggregate: String,
                edge_kind: String,
                include_fns: bool,
                include_types: bool,
                include_traits: bool,
                cache: bool,
                cache_refresh: bool,
            }

            let mut pkgs: HashMap<String, ModulesSweepPackageOut> = HashMap::new();
            for r in results {
                match r {
                    SweepOne::Ok(ok) => {
                        pkgs.insert(
                            ok.pkg,
                            ModulesSweepPackageOut {
                                ok: true,
                                error: None,
                                nodes: Some(ok.nodes),
                                edges: Some(ok.edges),
                                aggregate_label: Some(ok.aggregate_label),
                                rows: Some(ok.rows),
                            },
                        );
                    }
                    SweepOne::Err(er) => {
                        pkgs.insert(
                            er.pkg,
                            ModulesSweepPackageOut {
                                ok: false,
                                error: Some(er.err),
                                nodes: None,
                                edges: None,
                                aggregate_label: None,
                                rows: None,
                            },
                        );
                    }
                }
            }

            let out = ModulesSweepOut {
                schema_version: 1,
                ok: pkgs.values().all(|p| p.ok),
                command: "modules-sweep",
                manifest_path: args.manifest_path.display().to_string(),
                preset: match args.preset {
                    ModulesPreset::None => None,
                    p => Some(format!("{p:?}")),
                },
                effective: ModulesSweepEffective {
                    aggregate: format!("{:?}", template.aggregate),
                    edge_kind: format!("{:?}", template.edge_kind),
                    include_fns: template.include_fns,
                    include_types: template.include_types,
                    include_traits: template.include_traits,
                    cache: template.cache,
                    cache_refresh: template.cache_refresh,
                },
                packages: pkgs,
            };

            println!("{}", serde_json::to_string_pretty(&out)?);
            Ok(())
        }
        OutputFormat::Text => {
            println!("pkgrank modules-sweep");
            println!("  manifest: {}", args.manifest_path.display());
            println!("  packages: {}  ({})", packages.len(), packages.join(", "));
            if !matches!(args.preset, ModulesPreset::None) {
                println!("  preset:   {:?}", args.preset);
            }
            println!(
                "  effective: aggregate={:?} edge_kind={:?} include=[fns:{} types:{} traits:{}] cache={} refresh={}",
                template.aggregate,
                template.edge_kind,
                template.include_fns,
                template.include_types,
                template.include_traits,
                template.cache,
                template.cache_refresh
            );
            println!(
                "  mode: summary_only={}  continue_on_error={}\n",
                args.summary_only, continue_on_error
            );

            // Compact summary table.
            let header = format!(
                "{:<14} {:>6} {:>5} {:>5}  {:>10}  {:>10}  {:>10}  {}",
                "package", "status", "nodes", "edges", "top_pr", "top_cons", "top_between", "error"
            );
            println!("{header}");
            println!("{:─<width$}", "", width = header.chars().count());
            for r in &results {
                match r {
                    SweepOne::Ok(ok) => {
                        let top_pr = ok
                            .rows
                            .iter()
                            .max_by(|a, b| {
                                a.pagerank
                                    .partial_cmp(&b.pagerank)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|x| format!("{}({:.3})", x.node, x.pagerank))
                            .unwrap_or_else(|| "-".to_string());
                        let top_cons = ok
                            .rows
                            .iter()
                            .max_by(|a, b| {
                                a.consumers_pagerank
                                    .partial_cmp(&b.consumers_pagerank)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|x| format!("{}({:.3})", x.node, x.consumers_pagerank))
                            .unwrap_or_else(|| "-".to_string());
                        let top_between = ok
                            .rows
                            .iter()
                            .max_by(|a, b| {
                                a.betweenness
                                    .partial_cmp(&b.betweenness)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|x| format!("{}({:.3})", x.node, x.betweenness))
                            .unwrap_or_else(|| "-".to_string());
                        println!(
                            "{:<14} {:>6} {:>5} {:>5}  {:>10}  {:>10}  {:>10}",
                            ok.pkg,
                            "ok",
                            ok.nodes,
                            ok.edges,
                            truncate_cell(&top_pr, 10),
                            truncate_cell(&top_cons, 10),
                            truncate_cell(&top_between, 10)
                        );
                    }
                    SweepOne::Err(er) => {
                        let first_line = er.err.lines().next().unwrap_or(&er.err);
                        println!(
                            "{:<14} {:>6} {:>5} {:>5}  {:>10}  {:>10}  {:>10}  {}",
                            er.pkg,
                            "err",
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                            truncate_cell(first_line, 60)
                        );
                    }
                }
            }

            if args.summary_only {
                Ok(())
            } else {
                // Full per-package sections.
                for r in &results {
                    match r {
                        SweepOne::Ok(ok) => {
                            println!("\n{:═<110}", "");
                            println!("package: {}", ok.pkg);
                            print_modules_text(
                                &ok.args,
                                &ok.rows,
                                ok.nodes,
                                ok.edges,
                                &ok.aggregate_label,
                                &ok.top_edges,
                            );
                        }
                        SweepOne::Err(er) => {
                            println!("\n{:═<110}", "");
                            println!("package: {}", er.pkg);
                            println!("status:  error");
                            println!("error:\n{}", er.err.trim());
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

fn truncate_cell(s: &str, max: usize) -> String {
    // Keep table stable-width without hiding the real data elsewhere (full tables follow).
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    for c in s.chars().take(max.saturating_sub(1)) {
        out.push(c);
    }
    out.push('…');
    out
}

#[derive(Debug, Serialize)]
struct ModuleRow {
    /// Node identifier (or an aggregate key).
    node: String,
    /// Aggregate kind (node/module/file), for interpretability.
    aggregate: String,
    /// Node kind, when available (crate/mod/struct/enum/trait/fn/...).
    kind: Option<String>,
    /// Visibility string, when available (`pub`, `pub(crate)`, `pub(self)`, ...).
    visibility: Option<String>,
    /// Group size (only meaningful when aggregate != node).
    group_size: Option<usize>,
    /// Small member preview (only meaningful when aggregate != node).
    members_preview: Option<String>,
    /// Small “top members” preview (aggregate != node): member items with highest node-level PageRank.
    top_members_pr: Option<String>,
    /// Transitive dependencies within the chosen graph (reachability following A → B).
    transitive_dependencies: usize,
    /// Transitive dependents within the chosen graph (reachability in reversed graph).
    transitive_dependents: usize,
    in_degree: usize,
    out_degree: usize,
    pagerank: f64,
    consumers_pagerank: f64,
    betweenness: f64,
}

#[derive(Debug, Clone)]
struct CargoModulesNodeMeta {
    kind: Option<String>,
    visibility: Option<String>,
}

fn run_modules(args: &ModulesArgs) -> Result<()> {
    // Printing wrapper around `run_modules_rows`.
    let eff = apply_modules_preset(args);
    let (rows, nodes, edges, aggregate_label, top_edges) = run_modules_core(&eff)?;

    match eff.format {
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct ModulesJsonOut<'a> {
                schema_version: u32,
                ok: bool,
                command: &'a str,
                rows: Vec<ModuleRow>,
            }
            let out = ModulesJsonOut {
                schema_version: 1,
                ok: true,
                command: "modules",
                rows,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Text => {
            print_modules_text(&eff, &rows, nodes, edges, &aggregate_label, &top_edges);
        }
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn run_modules_core(
    args: &ModulesArgs,
) -> Result<(
    Vec<ModuleRow>,
    usize,
    usize,
    String,
    Vec<(String, String, f64)>,
)> {
    // We intentionally shell out to cargo-modules:
    // - avoids embedding rust-analyzer crates into pkgrank
    // - keeps pkgrank MIT/Apache; cargo-modules is MPL-2.0
    let mut cmd = ProcessCommand::new("cargo");
    cmd.args(["modules", "dependencies"]);
    cmd.args(["--manifest-path", &args.manifest_path.to_string_lossy()]);

    // cargo-modules requires --package when `--manifest-path` points at a workspace.
    // Do not guess a package name: it's often wrong (workspace roots, repo folder names, etc).
    // Instead, fail fast with an actionable message.
    let selected_pkg: Option<String> = args.package.clone();
    if selected_pkg.is_none() {
        // Cheap heuristic: if the manifest text contains `[workspace]`, cargo-modules will require `-p`.
        // Even when it doesn't, requiring an explicit `package` is clearer than guessing.
        if let Ok(raw) = fs::read_to_string(&args.manifest_path) {
            if raw.contains("[workspace]") {
                anyhow::bail!(
                    "cargo-modules requires an explicit package when analyzing a workspace manifest.\n\
                     Fix: pass `--package <crate>` (or point `--manifest-path` at the crate's Cargo.toml)."
                );
            }
        }
    }
    if let Some(pkg) = &selected_pkg {
        cmd.args(["-p", pkg]);
    }
    if args.lib {
        cmd.arg("--lib");
    }
    if let Some(bin) = &args.bin {
        cmd.args(["--bin", bin]);
    }
    if args.cfg_test {
        cmd.arg("--cfg-test");
    }

    // Selection filters:
    let no_externs = if args.include_externs {
        false
    } else {
        args.no_externs
    };
    let no_sysroot = if args.include_sysroot {
        false
    } else {
        args.no_sysroot
    };
    let no_fns = if args.include_fns { false } else { args.no_fns };
    let no_traits = if args.include_traits {
        false
    } else {
        args.no_traits
    };
    let no_types = if args.include_types {
        false
    } else {
        args.no_types
    };

    if no_externs {
        cmd.arg("--no-externs");
    }
    if no_sysroot {
        cmd.arg("--no-sysroot");
    }
    if no_fns {
        cmd.arg("--no-fns");
    }
    if no_traits {
        cmd.arg("--no-traits");
    }
    if no_types {
        cmd.arg("--no-types");
    }

    // Edge kinds:
    match args.edge_kind {
        ModuleEdgeKind::Uses => {
            // keep uses edges, drop owns edges
            cmd.arg("--no-owns");
        }
        ModuleEdgeKind::Owns => {
            // keep owns edges, drop uses edges
            cmd.arg("--no-uses");
        }
        ModuleEdgeKind::Both => {}
    }

    let dot = cargo_modules_dot_cached(args, &cmd, selected_pkg.as_deref())?;
    let (node_names, edges, node_meta) = parse_cargo_modules_dot(&dot, args.edge_kind);
    let node_index_by_name: HashMap<String, usize> = node_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Build node-level graph first.
    let mut g: DiGraph<String, f64> = DiGraph::new();
    let mut idx: Vec<NodeIndex> = Vec::with_capacity(node_names.len());
    for n in &node_names {
        idx.push(g.add_node(n.clone()));
    }
    for e in edges {
        let w = match (args.edge_kind, e.label.as_str()) {
            (ModuleEdgeKind::Both, "uses") => args.uses_weight,
            (ModuleEdgeKind::Both, "owns") => args.owns_weight,
            (_, _) => 1.0,
        };
        if w > 0.0 {
            g.update_edge(idx[e.u], idx[e.v], w);
        }
    }

    // Node-level centralities (used for “top members” previews during aggregation).
    // Note: when aggregating, these remain useful because they identify which items
    // inside a file/module are carrying the coupling.
    let node_pr = pagerank_auto(&g);

    // Optionally aggregate.
    let (g2, members_map, aggregate_label) = if matches!(args.aggregate, ModuleAggregate::Node) {
        (g, None, "node".to_string())
    } else if matches!(args.aggregate, ModuleAggregate::Module) {
        let (ng, labels) = contract_graph(&g, owning_module);
        (ng, Some(labels), "module".to_string())
    } else {
        // File aggregation must be anchored to the *package*, not the workspace root.
        let crate_name = selected_pkg.clone().unwrap_or_else(|| "crate".to_string());
        let crate_dir =
            resolve_package_dir(&args.manifest_path, &crate_name).unwrap_or_else(|| {
                args.manifest_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            });
        let root_file = infer_rust_crate_root_file(&crate_dir, args, &crate_name);
        let (ng, labels) = contract_graph(&g, |name| {
            let module = owning_module(name);
            module_to_file_key(&crate_dir, &crate_name, &root_file, &module)
        });
        (ng, Some(labels), "file".to_string())
    };

    // Compute transitive reachability counts (blast radius within the chosen graph).
    let mut edge_pairs: Vec<(usize, usize)> = Vec::new();
    for e in g2.edge_references() {
        edge_pairs.push((e.source().index(), e.target().index()));
    }
    let (transitive_dependents, transitive_dependencies) =
        reachability_counts_edges(g2.node_count(), &edge_pairs);

    // Compute centralities on the chosen graph.
    let pr = pagerank_auto(&g2);
    let consumers_pr = pagerank_auto(&reverse_graph(&g2));
    let bc = betweenness_centrality(&g2);

    let mut rows: Vec<ModuleRow> = g2
        .node_indices()
        .map(|n| {
            let in_degree = g2.neighbors_directed(n, Direction::Incoming).count();
            let out_degree = g2.neighbors_directed(n, Direction::Outgoing).count();
            let node = g2.node_weight(n).expect("node weight").clone();
            let (kind, visibility, group_size, members_preview, top_members_pr) =
                match (&args.aggregate, &members_map) {
                    (ModuleAggregate::Node, _) => {
                        let meta = node_meta.get(&node);
                        (
                            meta.and_then(|m| m.kind.clone()),
                            meta.and_then(|m| m.visibility.clone()),
                            None,
                            None,
                            None,
                        )
                    }
                    (_, Some(map)) => {
                        let members = map.get(&node).cloned().unwrap_or_default();
                        let group_size = members.len();
                        let mut preview = String::new();
                        for (i, m) in members.iter().take(3).enumerate() {
                            if i > 0 {
                                preview.push_str(", ");
                            }
                            preview.push_str(m);
                        }
                        if group_size > 3 {
                            preview.push_str(&format!(", …(+{})", group_size - 3));
                        }
                        // Top members by node-level PageRank.
                        let mut scored: Vec<(&str, f64)> = members
                            .iter()
                            .filter_map(|m| {
                                node_index_by_name.get(m).map(|&i| (m.as_str(), node_pr[i]))
                            })
                            .collect();
                        scored.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let mut tops = String::new();
                        for (i, (m, s)) in scored.into_iter().take(args.members_top).enumerate() {
                            if i > 0 {
                                tops.push_str(", ");
                            }
                            tops.push_str(&format!("{}({:.4})", m, s));
                        }
                        let top_members_pr = if tops.is_empty() { None } else { Some(tops) };

                        (None, None, Some(group_size), Some(preview), top_members_pr)
                    }
                    _ => (None, None, None, None, None),
                };
            ModuleRow {
                node,
                aggregate: aggregate_label.clone(),
                kind,
                visibility,
                group_size,
                members_preview,
                top_members_pr,
                transitive_dependencies: transitive_dependencies[n.index()],
                transitive_dependents: transitive_dependents[n.index()],
                in_degree,
                out_degree,
                pagerank: pr[n.index()],
                consumers_pagerank: consumers_pr[n.index()],
                betweenness: bc[n.index()],
            }
        })
        .collect();

    // Default ordering: dependency PageRank.
    rows.sort_by(|a, b| {
        b.pagerank
            .partial_cmp(&a.pagerank)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut top_edges: Vec<(String, String, f64)> = Vec::new();
    if args.edges_top > 0 {
        for e in g2.edge_references() {
            let u = g2.node_weight(e.source()).expect("node").clone();
            let v = g2.node_weight(e.target()).expect("node").clone();
            let w = (*e.weight()).max(0.0);
            top_edges.push((u, v, w));
        }
        top_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        top_edges.truncate(args.edges_top);
    }

    Ok((
        rows,
        g2.node_count(),
        g2.edge_count(),
        aggregate_label,
        top_edges,
    ))
}

fn apply_modules_preset(args: &ModulesArgs) -> ModulesArgs {
    // Preset is a “default bundle”: apply only when a field is still at its default.
    let mut out = args.clone();
    let p = args.preset;

    // Helper: only set if still default value.
    let mut set_if = |cond: bool, f: &mut dyn FnMut(&mut ModulesArgs)| {
        if cond {
            f(&mut out);
        }
    };

    match p {
        ModulesPreset::None => return out,
        ModulesPreset::FileFull => {
            // If caller left aggregate at a non-file default, steer to file.
            set_if(!matches!(args.aggregate, ModuleAggregate::File), &mut |a| {
                a.aggregate = ModuleAggregate::File
            });
            set_if(matches!(args.edge_kind, ModuleEdgeKind::Uses), &mut |_a| {});
            // include fns/types/traits
            set_if(!args.include_fns, &mut |a| a.include_fns = true);
            set_if(!args.include_types, &mut |a| a.include_types = true);
            set_if(!args.include_traits, &mut |a| a.include_traits = true);
            // show edges by default
            set_if(args.edges_top == 0, &mut |a| a.edges_top = 5);
            set_if(args.members_top == 3, &mut |_a| {});
        }
        ModulesPreset::FileApi => {
            set_if(!matches!(args.aggregate, ModuleAggregate::File), &mut |a| {
                a.aggregate = ModuleAggregate::File
            });
            set_if(!args.include_types, &mut |a| a.include_types = true);
            set_if(!args.include_traits, &mut |a| a.include_traits = true);
            // keep functions hidden by default
            set_if(args.edges_top == 0, &mut |a| a.edges_top = 3);
            set_if(args.members_top == 3, &mut |a| a.members_top = 2);
        }
        ModulesPreset::NodeFull => {
            // If caller left aggregate at a non-node default, steer to node.
            set_if(!matches!(args.aggregate, ModuleAggregate::Node), &mut |a| {
                a.aggregate = ModuleAggregate::Node
            });
            set_if(!args.include_fns, &mut |a| a.include_fns = true);
            set_if(!args.include_types, &mut |a| a.include_types = true);
            set_if(!args.include_traits, &mut |a| a.include_traits = true);
            set_if(args.edge_kind == ModuleEdgeKind::Uses, &mut |a| {
                a.edge_kind = ModuleEdgeKind::Both
            });
            set_if(args.edges_top == 0, &mut |a| a.edges_top = 8);
        }
        ModulesPreset::NodeApi => {
            set_if(!matches!(args.aggregate, ModuleAggregate::Node), &mut |a| {
                a.aggregate = ModuleAggregate::Node
            });
            set_if(!args.include_types, &mut |a| a.include_types = true);
            set_if(!args.include_traits, &mut |a| a.include_traits = true);
            // keep fns hidden
            set_if(args.edges_top == 0, &mut |a| a.edges_top = 5);
        }
    }

    out
}

#[derive(Debug, Serialize)]
struct ModulesCacheMeta {
    generated_at_unix: i64,
    pkg: Option<String>,
    manifest_path: String,
    target: String,
    cmd: String,
    key: String,
}

fn cargo_modules_dot_cached(
    args: &ModulesArgs,
    cmd: &ProcessCommand,
    pkg: Option<&str>,
) -> Result<String> {
    // Cache root defaults to `<workspace_root>/evals/pkgrank/modules_cache`.
    // If manifest_path is a workspace root, this keeps cache co-located with other pkgrank artifacts.
    let workspace_root = args
        .manifest_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let cache_root = workspace_root.join("evals/pkgrank/modules_cache");
    fs::create_dir_all(&cache_root).ok();

    if !args.cache {
        return run_cargo_modules_dot(cmd);
    }

    // Stable key over command + relevant options.
    let target = if args.lib {
        "lib".to_string()
    } else if let Some(bin) = &args.bin {
        format!("bin={bin}")
    } else {
        "default".to_string()
    };

    let key_material = format!(
        "pkg={:?}\nmanifest={}\ntarget={}\nedge={:?}\nagg={:?}\nweights={:.3}/{:.3}\nfilters=no_externs:{} no_sysroot:{} no_fns:{} no_types:{} no_traits:{}\ninclude=externs:{} sysroot:{} fns:{} types:{} traits:{}\ncmd={:?}",
        pkg,
        args.manifest_path.display(),
        target,
        args.edge_kind,
        args.aggregate,
        args.uses_weight,
        args.owns_weight,
        args.no_externs,
        args.no_sysroot,
        args.no_fns,
        args.no_types,
        args.no_traits,
        args.include_externs,
        args.include_sysroot,
        args.include_fns,
        args.include_types,
        args.include_traits,
        cmd
    );

    let key = format!("{:016x}", fnv1a64(key_material.as_bytes()));
    let dot_path = cache_root.join(format!("modules_{key}.dot"));
    let meta_path = cache_root.join(format!("modules_{key}.json"));

    if !args.cache_refresh && dot_path.exists() {
        return fs::read_to_string(&dot_path).map_err(Into::into);
    }

    let dot = run_cargo_modules_dot(cmd)?;
    fs::write(&dot_path, &dot).ok();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let meta = ModulesCacheMeta {
        generated_at_unix: now,
        pkg: pkg.map(|s| s.to_string()),
        manifest_path: args.manifest_path.display().to_string(),
        target,
        cmd: format!("{:?}", cmd),
        key,
    };
    fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta).unwrap_or_default(),
    )
    .ok();

    Ok(dot)
}

fn run_cargo_modules_dot(cmd: &ProcessCommand) -> Result<String> {
    // `std::process::Command` is not cloneable; rebuild a fresh command from `args`.
    let program = cmd.get_program().to_string_lossy().to_string();
    let mut cmd2 = ProcessCommand::new(program);
    cmd2.args(cmd.get_args());
    cmd2.envs(cmd.get_envs().filter_map(|(k, v)| v.map(|vv| (k, vv))));
    cmd2.current_dir(
        cmd.get_current_dir()
            .unwrap_or_else(|| std::path::Path::new(".")),
    );
    let out = cmd2
        .output()
        .with_context(|| format!("failed to spawn: {:?}", cmd))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!(
            "cargo modules dependencies failed (exit={:?}): {}",
            out.status.code(),
            stderr.trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn print_modules_text(
    args: &ModulesArgs,
    rows: &[ModuleRow],
    nodes: usize,
    edges: usize,
    aggregate_label: &str,
    top_edges: &[(String, String, f64)],
) {
    let mut sorted: Vec<&ModuleRow> = rows.iter().collect();
    sorted.sort_by(|a, b| match args.metric {
        Metric::Pagerank => b.pagerank.partial_cmp(&a.pagerank).unwrap(),
        Metric::ConsumersPagerank => b
            .consumers_pagerank
            .partial_cmp(&a.consumers_pagerank)
            .unwrap(),
        Metric::Indegree => b.in_degree.cmp(&a.in_degree),
        Metric::Outdegree => b.out_degree.cmp(&a.out_degree),
        Metric::Betweenness => b.betweenness.partial_cmp(&a.betweenness).unwrap(),
    });

    let target = if args.lib {
        "lib".to_string()
    } else if let Some(bin) = &args.bin {
        format!("bin={bin}")
    } else {
        "default".to_string()
    };

    println!("pkgrank modules");
    println!("  manifest: {}", args.manifest_path.display());
    if let Some(p) = &args.package {
        println!("  package:  {}", p);
    }
    println!("  target:   {}", target);
    if !matches!(args.preset, ModulesPreset::None) {
        println!("  preset:   {:?}", args.preset);
    }
    println!("  edges:    {:?} (from cargo-modules)", args.edge_kind);
    println!("  aggregate:{}", aggregate_label);
    println!(
        "  include:  fns={} types={} traits={} externs={} sysroot={}  cache={} refresh={}\n",
        args.include_fns,
        args.include_types,
        args.include_traits,
        args.include_externs,
        args.include_sysroot,
        args.cache,
        args.cache_refresh
    );

    // Header row: show all scalar signals; humans can pick what they care about.
    println!(
        "{:>4}  {:>10} {:>10} {:>9} {:>6} {:>6} {:>3} {:>3}  {:<10} {:<8} node",
        "rank", "pr", "cons_pr", "between", "depsT", "consT", "in", "out", "kind", "vis"
    );
    println!("{:─<110}", "");

    for (i, r) in sorted.into_iter().take(args.top).enumerate() {
        let kind = r.kind.as_deref().unwrap_or("-");
        let vis = r.visibility.as_deref().unwrap_or("-");
        let mut node = r.node.clone();
        if let Some(gs) = r.group_size {
            node.push_str(&format!("  [n={}]", gs));
        }
        if let Some(prev) = &r.members_preview {
            node.push_str(&format!("  members: {}", prev));
        }
        if let Some(tops) = &r.top_members_pr {
            node.push_str(&format!("  top_pr: {}", tops));
        }

        println!(
            "{:>4}. {:>10.6} {:>10.6} {:>9.6} {:>6} {:>6} {:>3} {:>3}  {:<10} {:<8} {}",
            i + 1,
            r.pagerank,
            r.consumers_pagerank,
            r.betweenness,
            r.transitive_dependencies,
            r.transitive_dependents,
            r.in_degree,
            r.out_degree,
            kind,
            vis,
            node
        );
    }

    println!(
        "\n{} nodes, {} edges\nEdge semantics: A → B means A {} B",
        nodes,
        edges,
        match args.edge_kind {
            ModuleEdgeKind::Uses => "uses",
            ModuleEdgeKind::Owns => "owns",
            ModuleEdgeKind::Both => "relates to",
        }
    );

    if args.edges_top > 0 {
        println!("\nTop {} edges (by weight):", args.edges_top);
        println!("{:─<110}", "");
        for (i, (u, v, w)) in top_edges.iter().take(args.edges_top).enumerate() {
            println!("{:>4}. w={:>7.3}  {}  ->  {}", i + 1, w, u, v);
        }
    }
}

fn owning_module(node: &str) -> String {
    // Best-effort: for `crate::mod::Item`, owner module is `crate::mod`.
    // For a crate root `crate`, owner is itself.
    node.rsplit_once("::")
        .map(|(p, _)| p.to_string())
        .unwrap_or_else(|| node.to_string())
}

fn contract_graph<F>(
    g: &DiGraph<String, f64>,
    key_fn: F,
) -> (DiGraph<String, f64>, HashMap<String, Vec<String>>)
where
    F: Fn(&str) -> String,
{
    let mut members: HashMap<String, Vec<String>> = HashMap::new();

    // Assign each node to a group.
    for n in g.node_indices() {
        let name = g.node_weight(n).expect("node weight");
        let key = key_fn(name);
        members.entry(key).or_default().push(name.clone());
    }

    // Stable group ordering (deterministic across runs).
    let mut keys: Vec<String> = members.keys().cloned().collect();
    keys.sort();
    let mut groups: HashMap<String, usize> = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        groups.insert(k.clone(), i);
    }

    // Build new graph.
    let mut ng: DiGraph<String, f64> = DiGraph::new();
    let mut idx: Vec<NodeIndex> = vec![NodeIndex::new(0); groups.len()];
    for (i, k) in keys.iter().enumerate() {
        idx[i] = ng.add_node(k.clone());
    }

    for e in g.edge_references() {
        let u = e.source().index();
        let v = e.target().index();
        let from = g.node_weight(NodeIndex::new(u)).expect("node");
        let to = g.node_weight(NodeIndex::new(v)).expect("node");
        let gu = groups[&key_fn(from)];
        let gv = groups[&key_fn(to)];
        if gu == gv {
            continue;
        }
        // Edge weight = number of induced edges (lets weighted PageRank reflect multiplicity).
        let w = (*e.weight()).max(0.0);
        let cur = ng
            .find_edge(idx[gu], idx[gv])
            .and_then(|ei| ng.edge_weight(ei).copied())
            .unwrap_or(0.0);
        ng.update_edge(idx[gu], idx[gv], cur + w);
    }

    // Sort member lists for stable output / debugging.
    for v in members.values_mut() {
        v.sort();
    }

    (ng, members)
}

fn infer_rust_crate_root_file(crate_dir: &Path, args: &ModulesArgs, _crate_name: &str) -> PathBuf {
    // Best-effort; intended only for file aggregation heuristics.
    let src = crate_dir.join("src");
    if let Some(bin) = &args.bin {
        let candidate = src.join("bin").join(format!("{bin}.rs"));
        if candidate.exists() {
            return candidate;
        }
        let candidate = src.join("main.rs");
        if candidate.exists() {
            return candidate;
        }
    }
    if args.lib {
        let candidate = src.join("lib.rs");
        if candidate.exists() {
            return candidate;
        }
    }
    // default guess: lib.rs if it exists, else main.rs
    let lib = src.join("lib.rs");
    if lib.exists() {
        return lib;
    }
    src.join("main.rs")
}

fn module_to_file_key(
    crate_dir: &Path,
    crate_name: &str,
    root_file: &Path,
    module: &str,
) -> String {
    // Best-effort mapping of module path to a source file path.
    // If mapping fails, we fall back to the root file, then to "<unknown>".
    let src = crate_dir.join("src");

    if module == crate_name {
        return rel_path_display(crate_dir, root_file);
    }

    // strip leading "<crate_name>::"
    let rel = module
        .strip_prefix(&format!("{crate_name}::"))
        .unwrap_or(module);
    let segs: Vec<&str> = rel.split("::").filter(|s| !s.is_empty()).collect();
    if segs.is_empty() {
        return rel_path_display(crate_dir, root_file);
    }

    // Candidates for exact module path:
    let mut cur = segs.as_slice();
    while !cur.is_empty() {
        let as_file = src.join(cur.join("/")).with_extension("rs");
        if as_file.exists() {
            return rel_path_display(crate_dir, &as_file);
        }
        let as_mod = src.join(cur.join("/")).join("mod.rs");
        if as_mod.exists() {
            return rel_path_display(crate_dir, &as_mod);
        }
        // Fall back to parent module file (handles inline modules in same file reasonably well).
        cur = &cur[..cur.len() - 1];
    }

    if root_file.exists() {
        return rel_path_display(crate_dir, root_file);
    }
    "<unknown>".to_string()
}

fn rel_path_display(base: &Path, p: &Path) -> String {
    // Prefer repo-relative paths in output; absolute paths are noisy and not diff-stable.
    match p.strip_prefix(base) {
        Ok(rp) => rp.display().to_string(),
        Err(_) => p.display().to_string(),
    }
}

fn resolve_package_dir(workspace_manifest: &Path, package_name: &str) -> Option<PathBuf> {
    // Use cargo metadata to find the actual package manifest path, then take its parent.
    // This fixes the super-workspace / virtual-workspace confusion where `--manifest-path`
    // may be the workspace root `Cargo.toml`, not the package’s own `Cargo.toml`.
    let mut cmd = MetadataCommand::new();
    cmd.manifest_path(workspace_manifest);
    let md = cmd.exec().ok()?;
    let pkg = md
        .packages
        .iter()
        .find(|p| p.name.as_str() == package_name)?;
    let mp = PathBuf::from(pkg.manifest_path.to_string());
    mp.parent().map(|p| p.to_path_buf())
}

#[derive(Debug, Clone)]
struct ParsedEdge {
    u: usize,
    v: usize,
    label: String,
}

fn parse_cargo_modules_dot(
    dot: &str,
    edge_kind: ModuleEdgeKind,
) -> (
    Vec<String>,
    Vec<ParsedEdge>,
    HashMap<String, CargoModulesNodeMeta>,
) {
    // cargo-modules prints DOT with quoted node IDs:
    //   "a::b" -> "c::d" [label="uses", ...] ...
    //
    // We parse a conservative subset: only edges where we can extract
    // (source, target, label). Everything else is ignored.
    let want_uses = matches!(edge_kind, ModuleEdgeKind::Uses | ModuleEdgeKind::Both);
    let want_owns = matches!(edge_kind, ModuleEdgeKind::Owns | ModuleEdgeKind::Both);

    let mut nodes: HashMap<String, usize> = HashMap::new();
    let mut edges: Vec<ParsedEdge> = Vec::new();
    let mut meta: HashMap<String, CargoModulesNodeMeta> = HashMap::new();

    let intern = |s: &str, nodes: &mut HashMap<String, usize>| -> usize {
        if let Some(&i) = nodes.get(s) {
            return i;
        }
        let i = nodes.len();
        nodes.insert(s.to_string(), i);
        i
    };

    for line in dot.lines() {
        // quick reject
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Node line: `"foo::bar" [label="pub(crate) struct|Bar", ...];`
        if line.starts_with('"')
            && line.contains(" [label=\"")
            && line.contains("];")
            && !line.contains("->")
        {
            if let Some((id, m)) = parse_cargo_modules_node_line(line) {
                meta.insert(id.clone(), m);
                // Ensure isolated nodes are included (cargo-modules prints nodes even if they have no edges).
                // This matters for centrality baselines and for “what exists but is disconnected?”.
                let _ = intern(&id, &mut nodes);
            }
            continue;
        }

        // Edge line:
        if !line.contains("->") || !line.contains("label=") {
            continue;
        }

        // Parse first quoted string (source)
        let Some(s1) = line.find('"') else { continue };
        let rest = &line[s1 + 1..];
        let Some(e1) = rest.find('"') else { continue };
        let source = &rest[..e1];

        // Find the target quoted string after ->
        let after_src = &rest[e1 + 1..];
        let Some(arrow) = after_src.find("->") else {
            continue;
        };
        let after_arrow = &after_src[arrow + 2..];
        let Some(s2q) = after_arrow.find('"') else {
            continue;
        };
        let rest2 = &after_arrow[s2q + 1..];
        let Some(e2) = rest2.find('"') else { continue };
        let target = &rest2[..e2];

        // Parse label="uses"/"owns"
        let Some(li) = line.find("label=\"") else {
            continue;
        };
        let restl = &line[li + "label=\"".len()..];
        let Some(le) = restl.find('"') else { continue };
        let label = &restl[..le];

        let include = match label {
            "uses" => want_uses,
            "owns" => want_owns,
            _ => false,
        };
        if !include {
            continue;
        }

        let u = intern(source, &mut nodes);
        let v = intern(target, &mut nodes);
        edges.push(ParsedEdge {
            u,
            v,
            label: label.to_string(),
        });
    }

    // Convert HashMap to Vec in index order.
    let mut node_names = vec![String::new(); nodes.len()];
    for (name, i) in nodes {
        node_names[i] = name;
    }

    (node_names, edges, meta)
}

fn parse_cargo_modules_node_line(line: &str) -> Option<(String, CargoModulesNodeMeta)> {
    // Example:
    //   "pkgrank::AnalyzeArgs" [label="pub(crate) struct|AnalyzeArgs", ...]; // "struct" node
    let s1 = line.find('"')?;
    let rest = &line[s1 + 1..];
    let e1 = rest.find('"')?;
    let id = rest[..e1].to_string();

    let li = line.find("label=\"")?;
    let restl = &line[li + "label=\"".len()..];
    let le = restl.find('"')?;
    let label = &restl[..le];
    // label is "crate|pkgrank" OR "pub(crate) struct|AnalyzeArgs"
    let (header, _body) = label.split_once('|').unwrap_or((label, ""));
    let header = header.trim();
    if header.is_empty() {
        return Some((
            id,
            CargoModulesNodeMeta {
                kind: None,
                visibility: None,
            },
        ));
    }
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.is_empty() {
        return Some((
            id,
            CargoModulesNodeMeta {
                kind: None,
                visibility: None,
            },
        ));
    }
    let kind = parts.last().map(|s| s.to_string());
    let visibility = if parts.len() >= 2 {
        Some(parts[..parts.len() - 1].join(" "))
    } else {
        None
    };
    Some((id, CargoModulesNodeMeta { kind, visibility }))
}

fn run_analyze(args: &AnalyzeArgs) -> Result<()> {
    let started_at = Instant::now();
    let manifest_path = manifest_path(&args.path)?;
    let t_metadata = Instant::now();
    let metadata = metadata_for(&manifest_path, args)
        .with_context(|| format!("cargo metadata failed for {}", manifest_path.display()))?;
    let dt_metadata = t_metadata.elapsed();

    let t_graph = Instant::now();
    let (graph, _nodes) = build_graph(&metadata, args)?;
    let dt_graph = t_graph.elapsed();

    let t_score = Instant::now();
    let (mut rows, convergence) = compute_rows_with_convergence(&metadata, &graph);
    sort_rows_by_metric(&mut rows, args.metric);
    let dt_score = t_score.elapsed();

    let mut bytes_printed: Option<u64> = None;

    match args.format {
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct AnalyzeJsonOut<'a> {
                schema_version: u32,
                ok: bool,
                command: &'a str,
                metric: String,
                sorted_by: String,
                rows_total: usize,
                rows_returned: usize,
                truncated: bool,
                json_limit: Option<usize>,
                convergence: serde_json::Value,
                rows: Vec<Row>,
            }
            let rows_total = rows.len();
            let limit = args.json_limit.unwrap_or(rows_total);
            let rows_returned = rows_total.min(limit);
            let rows = rows.into_iter().take(limit).collect::<Vec<_>>();
            let out = AnalyzeJsonOut {
                schema_version: 1,
                ok: true,
                command: "analyze",
                metric: format!("{:?}", args.metric),
                sorted_by: format!("{:?}", args.metric),
                rows_total,
                rows_returned,
                truncated: rows_returned < rows_total,
                json_limit: args.json_limit,
                convergence,
                rows,
            };
            let s = serde_json::to_string_pretty(&out)?;
            bytes_printed = Some(s.len() as u64);
            println!("{s}");
        }
        OutputFormat::Text => print_text(
            &rows,
            args.metric,
            args.top,
            graph.node_count(),
            graph.edge_count(),
        ),
    }

    if args.stats {
        let stats = PkgrankStatsSummary {
            command: "analyze",
            node_count: graph.node_count() as u64,
            edge_count: graph.edge_count() as u64,
            package_count_total: metadata.packages.len() as u64,
            package_count_workspace: metadata.workspace_members.len() as u64,
            bytes_printed,
            elapsed_metadata: dt_metadata,
            elapsed_build_graph: dt_graph,
            elapsed_score: dt_score,
            elapsed_total: started_at.elapsed(),
        };
        let _ = print_pkgrank_stats(args.format, &stats);
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct PkgrankStatsSummary<'a> {
    command: &'a str,
    node_count: u64,
    edge_count: u64,
    package_count_total: u64,
    package_count_workspace: u64,
    bytes_printed: Option<u64>,
    #[serde(skip)]
    elapsed_metadata: Duration,
    #[serde(skip)]
    elapsed_build_graph: Duration,
    #[serde(skip)]
    elapsed_score: Duration,
    #[serde(skip)]
    elapsed_total: Duration,
}

fn print_pkgrank_stats(
    format: OutputFormat,
    stats: &PkgrankStatsSummary<'_>,
) -> std::io::Result<()> {
    // IMPORTANT: keep stats on stderr so stdout remains a stable artifact surface.
    if matches!(format, OutputFormat::Json) {
        // Mirror ripgrep's "summary" JSON convention: machine-readable, one object.
        let out = serde_json::json!({
            "type": "pkgrank_stats",
            "data": {
                "command": stats.command,
                "graph": {
                    "nodes": stats.node_count,
                    "edges": stats.edge_count,
                },
                "packages": {
                    "total": stats.package_count_total,
                    "workspace": stats.package_count_workspace,
                },
                "bytes_printed": stats.bytes_printed,
                "elapsed": {
                    "metadata_human": format!("{:0.6}s", stats.elapsed_metadata.as_secs_f64()),
                    "build_graph_human": format!("{:0.6}s", stats.elapsed_build_graph.as_secs_f64()),
                    "score_human": format!("{:0.6}s", stats.elapsed_score.as_secs_f64()),
                    "total_human": format!("{:0.6}s", stats.elapsed_total.as_secs_f64()),
                    "total_secs": stats.elapsed_total.as_secs(),
                    "total_nanos": stats.elapsed_total.subsec_nanos(),
                }
            }
        });
        eprintln!(
            "{}",
            serde_json::to_string(&out).unwrap_or_else(|_| "{}".to_string())
        );
        Ok(())
    } else {
        eprintln!(
            "
pkgrank stats
command: {command}
packages: {pkg_total} total ({pkg_ws} workspace)
graph: {nodes} nodes, {edges} edges
bytes printed: {bytes_printed}
{t_meta:0.6} seconds cargo metadata
{t_graph:0.6} seconds build graph
{t_score:0.6} seconds score centrality
{t_total:0.6} seconds total
",
            command = stats.command,
            pkg_total = stats.package_count_total,
            pkg_ws = stats.package_count_workspace,
            nodes = stats.node_count,
            edges = stats.edge_count,
            bytes_printed = stats
                .bytes_printed
                .map(|b| b.to_string())
                .unwrap_or_else(|| "-".to_string()),
            t_meta = stats.elapsed_metadata.as_secs_f64(),
            t_graph = stats.elapsed_build_graph.as_secs_f64(),
            t_score = stats.elapsed_score.as_secs_f64(),
            t_total = stats.elapsed_total.as_secs_f64(),
        );
        Ok(())
    }
}

fn run_mcp_stdio() -> Result<()> {
    // IMPORTANT: stdout is reserved for MCP JSON-RPC frames.
    // If you need diagnostics, use stderr.
    #[cfg(feature = "stdio")]
    {
        let rt = tokio::runtime::Runtime::new().context("failed to build tokio runtime")?;
        rt.block_on(async {
            let toolset_raw =
                std::env::var("PKGRANK_MCP_TOOLSET").unwrap_or_else(|_| "slim".to_string());
            let toolset = toolset_raw.trim().to_ascii_lowercase();

            // Default should "just work": keep tool surface small unless explicitly opted-in.
            match toolset.as_str() {
                "" | "slim" | "minimal" => {
                    let service = PkgrankStdioMcpSlim::new();
                    let running = service
                        .serve(stdio())
                        .await
                        .context("failed to start stdio MCP server (toolset=slim)")?;
                    let _ = running
                        .waiting()
                        .await
                        .context("stdio MCP server task join failed (toolset=slim)")?;
                }
                "full" => {
                    let service = PkgrankStdioMcpFull::new();
                    let running = service
                        .serve(stdio())
                        .await
                        .context("failed to start stdio MCP server (toolset=full)")?;
                    let _ = running
                        .waiting()
                        .await
                        .context("stdio MCP server task join failed (toolset=full)")?;
                }
                "debug" => {
                    let service = PkgrankStdioMcpDebug::new();
                    let running = service
                        .serve(stdio())
                        .await
                        .context("failed to start stdio MCP server (toolset=debug)")?;
                    let _ = running
                        .waiting()
                        .await
                        .context("stdio MCP server task join failed (toolset=debug)")?;
                }
                other => {
                    eprintln!(
                        "warning: PKGRANK_MCP_TOOLSET must be one of: slim, full, debug (got {other}); defaulting to slim"
                    );
                    let service = PkgrankStdioMcpSlim::new();
                    let running = service
                        .serve(stdio())
                        .await
                        .context("failed to start stdio MCP server (toolset=slim)")?;
                    let _ = running
                        .waiting()
                        .await
                        .context("stdio MCP server task join failed (toolset=slim)")?;
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;
        Ok(())
    }
    #[cfg(not(feature = "stdio"))]
    {
        anyhow::bail!(
            "mcp-stdio requires compile-time feature `stdio` (rebuild: cargo build -p pkgrank --features stdio)"
        );
    }
}

fn manifest_path(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    Ok(path.join("Cargo.toml"))
}

fn metadata_for(manifest_path: &PathBuf, args: &AnalyzeArgs) -> Result<Metadata> {
    let mut cmd = MetadataCommand::new();
    cmd.manifest_path(manifest_path);

    // Order is chosen to mimic cargo’s precedence:
    // - `--all-features` dominates
    // - otherwise: optional `--no-default-features` + optional `--features ...`
    if args.all_features {
        cmd.features(CargoOpt::AllFeatures);
        return cmd.exec().map_err(Into::into);
    }

    let feats: Vec<String> = args
        .features
        .as_deref()
        .unwrap_or("")
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    match (args.no_default_features, feats.is_empty()) {
        (false, true) => cmd.exec().map_err(Into::into),
        (true, true) => cmd
            .features(CargoOpt::NoDefaultFeatures)
            .exec()
            .map_err(Into::into),
        (false, false) => cmd
            .features(CargoOpt::SomeFeatures(feats))
            .exec()
            .map_err(Into::into),
        (true, false) => {
            // `cargo_metadata` does not support "no-default-features + some features"
            // in one CargoOpt. Cargo itself supports it, so we shell out and parse.
            let mut cmd = ProcessCommand::new("cargo");
            cmd.args(["metadata", "--format-version", "1"]);
            cmd.args(["--manifest-path", &manifest_path.to_string_lossy()]);
            cmd.arg("--no-default-features");
            cmd.arg("--features");
            cmd.arg(feats.join(","));
            let out = cmd
                .output()
                .with_context(|| format!("failed to spawn: {:?}", cmd))?;
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                return Err(anyhow!(
                    "cargo metadata failed (exit={:?}): {}",
                    out.status.code(),
                    stderr.trim()
                ));
            }
            let md: Metadata = serde_json::from_slice(&out.stdout)
                .with_context(|| "failed to parse `cargo metadata` JSON output")?;
            Ok(md)
        }
    }
}

fn build_graph(
    metadata: &Metadata,
    args: &AnalyzeArgs,
) -> Result<(DiGraph<PackageId, f64>, HashMap<PackageId, NodeIndex>)> {
    let resolve = metadata
        .resolve
        .as_ref()
        .ok_or_else(|| anyhow!("cargo metadata did not include `resolve` (unexpected)"))?;

    let workspace: HashSet<PackageId> = metadata.workspace_members.iter().cloned().collect();

    // Decide which packages are nodes.
    let include_node = |id: &PackageId| -> bool { !args.workspace_only || workspace.contains(id) };

    let mut graph: DiGraph<PackageId, f64> = DiGraph::new();
    let mut node_map: HashMap<PackageId, NodeIndex> = HashMap::new();

    for pkg in &metadata.packages {
        if include_node(&pkg.id) {
            let idx = graph.add_node(pkg.id.clone());
            node_map.insert(pkg.id.clone(), idx);
        }
    }

    // Add edges from resolved dependency graph.
    for node in &resolve.nodes {
        if !include_node(&node.id) {
            continue;
        }
        let from_idx = match node_map.get(&node.id) {
            Some(i) => *i,
            None => continue,
        };

        for dep in deps_for_node(node, args) {
            if !include_node(&dep) {
                continue;
            }
            if let Some(&to_idx) = node_map.get(&dep) {
                // Unweighted at package level: presence/absence edge.
                graph.update_edge(from_idx, to_idx, 1.0);
            }
        }
    }

    Ok((graph, node_map))
}

fn deps_for_node(node: &Node, args: &AnalyzeArgs) -> Vec<PackageId> {
    // `cargo_metadata` gives structured dep kinds per edge. We include an edge iff
    // any dep-kind on that edge is allowed.
    let mut out = Vec::new();
    for dep in &node.deps {
        if dep_kind_allowed(&dep.dep_kinds, args) {
            out.push(dep.pkg.clone());
        }
    }
    out
}

fn dep_kind_allowed(kinds: &[DepKindInfo], args: &AnalyzeArgs) -> bool {
    // If any kind is allowed, we include the edge.
    kinds.iter().any(|k| match k.kind {
        DependencyKind::Normal => true,
        DependencyKind::Development => args.dev,
        DependencyKind::Build => args.build,
        DependencyKind::Unknown => false,
    })
}

fn compute_rows(
    metadata: &Metadata,
    graph: &DiGraph<PackageId, f64>,
    _node_map: &HashMap<PackageId, NodeIndex>,
) -> Vec<Row> {
    let pkg_by_id: HashMap<&PackageId, &cargo_metadata::Package> =
        metadata.packages.iter().map(|p| (&p.id, p)).collect();

    let workspace: HashSet<&PackageId> = metadata.workspace_members.iter().collect();

    let pr = pagerank_auto(graph);
    let consumers_pr = pagerank_auto(&reverse_graph(graph));
    let betweenness = betweenness_centrality(graph);

    let mut rows = Vec::with_capacity(graph.node_count());
    for node in graph.node_indices() {
        let id = graph.node_weight(node).expect("node weight").clone();
        let pkg = pkg_by_id
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("missing package for id {}", id));

        let in_degree = graph.neighbors_directed(node, Direction::Incoming).count();
        let out_degree = graph.neighbors_directed(node, Direction::Outgoing).count();

        let origin = if workspace.contains(&id) {
            PackageOrigin::WorkspaceMember
        } else if pkg.source.is_none() {
            // In cargo metadata, local path dependencies typically have no "source".
            PackageOrigin::Path
        } else {
            // `source` is a URL-ish identifier like:
            // - registry+https://github.com/rust-lang/crates.io-index
            // - git+https://github.com/...#rev
            let s = pkg
                .source
                .as_ref()
                .map(|x| x.to_string())
                .unwrap_or_default();
            if s.starts_with("registry+") {
                PackageOrigin::Registry
            } else if s.starts_with("git+") {
                PackageOrigin::Git
            } else {
                PackageOrigin::Other
            }
        };

        rows.push(Row {
            id: id.to_string(),
            name: pkg.name.to_string(),
            version: pkg.version.to_string(),
            manifest_path: pkg.manifest_path.to_string(),
            origin,
            in_degree,
            out_degree,
            pagerank: pr[node.index()],
            consumers_pagerank: consumers_pr[node.index()],
            betweenness: betweenness[node.index()],
        });
    }

    rows.sort_by(|a, b| {
        b.pagerank
            .partial_cmp(&a.pagerank)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

fn compute_rows_with_convergence(
    metadata: &Metadata,
    graph: &DiGraph<PackageId, f64>,
) -> (Vec<Row>, serde_json::Value) {
    let pkg_by_id: HashMap<&PackageId, &cargo_metadata::Package> =
        metadata.packages.iter().map(|p| (&p.id, p)).collect();

    let workspace: HashSet<&PackageId> = metadata.workspace_members.iter().collect();

    let pr_run = pagerank_auto_run(graph);
    let consumers_run = pagerank_auto_run(&reverse_graph(graph));
    let betweenness = betweenness_centrality(graph);

    let pr = pr_run.scores.clone();
    let consumers_pr = consumers_run.scores.clone();

    let mut rows = Vec::with_capacity(graph.node_count());
    for node in graph.node_indices() {
        let id = graph.node_weight(node).expect("node weight").clone();
        let pkg = pkg_by_id
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("missing package for id {}", id));

        let in_degree = graph.neighbors_directed(node, Direction::Incoming).count();
        let out_degree = graph.neighbors_directed(node, Direction::Outgoing).count();

        let origin = if workspace.contains(&id) {
            PackageOrigin::WorkspaceMember
        } else if pkg.source.is_none() {
            PackageOrigin::Path
        } else {
            let s = pkg
                .source
                .as_ref()
                .map(|x| x.to_string())
                .unwrap_or_default();
            if s.starts_with("registry+") {
                PackageOrigin::Registry
            } else if s.starts_with("git+") {
                PackageOrigin::Git
            } else {
                PackageOrigin::Other
            }
        };

        rows.push(Row {
            id: id.to_string(),
            name: pkg.name.to_string(),
            version: pkg.version.to_string(),
            manifest_path: pkg.manifest_path.to_string(),
            origin,
            in_degree,
            out_degree,
            pagerank: pr[node.index()],
            consumers_pagerank: consumers_pr[node.index()],
            betweenness: betweenness[node.index()],
        });
    }

    // Note: sorting for presentation is a policy choice.
    // We sort by the requested metric elsewhere (CLI/MCP), but keep a deterministic default here.
    rows.sort_by(|a, b| {
        b.pagerank
            .partial_cmp(&a.pagerank)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let convergence = serde_json::json!({
        "pagerank": convergence_report(&pr_run),
        "consumers_pagerank": convergence_report(&consumers_run),
    });

    (rows, convergence)
}

fn sort_rows_by_metric(rows: &mut [Row], metric: Metric) {
    rows.sort_by(|a, b| {
        let ord = match metric {
            Metric::Pagerank => b.pagerank.partial_cmp(&a.pagerank),
            Metric::ConsumersPagerank => b.consumers_pagerank.partial_cmp(&a.consumers_pagerank),
            Metric::Indegree => Some(b.in_degree.cmp(&a.in_degree)),
            Metric::Outdegree => Some(b.out_degree.cmp(&a.out_degree)),
            Metric::Betweenness => b.betweenness.partial_cmp(&a.betweenness),
        }
        .unwrap_or(std::cmp::Ordering::Equal);

        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
        // Stable tie-break for determinism.
        a.name.cmp(&b.name).then_with(|| a.id.cmp(&b.id))
    });
}

fn print_text(rows: &[Row], metric: Metric, top: usize, nodes: usize, edges: usize) {
    let mut sorted: Vec<&Row> = rows.iter().collect();
    sorted.sort_by(|a, b| match metric {
        Metric::Pagerank => b.pagerank.partial_cmp(&a.pagerank).unwrap(),
        Metric::ConsumersPagerank => b
            .consumers_pagerank
            .partial_cmp(&a.consumers_pagerank)
            .unwrap(),
        Metric::Indegree => b.in_degree.cmp(&a.in_degree),
        Metric::Outdegree => b.out_degree.cmp(&a.out_degree),
        Metric::Betweenness => b.betweenness.partial_cmp(&a.betweenness).unwrap(),
    });

    println!("Top {} by {:?}:", top, metric);
    println!("{:─<72}", "");
    for (i, r) in sorted.into_iter().take(top).enumerate() {
        let score = match metric {
            Metric::Pagerank => r.pagerank,
            Metric::ConsumersPagerank => r.consumers_pagerank,
            Metric::Indegree => r.in_degree as f64,
            Metric::Outdegree => r.out_degree as f64,
            Metric::Betweenness => r.betweenness,
        };
        println!(
            "{:3}. {:28} {:10}  in={:2} out={:2}  score={:.6}",
            i + 1,
            r.name,
            r.version,
            r.in_degree,
            r.out_degree,
            score
        );
    }
    println!(
        "\n{} nodes, {} edges\nEdges are A → B meaning A depends on B.\n- pagerank: central shared dependencies (mass flows toward dependencies)\n- consumers_pagerank: orchestrators (PageRank on reversed graph)",
        nodes, edges
    );
}

fn default_overview_path(root: &Path) -> PathBuf {
    root.join("evals/arch/dev_repos_overview.json")
}

#[derive(Debug, Clone, Copy, Default)]
struct GitRepoStats {
    commits_30d: Option<u64>,
    days_since_last_commit: Option<u64>,
}

fn git_repo_stats(repo_dir: &Path) -> GitRepoStats {
    // Only attempt git if this looks like a repo/worktree.
    if !repo_dir.join(".git").exists() {
        return GitRepoStats::default();
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let last_commit_secs = ProcessCommand::new("git")
        .args([
            "-C",
            &repo_dir.to_string_lossy(),
            "log",
            "-1",
            "--format=%ct",
        ])
        .output()
        .ok()
        .and_then(|o| {
            if !o.status.success() {
                return None;
            }
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<u64>().ok()
        });

    let days_since_last_commit =
        last_commit_secs.map(|t| if now >= t { (now - t) / 86_400 } else { 0 });

    // commits over last 30 days (fast count)
    let commits_30d = ProcessCommand::new("git")
        .args([
            "-C",
            &repo_dir.to_string_lossy(),
            "rev-list",
            "--count",
            "--since=30.days",
            "HEAD",
        ])
        .output()
        .ok()
        .and_then(|o| {
            if !o.status.success() {
                return None;
            }
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<u64>().ok()
        });

    GitRepoStats {
        commits_30d,
        days_since_last_commit,
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RepoRow {
    repo: String,
    axis: String,
    in_degree: usize,
    out_degree: usize,
    in_weight: f64,
    out_weight: f64,
    pagerank: f64,
    consumers_pagerank: f64,
    transitive_dependents: usize,
    transitive_dependencies: usize,
    third_party_deps: usize,
    git_commits_30d: Option<u64>,
    git_days_since_last_commit: Option<u64>,
}

#[derive(Debug, Serialize)]
struct RepoAxesSummary {
    totals: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RepoInvariantViolation {
    rule: String,
    from_repo: String,
    from_axis: String,
    to_repo: String,
    to_axis: String,
    weight: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct TlcRepoRow {
    repo: String,
    axis: String,
    deps_pagerank: f64,
    consumers_pagerank: f64,
    transitive_dependents: usize,
    transitive_dependencies: usize,
    third_party_deps: usize,
    violation_weight: usize,
    score: f64,
    why: String,
    git_commits_30d: Option<u64>,
    git_days_since_last_commit: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[cfg(feature = "stdio")]
struct PkgrankCompareRunsArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    root: Option<String>,
    /// "new" artifact directory (relative to root if not absolute).
    new_out: String,
    /// "old" artifact directory (relative to root if not absolute).
    old_out: String,
    /// Limit number of deltas returned.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[cfg(feature = "stdio")]
struct PkgrankSnapshotArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    root: Option<String>,
    /// Current artifact directory to snapshot (relative to root if not absolute).
    #[serde(default)]
    out: Option<String>,
    /// Destination directory (relative to root if not absolute). Defaults to `<out>/runs/<label>`.
    #[serde(default)]
    dest: Option<String>,
    /// Label used for default dest.
    #[serde(default)]
    label: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[cfg(feature = "stdio")]
struct PkgrankTriageArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    root: Option<String>,
    /// Artifact directory (relative to root if not absolute).
    #[serde(default)]
    out: Option<String>,
    /// If required artifacts are missing, re-run `pkgrank_view` first.
    #[serde(default)]
    refresh_if_missing: Option<bool>,
    /// View mode used for refresh (local/cratesio/both). Defaults to "local".
    #[serde(default)]
    mode: Option<String>,
    /// Mark artifacts stale if older than this many minutes. Defaults to 60.
    #[serde(default)]
    stale_minutes: Option<u64>,
    /// Limit returned rows.
    #[serde(default)]
    limit: Option<usize>,
    /// Optional axis filter for TLC tables.
    #[serde(default)]
    axis: Option<String>,
    /// Include PPR aggregate top-k.
    #[serde(default)]
    ppr_top: Option<usize>,

    /// Summarize READMEs (bounded by the *_top_* limits below). Default: false.
    #[serde(default)]
    summarize_readmes: Option<bool>,
    /// Number of top repos (post-filter) to summarize. Default: 0.
    #[serde(default)]
    summarize_repos_top: Option<usize>,
    /// Number of top crates (post-filter) to summarize. Default: 0.
    #[serde(default)]
    summarize_crates_top: Option<usize>,
    /// Max chars of README fed into LLM (default: 12000).
    #[serde(default)]
    llm_input_max_chars: Option<usize>,
    /// Timeout seconds for the LLM command (default: 30).
    #[serde(default)]
    llm_timeout_secs: Option<u64>,
    /// Cache summaries under `<out>/readme_ai_cache/` (default: true).
    #[serde(default)]
    llm_cache: Option<bool>,
    /// Include raw LLM output in triage results (default: false).
    #[serde(default)]
    llm_include_raw: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[cfg(feature = "stdio")]
struct PkgrankRepoDetailArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    root: Option<String>,
    /// Artifact directory (relative to root if not absolute).
    #[serde(default)]
    out: Option<String>,
    /// Repo name (e.g. "anno", "hop", "innr").
    repo: String,
    /// Include README snippet (default: false).
    #[serde(default)]
    include_readme: Option<bool>,
    /// Max README snippet chars (default: 4000).
    #[serde(default)]
    readme_max_chars: Option<usize>,
    /// Summarize README via configured local LLM (default: false).
    #[serde(default)]
    summarize_readme: Option<bool>,
    /// Max chars of README fed into LLM (default: 12000).
    #[serde(default)]
    llm_input_max_chars: Option<usize>,
    /// Timeout seconds for the LLM command (default: 30).
    #[serde(default)]
    llm_timeout_secs: Option<u64>,
    /// Cache summaries under `<out>/readme_ai_cache/` (default: true).
    #[serde(default)]
    llm_cache: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[cfg(feature = "stdio")]
struct PkgrankCrateDetailArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    root: Option<String>,
    /// Artifact directory (relative to root if not absolute).
    #[serde(default)]
    out: Option<String>,
    /// Crate name (Cargo package name).
    #[serde(rename = "crate")]
    krate: String,
    /// Include README snippet (default: false).
    #[serde(default)]
    include_readme: Option<bool>,
    /// Max README snippet chars (default: 4000).
    #[serde(default)]
    readme_max_chars: Option<usize>,
    /// Summarize README via configured local LLM (default: false).
    #[serde(default)]
    summarize_readme: Option<bool>,
    /// Max chars of README fed into LLM (default: 12000).
    #[serde(default)]
    llm_input_max_chars: Option<usize>,
    /// Timeout seconds for the LLM command (default: 30).
    #[serde(default)]
    llm_timeout_secs: Option<u64>,
    /// Cache summaries under `<out>/readme_ai_cache/` (default: true).
    #[serde(default)]
    llm_cache: Option<bool>,
}

fn axis_for_repo(
    repo: &str,
    token_axis: &HashMap<String, String>,
    repo_axis_mass: &HashMap<String, HashMap<String, usize>>,
) -> String {
    // Prefer an explicit repo-level label when present.
    if let Some(ax) = token_axis.get(repo) {
        return ax.clone();
    }
    // Otherwise majority-vote over package labels we inferred.
    let Some(m) = repo_axis_mass.get(repo) else {
        return "other".to_string();
    };
    let mut best = ("other".to_string(), 0usize);
    for (axis, c) in m {
        if *c > best.1 {
            best = (axis.clone(), *c);
        }
    }
    best.0
}

fn reachability_counts(
    nodes: &[String],
    edges: &[(String, String)],
) -> (HashMap<String, usize>, HashMap<String, usize>) {
    // Graph edges are A -> B meaning "A depends on B".
    // - transitive_dependencies(A) = reachable forward from A
    // - transitive_dependents(A) = reachable forward in the reversed graph (who depends on A)
    let mut fwd: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut rev: HashMap<&str, Vec<&str>> = HashMap::new();

    for (a, b) in edges {
        fwd.entry(a.as_str()).or_default().push(b.as_str());
        rev.entry(b.as_str()).or_default().push(a.as_str());
    }

    let mut deps = HashMap::new();
    let mut dependents = HashMap::new();

    for n in nodes {
        let mut seen = HashSet::new();
        let mut q = VecDeque::new();
        q.push_back(n.as_str());
        while let Some(cur) = q.pop_front() {
            for &nx in fwd.get(cur).map(|v| v.as_slice()).unwrap_or(&[]) {
                if seen.insert(nx) {
                    q.push_back(nx);
                }
            }
        }
        deps.insert(n.clone(), seen.len());

        let mut seen = HashSet::new();
        let mut q = VecDeque::new();
        q.push_back(n.as_str());
        while let Some(cur) = q.pop_front() {
            for &nx in rev.get(cur).map(|v| v.as_slice()).unwrap_or(&[]) {
                if seen.insert(nx) {
                    q.push_back(nx);
                }
            }
        }
        dependents.insert(n.clone(), seen.len());
    }

    (dependents, deps)
}

fn pos_zero(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

// Reachability counts live in `walk` so other crates can reuse them without
// copying (and without adding new graph dependencies at higher layers).

fn compute_tlc_crates(root: &Path, out_dir: &Path) -> Result<Vec<TlcCrateRow>> {
    // TLC heuristic: highlight crates that are important (central / many dependents)
    // and/or complex (many third-party deps).
    //
    // This is a heuristic ranking, not a proof.
    let analyze = AnalyzeArgs {
        path: root.to_path_buf(),
        metric: Metric::Pagerank,
        top: 25,
        dev: false,
        build: false,
        workspace_only: false,
        all_features: false,
        no_default_features: false,
        features: None,
        format: OutputFormat::Json,
        stats: false,
        json_limit: None,
    };
    let manifest = manifest_path(&analyze.path)?;
    let metadata = metadata_for(&manifest, &analyze)?;
    let (graph, node_map) = build_graph(&metadata, &analyze)?;
    let rows = compute_rows(&metadata, &graph, &node_map);

    // Repo axis mapping (optional), pulled from dev_repos_overview.json.
    // We use this only for labeling in the TLC output; it must not affect the graph itself.
    let mut token_axis: HashMap<String, String> = HashMap::new();
    let overview_path = default_overview_path(root);
    if overview_path.exists() {
        if let Ok(raw) = fs::read_to_string(&overview_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(axes) = v.get("axes").and_then(|a| a.as_object()) {
                    for (axis, arr) in axes {
                        if let Some(arr) = arr.as_array() {
                            for x in arr {
                                if let Some(name) = x.as_str() {
                                    token_axis.insert(name.to_string(), axis.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Index rows by package id string.
    let mut row_by_id: HashMap<&str, &Row> = HashMap::new();
    for r in &rows {
        row_by_id.insert(r.id.as_str(), r);
    }

    // Build an index mapping NodeIndex -> Row (if any). Some nodes can exist without a row
    // if metadata is odd, but we try to be robust.
    let mut idx_to_row: Vec<Option<&Row>> = vec![None; graph.node_count()];
    for node in graph.node_indices() {
        let id = &graph[node];
        if let Some(r) = row_by_id.get(id.repr.as_str()).copied() {
            idx_to_row[node.index()] = Some(r);
        }
    }

    // Build a first-party induced graph (WorkspaceMember/Path), and compute:
    // - transitive reachability counts within first-party
    // - centrality within first-party (pagerank, betweenness)
    // - boundary complexity (# unique third-party deps)
    let mut fp_nodes: Vec<usize> = Vec::new();
    let mut fp_index: HashMap<usize, usize> = HashMap::new();
    for (i, r) in idx_to_row.iter().enumerate() {
        if let Some(r) = r {
            if matches!(
                r.origin,
                PackageOrigin::WorkspaceMember | PackageOrigin::Path
            ) {
                fp_index.insert(i, fp_nodes.len());
                fp_nodes.push(i);
            }
        }
    }

    let mut fp_edges: Vec<(usize, usize)> = Vec::new();
    // Also count unique third-party direct deps per first-party crate.
    let mut third_party_sets: Vec<HashSet<String>> = vec![HashSet::new(); fp_nodes.len()];

    for u in graph.node_indices() {
        let Some(from_r) = idx_to_row[u.index()] else {
            continue;
        };
        let Some(&u_fp) = fp_index.get(&u.index()) else {
            continue;
        };
        for e in graph.edges_directed(u, Direction::Outgoing) {
            let v = e.target();
            if let Some(to_r) = idx_to_row[v.index()] {
                if matches!(
                    to_r.origin,
                    PackageOrigin::WorkspaceMember | PackageOrigin::Path
                ) {
                    if let Some(&v_fp) = fp_index.get(&v.index()) {
                        fp_edges.push((u_fp, v_fp));
                    }
                } else {
                    // first-party -> third-party boundary
                    third_party_sets[u_fp].insert(to_r.name.clone());
                }
            } else {
                // Unknown target; treat as third-party-ish.
                third_party_sets[u_fp].insert("<unknown>".to_string());
            }
        }
        // suppress unused warning in case of future edits
        let _ = from_r;
    }

    let (fp_dependents, fp_deps) = reachability_counts_edges(fp_nodes.len(), &fp_edges);

    // Export first-party crate dependency adjacency (JSON adjacency list).
    //
    // This is intentionally a *data boundary*, not a Rust dependency boundary:
    // pkgrank should not depend on higher-layer crates (e.g. `webs`/`webs-core` in L5).
    //
    // The export happens to be compatible with `webs_core::KnowledgeGraph::from_json_adjacency_file`,
    // so you can run L5 analysis tools *externally* on the exact same graph.
    let mut adj: HashMap<String, HashSet<String>> = HashMap::new();
    for (u, v) in &fp_edges {
        let Some(u_node) = fp_nodes.get(*u).copied() else {
            continue;
        };
        let Some(v_node) = fp_nodes.get(*v).copied() else {
            continue;
        };
        let (Some(ur), Some(vr)) = (idx_to_row[u_node], idx_to_row[v_node]) else {
            continue;
        };
        adj.entry(ur.name.clone())
            .or_default()
            .insert(vr.name.clone());
    }
    let mut adj_out: HashMap<String, Vec<String>> = HashMap::new();
    for (k, vs) in adj {
        let mut v: Vec<String> = vs.into_iter().collect();
        v.sort();
        adj_out.insert(k, v);
    }
    fs::write(
        out_dir.join("local.first_party.adj.json"),
        serde_json::to_string_pretty(&adj_out)?,
    )?;

    // Compute centrality on the first-party induced graph (so scores are interpretable).
    let mut fp_g: DiGraph<usize, f64> = DiGraph::new();
    let mut fp_idx: Vec<NodeIndex> = Vec::with_capacity(fp_nodes.len());
    for i in 0..fp_nodes.len() {
        fp_idx.push(fp_g.add_node(i));
    }
    for (u, v) in &fp_edges {
        fp_g.update_edge(fp_idx[*u], fp_idx[*v], 1.0);
    }
    let fp_pr = pagerank_auto(&fp_g);
    let fp_bc = betweenness_centrality(&fp_g);

    // Personalized PageRank “what do boundary-heavy entrypoints lean on?”
    //
    // This is a cheap approximation of “cargo tree transitive importance” that
    // respects graph structure without enumerating full trees in the UI.
    #[derive(Debug, Serialize)]
    struct PprSeed {
        seed: String,
        reachable_first_party: usize,
        top: Vec<(String, f64)>,
    }

    let cfg = PageRankConfig::default();
    let mut candidates: Vec<(usize, usize, usize, String)> = Vec::new(); // (fp_i, third, fp_deps, name)
    for (fp_i, &node_i) in fp_nodes.iter().enumerate() {
        let Some(r) = idx_to_row[node_i] else {
            continue;
        };
        if fp_dependents[fp_i] == 0 && fp_deps[fp_i] > 0 {
            candidates.push((
                fp_i,
                third_party_sets[fp_i].len(),
                fp_deps[fp_i],
                r.name.clone(),
            ));
        }
    }
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    let seeds = candidates.into_iter().take(6).collect::<Vec<_>>();

    let mut ppr_out: Vec<PprSeed> = Vec::new();
    let mut ppr_sum = vec![0.0_f64; fp_nodes.len()];
    for (seed_fp, _third, _deps, seed_name) in seeds {
        let mut personalization = vec![0.0; fp_nodes.len()];
        personalization[seed_fp] = 1.0;
        let scores = personalized_pagerank(&fp_g, cfg, &personalization);

        for (i, &s) in scores.iter().enumerate() {
            ppr_sum[i] += s;
        }

        let reachable_first_party = scores.iter().filter(|&&s| s > 1e-12).count();
        let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut top = Vec::new();
        for (i, s) in items {
            if i == seed_fp {
                continue;
            }
            if s <= 1e-12 {
                break; // everything after this is zero (disconnected components)
            }
            let node_i = fp_nodes[i];
            if let Some(r) = idx_to_row[node_i] {
                top.push((r.name.clone(), s));
            }
            if top.len() >= 20 {
                break;
            }
        }
        ppr_out.push(PprSeed {
            seed: seed_name,
            reachable_first_party,
            top,
        });
    }
    fs::write(
        out_dir.join("ppr.entrypoints.json"),
        serde_json::to_string_pretty(&ppr_out)?,
    )?;

    // Aggregate PPR mass across the chosen entrypoints (heuristic).
    // We filter out the seeds themselves (and other leaf/entrypoint crates) so this reads as
    // “shared foundations leaned on by selected entrypoints”, not “the entrypoints again”.
    let seed_names: HashSet<String> = ppr_out.iter().map(|s| s.seed.clone()).collect();
    let mut ppr_agg: Vec<(String, f64)> = Vec::new();
    for (i, &mass) in ppr_sum.iter().enumerate() {
        if mass <= 1e-12 {
            continue;
        }
        if fp_dependents[i] == 0 {
            continue;
        }
        let node_i = fp_nodes[i];
        if let Some(r) = idx_to_row[node_i] {
            if seed_names.contains(&r.name) {
                continue;
            }
            ppr_agg.push((r.name.clone(), mass));
        }
    }
    ppr_agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fs::write(
        out_dir.join("ppr.aggregate.json"),
        serde_json::to_string_pretty(&ppr_agg)?,
    )?;

    // Build TLC rows.
    let mut tlc = Vec::new();
    let mut git_by_repo: HashMap<String, GitRepoStats> = HashMap::new();
    for (fp_i, &node_i) in fp_nodes.iter().enumerate() {
        let Some(r) = idx_to_row[node_i] else {
            continue;
        };
        let depnt = fp_dependents[fp_i];
        let deps = fp_deps[fp_i];
        let third = third_party_sets[fp_i].len();
        let pr = fp_pr[fp_i];
        let bc = fp_bc[fp_i];

        // Simple scoring: emphasize “blast radius”, then centrality, then boundary complexity.
        // (log keeps the dependents term from dominating everything.)
        let score = (10.0 * ((depnt as f64) + 1.0).ln()) + (1000.0 * pr) + (1.0 * third as f64);

        let mut why_bits = Vec::new();
        if depnt >= 5 {
            why_bits.push(format!("many dependents ({depnt})"));
        }
        if pr >= 0.02 {
            why_bits.push(format!("high pagerank ({:.4})", pr));
        }
        if third >= 10 {
            why_bits.push(format!("many third-party deps ({third})"));
        }
        if why_bits.is_empty() {
            why_bits.push("moderate centrality/complexity".to_string());
        }

        // Infer repo from manifest_path.
        // The "super-workspace" contains nested repo layouts; special-case `_mcp/*` so those
        // crates become their own repos (e.g. `_mcp/axum-mcp` -> repo `axum-mcp`).
        let repo = infer_repo_for_manifest(root, &r.manifest_path);
        let axis = token_axis
            .get(&r.name)
            .or_else(|| token_axis.get(&repo))
            .cloned()
            .unwrap_or_else(|| "other".to_string());

        let gs = git_by_repo
            .entry(repo.clone())
            .or_insert_with(|| git_repo_stats(&root.join(&repo)));

        tlc.push(TlcCrateRow {
            repo,
            axis,
            name: r.name.clone(),
            manifest_path: r.manifest_path.clone(),
            origin: r.origin,
            pagerank: pr,
            betweenness: bc,
            transitive_dependents: depnt,
            transitive_dependencies: deps,
            third_party_deps: third,
            score,
            why: why_bits.join("; "),
            repo_git_commits_30d: gs.commits_30d,
            repo_git_days_since_last_commit: gs.days_since_last_commit,
        });
    }

    tlc.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fs::write(
        out_dir.join("tlc.crates.json"),
        serde_json::to_string_pretty(&tlc)?,
    )?;
    Ok(tlc)
}

fn compute_repo_graph_from_live_metadata(
    root: &Path,
    out_dir: &Path,
) -> Result<(Vec<RepoRow>, RepoAxesSummary)> {
    // Critique of the old approach:
    // - Using `repo_dependency_edges` was a derived artifact that can drift/stale.
    // - Collapsing A->B edges with `update_edge` silently threw away multiplicity.
    //
    // New approach:
    // - Use live `cargo metadata` from the root workspace.
    // - Collapse package-level edges to repo-level edges with integer weights (# of package edges).

    // Load axes mapping from dev_repos_overview.json if present (optional).
    let overview_path = default_overview_path(root);
    // Note: the overview "axes" lists are *usually repo names*, but often overlap
    // with package names. We support both, and prefer a repo-level match when available.
    let mut token_axis: HashMap<String, String> = HashMap::new();
    if overview_path.exists() {
        let raw = fs::read_to_string(&overview_path)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        if let Some(axes) = v.get("axes").and_then(|a| a.as_object()) {
            for (axis, arr) in axes {
                if let Some(arr) = arr.as_array() {
                    for x in arr {
                        if let Some(name) = x.as_str() {
                            token_axis.insert(name.to_string(), axis.clone());
                        }
                    }
                }
            }
        }
    }

    // Get full workspace metadata once.
    //
    // Important nuance: Cargo's resolved graph depends on:
    // - feature selection
    // - target triple (cargo metadata has --filter-platform)
    //
    // For now we keep this aligned with pkgrank defaults: normal deps only.
    let analyze = AnalyzeArgs {
        path: root.to_path_buf(),
        metric: Metric::Pagerank,
        top: 25,
        dev: false,
        build: false,
        // We'll compute a "local graph" from the resolve graph instead of restricting
        // to workspace_members only.
        workspace_only: false,
        all_features: false,
        no_default_features: false,
        features: None,
        format: OutputFormat::Json,
        stats: false,
        json_limit: None,
    };
    let manifest_path = manifest_path(&analyze.path)?;
    let metadata = metadata_for(&manifest_path, &analyze)?;
    let resolve = metadata
        .resolve
        .as_ref()
        .ok_or_else(|| anyhow!("cargo metadata did not include `resolve`"))?;

    // Build the set of "local" packages: any package whose manifest_path is under `root`.
    // This includes workspace members and any path deps pulled in from other local repos.
    let pkg_by_id: HashMap<&PackageId, &cargo_metadata::Package> =
        metadata.packages.iter().map(|p| (&p.id, p)).collect();
    let mut local_pkg_ids: HashSet<PackageId> = HashSet::new();
    for p in &metadata.packages {
        let mp = PathBuf::from(p.manifest_path.to_string());
        if mp.strip_prefix(root).is_ok() {
            local_pkg_ids.insert(p.id.clone());
        }
    }

    // Map package id -> repo dir (heuristic rooted at `root`).
    let mut repo_for_pkg: HashMap<&PackageId, String> = HashMap::new();
    let mut pkgs_in_repo: HashMap<String, Vec<String>> = HashMap::new();
    for id in &local_pkg_ids {
        let pkg = pkg_by_id
            .get(id)
            .copied()
            .ok_or_else(|| anyhow!("missing pkg for local id"))?;
        let mp = PathBuf::from(pkg.manifest_path.to_string());
        let repo = infer_repo_for_manifest(root, &mp.to_string_lossy());
        repo_for_pkg.insert(id, repo.clone());
        pkgs_in_repo
            .entry(repo)
            .or_default()
            .push(pkg.name.to_string());
    }

    // Repo axis mass: count how many packages in each repo are in each axis.
    let mut repo_axis_mass: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for (repo, pkgs) in &pkgs_in_repo {
        let mut m: HashMap<String, usize> = HashMap::new();
        for p in pkgs {
            let axis = token_axis
                .get(p)
                .cloned()
                .unwrap_or_else(|| "other".to_string());
            *m.entry(axis).or_insert(0) += 1;
        }
        repo_axis_mass.insert(repo.clone(), m);
    }

    // Collapse edges with multiplicity as weight.
    let mut edge_w: HashMap<(String, String), usize> = HashMap::new();
    // Boundary complexity: unique third-party deps per repo.
    let mut repo_third_party: HashMap<String, HashSet<String>> = HashMap::new();
    for node in &resolve.nodes {
        if !local_pkg_ids.contains(&node.id) {
            continue;
        }
        let from_repo = repo_for_pkg
            .get(&node.id)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        for dep in &node.deps {
            if !dep_kind_allowed(&dep.dep_kinds, &analyze) {
                continue;
            }
            let dep_id = &dep.pkg;
            if !local_pkg_ids.contains(dep_id) {
                // local -> third-party boundary (still record it for TLC)
                if let Some(p) = pkg_by_id.get(dep_id).copied() {
                    repo_third_party
                        .entry(from_repo.clone())
                        .or_default()
                        .insert(p.name.to_string());
                } else {
                    repo_third_party
                        .entry(from_repo.clone())
                        .or_default()
                        .insert("<unknown>".to_string());
                }
                continue;
            }
            let to_repo = repo_for_pkg
                .get(dep_id)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            if from_repo == to_repo {
                continue;
            }
            *edge_w.entry((from_repo.clone(), to_repo)).or_insert(0) += 1;
        }
    }

    // Nodes = repos present in workspace.
    let mut repos: Vec<String> = pkgs_in_repo.keys().cloned().collect();
    repos.sort();

    let mut g: DiGraph<String, f64> = DiGraph::new();
    let mut idx: HashMap<String, NodeIndex> = HashMap::new();
    for r in &repos {
        idx.insert(r.clone(), g.add_node(r.clone()));
    }
    let mut edges: Vec<(String, String)> = Vec::new();
    for ((a, b), w) in &edge_w {
        if let (Some(&ia), Some(&ib)) = (idx.get(a), idx.get(b)) {
            g.update_edge(ia, ib, *w as f64);
            edges.push((a.clone(), b.clone()));
        }
    }

    // Two interpretations:
    // - depends_pr: PageRank on A->B (dependencies)
    // - consumers_pr: PageRank on reversed graph (consumers/orchestrators)
    let depends_pr = pagerank_auto(&g);
    let mut g_rev: DiGraph<String, f64> = DiGraph::new();
    let mut idx2: HashMap<String, NodeIndex> = HashMap::new();
    for r in &repos {
        idx2.insert(r.clone(), g_rev.add_node(r.clone()));
    }
    for ((a, b), w) in &edge_w {
        // reverse edge
        if let (Some(&ib), Some(&ia)) = (idx2.get(b), idx2.get(a)) {
            g_rev.update_edge(ib, ia, *w as f64);
        }
    }
    let consumers_pr = pagerank_auto(&g_rev);

    // Transitive counts computed on unweighted reachability.
    let (dependents, deps) = reachability_counts(&repos, &edges);

    // Axis label per repo.
    let mut axis_by_repo: HashMap<String, String> = HashMap::new();
    for r in &repos {
        axis_by_repo.insert(r.clone(), axis_for_repo(r, &token_axis, &repo_axis_mass));
    }

    // Git stats per repo (best-effort).
    let mut git_by_repo: HashMap<String, GitRepoStats> = HashMap::new();
    for r in &repos {
        git_by_repo.insert(r.clone(), git_repo_stats(&root.join(r)));
    }

    let mut rows = Vec::new();
    for r in &repos {
        let i = idx[r];
        let indeg = g.neighbors_directed(i, Direction::Incoming).count();
        let outdeg = g.neighbors_directed(i, Direction::Outgoing).count();
        let in_w: f64 = g
            .edges_directed(i, Direction::Incoming)
            .map(|e| *e.weight())
            .sum();
        let out_w: f64 = g
            .edges_directed(i, Direction::Outgoing)
            .map(|e| *e.weight())
            .sum();
        let axis = axis_by_repo
            .get(r)
            .cloned()
            .unwrap_or_else(|| "other".to_string());
        let consumers = consumers_pr[idx2[r].index()];
        let third = repo_third_party.get(r).map(|s| s.len()).unwrap_or(0);
        let gs = git_by_repo.get(r).copied().unwrap_or_default();
        rows.push(RepoRow {
            repo: r.clone(),
            axis,
            in_degree: indeg,
            out_degree: outdeg,
            in_weight: pos_zero(in_w),
            out_weight: pos_zero(out_w),
            pagerank: depends_pr[i.index()],
            consumers_pagerank: consumers,
            transitive_dependents: *dependents.get(r).unwrap_or(&0),
            transitive_dependencies: *deps.get(r).unwrap_or(&0),
            third_party_deps: third,
            git_commits_30d: gs.commits_30d,
            git_days_since_last_commit: gs.days_since_last_commit,
        });
    }
    rows.sort_by(|a, b| {
        b.pagerank
            .partial_cmp(&a.pagerank)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Axis totals for depends_pr.
    let mut totals: HashMap<String, f64> = HashMap::new();
    for rr in &rows {
        *totals.entry(rr.axis.clone()).or_insert(0.0) += rr.pagerank;
    }
    let axes_summary = RepoAxesSummary { totals };

    // Invariants / forbidden cross-axis edges (repo-level).
    //
    // These are workspace-level “shape constraints”, not Rust type constraints:
    // - Tekne should not depend on Agents or Governance.
    // - Governance should not depend on Agents.
    //
    // (Agents depending on Tekne is fine; Agents depending on Governance can be OK,
    // but it’s still worth surfacing separately later.)
    let mut violations: Vec<RepoInvariantViolation> = Vec::new();
    for ((a, b), w) in &edge_w {
        let ax = axis_by_repo
            .get(a)
            .cloned()
            .unwrap_or_else(|| "other".to_string());
        let bx = axis_by_repo
            .get(b)
            .cloned()
            .unwrap_or_else(|| "other".to_string());

        let tekne_forbidden = ax == "tekne" && (bx == "agents" || bx == "governance");
        let governance_forbidden = ax == "governance" && bx == "agents";

        if tekne_forbidden {
            violations.push(RepoInvariantViolation {
                rule: "tekne_must_not_depend_on_agents_or_governance".to_string(),
                from_repo: a.clone(),
                from_axis: ax.clone(),
                to_repo: b.clone(),
                to_axis: bx.clone(),
                weight: *w,
            });
        }

        if governance_forbidden {
            violations.push(RepoInvariantViolation {
                rule: "governance_must_not_depend_on_agents".to_string(),
                from_repo: a.clone(),
                from_axis: ax,
                to_repo: b.clone(),
                to_axis: bx,
                weight: *w,
            });
        }
    }
    violations.sort_by(|a, b| b.weight.cmp(&a.weight));

    // TLC for repos: central + boundary-heavy + violations.
    let mut violation_weight_by_repo: HashMap<String, usize> = HashMap::new();
    for v in &violations {
        *violation_weight_by_repo
            .entry(v.from_repo.clone())
            .or_insert(0) += v.weight;
    }
    let mut tlc_repos: Vec<TlcRepoRow> = Vec::new();
    for r in &rows {
        let vw = *violation_weight_by_repo.get(&r.repo).unwrap_or(&0);
        let score = (500.0 * r.pagerank)
            + (500.0 * r.consumers_pagerank)
            + (10.0 * ((r.transitive_dependents as f64) + 1.0).ln())
            + (r.third_party_deps as f64)
            + (vw as f64);

        let mut why_bits = Vec::new();
        if r.consumers_pagerank >= 0.03 {
            why_bits.push(format!("high consumers_pr ({:.4})", r.consumers_pagerank));
        }
        if r.pagerank >= 0.03 {
            why_bits.push(format!("high deps_pr ({:.4})", r.pagerank));
        }
        if r.transitive_dependents >= 5 {
            why_bits.push(format!("many dependents ({})", r.transitive_dependents));
        }
        if r.third_party_deps >= 10 {
            why_bits.push(format!("many third-party deps ({})", r.third_party_deps));
        }
        if vw > 0 {
            why_bits.push(format!("invariant violations weight ({vw})"));
        }
        if why_bits.is_empty() {
            why_bits.push("moderate centrality/complexity".to_string());
        }

        tlc_repos.push(TlcRepoRow {
            repo: r.repo.clone(),
            axis: r.axis.clone(),
            deps_pagerank: r.pagerank,
            consumers_pagerank: r.consumers_pagerank,
            transitive_dependents: r.transitive_dependents,
            transitive_dependencies: r.transitive_dependencies,
            third_party_deps: r.third_party_deps,
            violation_weight: vw,
            score,
            why: why_bits.join("; "),
            git_commits_30d: r.git_commits_30d,
            git_days_since_last_commit: r.git_days_since_last_commit,
        });
    }
    tlc_repos.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Write artifacts including both PR directions.
    fs::write(
        out_dir.join("ecosystem.repo_graph.rows.json"),
        serde_json::to_string_pretty(&rows)?,
    )?;
    fs::write(
        out_dir.join("ecosystem.axes.pagerank.json"),
        serde_json::to_string_pretty(&axes_summary)?,
    )?;
    fs::write(
        out_dir.join("ecosystem.invariants.violations.json"),
        serde_json::to_string_pretty(&violations)?,
    )?;
    fs::write(
        out_dir.join("tlc.repos.json"),
        serde_json::to_string_pretty(&tlc_repos)?,
    )?;
    // Keep a dedicated consumer-PR artifact too (easy diff/debug).
    let mut consumers: Vec<(String, f64)> = rows
        .iter()
        .map(|r| (r.repo.clone(), r.consumers_pagerank))
        .collect();
    consumers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fs::write(
        out_dir.join("ecosystem.repo_graph.consumers_pagerank.json"),
        serde_json::to_string_pretty(&consumers)?,
    )?;

    Ok((rows, axes_summary))
}

#[derive(Debug, Serialize)]
struct SweepSummary {
    mode: SweepMode,
    ok: Vec<String>,
    failed: Vec<SweepFailure>,
    missing: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SweepFailure {
    repo: String,
    error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecentFileRow {
    /// Absolute path.
    path: String,
    /// Top-level repo directory under the dev root (first path component).
    repo: String,
    /// Best-effort crate name (workspace member) owning this path.
    crate_name: Option<String>,
    /// Manifest path for the owning crate (if any).
    crate_manifest_path: Option<String>,
    /// Modified time (unix seconds).
    modified_unix: i64,
    /// Age at generation time (seconds).
    age_seconds: i64,
    /// File size (bytes).
    size_bytes: u64,
    /// TLC score for the owning crate (if `tlc.crates.json` exists).
    tlc_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecentFilesSummary {
    generated_at_unix: i64,
    window_days: u64,
    scanned_roots: Vec<String>,
    rows: usize,
    by_repo: Vec<(String, usize)>,
    by_crate: Vec<(String, usize)>,
}

fn should_skip_dir_name(name: &str) -> bool {
    matches!(
        name,
        ".git" | "target" | "node_modules" | ".venv" | "__pycache__"
    )
}

fn write_recent_files_artifacts(
    root: &Path,
    out_dir: &Path,
    recent_days: u64,
    recent_max: usize,
) -> Result<()> {
    let now = SystemTime::now();
    let window = Duration::from_secs(recent_days.saturating_mul(24 * 60 * 60));
    let cutoff = now.checked_sub(window).unwrap_or(UNIX_EPOCH);

    // Build a best-effort crate-root map from workspace metadata.
    // We only need workspace member manifests and their directories.
    let analyze = AnalyzeArgs {
        path: root.to_path_buf(),
        metric: Metric::Pagerank,
        top: 10,
        dev: false,
        build: false,
        workspace_only: true,
        all_features: false,
        no_default_features: false,
        features: None,
        format: OutputFormat::Json,
        stats: false,
        json_limit: None,
    };
    let rows = analyze_rows(&analyze).unwrap_or_default();
    let mut crate_roots: Vec<(PathBuf, String, String)> = rows
        .into_iter()
        .filter(|r| matches!(r.origin, PackageOrigin::WorkspaceMember))
        .map(|r| {
            let mp = PathBuf::from(&r.manifest_path);
            let dir = mp.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
            (dir, r.name, r.manifest_path)
        })
        .collect();
    // Prefer longest prefix match.
    crate_roots.sort_by_key(|(p, _, _)| std::cmp::Reverse(p.components().count()));

    // Optional TLC mapping: manifest_path -> score
    let mut tlc_by_manifest: HashMap<String, f64> = HashMap::new();
    let tlc_path = out_dir.join("tlc.crates.json");
    if tlc_path.exists() {
        if let Ok(raw) = fs::read_to_string(&tlc_path) {
            if let Ok(v) = serde_json::from_str::<Vec<TlcCrateRow>>(&raw) {
                for row in v {
                    tlc_by_manifest.insert(row.manifest_path, row.score);
                }
            }
        }
    }

    // Scan roots:
    // - each workspace member crate dir
    // - plus a couple “context” dirs that influence this workspace
    let mut scan_roots: Vec<PathBuf> = crate_roots.iter().map(|(p, _, _)| p.clone()).collect();
    let rules_dir = root.join(".cursor").join("rules");
    if rules_dir.exists() {
        scan_roots.push(rules_dir);
    }
    let docs_dir = root.join("docs");
    if docs_dir.exists() {
        scan_roots.push(docs_dir);
    }
    // Dedup roots.
    scan_roots.sort();
    scan_roots.dedup();

    let mut out: Vec<RecentFileRow> = Vec::new();

    // Directory walk (bounded by scan_roots, with a few hard skips).
    for scan_root in &scan_roots {
        let mut q: VecDeque<PathBuf> = VecDeque::new();
        q.push_back(scan_root.clone());
        while let Some(dir) = q.pop_front() {
            let read_dir = match fs::read_dir(&dir) {
                Ok(v) => v,
                Err(_) => continue,
            };
            for entry in read_dir.flatten() {
                let path = entry.path();
                let ft = match entry.file_type() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if ft.is_dir() {
                    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                        if should_skip_dir_name(name) {
                            continue;
                        }
                    }
                    q.push_back(path);
                    continue;
                }
                if !ft.is_file() {
                    continue;
                }
                let md = match entry.metadata() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let modified = match md.modified() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if modified < cutoff {
                    continue;
                }

                let modified_unix = modified
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                let age_seconds = now
                    .duration_since(modified)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);

                let rel = path.strip_prefix(root).ok();
                let repo = rel
                    .and_then(|rp| {
                        rp.components()
                            .next()
                            .map(|c| c.as_os_str().to_string_lossy().to_string())
                    })
                    .unwrap_or_else(|| "<outside-root>".to_string());

                // Find owning crate via longest prefix match.
                let mut crate_name: Option<String> = None;
                let mut crate_manifest_path: Option<String> = None;
                for (crate_dir, name, mp) in &crate_roots {
                    if path.starts_with(crate_dir) {
                        crate_name = Some(name.clone());
                        crate_manifest_path = Some(mp.clone());
                        break;
                    }
                }
                let tlc_score = crate_manifest_path
                    .as_ref()
                    .and_then(|mp| tlc_by_manifest.get(mp).copied());

                out.push(RecentFileRow {
                    path: path.display().to_string(),
                    repo,
                    crate_name,
                    crate_manifest_path,
                    modified_unix,
                    age_seconds,
                    size_bytes: md.len(),
                    tlc_score,
                });
            }
        }
    }

    out.sort_by(|a, b| b.modified_unix.cmp(&a.modified_unix));
    if out.len() > recent_max {
        out.truncate(recent_max);
    }

    // Aggregate counts.
    let mut by_repo: HashMap<String, usize> = HashMap::new();
    let mut by_crate: HashMap<String, usize> = HashMap::new();
    for r in &out {
        *by_repo.entry(r.repo.clone()).or_default() += 1;
        if let Some(c) = &r.crate_name {
            *by_crate.entry(c.clone()).or_default() += 1;
        }
    }
    let mut by_repo = by_repo.into_iter().collect::<Vec<_>>();
    by_repo.sort_by(|a, b| b.1.cmp(&a.1));
    let mut by_crate = by_crate.into_iter().collect::<Vec<_>>();
    by_crate.sort_by(|a, b| b.1.cmp(&a.1));

    let generated_at_unix = now
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let summary = RecentFilesSummary {
        generated_at_unix,
        window_days: recent_days,
        scanned_roots: scan_roots.iter().map(|p| p.display().to_string()).collect(),
        rows: out.len(),
        by_repo,
        by_crate,
    };

    fs::write(
        out_dir.join("recent.files.json"),
        serde_json::to_string_pretty(&out)?,
    )?;
    fs::write(
        out_dir.join("recent.summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(())
}

fn run_sweep_local(args: &SweepLocalArgs) -> Result<()> {
    let root = args
        .root
        .canonicalize()
        .unwrap_or_else(|_| args.root.clone());
    let out_dir = root.join(&args.out);
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create output dir {}", out_dir.display()))?;

    let overview_path = args
        .overview
        .clone()
        .unwrap_or_else(|| default_overview_path(&root));

    let mut summary = SweepSummary {
        mode: args.mode,
        ok: vec![],
        failed: vec![],
        missing: vec![],
    };

    match args.mode {
        SweepMode::WorkspaceSlice => {
            // Run once on the root workspace.
            let analyze = AnalyzeArgs {
                path: root.clone(),
                metric: Metric::Pagerank,
                top: args.top,
                dev: args.dev,
                build: args.build,
                workspace_only: true,
                all_features: false,
                no_default_features: false,
                features: None,
                format: OutputFormat::Json,
                stats: false,
                json_limit: None,
            };
            let rows = analyze_rows(&analyze)?;

            // Write the full rows once (stable name).
            fs::write(
                out_dir.join("root.workspace_only.json"),
                serde_json::to_string_pretty(&rows)?,
            )?;

            // Slice rows by first path component (repo dir).
            let mut by_repo: HashMap<String, Vec<Row>> = HashMap::new();
            for r in rows {
                let mp = PathBuf::from(&r.manifest_path);
                let rel = match mp.strip_prefix(&root) {
                    Ok(rp) => rp,
                    Err(_) => continue,
                };
                let repo = rel
                    .components()
                    .next()
                    .map(|c| c.as_os_str().to_string_lossy().to_string());
                let Some(repo) = repo else {
                    continue;
                };
                by_repo.entry(repo).or_default().push(r);
            }

            // Write per-repo text summaries.
            let mut repos: Vec<String> = by_repo.keys().cloned().collect();
            repos.sort();
            for repo in repos {
                let mut rows = by_repo.remove(&repo).unwrap_or_default();
                rows.sort_by(|a, b| {
                    b.pagerank
                        .partial_cmp(&a.pagerank)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut lines = Vec::new();
                lines.push(format!("{}: {} workspace crates", repo, rows.len()));
                lines.push(
                    "------------------------------------------------------------------------"
                        .to_string(),
                );
                for (i, r) in rows.iter().take(args.top).enumerate() {
                    lines.push(format!(
                        "{:3}. {:28} [{:>12}] in={:2} out={:2} pr={:.6}",
                        i + 1,
                        r.name,
                        format!("{:?}", r.origin),
                        r.in_degree,
                        r.out_degree,
                        r.pagerank
                    ));
                }
                fs::write(
                    out_dir.join(format!("{}.top{}.txt", repo, args.top)),
                    lines.join("\n") + "\n",
                )?;
                summary.ok.push(repo);
            }
        }
        SweepMode::RepoRoots => {
            // Repo-local mode: use overview list if present, else fall back to immediate children with Cargo.toml.
            let repos = if overview_path.exists() {
                let raw = fs::read_to_string(&overview_path)
                    .with_context(|| format!("failed to read {}", overview_path.display()))?;
                let v: serde_json::Value = serde_json::from_str(&raw)?;
                let arr = v
                    .get("member_repos")
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| anyhow!("overview missing member_repos array"))?;
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            } else {
                // Limited fallback: immediate directories under root with Cargo.toml.
                let mut out = Vec::new();
                for entry in fs::read_dir(&root)? {
                    let entry = entry?;
                    if !entry.file_type()?.is_dir() {
                        continue;
                    }
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with('.') || name.starts_with('_') {
                        continue;
                    }
                    if entry.path().join("Cargo.toml").exists() {
                        out.push(name);
                    }
                }
                out
            };

            for repo in repos {
                let repo_dir = root.join(&repo);
                let cargo_toml = repo_dir.join("Cargo.toml");
                if !cargo_toml.exists() {
                    summary.missing.push(repo);
                    continue;
                }

                let analyze = AnalyzeArgs {
                    path: repo_dir.clone(),
                    metric: Metric::Pagerank,
                    top: args.top,
                    dev: args.dev,
                    build: args.build,
                    workspace_only: true,
                    all_features: false,
                    no_default_features: false,
                    features: None,
                    format: OutputFormat::Json,
                    stats: false,
                    json_limit: None,
                };

                match analyze_rows(&analyze) {
                    Ok(rows) => {
                        fs::write(
                            out_dir.join(format!("{}.workspace_only.json", repo)),
                            serde_json::to_string_pretty(&rows)?,
                        )?;
                        // Small text file too.
                        let mut lines = Vec::new();
                        let mut sorted = rows.clone();
                        sorted.sort_by(|a, b| {
                            b.pagerank
                                .partial_cmp(&a.pagerank)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        lines.push(format!("{}: {} workspace crates", repo, sorted.len()));
                        lines.push("------------------------------------------------------------------------".to_string());
                        for (i, r) in sorted.iter().take(args.top).enumerate() {
                            lines.push(format!(
                                "{:3}. {:28} [{:>12}] in={:2} out={:2} pr={:.6}",
                                i + 1,
                                r.name,
                                format!("{:?}", r.origin),
                                r.in_degree,
                                r.out_degree,
                                r.pagerank
                            ));
                        }
                        fs::write(
                            out_dir.join(format!("{}.top{}.txt", repo, args.top)),
                            lines.join("\n") + "\n",
                        )?;
                        summary.ok.push(repo);
                    }
                    Err(e) => {
                        summary.failed.push(SweepFailure {
                            repo,
                            error: format!("{:#}", e),
                        });
                    }
                }
            }
        }
    }

    fs::write(
        out_dir.join("sweep.summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;

    // Optional: “recently modified files” view (mtime-based, bounded scan).
    // This is an intentionally git-free signal: the dev folder is a super-workspace.
    if args.recent {
        write_recent_files_artifacts(&root, &out_dir, args.recent_days, args.recent_max)?;
    }
    Ok(())
}

fn analyze_rows(args: &AnalyzeArgs) -> Result<Vec<Row>> {
    let manifest_path = manifest_path(&args.path)?;
    let metadata = metadata_for(&manifest_path, args)
        .with_context(|| format!("cargo metadata failed for {}", manifest_path.display()))?;
    let (graph, nodes) = build_graph(&metadata, args)?;
    Ok(compute_rows(&metadata, &graph, &nodes))
}

fn analyze_rows_with_convergence(args: &AnalyzeArgs) -> Result<(Vec<Row>, serde_json::Value)> {
    let manifest_path = manifest_path(&args.path)?;
    let metadata = metadata_for(&manifest_path, args)
        .with_context(|| format!("cargo metadata failed for {}", manifest_path.display()))?;
    let (graph, _nodes) = build_graph(&metadata, args)?;
    let (mut rows, convergence) = compute_rows_with_convergence(&metadata, &graph);
    sort_rows_by_metric(&mut rows, args.metric);
    Ok((rows, convergence))
}

#[cfg(feature = "stdio")]
fn parse_modules_preset(s: &str) -> Result<ModulesPreset, McpError> {
    match s {
        "none" => Ok(ModulesPreset::None),
        "file-full" => Ok(ModulesPreset::FileFull),
        "file-api" => Ok(ModulesPreset::FileApi),
        "node-full" => Ok(ModulesPreset::NodeFull),
        "node-api" => Ok(ModulesPreset::NodeApi),
        other => Err(McpError::invalid_params(
            format!("preset must be one of: none, file-full, file-api, node-full, node-api (got {other})"),
            None,
        )),
    }
}

#[cfg(feature = "stdio")]
fn parse_metric(s: &str) -> Result<Metric, McpError> {
    match s {
        "pagerank" => Ok(Metric::Pagerank),
        "consumers-pagerank" => Ok(Metric::ConsumersPagerank),
        "indegree" => Ok(Metric::Indegree),
        "outdegree" => Ok(Metric::Outdegree),
        "betweenness" => Ok(Metric::Betweenness),
        other => Err(McpError::invalid_params(
            format!("metric must be one of: pagerank, consumers-pagerank, indegree, outdegree, betweenness (got {other})"),
            None,
        )),
    }
}

#[cfg(feature = "stdio")]
fn parse_edge_kind(s: &str) -> Result<ModuleEdgeKind, McpError> {
    match s {
        "uses" => Ok(ModuleEdgeKind::Uses),
        "owns" => Ok(ModuleEdgeKind::Owns),
        "both" => Ok(ModuleEdgeKind::Both),
        other => Err(McpError::invalid_params(
            format!("edge_kind must be one of: uses, owns, both (got {other})"),
            None,
        )),
    }
}

#[cfg(feature = "stdio")]
fn parse_aggregate(s: &str) -> Result<ModuleAggregate, McpError> {
    match s {
        "node" => Ok(ModuleAggregate::Node),
        "module" => Ok(ModuleAggregate::Module),
        "file" => Ok(ModuleAggregate::File),
        other => Err(McpError::invalid_params(
            format!("aggregate must be one of: node, module, file (got {other})"),
            None,
        )),
    }
}

#[cfg(feature = "stdio")]
fn modules_args_from_tool_params(p: &PkgrankModulesToolArgs) -> Result<ModulesArgs, McpError> {
    let mut args = ModulesArgs {
        manifest_path: PathBuf::from(
            p.manifest_path
                .clone()
                .unwrap_or_else(|| "Cargo.toml".to_string()),
        ),
        package: p.package.clone(),
        lib: p.lib.unwrap_or(false),
        bin: p.bin.clone(),
        cfg_test: p.cfg_test.unwrap_or(false),
        metric: p
            .metric
            .as_deref()
            .map(parse_metric)
            .transpose()?
            .unwrap_or(Metric::Pagerank),
        preset: p
            .preset
            .as_deref()
            .map(parse_modules_preset)
            .transpose()?
            .unwrap_or(ModulesPreset::None),
        top: p.top.unwrap_or(25).min(500),
        format: OutputFormat::Json,
        edge_kind: p
            .edge_kind
            .as_deref()
            .map(parse_edge_kind)
            .transpose()?
            .unwrap_or(ModuleEdgeKind::Uses),
        aggregate: p
            .aggregate
            .as_deref()
            .map(parse_aggregate)
            .transpose()?
            .unwrap_or(ModuleAggregate::Node),
        uses_weight: 1.0,
        owns_weight: 0.2,
        edges_top: 0,
        members_top: 3,
        cache: p.cache.unwrap_or(true),
        cache_refresh: p.cache_refresh.unwrap_or(false),
        no_externs: true,
        include_externs: p.include_externs.unwrap_or(false),
        no_sysroot: true,
        include_sysroot: p.include_sysroot.unwrap_or(false),
        no_fns: true,
        include_fns: p.include_fns.unwrap_or(false),
        no_traits: true,
        include_traits: p.include_traits.unwrap_or(false),
        no_types: true,
        include_types: p.include_types.unwrap_or(false),
    };

    // Override include flags if requested.
    if let Some(v) = p.include_fns {
        args.include_fns = v;
    }
    if let Some(v) = p.include_types {
        args.include_types = v;
    }
    if let Some(v) = p.include_traits {
        args.include_traits = v;
    }

    // Better UX: fail fast with invalid_params (not internal_error) when `package` is required.
    if args.package.is_none() {
        if let Ok(raw) = fs::read_to_string(&args.manifest_path) {
            if raw.contains("[workspace]") {
                return Err(McpError::invalid_params(
                    "missing required parameter `package` when `manifest_path` points at a workspace.\n\
                     Fix: pass `package` (crate name), or set `manifest_path` to the crate's Cargo.toml."
                        .to_string(),
                    None,
                ));
            }
        }
    }

    Ok(apply_modules_preset(&args))
}

#[cfg(feature = "stdio")]
fn modules_sweep_args_from_tool_params(
    p: &PkgrankModulesSweepToolArgs,
) -> Result<ModulesSweepArgs, McpError> {
    let packages = p.packages.clone().unwrap_or_default();
    // Better UX: if this looks like a workspace, require either explicit packages or all_packages=true.
    let manifest_path = PathBuf::from(
        p.manifest_path
            .clone()
            .unwrap_or_else(|| "Cargo.toml".to_string()),
    );
    if !p.all_packages.unwrap_or(false) && packages.is_empty() {
        if let Ok(raw) = fs::read_to_string(&manifest_path) {
            if raw.contains("[workspace]") {
                return Err(McpError::invalid_params(
                    "modules-sweep requires either `packages` or `all_packages=true` when `manifest_path` points at a workspace."
                        .to_string(),
                    None,
                ));
            }
        }
    }
    Ok(ModulesSweepArgs {
        manifest_path,
        package: packages,
        all_packages: p.all_packages.unwrap_or(false),
        lib: p.lib.unwrap_or(false),
        bin: p.bin.clone(),
        cfg_test: p.cfg_test.unwrap_or(false),
        metric: p
            .metric
            .as_deref()
            .map(parse_metric)
            .transpose()?
            .unwrap_or(Metric::Pagerank),
        preset: p
            .preset
            .as_deref()
            .map(parse_modules_preset)
            .transpose()?
            .unwrap_or(ModulesPreset::None),
        summary_only: true,
        continue_on_error: p.continue_on_error.unwrap_or(true),
        fail_fast: p.fail_fast.unwrap_or(false),
        top: p.top.unwrap_or(12).min(200),
        format: OutputFormat::Json,
        edge_kind: ModuleEdgeKind::Uses,
        aggregate: ModuleAggregate::File,
        uses_weight: 1.0,
        owns_weight: 0.2,
        edges_top: 0,
        members_top: 3,
        cache: p.cache.unwrap_or(true),
        cache_refresh: p.cache_refresh.unwrap_or(false),
        no_externs: true,
        include_externs: false,
        no_sysroot: true,
        include_sysroot: false,
        no_fns: true,
        include_fns: false,
        no_traits: true,
        include_traits: false,
        no_types: true,
        include_types: false,
    })
}

#[cfg(feature = "stdio")]
fn modules_sweep_payload(
    args: &ModulesSweepArgs,
    include_rows: bool,
    include_top_edges: bool,
) -> Result<serde_json::Value> {
    // Reuse the existing CLI logic by running the per-package loop and constructing
    // a JSON payload analogous to the CLI JSON mode, but kept in-memory.
    //
    // Note: This intentionally uses only package-level return values (no stdout printing).
    let packages: Vec<String> = if !args.package.is_empty() {
        args.package.clone()
    } else if args.all_packages {
        let mut cmd = MetadataCommand::new();
        cmd.manifest_path(&args.manifest_path);
        let md = cmd
            .exec()
            .map_err(|e| anyhow!(e))
            .with_context(|| "cargo metadata failed for modules-sweep")?;
        let mut names: Vec<String> = md
            .workspace_members
            .iter()
            .filter_map(|id| {
                md.packages
                    .iter()
                    .find(|p| &p.id == id)
                    .map(|p| p.name.to_string())
            })
            .collect();
        names.sort();
        names
    } else {
        return Err(anyhow!(
            "modules-sweep requires at least one package, or all_packages=true"
        ));
    };

    let template = apply_modules_preset(&ModulesArgs {
        manifest_path: args.manifest_path.clone(),
        package: None,
        lib: args.lib,
        bin: args.bin.clone(),
        cfg_test: args.cfg_test,
        metric: args.metric,
        preset: args.preset,
        top: args.top,
        format: OutputFormat::Json,
        edge_kind: args.edge_kind,
        aggregate: args.aggregate,
        uses_weight: args.uses_weight,
        owns_weight: args.owns_weight,
        edges_top: args.edges_top,
        members_top: args.members_top,
        cache: args.cache,
        cache_refresh: args.cache_refresh,
        no_externs: args.no_externs,
        include_externs: args.include_externs,
        no_sysroot: args.no_sysroot,
        include_sysroot: args.include_sysroot,
        no_fns: args.no_fns,
        include_fns: args.include_fns,
        no_traits: args.no_traits,
        include_traits: args.include_traits,
        no_types: args.no_types,
        include_types: args.include_types,
    });

    let continue_on_error = if args.fail_fast {
        false
    } else {
        args.continue_on_error
    };

    let mut out_pkgs = serde_json::Map::new();
    for pkg in packages {
        let mut one_args = template.clone();
        one_args.package = Some(pkg.clone());
        match run_modules_core(&one_args) {
            Ok((mut rows, nodes, edges, aggregate_label, top_edges)) => {
                let rows_total = rows.len();
                let top_pr = rows
                    .iter()
                    .max_by(|a, b| {
                        a.pagerank
                            .partial_cmp(&b.pagerank)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| serde_json::json!({"node": r.node, "score": r.pagerank}));
                let top_cons = rows
                    .iter()
                    .max_by(|a, b| {
                        a.consumers_pagerank
                            .partial_cmp(&b.consumers_pagerank)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| serde_json::json!({"node": r.node, "score": r.consumers_pagerank}));
                let top_between = rows
                    .iter()
                    .max_by(|a, b| {
                        a.betweenness
                            .partial_cmp(&b.betweenness)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| serde_json::json!({"node": r.node, "score": r.betweenness}));

                // Respect top: sort by requested metric and truncate.
                rows.sort_by(|a, b| match args.metric {
                    Metric::Pagerank => b
                        .pagerank
                        .partial_cmp(&a.pagerank)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    Metric::ConsumersPagerank => b
                        .consumers_pagerank
                        .partial_cmp(&a.consumers_pagerank)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    Metric::Indegree => b.in_degree.cmp(&a.in_degree),
                    Metric::Outdegree => b.out_degree.cmp(&a.out_degree),
                    Metric::Betweenness => b
                        .betweenness
                        .partial_cmp(&a.betweenness)
                        .unwrap_or(std::cmp::Ordering::Equal),
                });
                let rows_returned = rows_total.min(args.top);
                let truncated = rows_returned < rows_total;
                if include_rows {
                    rows.truncate(args.top);
                }
                out_pkgs.insert(
                    pkg,
                    serde_json::json!({
                        "ok": true,
                        "error": null,
                        "nodes": nodes,
                        "edges": edges,
                        "aggregate_label": aggregate_label,
                        "rows_total": rows_total,
                        "rows_returned": if include_rows { Some(rows_returned) } else { None },
                        "truncated": if include_rows { Some(truncated) } else { None },
                        "limit": if include_rows { Some(args.top) } else { None },
                        "tops": {
                            "pagerank": top_pr,
                            "consumers_pagerank": top_cons,
                            "betweenness": top_between,
                        },
                        "rows": if include_rows { serde_json::to_value(&rows).unwrap_or(serde_json::Value::Null) } else { serde_json::Value::Null },
                        "top_edges": if include_top_edges { serde_json::to_value(&top_edges).unwrap_or(serde_json::Value::Null) } else { serde_json::Value::Null },
                    }),
                );
            }
            Err(e) => {
                if !continue_on_error {
                    return Err(e)
                        .with_context(|| format!("modules-sweep failed for package `{pkg}`"));
                }
                out_pkgs.insert(
                    pkg,
                    serde_json::json!({
                        "ok": false,
                        "error": format!("{:#}", e),
                        "nodes": null,
                        "edges": null,
                        "aggregate_label": null,
                        "tops": null,
                        "rows": null,
                        "top_edges": null,
                    }),
                );
            }
        }
    }

    Ok(serde_json::json!({
        "ok": true,
        "manifest_path": args.manifest_path.display().to_string(),
        "preset": match args.preset {
            ModulesPreset::None => None,
            p => Some(format!("{p:?}")),
        },
        "effective": {
            "aggregate": format!("{:?}", template.aggregate),
            "edge_kind": format!("{:?}", template.edge_kind),
            "include_fns": template.include_fns,
            "include_types": template.include_types,
            "include_traits": template.include_traits,
            "cache": template.cache,
            "cache_refresh": template.cache_refresh,
        },
        "mode": {
            "top": args.top,
            "metric": format!("{:?}", args.metric),
            "continue_on_error": continue_on_error,
            "include_rows": include_rows,
            "include_top_edges": include_top_edges,
        },
        "packages": out_pkgs,
    }))
}

#[derive(Debug)]
struct CratesIoClient {
    base: String,
    agent: ureq::Agent,
}

impl CratesIoClient {
    fn new() -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(30))
            .timeout_write(Duration::from_secs(30))
            .build();
        Self {
            base: "https://crates.io/api/v1".to_string(),
            agent,
        }
    }

    fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}/{}", self.base, path.trim_start_matches('/'));
        let resp = match self
            .agent
            .get(&url)
            .set("Accept", "application/json")
            .set("User-Agent", "pkgrank (local analysis)")
            .call()
        {
            Ok(resp) => resp,
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                return Err(anyhow!("request failed: {}: HTTP {}: {}", url, code, body));
            }
            Err(e) => return Err(anyhow!("request failed: {}: {}", url, e)),
        };

        resp.into_json::<T>()
            .map_err(|e| anyhow!("failed to parse JSON from {}: {}", url, e))
    }
}

#[derive(Debug, serde::Deserialize)]
struct CrateInfoResponse {
    #[serde(rename = "crate")]
    krate: CrateInfo,
}

#[derive(Debug, serde::Deserialize)]
struct CrateInfo {
    max_version: String,
}

#[derive(Debug, serde::Deserialize)]
struct DepsResponse {
    dependencies: Vec<CratesIoDep>,
}

#[derive(Debug, serde::Deserialize)]
struct CratesIoDep {
    crate_id: String,
    kind: String,
    optional: bool,
}

fn run_cratesio(args: &CratesIoArgs) -> Result<()> {
    let root = args
        .root
        .canonicalize()
        .unwrap_or_else(|_| args.root.clone());
    let out_dir = root.join(&args.out);
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create output dir {}", out_dir.display()))?;

    // Seed discovery.
    let mut seeds = args.seed.clone();
    if seeds.is_empty() {
        let analyze = AnalyzeArgs {
            path: root.clone(),
            metric: Metric::Pagerank,
            top: 25,
            dev: false,
            build: false,
            workspace_only: true,
            all_features: false,
            no_default_features: false,
            features: None,
            format: OutputFormat::Json,
            stats: false,
            json_limit: None,
        };
        let rows = analyze_rows(&analyze)?;
        let mut uniq = HashSet::new();
        for r in rows {
            uniq.insert(r.name);
        }
        seeds = uniq.into_iter().collect();
        seeds.sort();
    }

    let client = CratesIoClient::new();

    // BFS over crates.io deps.
    let mut graph: DiGraph<String, f64> = DiGraph::new();
    let mut node_idx: HashMap<String, NodeIndex> = HashMap::new();

    let mut seen: HashSet<String> = HashSet::new();
    let mut q: VecDeque<(String, usize)> = VecDeque::new();
    for s in &seeds {
        q.push_back((s.clone(), 0));
        seen.insert(s.clone());
    }

    while let Some((name, depth)) = q.pop_front() {
        // Ensure node exists.
        let from = *node_idx
            .entry(name.clone())
            .or_insert_with(|| graph.add_node(name.clone()));

        if depth >= args.depth {
            continue;
        }

        // Fetch latest version and deps.
        let info: CrateInfoResponse = match client.get_json(&format!("crates/{}", name)) {
            Ok(v) => v,
            Err(_) => continue, // not on crates.io; skip
        };
        let deps: DepsResponse = match client.get_json(&format!(
            "crates/{}/{}/dependencies",
            name, info.krate.max_version
        )) {
            Ok(v) => v,
            Err(_) => continue,
        };

        for d in deps.dependencies {
            let kind_ok = match d.kind.as_str() {
                "normal" => true,
                "dev" => args.dev,
                "build" => args.build,
                _ => false,
            };
            if !kind_ok {
                continue;
            }
            if d.optional && !args.optional {
                continue;
            }

            let to_name = d.crate_id;
            let to = *node_idx
                .entry(to_name.clone())
                .or_insert_with(|| graph.add_node(to_name.clone()));
            graph.update_edge(from, to, 1.0);

            if seen.insert(to_name.clone()) {
                q.push_back((to_name, depth + 1));
            }
        }
    }

    let pr = pagerank_auto(&graph);
    let bc = betweenness_centrality(&graph);

    #[derive(Debug, Serialize)]
    struct CratesIoRow {
        name: String,
        is_seed: bool,
        in_degree: usize,
        out_degree: usize,
        pagerank: f64,
        betweenness: f64,
    }

    let seed_set: HashSet<String> = seeds.iter().cloned().collect();

    let mut rows = Vec::with_capacity(graph.node_count());
    for node in graph.node_indices() {
        let name = graph.node_weight(node).expect("node").clone();
        rows.push(CratesIoRow {
            is_seed: seed_set.contains(&name),
            name,
            in_degree: graph.neighbors_directed(node, Direction::Incoming).count(),
            out_degree: graph.neighbors_directed(node, Direction::Outgoing).count(),
            pagerank: pr[node.index()],
            betweenness: bc[node.index()],
        });
    }
    rows.sort_by(|a, b| {
        b.pagerank
            .partial_cmp(&a.pagerank)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    match args.format {
        OutputFormat::Json => {
            // Stable artifact name.
            fs::write(
                out_dir.join("cratesio.rows.json"),
                serde_json::to_string_pretty(&rows)?,
            )?;
            fs::write(
                out_dir.join("cratesio.seeds.json"),
                serde_json::to_string_pretty(&seeds)?,
            )?;
            if !args.quiet {
                #[derive(Serialize)]
                struct CratesIoJsonOut<'a> {
                    schema_version: u32,
                    ok: bool,
                    command: &'a str,
                    rows: Vec<CratesIoRow>,
                    seeds: Vec<String>,
                }
                let out = CratesIoJsonOut {
                    schema_version: 1,
                    ok: true,
                    command: "cratesio",
                    rows,
                    seeds,
                };
                println!("{}", serde_json::to_string_pretty(&out)?);
            }
        }
        OutputFormat::Text => {
            if !args.quiet {
                println!(
                    "crates.io graph: {} nodes, {} edges",
                    graph.node_count(),
                    graph.edge_count()
                );
                println!("seeds ({}): {}", seeds.len(), seeds.join(", "));
                println!("{:─<72}", "");
                for (i, r) in rows.iter().take(25).enumerate() {
                    println!(
                        "{:3}. {:28} seed={} in={:3} out={:3} pr={:.6} bc={:.6}",
                        i + 1,
                        r.name,
                        r.is_seed,
                        r.in_degree,
                        r.out_degree,
                        r.pagerank,
                        r.betweenness
                    );
                }
            }
        }
    }

    Ok(())
}

fn run_view(args: &ViewArgs) -> Result<()> {
    let root = args
        .root
        .canonicalize()
        .unwrap_or_else(|_| args.root.clone());
    let out_dir = if args.out.is_absolute() {
        args.out.clone()
    } else {
        root.join(&args.out)
    };
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create output dir {}", out_dir.display()))?;

    // Always write a small JSON manifest describing what we ran.
    #[derive(Serialize)]
    struct ViewManifest {
        root: String,
        mode: ViewMode,
        local_top: usize,
        cratesio_depth: usize,
        cratesio_dev: bool,
        cratesio_build: bool,
        cratesio_optional: bool,
    }
    fs::write(
        out_dir.join("view.manifest.json"),
        serde_json::to_string_pretty(&ViewManifest {
            root: root.display().to_string(),
            mode: args.mode,
            local_top: args.local_top,
            cratesio_depth: args.cratesio_depth,
            cratesio_dev: args.cratesio_dev,
            cratesio_build: args.cratesio_build,
            cratesio_optional: args.cratesio_optional,
        })?,
    )?;

    if matches!(args.mode, ViewMode::Local | ViewMode::Both) {
        let sweep = SweepLocalArgs {
            root: root.clone(),
            out: out_dir.clone(),
            overview: None,
            mode: SweepMode::WorkspaceSlice,
            top: args.local_top,
            dev: false,
            build: false,
            recent: true,
            recent_days: 14,
            recent_max: 200,
        };
        run_sweep_local(&sweep)?;
    }

    if matches!(args.mode, ViewMode::CratesIo | ViewMode::Both) {
        let cratesio = CratesIoArgs {
            root: root.clone(),
            seed: vec![],
            depth: args.cratesio_depth,
            dev: args.cratesio_dev,
            build: args.cratesio_build,
            optional: args.cratesio_optional,
            format: OutputFormat::Json,
            out: out_dir.clone(),
            quiet: args.quiet,
        };
        run_cratesio(&cratesio)?;
    }

    // Compose a minimal HTML view from artifacts we wrote.
    let local_summary_path = out_dir.join("by_repo.summary.json");
    let cratesio_rows_path = out_dir.join("cratesio.rows.json");

    let local_summary: Option<serde_json::Value> = if local_summary_path.exists() {
        Some(serde_json::from_str(&fs::read_to_string(
            &local_summary_path,
        )?)?)
    } else {
        None
    };
    let cratesio_rows: Option<Vec<serde_json::Value>> = if cratesio_rows_path.exists() {
        Some(serde_json::from_str(&fs::read_to_string(
            &cratesio_rows_path,
        )?)?)
    } else {
        None
    };

    // Also compute a “full local graph” view (includes third-party deps),
    // so origin labeling is actually visible in the combined view.
    let local_full_rows = if matches!(args.mode, ViewMode::Local | ViewMode::Both) {
        let analyze = AnalyzeArgs {
            path: root.clone(),
            metric: Metric::Pagerank,
            top: 25,
            dev: false,
            build: false,
            workspace_only: false,
            all_features: false,
            no_default_features: false,
            features: None,
            format: OutputFormat::Json,
            stats: false,
            json_limit: None,
        };
        Some(analyze_rows(&analyze)?)
    } else {
        None
    };

    let mut html = String::new();
    html.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("<meta charset=\"utf-8\" />\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n");
    html.push_str("<title>pkgrank: local + crates.io</title>\n");
    html.push_str("<style>body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:0;background:#eee;}header{padding:10px 12px;border-bottom:1px solid #d6d6d6;background:#fff;}main{padding:12px;max-width:1100px;}section{background:#fff;border:1px solid #d6d6d6;border-radius:6px;padding:10px 12px;margin-bottom:10px;}h2{margin:0 0 8px 0;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:13px;}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px;}table{width:100%;border-collapse:collapse;}th,td{border-top:1px solid #d6d6d6;padding:6px;vertical-align:top;text-align:left;}th{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px;}details.readme{margin-top:6px;}details.readme summary{cursor:pointer;color:#2b2b2b;}pre.readme{white-space:pre-wrap;max-height:260px;overflow:auto;background:#f7f7f7;border:1px solid #e1e1e1;border-radius:4px;padding:6px;}</style>\n");
    html.push_str("</head><body>\n<header>\n<div><code>pkgrank view</code></div>\n<div style=\"color:#3d3d3d;font-size:12px;margin-top:4px;\">Local (Cargo metadata) + crates.io (bounded crawl). No .gitignore / ignore-file walking is used; inputs are Cargo manifests and explicit repo lists.</div>\n<div style=\"margin-top:8px;font-size:12px;display:flex;gap:10px;flex-wrap:wrap;\">\n<a href=\"#local-per-repo\">local per-repo</a>\n<a href=\"#local-whole\">local whole-graph</a>\n<a href=\"#tlc-crates\">tlc crates</a>\n<a href=\"#ppr\">ppr</a>\n<a href=\"#ecosystem\">ecosystem</a>\n<a href=\"#cratesio\">crates.io</a>\n</div>\n</header>\n<main>\n");

    if let Some(v) = local_summary {
        html.push_str(
            "<section id=\"local-per-repo\"><h2>Local: per-repo top crates (PageRank)</h2>\n",
        );
        html.push_str("<p><code>evals/pkgrank/by_repo.summary.json</code></p>\n");
        html.push_str("<table><thead><tr><th>repo</th><th>count</th><th>top (name → pagerank)</th></tr></thead><tbody>\n");
        if let Some(obj) = v.as_object() {
            // Sort repos by name.
            let mut keys: Vec<_> = obj.keys().cloned().collect();
            keys.sort();
            for k in keys {
                let entry = &obj[&k];
                let count = entry.get("count").and_then(|x| x.as_u64()).unwrap_or(0);
                let mut top = String::new();
                if let Some(arr) = entry.get("top_pagerank").and_then(|x| x.as_array()) {
                    for (i, item) in arr.iter().take(args.local_top).enumerate() {
                        let name = item.get("name").and_then(|x| x.as_str()).unwrap_or("?");
                        let pr = item.get("pagerank").and_then(|x| x.as_f64()).unwrap_or(0.0);
                        if i > 0 {
                            top.push_str(", ");
                        }
                        top.push_str(&format!("{}→{:.4}", name, pr));
                    }
                }
                html.push_str(&format!(
                    "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>\n",
                    k, count, top
                ));
            }
        }
        html.push_str("</tbody></table></section>\n");
    }

    // Recent file activity (mtime-based), if available.
    let recent_path = out_dir.join("recent.files.json");
    if recent_path.exists() {
        if let Ok(raw) = fs::read_to_string(&recent_path) {
            if let Ok(rows) = serde_json::from_str::<Vec<RecentFileRow>>(&raw) {
                html.push_str(
                    "<section id=\"recent\"><h2>Recent: modified files (mtime; bounded)</h2>",
                );
                html.push_str("<p><code>evals/pkgrank/recent.files.json</code>, <code>evals/pkgrank/recent.summary.json</code></p>");
                html.push_str("<p>Interpretation: this is filesystem time, not Git history. It’s useful for “don’t miss context” in a super-workspace.</p>");
                html.push_str("<table><thead><tr><th>age</th><th>repo</th><th>crate</th><th>tlc</th><th>size</th><th>path</th></tr></thead><tbody>");
                for r in rows.iter().take(50) {
                    let age_hours = (r.age_seconds as f64) / 3600.0;
                    html.push_str(&format!(
                        "<tr><td><code>{:.1}h</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td></tr>",
                        age_hours,
                        html_escape(&r.repo),
                        html_escape(r.crate_name.as_deref().unwrap_or("-")),
                        r.tlc_score.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".to_string()),
                        r.size_bytes,
                        html_escape(&r.path),
                    ));
                }
                html.push_str("</tbody></table></section>");
            }
        }
    }

    if let Some(rows) = local_full_rows {
        // Split local full graph into first-party vs third-party.
        let mut first_party: Vec<&Row> = rows
            .iter()
            .filter(|r| {
                matches!(
                    r.origin,
                    PackageOrigin::WorkspaceMember | PackageOrigin::Path
                )
            })
            .collect();
        let mut third_party: Vec<&Row> = rows
            .iter()
            .filter(|r| {
                matches!(
                    r.origin,
                    PackageOrigin::Registry | PackageOrigin::Git | PackageOrigin::Other
                )
            })
            .collect();

        first_party.sort_by(|a, b| {
            b.pagerank
                .partial_cmp(&a.pagerank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        third_party.sort_by(|a, b| {
            b.pagerank
                .partial_cmp(&a.pagerank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        html.push_str("<section id=\"local-whole\"><h2>Local: whole-graph ranking (first-party vs third-party)</h2>\n");
        html.push_str("<p>This uses <code>cargo metadata</code> on the root workspace with <code>--workspace-only=false</code>, so registry/git deps are included and labeled.</p>\n");
        html.push_str("<div style=\"display:grid;grid-template-columns:1fr 1fr;gap:10px;\">");

        html.push_str("<div><h2>Top first-party</h2><table><thead><tr><th>rank</th><th>crate</th><th>origin</th><th>pr</th></tr></thead><tbody>");
        for (i, r) in first_party.iter().take(25).enumerate() {
            html.push_str(&format!(
                "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{:?}</code></td><td><code>{:.6}</code></td></tr>",
                i + 1,
                r.name,
                r.origin,
                r.pagerank
            ));
        }
        html.push_str("</tbody></table></div>");

        html.push_str("<div><h2>Top third-party</h2><table><thead><tr><th>rank</th><th>crate</th><th>origin</th><th>pr</th></tr></thead><tbody>");
        for (i, r) in third_party.iter().take(25).enumerate() {
            html.push_str(&format!(
                "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{:?}</code></td><td><code>{:.6}</code></td></tr>",
                i + 1,
                r.name,
                r.origin,
                r.pagerank
            ));
        }
        html.push_str("</tbody></table></div>");
        html.push_str("</div></section>\n");
    }

    // TLC: prioritize attention (heuristic).
    if matches!(args.mode, ViewMode::Local | ViewMode::Both) {
        match compute_tlc_crates(&root, &out_dir) {
            Ok(tlc) => {
                html.push_str("<section id=\"tlc-crates\"><h2>TLC: crates that are both central and risky</h2>");
                html.push_str("<p>Heuristic score combines: transitive dependents (blast radius), PageRank (centrality within first-party), and # of unique direct third-party deps (boundary complexity).</p>");
                html.push_str("<p>Artifacts: <code>evals/pkgrank/tlc.crates.json</code>, <code>evals/pkgrank/local.first_party.adj.json</code>, <code>evals/pkgrank/ppr.entrypoints.json</code>, <code>evals/pkgrank/ppr.aggregate.json</code>.</p>");
                html.push_str("<p>Interpretation note: high <code>dependents</code> often means “foundational; keep stable”; high <code>3p deps</code> often means “boundary-heavy; keep interfaces tight”. Treat this as triage, not a moral judgment.</p>");
                html.push_str("<table><thead><tr><th>rank</th><th>crate</th><th>repo</th><th>axis</th><th>origin</th><th>score</th><th>deps_pr</th><th>betweenness</th><th>dependents</th><th>deps</th><th>3p deps</th><th>repo commits (30d)</th><th>repo days since</th><th>why</th></tr></thead><tbody>");
                for (i, r) in tlc.iter().take(50).enumerate() {
                    let readme = readme_details_html(
                        &root,
                        find_readme_for_manifest(&root, &r.manifest_path),
                    );
                    html.push_str("<tr>");
                    html.push_str(&format!("<td><code>{}</code></td>", i + 1));
                    html.push_str("<td>");
                    html.push_str(&format!("<code>{}</code>", html_escape(&r.name)));
                    html.push_str(&readme);
                    html.push_str("</td>");
                    html.push_str(&format!("<td><code>{}</code></td>", html_escape(&r.repo)));
                    html.push_str(&format!("<td><code>{}</code></td>", html_escape(&r.axis)));
                    html.push_str(&format!("<td><code>{:?}</code></td>", r.origin));
                    html.push_str(&format!("<td><code>{:.2}</code></td>", r.score));
                    html.push_str(&format!("<td><code>{:.6}</code></td>", r.pagerank));
                    html.push_str(&format!("<td><code>{:.6}</code></td>", r.betweenness));
                    html.push_str(&format!(
                        "<td><code>{}</code></td>",
                        r.transitive_dependents
                    ));
                    html.push_str(&format!(
                        "<td><code>{}</code></td>",
                        r.transitive_dependencies
                    ));
                    html.push_str(&format!("<td><code>{}</code></td>", r.third_party_deps));
                    match r.repo_git_commits_30d {
                        Some(v) => html.push_str(&format!("<td><code>{}</code></td>", v)),
                        None => html.push_str("<td><code>-</code></td>"),
                    }
                    match r.repo_git_days_since_last_commit {
                        Some(v) => html.push_str(&format!("<td><code>{}</code></td>", v)),
                        None => html.push_str("<td><code>-</code></td>"),
                    }
                    html.push_str(&format!("<td>{}</td>", html_escape(&r.why)));
                    html.push_str("</tr>");
                }
                html.push_str("</tbody></table>");

                // TLC slices: show why things appear without hiding it in a combined score.
                html.push_str("<h2>TLC slices</h2>");
                html.push_str(
                    "<div style=\"display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;\">",
                );

                let mut by_dependents = tlc.clone();
                by_dependents.sort_by(|a, b| b.transitive_dependents.cmp(&a.transitive_dependents));
                html.push_str("<div><h2>Blast radius (dependents)</h2><table><thead><tr><th>rank</th><th>crate</th><th>dependents</th></tr></thead><tbody>");
                for (i, r) in by_dependents.iter().take(15).enumerate() {
                    html.push_str(&format!(
                        "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td></tr>",
                        i + 1,
                        html_escape(&r.name),
                        r.transitive_dependents
                    ));
                }
                html.push_str("</tbody></table></div>");

                let mut by_boundary = tlc.clone();
                by_boundary.sort_by(|a, b| b.third_party_deps.cmp(&a.third_party_deps));
                html.push_str("<div><h2>Boundary load (3p deps)</h2><table><thead><tr><th>rank</th><th>crate</th><th>3p deps</th></tr></thead><tbody>");
                for (i, r) in by_boundary.iter().take(15).enumerate() {
                    html.push_str(&format!(
                        "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td></tr>",
                        i + 1,
                        html_escape(&r.name),
                        r.third_party_deps
                    ));
                }
                html.push_str("</tbody></table></div>");

                let mut by_bridge = tlc.clone();
                by_bridge.sort_by(|a, b| {
                    b.betweenness
                        .partial_cmp(&a.betweenness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                html.push_str("<div><h2>Bridge nodes (betweenness)</h2><table><thead><tr><th>rank</th><th>crate</th><th>betweenness</th></tr></thead><tbody>");
                for (i, r) in by_bridge.iter().take(15).enumerate() {
                    html.push_str(&format!(
                        "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{:.6}</code></td></tr>",
                        i + 1,
                        html_escape(&r.name),
                        r.betweenness
                    ));
                }
                html.push_str("</tbody></table></div>");

                html.push_str("</div></section>");

                // PPR entrypoints: which foundations boundary-heavy entrypoints lean on.
                let ppr_path = out_dir.join("ppr.entrypoints.json");
                if ppr_path.exists() {
                    if let Ok(raw) = fs::read_to_string(&ppr_path) {
                        if let Ok(v) = serde_json::from_str::<Vec<serde_json::Value>>(&raw) {
                            html.push_str(
                                "<section id=\"ppr\"><h2>PPR: what entrypoints lean on (walk)</h2>",
                            );
                            html.push_str("<p>Seeds are chosen heuristically (boundary-heavy crates with 0 first-party dependents). Scores approximate “influence” under a random-surfer model over dependency edges. Many crates are in disconnected components; we only show non-zero results.</p>");
                            let agg_path = out_dir.join("ppr.aggregate.json");
                            if agg_path.exists() {
                                if let Ok(raw_agg) = fs::read_to_string(&agg_path) {
                                    if let Ok(agg) =
                                        serde_json::from_str::<Vec<(String, f64)>>(&raw_agg)
                                    {
                                        html.push_str("<h2>PPR aggregate: shared foundations</h2>");
                                        html.push_str("<p>This sums PPR mass across the chosen seeds, filtering out the seeds and other leaf/entrypoint crates. It’s a quick proxy for “what repeatedly shows up under entrypoints”.</p>");
                                        html.push_str("<table><thead><tr><th>rank</th><th>crate</th><th>ppr_sum</th></tr></thead><tbody>");
                                        for (i, (name, mass)) in agg.iter().take(25).enumerate() {
                                            html.push_str(&format!(
                                                "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{:.6}</code></td></tr>",
                                                i + 1,
                                                html_escape(name),
                                                mass
                                            ));
                                        }
                                        html.push_str("</tbody></table>");
                                    }
                                }
                            }
                            for seed in v.iter().take(6) {
                                let name = seed.get("seed").and_then(|x| x.as_str()).unwrap_or("?");
                                let reachable = seed
                                    .get("reachable_first_party")
                                    .and_then(|x| x.as_u64())
                                    .unwrap_or(0);
                                html.push_str(&format!(
                                    "<h2><code>{}</code></h2>",
                                    html_escape(name)
                                ));
                                html.push_str(&format!("<p>Reachable first-party nodes (including seed): <code>{}</code></p>", reachable));
                                html.push_str("<table><thead><tr><th>rank</th><th>crate</th><th>ppr</th></tr></thead><tbody>");
                                if let Some(arr) = seed.get("top").and_then(|x| x.as_array()) {
                                    for (i, item) in arr.iter().take(15).enumerate() {
                                        let crate_name =
                                            item.get(0).and_then(|x| x.as_str()).unwrap_or("?");
                                        let score =
                                            item.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0);
                                        html.push_str(&format!(
                                            "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{:.6}</code></td></tr>",
                                            i + 1,
                                            html_escape(crate_name),
                                            score
                                        ));
                                    }
                                }
                                html.push_str("</tbody></table>");
                            }
                            html.push_str("</section>");
                        }
                    }
                }
            }
            Err(e) => {
                html.push_str("<section><h2>TLC: crates</h2>");
                html.push_str(&format!(
                    "<p>Failed to compute TLC crates: <code>{}</code></p>",
                    html_escape(&format!("{:#}", e))
                ));
                html.push_str("</section>");
            }
        }
    }

    // Ecosystem (repo-level) multi-axis view, based on dev_repos_overview.json.
    if default_overview_path(&root).exists() {
        match compute_repo_graph_from_live_metadata(&root, &out_dir) {
            Ok((repo_rows, axes_summary)) => {
                html.push_str("<section id=\"ecosystem\"><h2>Ecosystem: repo-level dependency graph (transitive)</h2>");
                html.push_str("<p>Edges are <code>A → B</code> meaning <em>A depends on B</em>. PageRank flows toward dependencies.</p>");
                html.push_str("<p><code>transitive_dependents</code> is the size of the reverse-reachability set (who depends on you, transitively).</p>");

                html.push_str("<h2>Axis PageRank mass</h2>");
                html.push_str(
                    "<table><thead><tr><th>axis</th><th>pagerank mass</th></tr></thead><tbody>",
                );
                let mut keys: Vec<_> = axes_summary.totals.keys().cloned().collect();
                keys.sort();
                for k in keys {
                    let v = axes_summary.totals.get(&k).copied().unwrap_or(0.0);
                    html.push_str(&format!(
                        "<tr><td><code>{}</code></td><td><code>{:.6}</code></td></tr>",
                        k, v
                    ));
                }
                html.push_str("</tbody></table>");

                html.push_str("<h2>Repo scatter (deps_pr vs commits_30d)</h2>");
                html.push_str(
                    "<p>Point size ~ 3p deps; color by axis. Y is log1p(commits_30d).</p>",
                );
                html.push_str(&render_repo_scatter_svg(&repo_rows));

                // Invariants (cross-axis forbidden edges).
                let inv_path = out_dir.join("ecosystem.invariants.violations.json");
                if inv_path.exists() {
                    if let Ok(raw) = fs::read_to_string(&inv_path) {
                        if let Ok(v) = serde_json::from_str::<Vec<RepoInvariantViolation>>(&raw) {
                            html.push_str("<h2>Invariants: forbidden cross-axis edges</h2>");
                            html.push_str("<p>Rules enforced here:</p><ul>");
                            html.push_str("<li><code>tekne → agents</code> and <code>tekne → governance</code> forbidden</li>");
                            html.push_str("<li><code>governance → agents</code> forbidden</li>");
                            html.push_str("</ul>");
                            html.push_str(&format!("<p>Violations: <code>{}</code></p>", v.len()));

                            if !v.is_empty() {
                                html.push_str("<table><thead><tr><th>rule</th><th>from</th><th>from_axis</th><th>to</th><th>to_axis</th><th>weight</th></tr></thead><tbody>");
                                for row in v.iter().take(50) {
                                    html.push_str(&format!(
                                        "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td></tr>",
                                        html_escape(&row.rule),
                                        html_escape(&row.from_repo),
                                        html_escape(&row.from_axis),
                                        html_escape(&row.to_repo),
                                        html_escape(&row.to_axis),
                                        row.weight
                                    ));
                                }
                                html.push_str("</tbody></table>");
                                html.push_str("<p>Artifact: <code>evals/pkgrank/ecosystem.invariants.violations.json</code></p>");
                            }
                        }
                    }
                }

                // Repo TLC (heuristic).
                let tlc_repo_path = out_dir.join("tlc.repos.json");
                if tlc_repo_path.exists() {
                    if let Ok(raw) = fs::read_to_string(&tlc_repo_path) {
                        if let Ok(v) = serde_json::from_str::<Vec<TlcRepoRow>>(&raw) {
                            html.push_str(
                                "<h2>TLC: repos that are central and boundary-heavy</h2>",
                            );
                            html.push_str(
                                "<p>Artifact: <code>evals/pkgrank/tlc.repos.json</code>.</p>",
                            );
                            html.push_str("<table><thead><tr><th>rank</th><th>repo</th><th>axis</th><th>score</th><th>deps_pr</th><th>consumers_pr</th><th>dependents</th><th>3p deps</th><th>violations</th><th>commits (30d)</th><th>days since</th><th>why</th></tr></thead><tbody>");
                            for (i, r) in v.iter().take(30).enumerate() {
                                let readme = readme_details_html(
                                    &root,
                                    find_readme_for_repo(&root, &r.repo),
                                );
                                html.push_str("<tr>");
                                html.push_str(&format!("<td><code>{}</code></td>", i + 1));
                                html.push_str("<td>");
                                html.push_str(&format!("<code>{}</code>", html_escape(&r.repo)));
                                html.push_str(&readme);
                                html.push_str("</td>");
                                html.push_str(&format!(
                                    "<td><code>{}</code></td>",
                                    html_escape(&r.axis)
                                ));
                                html.push_str(&format!("<td><code>{:.2}</code></td>", r.score));
                                html.push_str(&format!(
                                    "<td><code>{:.6}</code></td>",
                                    r.deps_pagerank
                                ));
                                html.push_str(&format!(
                                    "<td><code>{:.6}</code></td>",
                                    r.consumers_pagerank
                                ));
                                html.push_str(&format!(
                                    "<td><code>{}</code></td>",
                                    r.transitive_dependents
                                ));
                                html.push_str(&format!(
                                    "<td><code>{}</code></td>",
                                    r.third_party_deps
                                ));
                                html.push_str(&format!(
                                    "<td><code>{}</code></td>",
                                    r.violation_weight
                                ));
                                match r.git_commits_30d {
                                    Some(v) => {
                                        html.push_str(&format!("<td><code>{}</code></td>", v))
                                    }
                                    None => html.push_str("<td><code>-</code></td>"),
                                }
                                match r.git_days_since_last_commit {
                                    Some(v) => {
                                        html.push_str(&format!("<td><code>{}</code></td>", v))
                                    }
                                    None => html.push_str("<td><code>-</code></td>"),
                                }
                                html.push_str(&format!("<td>{}</td>", html_escape(&r.why)));
                                html.push_str("</tr>");
                            }
                            html.push_str("</tbody></table>");
                        }
                    }
                }

                html.push_str(
                    "<h2>Top repos by dependency PageRank (A → B means A depends on B)</h2>",
                );
                html.push_str("<p>Also shown: consumer PageRank (reverse graph), which highlights orchestrators / top-level consumers.</p>");
                html.push_str("<table><thead><tr><th>rank</th><th>repo</th><th>axis</th><th>in</th><th>out</th><th>in_w</th><th>out_w</th><th>deps_pr</th><th>consumers_pr</th><th>dependents</th><th>deps</th><th>3p deps</th><th>commits (30d)</th><th>days since</th></tr></thead><tbody>");
                for (i, r) in repo_rows.iter().take(50).enumerate() {
                    html.push_str(&format!(
                        "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{:.0}</code></td><td><code>{:.0}</code></td><td><code>{:.6}</code></td><td><code>{:.6}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td></tr>",
                        i + 1,
                        r.repo,
                        r.axis,
                        r.in_degree,
                        r.out_degree,
                        r.in_weight,
                        r.out_weight,
                        r.pagerank,
                        r.consumers_pagerank,
                        r.transitive_dependents,
                        r.transitive_dependencies,
                        r.third_party_deps,
                        r.git_commits_30d.map(|x| x.to_string()).unwrap_or_else(|| "-".to_string()),
                        r.git_days_since_last_commit.map(|x| x.to_string()).unwrap_or_else(|| "-".to_string()),
                    ));
                }
                html.push_str("</tbody></table>");
                html.push_str("<p>Artifacts: <code>evals/pkgrank/ecosystem.repo_graph.rows.json</code>, <code>evals/pkgrank/ecosystem.axes.pagerank.json</code>, <code>evals/pkgrank/ecosystem.repo_graph.consumers_pagerank.json</code>, <code>evals/pkgrank/ecosystem.invariants.violations.json</code>, <code>evals/pkgrank/tlc.repos.json</code></p>");
                html.push_str("</section>");
            }
            Err(e) => {
                html.push_str("<section><h2>Ecosystem: repo-level dependency graph</h2>");
                html.push_str(&format!(
                    "<p>Failed to compute repo graph: <code>{}</code></p>",
                    html_escape(&format!("{:#}", e))
                ));
                html.push_str("</section>");
            }
        }
    }

    if let Some(rows) = cratesio_rows {
        let mut seeds = Vec::new();
        let mut non_seeds = Vec::new();
        for r in &rows {
            let is_seed = r.get("is_seed").and_then(|x| x.as_bool()).unwrap_or(false);
            if is_seed {
                seeds.push(r);
            } else {
                non_seeds.push(r);
            }
        }

        html.push_str("<section id=\"cratesio\"><h2>crates.io: top nodes (PageRank)</h2>\n");
        html.push_str("<p><code>evals/pkgrank/cratesio.rows.json</code> (bounded crawl)</p>\n");
        html.push_str("<div style=\"display:grid;grid-template-columns:1fr 1fr;gap:10px;\">");

        html.push_str("<div><h2>Top third-party (non-seeds)</h2>");
        html.push_str("<table><thead><tr><th>rank</th><th>name</th><th>in</th><th>out</th><th>pr</th></tr></thead><tbody>\n");
        for (i, r) in non_seeds.iter().take(25).enumerate() {
            let name = r.get("name").and_then(|x| x.as_str()).unwrap_or("?");
            let indeg = r.get("in_degree").and_then(|x| x.as_u64()).unwrap_or(0);
            let outdeg = r.get("out_degree").and_then(|x| x.as_u64()).unwrap_or(0);
            let pr = r.get("pagerank").and_then(|x| x.as_f64()).unwrap_or(0.0);
            html.push_str(&format!(
                "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{:.6}</code></td></tr>\n",
                i + 1,
                name,
                indeg,
                outdeg,
                pr
            ));
        }
        html.push_str("</tbody></table></div>");

        // “Seeds” table: in a depends-on crawl, seeds often have low PageRank.
        // Show them sorted by out-degree (how many deps they pull in at this depth).
        html.push_str("<div><h2>Seeds (sorted by out-degree)</h2>");
        let mut seed_sorted = seeds;
        seed_sorted.sort_by(|a, b| {
            let ao = a.get("out_degree").and_then(|x| x.as_u64()).unwrap_or(0);
            let bo = b.get("out_degree").and_then(|x| x.as_u64()).unwrap_or(0);
            bo.cmp(&ao)
        });
        html.push_str("<table><thead><tr><th>rank</th><th>name</th><th>out</th><th>pr</th></tr></thead><tbody>\n");
        for (i, r) in seed_sorted.iter().take(25).enumerate() {
            let name = r.get("name").and_then(|x| x.as_str()).unwrap_or("?");
            let outdeg = r.get("out_degree").and_then(|x| x.as_u64()).unwrap_or(0);
            let pr = r.get("pagerank").and_then(|x| x.as_f64()).unwrap_or(0.0);
            html.push_str(&format!(
                "<tr><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td><td><code>{:.6}</code></td></tr>\n",
                i + 1,
                name,
                outdeg,
                pr
            ));
        }
        html.push_str("</tbody></table></div>");

        html.push_str("</div></section>\n");
    }

    html.push_str("</main></body></html>\n");

    // Make the artifact somewhat readable in an editor:
    // - avoid a single gigantic line (hard to diff / inspect)
    // - keep it deterministic (simple replacement, no pretty-printer dependency)
    let html = html.replace("><", ">\n<");

    let html_path = out_dir.join("pkgrank_overview.html");
    fs::write(&html_path, html)?;
    eprintln!("wrote {}", html_path.display());
    Ok(())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(feature = "stdio")]
fn p_display(root: &Path, p: Option<&PathBuf>) -> Option<String> {
    p.map(|p| {
        p.strip_prefix(root)
            .ok()
            .map(|x| x.display().to_string())
            .unwrap_or_else(|| p.display().to_string())
    })
}

// --- MCP stdio server (Cursor integration) ---
//
// Mirrors `threadlog mcp-stdio`:
// - keep stdout clean (transport)
// - return JSON payloads as Content::text
// - keep tool surface small and stable

#[cfg(feature = "stdio")]
#[derive(Clone)]
struct PkgrankStdioMcpFull {
    tool_router: RmcpToolRouter<Self>,
}

// Toolset selection is runtime via `PKGRANK_MCP_TOOLSET` (default: slim).

#[cfg(feature = "stdio")]
#[derive(Clone)]
#[allow(dead_code)]
struct PkgrankStdioMcpDebug {
    tool_router: RmcpToolRouter<Self>,
}

#[cfg(feature = "stdio")]
#[tool_router]
impl PkgrankStdioMcpDebug {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    // Debug toolset: include the full surface plus internal artifact inspection tools.
    // Delegate “full” tools to `PkgrankStdioMcpFull` to avoid duplicating logic.

    #[tool(description = "Check pkgrank artifact status (what exists, where, and basic metadata)")]
    async fn pkgrank_status(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_status(params).await
    }

    #[tool(description = "Generate pkgrank HTML/JSON artifacts (pkgrank view)")]
    async fn pkgrank_view(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_view(params).await
    }

    #[tool(
        description = "Triage bundle: top TLC crates/repos + invariants + PPR top-k (artifact-backed)"
    )]
    async fn pkgrank_triage(
        &self,
        params: Parameters<PkgrankTriageArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpSlim::new().pkgrank_triage(params).await
    }

    #[tool(description = "Get repo details from artifacts")]
    async fn pkgrank_repo_detail(
        &self,
        params: Parameters<PkgrankRepoDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_repo_detail(params).await
    }

    #[tool(description = "Get crate details from artifacts")]
    async fn pkgrank_crate_detail(
        &self,
        params: Parameters<PkgrankCrateDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new()
            .pkgrank_crate_detail(params)
            .await
    }

    #[tool(description = "Snapshot pkgrank artifacts into a new directory (copy selected files)")]
    async fn pkgrank_snapshot(
        &self,
        params: Parameters<PkgrankSnapshotArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_snapshot(params).await
    }

    #[tool(description = "Compare two pkgrank artifact directories (TLC crates/repos deltas)")]
    async fn pkgrank_compare_runs(
        &self,
        params: Parameters<PkgrankCompareRunsArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new()
            .pkgrank_compare_runs(params)
            .await
    }

    #[tool(description = "Compute local crate ranking (pkgrank analyze)")]
    async fn pkgrank_analyze(
        &self,
        params: Parameters<PkgrankAnalyzeToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_analyze(params).await
    }

    #[tool(
        description = "Compute internal module/item centrality (pkgrank modules; cargo-modules-backed)"
    )]
    async fn pkgrank_modules(
        &self,
        params: Parameters<PkgrankModulesToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_modules(params).await
    }

    #[tool(
        description = "Compute internal module/item centrality across multiple packages (pkgrank modules-sweep)"
    )]
    async fn pkgrank_modules_sweep(
        &self,
        params: Parameters<PkgrankModulesSweepToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new()
            .pkgrank_modules_sweep(params)
            .await
    }

    // --- Internal artifact inspection tools (debug-only) ---
    // Keep these typed and schema-versioned; they are intentionally not in the default toolset.

    #[tool(description = "Return top TLC crates (artifact-backed) with optional filters (debug)")]
    async fn pkgrank_tlc_crates(
        &self,
        params: Parameters<PkgrankTlcCratesToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let limit = params.0.limit.unwrap_or(25).min(500);
        let axis = params.0.axis.clone();
        let repo = params.0.repo.clone();

        let path = PkgrankStdioMcpFull::artifact_path(&root, &out, "tlc.crates.json");
        let rows: Vec<TlcCrateRow> = PkgrankStdioMcpFull::read_json_file(&path)?;
        let mut out_rows = Vec::new();
        for r in rows {
            if let Some(ax) = axis.as_ref() {
                if &r.axis != ax {
                    continue;
                }
            }
            if let Some(rr) = repo.as_ref() {
                if &r.repo != rr {
                    continue;
                }
            }
            out_rows.push(r);
            if out_rows.len() >= limit {
                break;
            }
        }
        let result = serde_json::json!({
            "source": path.display().to_string(),
            "rows": out_rows,
        });
        mcp_ok("pkgrank_tlc_crates", result, None)
    }

    #[tool(description = "Return top TLC repos (artifact-backed) with optional filters (debug)")]
    async fn pkgrank_tlc_repos(
        &self,
        params: Parameters<PkgrankTlcReposToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let limit = params.0.limit.unwrap_or(25).min(500);
        let axis = params.0.axis.clone();

        let path = PkgrankStdioMcpFull::artifact_path(&root, &out, "tlc.repos.json");
        let rows: Vec<TlcRepoRow> = PkgrankStdioMcpFull::read_json_file(&path)?;
        let mut out_rows = Vec::new();
        for r in rows {
            if let Some(ax) = axis.as_ref() {
                if &r.axis != ax {
                    continue;
                }
            }
            out_rows.push(r);
            if out_rows.len() >= limit {
                break;
            }
        }
        let result = serde_json::json!({
            "source": path.display().to_string(),
            "rows": out_rows,
        });
        mcp_ok("pkgrank_tlc_repos", result, None)
    }

    #[tool(description = "List invariant violations (artifact-backed) (debug)")]
    async fn pkgrank_invariants(
        &self,
        params: Parameters<PkgrankInvariantsToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let path =
            PkgrankStdioMcpFull::artifact_path(&root, &out, "ecosystem.invariants.violations.json");
        let rows: Vec<RepoInvariantViolation> = PkgrankStdioMcpFull::read_json_file(&path)?;
        let result = serde_json::json!({
            "source": path.display().to_string(),
            "violations": rows,
        });
        mcp_ok("pkgrank_invariants", result, None)
    }

    #[tool(description = "Return PPR summaries (artifact-backed) (debug)")]
    async fn pkgrank_ppr_summary(
        &self,
        params: Parameters<PkgrankPprSummaryToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let entry_path = PkgrankStdioMcpFull::artifact_path(&root, &out, "ppr.entrypoints.json");
        let agg_path = PkgrankStdioMcpFull::artifact_path(&root, &out, "ppr.aggregate.json");
        let entry: serde_json::Value = PkgrankStdioMcpFull::read_json_file(&entry_path)?;
        let agg: serde_json::Value = PkgrankStdioMcpFull::read_json_file(&agg_path)?;
        let result = serde_json::json!({
            "entrypoints_source": entry_path.display().to_string(),
            "aggregate_source": agg_path.display().to_string(),
            "entrypoints": entry,
            "aggregate": agg,
        });
        mcp_ok("pkgrank_ppr_summary", result, None)
    }
}

#[cfg(feature = "stdio")]
#[tool_handler]
impl rmcp::ServerHandler for PkgrankStdioMcpDebug {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Tools for ranking crates/repos in the local Cargo workspace (pkgrank).\n\n\
                 Toolsets:\n\
                 - default: PKGRANK_MCP_TOOLSET=slim (small surface)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=full (includes module/type graph tools)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=debug (full + internal artifact inspection tools)"
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[cfg(feature = "stdio")]
fn mcp_ok(
    tool: &str,
    result: serde_json::Value,
    summary_text: Option<String>,
) -> Result<CallToolResult, McpError> {
    let payload = serde_json::json!({
        "schema_version": 1,
        "ok": true,
        "tool": tool,
        "summary_text": summary_text,
        "result": result,
    });
    Ok(CallToolResult::success(vec![Content::text(
        payload.to_string(),
    )]))
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct PkgrankViewToolArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    root: Option<String>,
    /// Output directory for artifacts (relative to root if not absolute).
    #[serde(default)]
    out: Option<String>,
    /// Mode: "local", "cratesio", or "both".
    #[serde(default)]
    mode: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct PkgrankAnalyzeToolArgs {
    /// Path to a Cargo.toml or directory containing one.
    #[serde(default)]
    path: Option<String>,
    /// Metric to sort by: pagerank, consumers-pagerank, indegree, outdegree, betweenness.
    #[serde(default)]
    metric: Option<String>,
    #[serde(default)]
    workspace_only: Option<bool>,
    #[serde(default)]
    dev: Option<bool>,
    #[serde(default)]
    build: Option<bool>,
    #[serde(default)]
    top: Option<usize>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct PkgrankModulesToolArgs {
    /// Path to a Cargo manifest to analyze.
    #[serde(default)]
    manifest_path: Option<String>,
    /// Package to analyze (same meaning as `cargo modules -p ...`).
    #[serde(default)]
    package: Option<String>,
    /// Analyze the package library target.
    #[serde(default)]
    lib: Option<bool>,
    /// Analyze the named binary target.
    #[serde(default)]
    bin: Option<String>,
    /// Analyze with `#[cfg(test)]` enabled.
    #[serde(default)]
    cfg_test: Option<bool>,
    /// Preset: "file-full", "file-api", "node-full", "node-api".
    #[serde(default)]
    preset: Option<String>,
    /// Centrality metric to sort by (same enum as CLI).
    #[serde(default)]
    metric: Option<String>,
    /// Aggregate: "node", "module", or "file".
    #[serde(default)]
    aggregate: Option<String>,
    /// Edge kind: "uses", "owns", or "both".
    #[serde(default)]
    edge_kind: Option<String>,
    /// Max rows to return (after sorting).
    #[serde(default)]
    top: Option<usize>,
    /// Include functions in graph.
    #[serde(default)]
    include_fns: Option<bool>,
    /// Include types in graph.
    #[serde(default)]
    include_types: Option<bool>,
    /// Include traits in graph.
    #[serde(default)]
    include_traits: Option<bool>,
    /// Include externs in graph.
    #[serde(default)]
    include_externs: Option<bool>,
    /// Include sysroot crates in graph.
    #[serde(default)]
    include_sysroot: Option<bool>,
    /// Cache cargo-modules DOT output on disk.
    #[serde(default)]
    cache: Option<bool>,
    /// Force refresh cache.
    #[serde(default)]
    cache_refresh: Option<bool>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct PkgrankModulesSweepToolArgs {
    /// Path to a Cargo manifest (workspace root or a single crate).
    #[serde(default)]
    manifest_path: Option<String>,
    /// Packages to analyze (repeatable).
    #[serde(default)]
    packages: Option<Vec<String>>,
    /// Analyze all workspace member packages under the manifest.
    #[serde(default)]
    all_packages: Option<bool>,
    /// Analyze the package library target.
    #[serde(default)]
    lib: Option<bool>,
    /// Analyze the named binary target (applies to all packages).
    #[serde(default)]
    bin: Option<String>,
    /// Analyze with `#[cfg(test)]` enabled.
    #[serde(default)]
    cfg_test: Option<bool>,
    /// Preset: "file-full", "file-api", "node-full", "node-api".
    #[serde(default)]
    preset: Option<String>,
    /// Centrality metric to sort by (same enum as CLI).
    #[serde(default)]
    metric: Option<String>,
    /// Max rows per package to return (after sorting).
    #[serde(default)]
    top: Option<usize>,
    /// Continue if a package analysis fails.
    #[serde(default)]
    continue_on_error: Option<bool>,
    /// Convenience alias: stop on first error.
    #[serde(default)]
    fail_fast: Option<bool>,
    /// Cache cargo-modules DOT output on disk.
    #[serde(default)]
    cache: Option<bool>,
    /// Force refresh cache.
    #[serde(default)]
    cache_refresh: Option<bool>,

    /// Include per-package `rows` (can be large). Default: false.
    #[serde(default)]
    include_rows: Option<bool>,
    /// Include per-package `top_edges` (can be large). Default: false.
    #[serde(default)]
    include_top_edges: Option<bool>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct PkgrankTlcCratesToolArgs {
    #[serde(default)]
    root: Option<String>,
    #[serde(default)]
    out: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    axis: Option<String>,
    #[serde(default)]
    repo: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct PkgrankTlcReposToolArgs {
    #[serde(default)]
    root: Option<String>,
    #[serde(default)]
    out: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    axis: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct PkgrankInvariantsToolArgs {
    #[serde(default)]
    root: Option<String>,
    #[serde(default)]
    out: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct PkgrankPprSummaryToolArgs {
    #[serde(default)]
    root: Option<String>,
    #[serde(default)]
    out: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Clone)]
#[allow(dead_code)] // constructed when `mcp-debug` is ON
struct PkgrankStdioMcpSlim {
    tool_router: RmcpToolRouter<Self>,
}

#[cfg(feature = "stdio")]
#[tool_router]
impl PkgrankStdioMcpFull {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    fn default_root(params_root: Option<&str>) -> PathBuf {
        if let Some(r) = params_root {
            return PathBuf::from(r);
        }
        if let Ok(r) = std::env::var("PKGRANK_ROOT") {
            if !r.trim().is_empty() {
                return PathBuf::from(r);
            }
        }
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }

    fn default_out(params_out: Option<&str>) -> PathBuf {
        if let Some(o) = params_out {
            return PathBuf::from(o);
        }
        if let Ok(o) = std::env::var("PKGRANK_OUT") {
            if !o.trim().is_empty() {
                return PathBuf::from(o);
            }
        }
        PathBuf::from("evals/pkgrank")
    }

    fn artifact_path(root: &Path, out: &Path, rel: &str) -> PathBuf {
        // If `out` is absolute, do not join with root.
        if out.is_absolute() {
            out.join(rel)
        } else {
            root.join(out).join(rel)
        }
    }

    fn ensure_dir(path: &Path) -> Result<(), McpError> {
        std::fs::create_dir_all(path).map_err(|e| {
            McpError::internal_error(
                format!("failed to create dir {}: {}", path.display(), e),
                None,
            )
        })
    }

    fn read_json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, McpError> {
        let raw = fs::read_to_string(path).map_err(|e| {
            McpError::internal_error(format!("failed to read {}: {}", path.display(), e), None)
        })?;
        serde_json::from_str::<T>(&raw).map_err(|e| {
            McpError::internal_error(format!("failed to parse {}: {}", path.display(), e), None)
        })
    }

    #[tool(
        description = "Triage bundle: top TLC crates/repos + invariants + PPR top-k (artifact-backed)"
    )]
    async fn pkgrank_triage(
        &self,
        params: Parameters<PkgrankTriageArgs>,
    ) -> Result<CallToolResult, McpError> {
        // Keep triage in all toolsets; delegate to the slim implementation.
        PkgrankStdioMcpSlim::new().pkgrank_triage(params).await
    }

    #[tool(description = "Check pkgrank artifact status (what exists, where, and basic metadata)")]
    async fn pkgrank_status(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());
        let out_dir = if out.is_absolute() {
            out.clone()
        } else {
            root.join(&out)
        };

        let files = [
            "pkgrank_overview.html",
            "by_repo.summary.json",
            "root.workspace_only.json",
            "ecosystem.repo_graph.rows.json",
            "ecosystem.axes.pagerank.json",
            "ecosystem.invariants.violations.json",
            "tlc.crates.json",
            "tlc.repos.json",
            "ppr.entrypoints.json",
            "ppr.aggregate.json",
        ];

        let mut status = Vec::new();
        for f in files {
            let p = out_dir.join(f);
            let meta = std::fs::metadata(&p).ok();
            status.push(serde_json::json!({
                "path": p.display().to_string(),
                "exists": meta.is_some(),
                "size_bytes": meta.as_ref().map(|m| m.len()),
                "mtime_epoch_secs": meta.and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs()),
            }));
        }

        let payload = serde_json::json!({
            "root": root.display().to_string(),
            "out_dir": out_dir.display().to_string(),
            "files": status,
        });
        mcp_ok("pkgrank_status", payload, None)
    }

    #[tool(description = "Generate pkgrank HTML/JSON artifacts (pkgrank view)")]
    async fn pkgrank_view(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());
        let mode = match params.0.mode.as_deref().unwrap_or("local") {
            "local" => ViewMode::Local,
            "cratesio" => ViewMode::CratesIo,
            "both" => ViewMode::Both,
            other => {
                return Err(McpError::invalid_params(
                    format!("mode must be one of: local, cratesio, both (got {other})"),
                    None,
                ));
            }
        };

        let args = ViewArgs {
            root,
            out,
            mode,
            local_top: 10,
            cratesio_depth: 2,
            cratesio_dev: false,
            cratesio_build: false,
            cratesio_optional: false,
            quiet: true,
        };

        run_view(&args).map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let html_path = Self::artifact_path(&args.root, &args.out, "pkgrank_overview.html");

        let payload = serde_json::json!({
            "root": args.root.display().to_string(),
            "out_dir": if args.out.is_absolute() { args.out.display().to_string() } else { args.root.join(&args.out).display().to_string() },
            "html_path": html_path.display().to_string(),
        });
        mcp_ok("pkgrank_view", payload, None)
    }

    #[tool(description = "Get repo details from artifacts")]
    async fn pkgrank_repo_detail(
        &self,
        params: Parameters<PkgrankRepoDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());
        let include_readme = params.0.include_readme.unwrap_or(false);
        let summarize_readme = params.0.summarize_readme.unwrap_or(false);
        let llm_input_max_chars = params.0.llm_input_max_chars.unwrap_or(12_000).min(80_000);
        let llm_timeout_secs = params.0.llm_timeout_secs.unwrap_or(30).min(600);
        let llm_cache = params.0.llm_cache.unwrap_or(true);
        let readme_max_chars = params.0.readme_max_chars.unwrap_or(4000).min(50_000);
        let mut payload = repo_detail_payload(
            &root,
            &out,
            &params.0.repo,
            include_readme,
            readme_max_chars,
        )?;

        let readme_path = find_readme_for_repo(&root, &params.0.repo);
        let ai = maybe_add_readme_llm_summary(
            &root,
            &out,
            "repo",
            &params.0.repo,
            readme_path.as_ref(),
            summarize_readme,
            llm_input_max_chars,
            llm_timeout_secs,
            llm_cache,
        )
        .await?;
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("readme_ai".to_string(), ai);
        }
        mcp_ok("pkgrank_repo_detail", payload, None)
    }

    #[tool(description = "Get crate details from artifacts")]
    async fn pkgrank_crate_detail(
        &self,
        params: Parameters<PkgrankCrateDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());
        let include_readme = params.0.include_readme.unwrap_or(false);
        let summarize_readme = params.0.summarize_readme.unwrap_or(false);
        let llm_input_max_chars = params.0.llm_input_max_chars.unwrap_or(12_000).min(80_000);
        let llm_timeout_secs = params.0.llm_timeout_secs.unwrap_or(30).min(600);
        let llm_cache = params.0.llm_cache.unwrap_or(true);
        let readme_max_chars = params.0.readme_max_chars.unwrap_or(4000).min(50_000);
        let mut payload = crate_detail_payload(
            &root,
            &out,
            &params.0.krate,
            include_readme,
            readme_max_chars,
        )?;

        // Use TLC row (manifest_path) to locate README if present.
        let readme_path = payload
            .get("tlc_row")
            .and_then(|v| v.get("manifest_path"))
            .and_then(|v| v.as_str())
            .and_then(|mp| find_readme_for_manifest(&root, mp));

        let ai = maybe_add_readme_llm_summary(
            &root,
            &out,
            "crate",
            &params.0.krate,
            readme_path.as_ref(),
            summarize_readme,
            llm_input_max_chars,
            llm_timeout_secs,
            llm_cache,
        )
        .await?;
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("readme_ai".to_string(), ai);
        }
        mcp_ok("pkgrank_crate_detail", payload, None)
    }

    // NOTE: internal artifact inspection tools live in `PkgrankStdioMcpDebug`.

    #[tool(description = "Snapshot pkgrank artifacts into a new directory (copy selected files)")]
    async fn pkgrank_snapshot(
        &self,
        params: Parameters<PkgrankSnapshotArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());

        let label = params.0.label.clone().unwrap_or_else(|| {
            format!(
                "run-{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            )
        });

        let dest = if let Some(d) = params.0.dest.as_deref() {
            PathBuf::from(d)
        } else {
            // Default: `<out>/runs/<label>`
            out.join("runs").join(&label)
        };

        let src_dir = if out.is_absolute() {
            out.clone()
        } else {
            root.join(&out)
        };
        let dst_dir = if dest.is_absolute() {
            dest.clone()
        } else {
            root.join(&dest)
        };
        Self::ensure_dir(&dst_dir)?;

        let files = [
            "pkgrank_overview.html",
            "by_repo.summary.json",
            "root.workspace_only.json",
            "ecosystem.repo_graph.rows.json",
            "ecosystem.axes.pagerank.json",
            "ecosystem.invariants.violations.json",
            "tlc.crates.json",
            "tlc.repos.json",
            "ppr.entrypoints.json",
            "ppr.aggregate.json",
        ];

        let mut copied = Vec::new();
        let mut missing = Vec::new();
        for f in files {
            let src = src_dir.join(f);
            let dst = dst_dir.join(f);
            if !src.exists() {
                missing.push(src.display().to_string());
                continue;
            }
            if let Some(parent) = dst.parent() {
                Self::ensure_dir(parent)?;
            }
            std::fs::copy(&src, &dst).map_err(|e| {
                McpError::internal_error(
                    format!(
                        "failed to copy {} -> {}: {}",
                        src.display(),
                        dst.display(),
                        e
                    ),
                    None,
                )
            })?;
            copied.push(serde_json::json!({
                "src": src.display().to_string(),
                "dst": dst.display().to_string(),
            }));
        }

        let payload = serde_json::json!({
            "root": root.display().to_string(),
            "source_dir": src_dir.display().to_string(),
            "dest_dir": dst_dir.display().to_string(),
            "label": label,
            "copied": copied,
            "missing": missing,
        });
        mcp_ok("pkgrank_snapshot", payload, None)
    }

    #[tool(description = "Compare two pkgrank artifact directories (TLC crates/repos deltas)")]
    async fn pkgrank_compare_runs(
        &self,
        params: Parameters<PkgrankCompareRunsArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let new_out = PathBuf::from(&params.0.new_out);
        let old_out = PathBuf::from(&params.0.old_out);
        let limit = params.0.limit.unwrap_or(25).min(200);

        let new_crates_path = Self::artifact_path(&root, &new_out, "tlc.crates.json");
        let old_crates_path = Self::artifact_path(&root, &old_out, "tlc.crates.json");
        let new_repos_path = Self::artifact_path(&root, &new_out, "tlc.repos.json");
        let old_repos_path = Self::artifact_path(&root, &old_out, "tlc.repos.json");

        let new_crates: Vec<TlcCrateRow> = Self::read_json_file(&new_crates_path)?;
        let old_crates: Vec<TlcCrateRow> = Self::read_json_file(&old_crates_path)?;
        let new_repos: Vec<TlcRepoRow> = Self::read_json_file(&new_repos_path)?;
        let old_repos: Vec<TlcRepoRow> = Self::read_json_file(&old_repos_path)?;

        let mut old_crate_rank: HashMap<String, usize> = HashMap::new();
        for (i, r) in old_crates.iter().enumerate() {
            old_crate_rank.insert(r.name.clone(), i + 1);
        }
        let mut old_repo_rank: HashMap<String, usize> = HashMap::new();
        for (i, r) in old_repos.iter().enumerate() {
            old_repo_rank.insert(r.repo.clone(), i + 1);
        }

        let mut crate_deltas = Vec::new();
        for (i, r) in new_crates.iter().enumerate() {
            let new_rank = i + 1;
            let old_rank = old_crate_rank.get(&r.name).copied();
            let delta_rank = old_rank.map(|o| (o as i64) - (new_rank as i64));
            crate_deltas.push(serde_json::json!({
                "crate": r.name,
                "repo": r.repo,
                "axis": r.axis,
                "new_rank": new_rank,
                "old_rank": old_rank,
                "delta_rank": delta_rank,
                "new_score": r.score,
            }));
        }
        crate_deltas.sort_by(|a, b| {
            let da = a
                .get("delta_rank")
                .and_then(|x| x.as_i64())
                .unwrap_or(0)
                .abs();
            let db = b
                .get("delta_rank")
                .and_then(|x| x.as_i64())
                .unwrap_or(0)
                .abs();
            db.cmp(&da)
        });

        let mut repo_deltas = Vec::new();
        for (i, r) in new_repos.iter().enumerate() {
            let new_rank = i + 1;
            let old_rank = old_repo_rank.get(&r.repo).copied();
            let delta_rank = old_rank.map(|o| (o as i64) - (new_rank as i64));
            repo_deltas.push(serde_json::json!({
                "repo": r.repo,
                "axis": r.axis,
                "new_rank": new_rank,
                "old_rank": old_rank,
                "delta_rank": delta_rank,
                "new_score": r.score,
            }));
        }
        repo_deltas.sort_by(|a, b| {
            let da = a
                .get("delta_rank")
                .and_then(|x| x.as_i64())
                .unwrap_or(0)
                .abs();
            let db = b
                .get("delta_rank")
                .and_then(|x| x.as_i64())
                .unwrap_or(0)
                .abs();
            db.cmp(&da)
        });

        let payload = serde_json::json!({
            "root": root.display().to_string(),
            "new_out": new_out.display().to_string(),
            "old_out": old_out.display().to_string(),
            "paths": {
                "new_tlc_crates": new_crates_path.display().to_string(),
                "old_tlc_crates": old_crates_path.display().to_string(),
                "new_tlc_repos": new_repos_path.display().to_string(),
                "old_tlc_repos": old_repos_path.display().to_string(),
            },
            "crate_rank_deltas": crate_deltas.into_iter().take(limit).collect::<Vec<_>>(),
            "repo_rank_deltas": repo_deltas.into_iter().take(limit).collect::<Vec<_>>(),
        });
        mcp_ok("pkgrank_compare_runs", payload, None)
    }

    #[tool(description = "Compute local crate ranking (pkgrank analyze)")]
    async fn pkgrank_analyze(
        &self,
        params: Parameters<PkgrankAnalyzeToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let path = params.0.path.clone().unwrap_or_else(|| ".".to_string());
        let metric = params
            .0
            .metric
            .as_deref()
            .map(parse_metric)
            .transpose()?
            .unwrap_or(Metric::Pagerank);
        let analyze = AnalyzeArgs {
            path: PathBuf::from(path),
            metric,
            top: params.0.top.unwrap_or(25),
            dev: params.0.dev.unwrap_or(false),
            build: params.0.build.unwrap_or(false),
            workspace_only: params.0.workspace_only.unwrap_or(true),
            all_features: false,
            no_default_features: false,
            features: None,
            format: OutputFormat::Json,
            stats: false,
            json_limit: None,
        };
        let (rows, convergence) = analyze_rows_with_convergence(&analyze)
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let rows_total = rows.len();
        let rows_returned = rows_total.min(analyze.top);
        let rows = rows.into_iter().take(analyze.top).collect::<Vec<_>>();
        let truncated = rows_returned < rows_total;
        let payload = serde_json::json!({
            "rows": rows,
            "rows_total": rows_total,
            "rows_returned": rows_returned,
            "truncated": truncated,
            "limit": analyze.top,
            "convergence": convergence,
            "metric": format!("{:?}", metric),
            "sorted_by": format!("{:?}", metric),
        });
        mcp_ok("pkgrank_analyze", payload, None)
    }

    #[tool(
        description = "Compute internal module/item centrality (pkgrank modules; cargo-modules-backed)"
    )]
    async fn pkgrank_modules(
        &self,
        params: Parameters<PkgrankModulesToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let mut args = modules_args_from_tool_params(&params.0)?;
        let (rows, nodes, edges, aggregate_label, top_edges) = match run_modules_core(&args) {
            Ok(v) => v,
            Err(e) => {
                // cargo-modules errors if a package has both --lib and --bin targets and neither
                // is explicitly selected. For MCP UX, auto-fallback to --lib when it looks safe.
                let msg = format!("{:#}", e);
                if !args.lib && args.bin.is_none() && msg.contains("Multiple targets present") {
                    args.lib = true;
                    run_modules_core(&args)
                        .map_err(|e2| McpError::internal_error(format!("{:#}", e2), None))?
                } else {
                    return Err(McpError::internal_error(msg, None));
                }
            }
        };
        let rows_total = rows.len();
        let rows_returned = rows_total.min(args.top);
        let rows = rows.into_iter().take(args.top).collect::<Vec<_>>();
        let truncated = rows_returned < rows_total;
        let payload = serde_json::json!({
            "effective": {
                "manifest_path": args.manifest_path.display().to_string(),
                "package": args.package,
                "target": if args.lib {
                    "lib".to_string()
                } else if let Some(bin) = args.bin.as_ref() {
                    format!("bin={bin}")
                } else {
                    "default".to_string()
                },
                "cfg_test": args.cfg_test,
                "preset": match args.preset {
                    ModulesPreset::None => None,
                    p => Some(format!("{p:?}")),
                },
                "metric": format!("{:?}", args.metric),
                "edge_kind": format!("{:?}", args.edge_kind),
                "aggregate": format!("{:?}", args.aggregate),
                "top": args.top,
                "include": {
                    "externs": args.include_externs,
                    "sysroot": args.include_sysroot,
                    "fns": args.include_fns,
                    "types": args.include_types,
                    "traits": args.include_traits,
                },
                "cache": args.cache,
                "cache_refresh": args.cache_refresh,
            },
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "aggregate_label": aggregate_label,
            },
            "rows": rows,
            "rows_total": rows_total,
            "rows_returned": rows_returned,
            "truncated": truncated,
            "limit": args.top,
            "top_edges": top_edges,
        });
        mcp_ok("pkgrank_modules", payload, None)
    }

    #[tool(
        description = "Compute internal module/item centrality across multiple packages (pkgrank modules-sweep)"
    )]
    async fn pkgrank_modules_sweep(
        &self,
        params: Parameters<PkgrankModulesSweepToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let args = modules_sweep_args_from_tool_params(&params.0)?;
        let include_rows = params.0.include_rows.unwrap_or(false);
        let include_top_edges = params.0.include_top_edges.unwrap_or(false);
        let payload = modules_sweep_payload(&args, include_rows, include_top_edges)
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        mcp_ok("pkgrank_modules_sweep", payload, None)
    }
}

#[cfg(feature = "stdio")]
#[tool_handler]
impl rmcp::ServerHandler for PkgrankStdioMcpFull {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Tools for ranking crates/repos in the local Cargo workspace (pkgrank).\n\n\
                 Toolsets:\n\
                 - default: PKGRANK_MCP_TOOLSET=slim (small surface)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=full (includes module/type graph tools)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=debug (internal artifact inspection tools)"
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

// Slim stdio MCP server: smaller tool surface for day-to-day use.
#[cfg(feature = "stdio")]
#[tool_router]
impl PkgrankStdioMcpSlim {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Triage bundle: top TLC crates/repos + invariants + PPR top-k (artifact-backed)"
    )]
    async fn pkgrank_triage(
        &self,
        params: Parameters<PkgrankTriageArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let mode = match params.0.mode.as_deref().unwrap_or("local") {
            "local" => ViewMode::Local,
            "cratesio" => ViewMode::CratesIo,
            "both" => ViewMode::Both,
            other => {
                return Err(McpError::invalid_params(
                    format!("mode must be one of: local, cratesio, both (got {other})"),
                    None,
                ));
            }
        };

        // Reuse the CLI triage implementation by mapping MCP params into CLI args.
        let cli = TriageCliArgs {
            root,
            out,
            refresh_if_missing: params.0.refresh_if_missing.unwrap_or(true),
            mode,
            stale_minutes: params.0.stale_minutes.unwrap_or(60),
            limit: params.0.limit.unwrap_or(15),
            axis: params.0.axis.clone(),
            ppr_top: params.0.ppr_top.unwrap_or(12),
            summarize_readmes: params.0.summarize_readmes.unwrap_or(false),
            summarize_repos_top: params.0.summarize_repos_top.unwrap_or(0),
            summarize_crates_top: params.0.summarize_crates_top.unwrap_or(0),
            llm_input_max_chars: params.0.llm_input_max_chars.unwrap_or(12_000),
            llm_timeout_secs: params.0.llm_timeout_secs.unwrap_or(30),
            llm_cache: params.0.llm_cache.unwrap_or(true),
            llm_include_raw: params.0.llm_include_raw.unwrap_or(false),
            format: OutputFormat::Json,
        };

        let (payload, summary) = triage_payload_from_cli(&cli)
            .await
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        mcp_ok("pkgrank_triage", payload, Some(summary))
    }

    #[tool(description = "Generate pkgrank artifacts (pkgrank view)")]
    async fn pkgrank_view(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        // Delegate to full implementation logic by constructing args locally (copy, not reuse trait objects).
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let mode = match params.0.mode.as_deref().unwrap_or("local") {
            "local" => ViewMode::Local,
            "cratesio" => ViewMode::CratesIo,
            "both" => ViewMode::Both,
            other => {
                return Err(McpError::invalid_params(
                    format!("mode must be one of: local, cratesio, both (got {other})"),
                    None,
                ));
            }
        };

        let args = ViewArgs {
            root,
            out,
            mode,
            local_top: 10,
            cratesio_depth: 2,
            cratesio_dev: false,
            cratesio_build: false,
            cratesio_optional: false,
            quiet: true,
        };

        run_view(&args).map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let html_path =
            PkgrankStdioMcpFull::artifact_path(&args.root, &args.out, "pkgrank_overview.html");
        let payload = serde_json::json!({
            "root": args.root.display().to_string(),
            "out_dir": if args.out.is_absolute() { args.out.display().to_string() } else { args.root.join(&args.out).display().to_string() },
            "html_path": html_path.display().to_string(),
        });
        mcp_ok("pkgrank_view", payload, None)
    }

    #[tool(description = "Compute local crate ranking (pkgrank analyze)")]
    async fn pkgrank_analyze(
        &self,
        params: Parameters<PkgrankAnalyzeToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull {
            tool_router: PkgrankStdioMcpFull::tool_router(),
        }
        .pkgrank_analyze(params)
        .await
    }

    #[tool(description = "Get repo details from artifacts")]
    async fn pkgrank_repo_detail(
        &self,
        params: Parameters<PkgrankRepoDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let include_readme = params.0.include_readme.unwrap_or(false);
        let readme_max_chars = params.0.readme_max_chars.unwrap_or(4000).min(50_000);
        let payload = repo_detail_payload(
            &root,
            &out,
            &params.0.repo,
            include_readme,
            readme_max_chars,
        )?;
        mcp_ok("pkgrank_repo_detail", payload, None)
    }

    #[tool(description = "Get crate details from artifacts")]
    async fn pkgrank_crate_detail(
        &self,
        params: Parameters<PkgrankCrateDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let include_readme = params.0.include_readme.unwrap_or(false);
        let readme_max_chars = params.0.readme_max_chars.unwrap_or(4000).min(50_000);
        let payload = crate_detail_payload(
            &root,
            &out,
            &params.0.krate,
            include_readme,
            readme_max_chars,
        )?;
        mcp_ok("pkgrank_crate_detail", payload, None)
    }

    #[tool(description = "Snapshot pkgrank artifacts (copy selected files)")]
    async fn pkgrank_snapshot(
        &self,
        params: Parameters<PkgrankSnapshotArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull {
            tool_router: PkgrankStdioMcpFull::tool_router(),
        }
        .pkgrank_snapshot(params)
        .await
    }

    #[tool(description = "Compare two artifact directories (TLC deltas)")]
    async fn pkgrank_compare_runs(
        &self,
        params: Parameters<PkgrankCompareRunsArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull {
            tool_router: PkgrankStdioMcpFull::tool_router(),
        }
        .pkgrank_compare_runs(params)
        .await
    }
}

#[cfg(feature = "stdio")]
#[tool_handler]
impl rmcp::ServerHandler for PkgrankStdioMcpSlim {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Tools for ranking crates/repos in the local Cargo workspace (pkgrank).\n\n\
                 Toolsets:\n\
                 - default: PKGRANK_MCP_TOOLSET=slim (small surface)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=full (includes module/type graph tools)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=debug (internal artifact inspection tools)"
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

fn find_readme_for_manifest(root: &Path, manifest_path: &str) -> Option<PathBuf> {
    let manifest = Path::new(manifest_path);
    let dir = manifest.parent()?;

    // Prefer crate-local README.{md,MD}.
    for name in ["README.md", "readme.md", "README.MD", "readme.MD"] {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }

    // Fallback: repo root README.{md,MD} where repo is the first path component under `root`.
    let repo = dir
        .strip_prefix(root)
        .ok()
        .and_then(|p| p.components().next())
        .map(|c| c.as_os_str().to_string_lossy().to_string())?;
    for name in ["README.md", "readme.md", "README.MD", "readme.MD"] {
        let p = root.join(&repo).join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn find_readme_for_repo(root: &Path, repo: &str) -> Option<PathBuf> {
    for name in ["README.md", "readme.md", "README.MD", "readme.MD"] {
        let p = root.join(repo).join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn read_text_snippet(path: &Path, max_chars: usize) -> Result<Option<(String, bool)>> {
    let Ok(s) = fs::read_to_string(path) else {
        return Ok(None);
    };
    if s.is_empty() {
        return Ok(None);
    }
    if s.chars().count() <= max_chars {
        return Ok(Some((s, false)));
    }
    let snippet: String = s.chars().take(max_chars).collect();
    Ok(Some((snippet, true)))
}

fn file_age_minutes(path: &Path) -> Option<u64> {
    let md = std::fs::metadata(path).ok()?;
    let m = md.modified().ok()?;
    let now = std::time::SystemTime::now();
    let dt = now.duration_since(m).ok()?;
    Some(dt.as_secs() / 60)
}

#[cfg(feature = "stdio")]
fn llm_summary_cmd_from_env() -> Result<Option<Vec<String>>, McpError> {
    // JSON array, e.g.:
    //   export PKGRANK_LLM_SUMMARY_CMD='["ollama","run","llama3.1"]'
    let Some(s) = std::env::var_os("PKGRANK_LLM_SUMMARY_CMD") else {
        return Ok(None);
    };
    let s = s.to_string_lossy().to_string();
    if s.trim().is_empty() {
        return Ok(None);
    }
    let v: Vec<String> = serde_json::from_str(&s).map_err(|e| {
        McpError::invalid_params(
            format!("PKGRANK_LLM_SUMMARY_CMD must be JSON array of strings: {e}"),
            None,
        )
    })?;
    if v.is_empty() {
        return Ok(None);
    }
    Ok(Some(v))
}

#[cfg(feature = "stdio")]
fn sanitize_id_piece(s: &str, max_len: usize) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if out.len() >= max_len {
            break;
        }
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "x".to_string()
    } else {
        out
    }
}

#[cfg(feature = "stdio")]
fn readme_ai_cache_path(
    root: &Path,
    out: &Path,
    kind: &str,
    name: &str,
    readme_path: &Path,
    cmd_id: &str,
) -> PathBuf {
    let out_dir = if out.is_absolute() {
        out.to_path_buf()
    } else {
        root.join(out)
    };
    let cache_dir = out_dir.join("readme_ai_cache");

    let md = std::fs::metadata(readme_path).ok();
    let len = md.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime_secs = md
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let name = sanitize_id_piece(name, 80);
    let cmd_id = sanitize_id_piece(cmd_id, 80);
    let fname = format!("{kind}__{name}__{mtime_secs}__{len}__{cmd_id}__v2.summary.json");
    cache_dir.join(fname)
}

#[cfg(feature = "stdio")]
async fn llm_summarize_readme(
    cmd: &[String],
    readme_text: &str,
    timeout_secs: u64,
) -> Result<String, McpError> {
    if cmd.is_empty() {
        return Err(McpError::invalid_params(
            "LLM cmd is empty".to_string(),
            None,
        ));
    }
    let mut c = TokioCommand::new(&cmd[0]);
    if cmd.len() > 1 {
        c.args(&cmd[1..]);
    }
    c.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = c
        .spawn()
        .map_err(|e| McpError::internal_error(format!("failed to spawn LLM cmd: {e}"), None))?;

    let prompt = r#"You are assisting with repository triage. Produce STRICT JSON ONLY (no markdown, no extra prose).

Goal: summarize a README for maintenance decisions.

Constraints:
- dry, minimal, no marketing
- keep strings short; lists <= 8 items each
- if unknown, use empty list or null

Output JSON schema:
{
  "what": "one sentence",
  "how_used": ["..."],
  "public_invariants": ["things that must not change"],
  "swappable_parts": ["things that can change without breaking users"],
  "risks": ["what is risky to change / likely footguns"],
  "entrypoints": ["commands / crates / binaries / main modules if visible"],
  "links": ["relevant docs paths if visible"]
}

README:
"#;

    {
        let mut stdin = child.stdin.take().ok_or_else(|| {
            McpError::internal_error("failed to open stdin for LLM process".to_string(), None)
        })?;
        let input = format!("{prompt}\n{readme_text}\n");
        tokio::io::AsyncWriteExt::write_all(&mut stdin, input.as_bytes())
            .await
            .map_err(|e| {
                McpError::internal_error(format!("failed writing to LLM stdin: {e}"), None)
            })?;
    }

    let out = timeout(Duration::from_secs(timeout_secs), child.wait_with_output())
        .await
        .map_err(|_| {
            McpError::internal_error(format!("LLM cmd timed out after {timeout_secs}s"), None)
        })?
        .map_err(|e| McpError::internal_error(format!("failed to wait for LLM cmd: {e}"), None))?;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(McpError::internal_error(
            format!("LLM cmd failed: status={} stderr={}", out.status, err),
            None,
        ));
    }

    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        return Err(McpError::internal_error(
            "LLM returned empty summary".to_string(),
            None,
        ));
    }
    Ok(s)
}

fn format_top_tlc_repos(rows: &[TlcRepoRow], k: usize) -> String {
    let mut out = String::new();
    for (i, r) in rows.iter().take(k).enumerate() {
        let _ = writeln!(
            &mut out,
            "{}. {} ({}) score={:.2} deps={} deps_pr={:.4} cons_pr={:.4} why={}",
            i + 1,
            r.repo,
            r.axis,
            r.score,
            r.third_party_deps,
            r.deps_pagerank,
            r.consumers_pagerank,
            r.why
        );
    }
    out
}

fn format_top_tlc_crates(rows: &[TlcCrateRow], k: usize) -> String {
    let mut out = String::new();
    for (i, r) in rows.iter().take(k).enumerate() {
        let _ = writeln!(
            &mut out,
            "{}. {} ({}) score={:.2} deps={} pr={:.4} why={}",
            i + 1,
            r.name,
            r.axis,
            r.score,
            r.third_party_deps,
            r.pagerank,
            r.why
        );
    }
    out
}

fn summarize_violation_rules(
    violations: &[RepoInvariantViolation],
    topk: usize,
) -> Vec<(String, usize)> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for v in violations {
        *counts.entry(v.rule.as_str()).or_insert(0) += 1;
    }
    let mut out: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.into_iter().take(topk).collect()
}

fn repo_detail_payload(
    root: &Path,
    out: &Path,
    repo: &str,
    include_readme: bool,
    readme_max_chars: usize,
) -> Result<serde_json::Value, McpError> {
    let rows_path = PkgrankStdioMcpFull::artifact_path(root, out, "ecosystem.repo_graph.rows.json");
    let rows: Vec<RepoRow> = PkgrankStdioMcpFull::read_json_file(&rows_path)?;
    let row = rows.into_iter().find(|r| r.repo == repo);

    let tlc_path = PkgrankStdioMcpFull::artifact_path(root, out, "tlc.repos.json");
    let tlc_rows: Vec<TlcRepoRow> = PkgrankStdioMcpFull::read_json_file(&tlc_path)?;
    let tlc = tlc_rows.into_iter().find(|r| r.repo == repo);

    let readme = if include_readme {
        let readme_path = find_readme_for_repo(root, repo);
        readme_path
            .as_ref()
            .and_then(|p| read_text_snippet(p, readme_max_chars).ok().flatten())
            .map(|(txt, truncated)| {
                serde_json::json!({
                    "path": p_display(root, readme_path.as_ref()),
                    "truncated": truncated,
                    "text": txt,
                })
            })
    } else {
        None
    };

    Ok(serde_json::json!({
        "ok": true,
        "repo": repo,
        "repo_row": row,
        "tlc_row": tlc,
        "readme": readme,
    }))
}

fn crate_detail_payload(
    root: &Path,
    out: &Path,
    name: &str,
    include_readme: bool,
    readme_max_chars: usize,
) -> Result<serde_json::Value, McpError> {
    let tlc_path = PkgrankStdioMcpFull::artifact_path(root, out, "tlc.crates.json");
    let tlc_rows: Vec<TlcCrateRow> = PkgrankStdioMcpFull::read_json_file(&tlc_path)?;
    let tlc = tlc_rows.into_iter().find(|r| r.name == name);

    let readme = if include_readme {
        tlc.as_ref()
            .and_then(|r| find_readme_for_manifest(root, &r.manifest_path))
            .and_then(|p| read_text_snippet(&p, readme_max_chars).ok().flatten().map(|(txt, truncated)| {
                serde_json::json!({
                    "path": p.strip_prefix(root).ok().map(|x| x.display().to_string()).unwrap_or_else(|| p.display().to_string()),
                    "truncated": truncated,
                    "text": txt,
                })
            }))
    } else {
        None
    };

    Ok(serde_json::json!({
        "ok": true,
        "crate": name,
        "tlc_row": tlc,
        "readme": readme,
    }))
}

#[cfg(feature = "stdio")]
#[allow(clippy::too_many_arguments)]
async fn maybe_add_readme_llm_summary(
    root: &Path,
    out: &Path,
    kind: &str,
    name: &str,
    readme_path: Option<&PathBuf>,
    summarize: bool,
    llm_input_max_chars: usize,
    llm_timeout_secs: u64,
    llm_cache: bool,
) -> Result<serde_json::Value, McpError> {
    if !summarize {
        return Ok(serde_json::json!({ "enabled": false }));
    }

    let Some(readme_path) = readme_path else {
        return Ok(
            serde_json::json!({ "enabled": true, "available": false, "reason": "no readme found" }),
        );
    };

    let Some(cmd) = llm_summary_cmd_from_env()? else {
        return Ok(serde_json::json!({
            "enabled": true,
            "available": false,
            "reason": "PKGRANK_LLM_SUMMARY_CMD not set (expected JSON array, e.g. [\"ollama\",\"run\",\"llama3.1\"])"
        }));
    };
    let cmd_id = cmd.join("_");

    let cache_path = readme_ai_cache_path(root, out, kind, name, readme_path, &cmd_id);
    if llm_cache {
        if let Ok(s) = std::fs::read_to_string(&cache_path) {
            if !s.trim().is_empty() {
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(&s);
                return Ok(serde_json::json!({
                    "enabled": true,
                    "available": true,
                    "cached": true,
                    "cache_path": p_display(root, Some(&cache_path)),
                    "cmd": cmd,
                    "schema_version": 2,
                    "raw": s,
                    "parsed": parsed.ok(),
                }));
            }
        }
    }

    let snippet = read_text_snippet(readme_path, llm_input_max_chars)
        .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
    let Some((txt, truncated)) = snippet else {
        return Ok(
            serde_json::json!({ "enabled": true, "available": false, "reason": "readme unreadable/empty" }),
        );
    };

    let summary = llm_summarize_readme(&cmd, &txt, llm_timeout_secs).await?;
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&summary);

    if llm_cache {
        if let Some(parent) = cache_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&cache_path, &summary);
    }

    Ok(serde_json::json!({
        "enabled": true,
        "available": true,
        "cached": false,
        "cmd": cmd,
        "readme_path": p_display(root, Some(readme_path)),
        "readme_truncated_for_llm": truncated,
        "schema_version": 2,
        "raw": summary,
        "parsed": parsed.ok(),
        "cache_path": if llm_cache {
            serde_json::json!(p_display(root, Some(&cache_path)))
        } else {
            serde_json::Value::Null
        },
    }))
}

fn readme_details_html(root: &Path, readme_path: Option<PathBuf>) -> String {
    let Some(path) = readme_path else {
        return String::new();
    };
    let rel = path.strip_prefix(root).unwrap_or(&path);
    let mut out = String::new();
    out.push_str("<details class=\"readme\"><summary>README</summary>");
    out.push_str(&format!(
        "<div style=\"color:#3d3d3d;font-size:11px;margin:4px 0;\"><code>{}</code></div>",
        html_escape(&rel.display().to_string())
    ));

    match read_text_snippet(&path, 12_000) {
        Ok(Some((txt, truncated))) => {
            out.push_str("<pre class=\"readme\">");
            out.push_str(&html_escape(&txt));
            out.push_str("</pre>");
            if truncated {
                out.push_str("<div style=\"color:#3d3d3d;font-size:11px;\">(truncated)</div>");
            }
        }
        Ok(None) => {
            out.push_str("<div style=\"color:#3d3d3d;font-size:11px;\">(empty)</div>");
        }
        Err(e) => {
            out.push_str(&format!(
                "<div style=\"color:#a00;font-size:11px;\">failed to read: <code>{}</code></div>",
                html_escape(&format!("{:#}", e))
            ));
        }
    }

    out.push_str("</details>");
    out
}

fn infer_repo_for_manifest(root: &Path, manifest_path: &str) -> String {
    let mp = Path::new(manifest_path);
    let rel = mp.strip_prefix(root).unwrap_or(mp);
    let mut it = rel.components();
    let first = it
        .next()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Treat `_mcp/*` crates as their own repos.
    if first == "_mcp" {
        if let Some(second) = it.next() {
            return second.as_os_str().to_string_lossy().to_string();
        }
        return "mcp".to_string();
    }

    first
}

fn render_repo_scatter_svg(rows: &[RepoRow]) -> String {
    // Very small, static visualization: deps PageRank vs commits_30d (log1p),
    // with point size ~ third_party_deps, color by axis.
    //
    // This is intentionally dependency-free (no JS), so the HTML artifact stays portable.
    let w = 860.0;
    let h = 260.0;
    let pad_l = 40.0;
    let pad_r = 10.0;
    let pad_t = 10.0;
    let pad_b = 30.0;

    let mut max_x = 0.0_f64;
    let mut max_y = 0.0_f64;
    let mut max_3p = 0usize;
    for r in rows {
        max_x = max_x.max(r.pagerank);
        let c = r.git_commits_30d.unwrap_or(0);
        let y = (c as f64 + 1.0).ln();
        max_y = max_y.max(y);
        max_3p = max_3p.max(r.third_party_deps);
    }
    if max_x <= 0.0 {
        max_x = 1e-9;
    }
    if max_y <= 0.0 {
        max_y = 1e-9;
    }

    let mut out = String::new();
    out.push_str(&format!(
        "<svg width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\"repo scatter\">\n"
    ));
    out.push_str("<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"#f7f7f7\" stroke=\"#e1e1e1\" />\n");

    // Axes
    out.push_str(&format!(
        "<line x1=\"{pad_l}\" y1=\"{pad_t}\" x2=\"{pad_l}\" y2=\"{}\" stroke=\"#777\" />\n",
        h - pad_b
    ));
    out.push_str(&format!(
        "<line x1=\"{pad_l}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#777\" />\n",
        h - pad_b,
        w - pad_r,
        h - pad_b
    ));
    out.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" font-family=\"ui-monospace,Menlo,monospace\" font-size=\"11\" fill=\"#333\">deps_pr</text>\n",
        w - pad_r - 60.0,
        h - 8.0
    ));
    out.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" font-family=\"ui-monospace,Menlo,monospace\" font-size=\"11\" fill=\"#333\" transform=\"rotate(-90 {x} {y})\">log1p(commits_30d)</text>\n",
        10.0,
        pad_t + 90.0,
        x = 10.0,
        y = pad_t + 90.0
    ));

    // Points (deterministic order)
    let mut rows_sorted: Vec<&RepoRow> = rows.iter().collect();
    rows_sorted.sort_by(|a, b| a.repo.cmp(&b.repo));

    for r in rows_sorted {
        let x = (r.pagerank / max_x) * (w - pad_l - pad_r) + pad_l;
        let c = r.git_commits_30d.unwrap_or(0);
        let yv = (c as f64 + 1.0).ln();
        let y = (h - pad_b) - (yv / max_y) * (h - pad_t - pad_b);
        let rad = if max_3p == 0 {
            3.0
        } else {
            3.0 + 9.0 * ((r.third_party_deps as f64) / (max_3p as f64)).sqrt()
        };
        let color = match r.axis.as_str() {
            "tekne" => "#2b6cb0",
            "agents" => "#c05621",
            "governance" => "#2f855a",
            _ => "#555555",
        };
        // Minimal tooltip via <title>
        out.push_str(&format!(
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.2}\" fill=\"{}\" fill-opacity=\"0.55\" stroke=\"#333\" stroke-opacity=\"0.2\"><title>{}</title></circle>\n",
            x,
            y,
            rad,
            color,
            html_escape(&format!(
                "{} axis={} deps_pr={:.6} commits_30d={} 3p={}",
                r.repo,
                r.axis,
                r.pagerank,
                r.git_commits_30d.unwrap_or(0),
                r.third_party_deps
            ))
        ));
    }

    out.push_str("</svg>\n");
    out
}
