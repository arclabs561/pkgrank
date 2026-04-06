use super::*;

#[derive(Parser, Debug, Clone)]
pub(crate) struct ModulesArgs {
    /// Path to a Cargo manifest to analyze.
    #[arg(long, default_value = "Cargo.toml")]
    pub(crate) manifest_path: PathBuf,

    /// Package to analyze (same meaning as `cargo modules -p ...`).
    #[arg(short = 'p', long)]
    pub(crate) package: Option<String>,

    /// Analyze the package library target.
    #[arg(long)]
    pub(crate) lib: bool,

    /// Analyze the named binary target.
    #[arg(long)]
    pub(crate) bin: Option<String>,

    /// Analyze with `#[cfg(test)]` enabled (as if built via `cargo test`).
    #[arg(long)]
    pub(crate) cfg_test: bool,

    /// Centrality metric to compute.
    #[arg(short, long, value_enum, default_value_t = Metric::Pagerank)]
    pub(crate) metric: Metric,

    /// Preset of common analysis settings.
    ///
    /// This exists to avoid long flag strings while still keeping every knob overrideable.
    ///
    /// Use `--preset none` to disable preset application and rely only on explicit flags.
    #[arg(long, value_enum, default_value_t = ModulesPreset::None)]
    pub(crate) preset: ModulesPreset,

    /// Number of top rows to show (text output only).
    #[arg(short = 'n', long, default_value_t = 25)]
    pub(crate) top: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub(crate) format: OutputFormat,

    /// Which edge kinds from cargo-modules to include.
    #[arg(long, value_enum, default_value_t = ModuleEdgeKind::Uses)]
    pub(crate) edge_kind: ModuleEdgeKind,

    /// Aggregate nodes before scoring.
    ///
    /// Why this exists: developers often organize work by *files* or *modules*, not by individual
    /// items. Aggregation provides a more "actionable" hotspot list.
    #[arg(long, value_enum, default_value_t = ModuleAggregate::File)]
    pub(crate) aggregate: ModuleAggregate,

    /// When `--edge-kind both`, weight for `uses` edges.
    ///
    /// Default: 1.0
    #[arg(long, default_value_t = 1.0)]
    pub(crate) uses_weight: f64,

    /// When `--edge-kind both`, weight for `owns` edges.
    ///
    /// Default: 0.2 (structural containment is informative but can dominate if unweighted).
    #[arg(long, default_value_t = 0.2)]
    pub(crate) owns_weight: f64,

    /// Show top inter-group edges (by total weight) after the table.
    #[arg(long, default_value_t = 3)]
    pub(crate) edges_top: usize,

    /// When aggregating, show the top-N member items by node-level PageRank.
    #[arg(long, default_value_t = 2)]
    pub(crate) members_top: usize,

    /// Cache `cargo modules dependencies` DOT output on disk (default: true).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) cache: bool,
    /// Force refresh cache (ignore existing cached DOT).
    #[arg(long, default_value_t = false)]
    pub(crate) cache_refresh: bool,

    /// Hide extern crates in the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_externs: bool,
    /// Include extern items (override for `--no-externs`).
    #[arg(long, default_value_t = false)]
    pub(crate) include_externs: bool,

    /// Hide sysroot crates (`std`, `core`, etc.) in the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_sysroot: bool,
    /// Include sysroot crates (override for `--no-sysroot`).
    #[arg(long, default_value_t = false)]
    pub(crate) include_sysroot: bool,

    /// Hide functions from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_fns: bool,
    /// Include functions (override for `--no-fns`).
    #[arg(long, default_value_t = false)]
    pub(crate) include_fns: bool,

    /// Hide traits from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_traits: bool,
    /// Include traits (override for `--no-traits`).
    #[arg(long, default_value_t = true)]
    pub(crate) include_traits: bool,

    /// Hide types (struct/enum/union) from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_types: bool,
    /// Include types (override for `--no-types`).
    #[arg(long, default_value_t = true)]
    pub(crate) include_types: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum ModuleEdgeKind {
    Uses,
    Owns,
    Both,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum ModulesPreset {
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
pub(crate) enum ModuleAggregate {
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
pub(crate) struct ModulesSweepArgs {
    /// Path to a Cargo manifest (workspace root or a single crate).
    #[arg(long, default_value = "Cargo.toml")]
    pub(crate) manifest_path: PathBuf,

    /// Package(s) to analyze (repeatable). If omitted, use `--all-packages`.
    #[arg(short = 'p', long)]
    pub(crate) package: Vec<String>,

    /// Analyze all workspace member packages under the manifest.
    #[arg(long, default_value_t = false)]
    pub(crate) all_packages: bool,

    /// Analyze the package library target.
    #[arg(long)]
    pub(crate) lib: bool,

    /// Analyze the named binary target.
    ///
    /// Note: for a sweep, specifying a single `--bin` applies to all packages (often not what you want).
    #[arg(long)]
    pub(crate) bin: Option<String>,

    /// Analyze with `#[cfg(test)]` enabled (as if built via `cargo test`).
    #[arg(long)]
    pub(crate) cfg_test: bool,

    /// Centrality metric to compute.
    #[arg(short, long, value_enum, default_value_t = Metric::Pagerank)]
    pub(crate) metric: Metric,

    /// Preset of common analysis settings.
    ///
    /// Use `--preset none` to disable preset application and rely only on explicit flags.
    #[arg(long, value_enum, default_value_t = ModulesPreset::None)]
    pub(crate) preset: ModulesPreset,

    /// Only print the compact summary table (no per-package tables).
    ///
    /// This is the "fast scan" mode for large workspaces.
    #[arg(long, default_value_t = true)]
    pub(crate) summary_only: bool,

    /// Continue the sweep if a package analysis fails.
    ///
    /// When enabled, failures are recorded in the summary table and (unless `--summary-only`)
    /// printed with their full error under a per-package section.
    ///
    /// Note: this is a boolean *value* (not a pure flag), because the default is `true`.
    /// Example: `--continue-on-error=false`
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) continue_on_error: bool,

    /// Convenience: stop on first package error (`--continue-on-error=false`).
    #[arg(long, default_value_t = false)]
    pub(crate) fail_fast: bool,

    /// Number of top rows per package to show (text output only).
    #[arg(short = 'n', long, default_value_t = 12)]
    pub(crate) top: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub(crate) format: OutputFormat,

    /// Which edge kinds from cargo-modules to include.
    #[arg(long, value_enum, default_value_t = ModuleEdgeKind::Uses)]
    pub(crate) edge_kind: ModuleEdgeKind,

    /// Aggregate nodes before scoring.
    #[arg(long, value_enum, default_value_t = ModuleAggregate::File)]
    pub(crate) aggregate: ModuleAggregate,

    /// When `--edge-kind both`, weight for `uses` edges.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) uses_weight: f64,

    /// When `--edge-kind both`, weight for `owns` edges.
    #[arg(long, default_value_t = 0.2)]
    pub(crate) owns_weight: f64,

    /// Show top inter-group edges (by total weight) after each package table.
    #[arg(long, default_value_t = 3)]
    pub(crate) edges_top: usize,

    /// When aggregating, show the top-N member items by node-level PageRank.
    #[arg(long, default_value_t = 2)]
    pub(crate) members_top: usize,

    /// Cache `cargo modules dependencies` DOT output on disk (default: true).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) cache: bool,
    /// Force refresh cache (ignore existing cached DOT).
    #[arg(long, default_value_t = false)]
    pub(crate) cache_refresh: bool,

    /// Hide extern crates in the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_externs: bool,
    /// Include extern items (override for `--no-externs`).
    #[arg(long, default_value_t = false)]
    pub(crate) include_externs: bool,

    /// Hide sysroot crates (`std`, `core`, etc.) in the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_sysroot: bool,
    /// Include sysroot crates (override for `--no-sysroot`).
    #[arg(long, default_value_t = false)]
    pub(crate) include_sysroot: bool,

    /// Hide functions from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_fns: bool,
    /// Include functions (override for `--no-fns`).
    #[arg(long, default_value_t = false)]
    pub(crate) include_fns: bool,

    /// Hide traits from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_traits: bool,
    /// Include traits (override for `--no-traits`).
    #[arg(long, default_value_t = true)]
    pub(crate) include_traits: bool,

    /// Hide types (struct/enum/union) from the cargo-modules graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub(crate) no_types: bool,
    /// Include types (override for `--no-types`).
    #[arg(long, default_value_t = true)]
    pub(crate) include_types: bool,
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

pub(crate) fn run_modules_sweep(args: &ModulesSweepArgs) -> Result<()> {
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
                            .max_by(|a, b| a.pagerank.total_cmp(&b.pagerank))
                            .map(|x| format!("{}({:.3})", x.node, x.pagerank))
                            .unwrap_or_else(|| "-".to_string());
                        let top_cons = ok
                            .rows
                            .iter()
                            .max_by(|a, b| a.consumers_pagerank.total_cmp(&b.consumers_pagerank))
                            .map(|x| format!("{}({:.3})", x.node, x.consumers_pagerank))
                            .unwrap_or_else(|| "-".to_string());
                        let top_between = ok
                            .rows
                            .iter()
                            .max_by(|a, b| a.betweenness.total_cmp(&b.betweenness))
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
    out.push('\u{2026}');
    out
}

#[derive(Debug, Serialize)]
pub(crate) struct ModuleRow {
    /// Node identifier (or an aggregate key).
    pub(crate) node: String,
    /// Aggregate kind (node/module/file), for interpretability.
    pub(crate) aggregate: String,
    /// Node kind, when available (crate/mod/struct/enum/trait/fn/...).
    pub(crate) kind: Option<String>,
    /// Visibility string, when available (`pub`, `pub(crate)`, `pub(self)`, ...).
    pub(crate) visibility: Option<String>,
    /// Group size (only meaningful when aggregate != node).
    pub(crate) group_size: Option<usize>,
    /// Small member preview (only meaningful when aggregate != node).
    pub(crate) members_preview: Option<String>,
    /// Small "top members" preview (aggregate != node): member items with highest node-level PageRank.
    pub(crate) top_members_pr: Option<String>,
    /// Transitive dependencies within the chosen graph (reachability following A -> B).
    pub(crate) transitive_dependencies: usize,
    /// Transitive dependents within the chosen graph (reachability in reversed graph).
    pub(crate) transitive_dependents: usize,
    pub(crate) in_degree: usize,
    pub(crate) out_degree: usize,
    pub(crate) pagerank: f64,
    pub(crate) consumers_pagerank: f64,
    pub(crate) betweenness: f64,
}

#[derive(Debug, Clone)]
struct CargoModulesNodeMeta {
    kind: Option<String>,
    visibility: Option<String>,
}

pub(crate) fn run_modules(args: &ModulesArgs) -> Result<()> {
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
pub(crate) fn run_modules_core(
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

    // Node-level centralities (used for "top members" previews during aggregation).
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
            let node = g2.nw(n).clone();
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
                        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
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
    rows.sort_by(|a, b| b.pagerank.total_cmp(&a.pagerank));

    let mut top_edges: Vec<(String, String, f64)> = Vec::new();
    if args.edges_top > 0 {
        for e in g2.edge_references() {
            let u = g2.nw(e.source()).clone();
            let v = g2.nw(e.target()).clone();
            let w = (*e.weight()).max(0.0);
            top_edges.push((u, v, w));
        }
        top_edges.sort_by(|a, b| b.2.total_cmp(&a.2));
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

pub(crate) fn apply_modules_preset(args: &ModulesArgs) -> ModulesArgs {
    // Preset is a "default bundle": apply only when a field is still at its default.
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

// fnv1a64 is now in dep_graph.rs (imported via `use super::*`).

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
        Metric::Pagerank => b.pagerank.total_cmp(&a.pagerank),
        Metric::ConsumersPagerank => b.consumers_pagerank.total_cmp(&a.consumers_pagerank),
        Metric::Indegree => b.in_degree.cmp(&a.in_degree),
        Metric::Outdegree => b.out_degree.cmp(&a.out_degree),
        Metric::Betweenness => b.betweenness.total_cmp(&a.betweenness),
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
        "\n{} nodes, {} edges\nEdge semantics: A -> B means A {} B",
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
        let name = g.nw(n);
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
        let from = g.nw(NodeIndex::new(u));
        let to = g.nw(NodeIndex::new(v));
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
    // may be the workspace root `Cargo.toml`, not the package's own `Cargo.toml`.
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
                // This matters for centrality baselines and for "what exists but is disconnected?".
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
