use anyhow::{anyhow, Context, Result};
use clap::Parser;
use petgraph::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use super::*;
use crate::dep_graph::{build_dep_graph, DepNode, Ecosystem};

type ParseResult = Result<(Vec<DepNode>, Vec<(String, String)>)>;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug, Clone)]
pub(crate) struct PolyglotArgs {
    /// Ecosystem: npm, python, go.
    #[arg(long, value_enum)]
    pub ecosystem: Ecosystem,

    /// Path to lock file or directory containing one.
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Centrality metric.
    #[arg(short, long, value_enum, default_value_t = Metric::Pagerank)]
    pub metric: Metric,

    /// Top-N rows.
    #[arg(short = 'n', long, default_value_t = 25)]
    pub top: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(Debug, Serialize)]
pub(crate) struct PolyglotRow {
    name: String,
    version: Option<String>,
    ecosystem: Ecosystem,
    in_degree: usize,
    out_degree: usize,
    pagerank: f64,
    consumers_pagerank: f64,
    betweenness: f64,
}

/// Load and rank a polyglot dependency graph. Returns the graph + ranked rows.
pub(crate) fn polyglot_analyze(
    ecosystem: Ecosystem,
    path: &Path,
    metric: Metric,
) -> Result<(dep_graph::DepGraph, dep_graph::DepNodeMap, Vec<PolyglotRow>)> {
    let (packages, edges) = match ecosystem {
        Ecosystem::Js => parse_npm(path)?,
        Ecosystem::Python => parse_python(path)?,
        Ecosystem::Go => parse_go_mod_graph(path)?,
        Ecosystem::Rust => {
            return Err(anyhow!(
                "Cargo workspaces use `cargo metadata` for richer analysis; run `pkgrank analyze` without --ecosystem"
            ));
        }
    };

    let (graph, map) = build_dep_graph(&packages, &edges);
    let pr = pagerank_auto(&graph);
    let rev = reverse_graph(&graph);
    let consumers_pr = pagerank_auto(&rev);
    let bc = graphops::betweenness_centrality(&graph);

    let mut rows: Vec<PolyglotRow> = graph
        .node_indices()
        .map(|n| {
            let node = graph.node_weight(n).expect("valid");
            PolyglotRow {
                name: node.name.clone(),
                version: node.version.clone(),
                ecosystem: node.ecosystem,
                in_degree: graph.neighbors_directed(n, Direction::Incoming).count(),
                out_degree: graph.neighbors_directed(n, Direction::Outgoing).count(),
                pagerank: pr[n.index()],
                consumers_pagerank: consumers_pr[n.index()],
                betweenness: bc[n.index()],
            }
        })
        .collect();

    rows.sort_by(|a, b| match metric {
        Metric::Pagerank => b.pagerank.total_cmp(&a.pagerank),
        Metric::ConsumersPagerank => b.consumers_pagerank.total_cmp(&a.consumers_pagerank),
        Metric::Betweenness => b.betweenness.total_cmp(&a.betweenness),
        Metric::Indegree => b.in_degree.cmp(&a.in_degree),
        Metric::Outdegree => b.out_degree.cmp(&a.out_degree),
    });

    Ok((graph, map, rows))
}

pub(crate) fn run_polyglot(args: &PolyglotArgs) -> Result<()> {
    let (graph, _map, rows) = polyglot_analyze(args.ecosystem, &args.path, args.metric)?;

    let fmt = effective_format(args.format);
    match fmt {
        OutputFormat::Json => {
            let rows_total = rows.len();
            let rows: Vec<_> = rows.into_iter().take(args.top).collect();
            #[derive(Serialize)]
            struct Out {
                schema_version: u32,
                ok: bool,
                command: &'static str,
                ecosystem: Ecosystem,
                rows_total: usize,
                rows_returned: usize,
                rows: Vec<PolyglotRow>,
            }
            let out = Out {
                schema_version: 1,
                ok: true,
                command: "polyglot",
                ecosystem: args.ecosystem,
                rows_total,
                rows_returned: rows.len(),
                rows,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Text => {
            println!(
                "pkgrank polyglot  ecosystem={}  metric={:?}\n",
                args.ecosystem, args.metric
            );
            println!(
                "{:>4}  {:>10}  {:>10}  {:>9}  {:>3}  {:>3}  name",
                "rank", "pr", "cons_pr", "between", "in", "out"
            );
            println!("{:\u{2500}<80}", "");
            for (i, r) in rows.iter().take(args.top).enumerate() {
                let ver = r.version.as_deref().unwrap_or("");
                println!(
                    "{:>4}. {:>10.6} {:>10.6} {:>9.6} {:>3} {:>3}  {} {}",
                    i + 1,
                    r.pagerank,
                    r.consumers_pagerank,
                    r.betweenness,
                    r.in_degree,
                    r.out_degree,
                    r.name,
                    ver
                );
            }
            println!(
                "\n{} packages, {} ecosystem",
                graph.node_count(),
                args.ecosystem
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// npm package-lock.json v3
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct NpmLockV3 {
    #[serde(default)]
    packages: HashMap<String, NpmPackageEntry>,
}

#[derive(Deserialize)]
struct NpmPackageEntry {
    version: Option<String>,
    #[serde(default)]
    dependencies: HashMap<String, String>,
}

pub(crate) fn parse_npm_lock(path: &Path) -> ParseResult {
    let path = if path.is_dir() {
        path.join("package-lock.json")
    } else {
        path.to_path_buf()
    };
    let raw =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let lock: NpmLockV3 =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;

    let mut packages = Vec::new();
    let mut edges = Vec::new();
    let mut seen_names: HashMap<String, bool> = HashMap::new();

    for (key, entry) in &lock.packages {
        // Keys are like "node_modules/express" or "node_modules/a/node_modules/b".
        // Extract the package name: last segment after "node_modules/".
        let name = if key.is_empty() {
            // Root project entry.
            continue;
        } else if let Some(pos) = key.rfind("node_modules/") {
            &key[pos + "node_modules/".len()..]
        } else {
            key.as_str()
        };

        if !seen_names.contains_key(name) {
            seen_names.insert(name.to_string(), true);
            packages.push(DepNode {
                name: name.to_string(),
                version: entry.version.clone(),
                ecosystem: Ecosystem::Js,
            });
        }

        for dep_name in entry.dependencies.keys() {
            edges.push((name.to_string(), dep_name.clone()));
        }
    }

    // Ensure all edge targets exist as nodes (some transitive deps may only appear as targets).
    for (_, to) in &edges {
        if !seen_names.contains_key(to) {
            seen_names.insert(to.clone(), true);
            packages.push(DepNode {
                name: to.clone(),
                version: None,
                ecosystem: Ecosystem::Js,
            });
        }
    }

    Ok((packages, edges))
}

/// Try lock file first, fall back to package.json (direct deps only).
pub(crate) fn parse_npm(path: &Path) -> ParseResult {
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(path).to_path_buf()
    };
    let lock = dir.join("package-lock.json");
    if lock.exists() {
        return parse_npm_lock(&lock);
    }
    let manifest = if path.is_file()
        && path
            .file_name()
            .map(|f| f == "package.json")
            .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        dir.join("package.json")
    };
    parse_npm_package_json(&manifest)
}

/// Parse package.json for direct dependencies only (no transitive graph).
fn parse_npm_package_json(path: &Path) -> ParseResult {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;

    eprintln!(
        "note: no package-lock.json found; using package.json (direct deps only, no transitive graph)"
    );

    let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("root");
    let version = v.get("version").and_then(|n| n.as_str());

    let mut packages = vec![DepNode {
        name: name.to_string(),
        version: version.map(|s| s.to_string()),
        ecosystem: Ecosystem::Js,
    }];
    let mut edges = Vec::new();

    for key in ["dependencies", "devDependencies"] {
        if let Some(deps) = v.get(key).and_then(|d| d.as_object()) {
            for (dep_name, ver_range) in deps {
                let ver = ver_range.as_str().map(|s| s.to_string());
                packages.push(DepNode {
                    name: dep_name.clone(),
                    version: ver,
                    ecosystem: Ecosystem::Js,
                });
                edges.push((name.to_string(), dep_name.clone()));
            }
        }
    }

    Ok((packages, edges))
}

// ---------------------------------------------------------------------------
// Python uv.lock
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct UvLock {
    #[serde(rename = "package", default)]
    packages: Vec<UvPackage>,
}

#[derive(Deserialize)]
struct UvPackage {
    name: String,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    dependencies: Vec<UvDep>,
}

#[derive(Deserialize)]
struct UvDep {
    name: String,
}

pub(crate) fn parse_uv_lock(path: &Path) -> ParseResult {
    let path = if path.is_dir() {
        path.join("uv.lock")
    } else {
        path.to_path_buf()
    };
    let raw =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let lock: UvLock =
        toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;

    let mut packages = Vec::new();
    let mut edges = Vec::new();
    let mut seen: HashMap<String, bool> = HashMap::new();

    for pkg in &lock.packages {
        if !seen.contains_key(&pkg.name) {
            seen.insert(pkg.name.clone(), true);
            packages.push(DepNode {
                name: pkg.name.clone(),
                version: pkg.version.clone(),
                ecosystem: Ecosystem::Python,
            });
        }
        for dep in &pkg.dependencies {
            edges.push((pkg.name.clone(), dep.name.clone()));
        }
    }

    for (_, to) in &edges {
        if !seen.contains_key(to) {
            seen.insert(to.clone(), true);
            packages.push(DepNode {
                name: to.clone(),
                version: None,
                ecosystem: Ecosystem::Python,
            });
        }
    }

    Ok((packages, edges))
}

/// Try uv.lock first, fall back to pyproject.toml (direct deps only).
pub(crate) fn parse_python(path: &Path) -> ParseResult {
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(path).to_path_buf()
    };
    let lock = dir.join("uv.lock");
    if lock.exists() {
        return parse_uv_lock(&lock);
    }
    let manifest = if path.is_file()
        && path
            .file_name()
            .map(|f| f == "pyproject.toml")
            .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        dir.join("pyproject.toml")
    };
    parse_pyproject_toml(&manifest)
}

/// Parse pyproject.toml for direct dependencies only.
fn parse_pyproject_toml(path: &Path) -> ParseResult {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: toml::Value =
        toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;

    eprintln!(
        "note: no uv.lock found; using pyproject.toml (direct deps only, no transitive graph)"
    );

    let project_name = v
        .get("project")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("root");
    let project_version = v
        .get("project")
        .and_then(|p| p.get("version"))
        .and_then(|n| n.as_str());

    let mut packages = vec![DepNode {
        name: project_name.to_string(),
        version: project_version.map(|s| s.to_string()),
        ecosystem: Ecosystem::Python,
    }];
    let mut edges = Vec::new();

    // [project.dependencies] is an array of PEP 508 strings like "requests>=2.0"
    if let Some(deps) = v
        .get("project")
        .and_then(|p| p.get("dependencies"))
        .and_then(|d| d.as_array())
    {
        for dep in deps {
            if let Some(spec) = dep.as_str() {
                // Extract package name (before any version specifier).
                let name = spec
                    .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                    .next()
                    .unwrap_or(spec)
                    .to_lowercase()
                    .replace('_', "-");
                if !name.is_empty() {
                    packages.push(DepNode {
                        name: name.clone(),
                        version: None,
                        ecosystem: Ecosystem::Python,
                    });
                    edges.push((project_name.to_string(), name));
                }
            }
        }
    }

    Ok((packages, edges))
}

// ---------------------------------------------------------------------------
// Go (go mod graph output)
// ---------------------------------------------------------------------------

pub(crate) fn parse_go_mod_graph(path: &Path) -> ParseResult {
    let output = if path.is_dir() || path.join("go.mod").exists() {
        // Shell out to `go mod graph`.
        let dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(path)
        };
        let out = ProcessCommand::new("go")
            .args(["mod", "graph"])
            .current_dir(dir)
            .output()
            .with_context(|| "failed to run `go mod graph`")?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(anyhow!(
                "go mod graph failed (exit {:?}): {}",
                out.status.code(),
                stderr.trim()
            ));
        }
        String::from_utf8_lossy(&out.stdout).to_string()
    } else {
        // Read from a file containing `go mod graph` output.
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
    };

    let mut seen: HashMap<String, Option<String>> = HashMap::new();
    let mut edges = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((from_raw, to_raw)) = line.split_once(' ') else {
            continue;
        };

        let (from_name, from_ver) = split_go_module(from_raw);
        let (to_name, to_ver) = split_go_module(to_raw);

        seen.entry(from_name.clone())
            .or_insert_with(|| from_ver.clone());
        seen.entry(to_name.clone())
            .or_insert_with(|| to_ver.clone());
        edges.push((from_name, to_name));
    }

    let packages: Vec<DepNode> = seen
        .into_iter()
        .map(|(name, version)| DepNode {
            name,
            version,
            ecosystem: Ecosystem::Go,
        })
        .collect();

    Ok((packages, edges))
}

fn split_go_module(s: &str) -> (String, Option<String>) {
    match s.split_once('@') {
        Some((name, ver)) => (name.to_string(), Some(ver.to_string())),
        None => (s.to_string(), None),
    }
}
