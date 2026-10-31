use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use super::*;
use crate::dep_graph::{build_dep_graph, dep_id, DepNode, Ecosystem};

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
    #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
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
        OutputFormat::Text | OutputFormat::Auto => {
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
    packages: BTreeMap<String, NpmPackageEntry>,
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

    // Keys are like "node_modules/express" or "node_modules/a/node_modules/b".
    // The package name is the part after the last "node_modules/".
    let npm_name = |key: &str| -> String {
        match key.rfind("node_modules/") {
            Some(pos) => key[pos + "node_modules/".len()..].to_string(),
            None => key.to_string(),
        }
    };

    let mut packages = Vec::new();
    let mut edges = Vec::new();

    for (key, entry) in &lock.packages {
        if key.is_empty() {
            // Root project entry.
            continue;
        }
        let node = DepNode {
            name: npm_name(key),
            version: entry.version.clone(),
            ecosystem: Ecosystem::Js,
        };
        let from = node.id();
        packages.push(node);

        for dep_name in entry.dependencies.keys() {
            let to = match resolve_npm_dep(&lock.packages, key, dep_name) {
                Some(dep_key) => dep_id(
                    &npm_name(dep_key),
                    lock.packages[dep_key].version.as_deref(),
                ),
                None => {
                    // Not installed (e.g. an unmet optional or peer dependency).
                    packages.push(DepNode {
                        name: dep_name.clone(),
                        version: None,
                        ecosystem: Ecosystem::Js,
                    });
                    dep_name.clone()
                }
            };
            edges.push((from.clone(), to));
        }
    }

    Ok((packages, edges))
}

/// Resolve `dep` the way Node does from the package installed at `from_key`:
/// the nearest `node_modules/<dep>` walking up from `from_key` to the root.
fn resolve_npm_dep<'a>(
    packages: &'a BTreeMap<String, NpmPackageEntry>,
    from_key: &str,
    dep: &str,
) -> Option<&'a str> {
    let mut base = from_key;
    loop {
        let candidate = if base.is_empty() {
            format!("node_modules/{dep}")
        } else {
            format!("{base}/node_modules/{dep}")
        };
        if let Some((key, _)) = packages.get_key_value(&candidate) {
            return Some(key.as_str());
        }
        if base.is_empty() {
            return None;
        }
        base = base
            .rsplit_once('/')
            .map(|(parent, _)| parent)
            .unwrap_or("");
    }
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

    let root = DepNode {
        name: name.to_string(),
        version: version.map(|s| s.to_string()),
        ecosystem: Ecosystem::Js,
    };
    let root_id = root.id();
    let mut packages = vec![root];
    let mut edges = Vec::new();

    for key in ["dependencies", "devDependencies"] {
        if let Some(deps) = v.get(key).and_then(|d| d.as_object()) {
            for (dep_name, ver_range) in deps {
                let ver = ver_range.as_str().map(|s| s.to_string());
                let dep = DepNode {
                    name: dep_name.clone(),
                    version: ver,
                    ecosystem: Ecosystem::Js,
                };
                edges.push((root_id.clone(), dep.id()));
                packages.push(dep);
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
    /// Present when the lock holds more than one version of `name`.
    #[serde(default)]
    version: Option<String>,
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

    // Versions locked per name, to resolve dependencies that omit a version
    // (uv only writes one when the name is ambiguous).
    let mut versions: HashMap<&str, Vec<Option<&str>>> = HashMap::new();
    for pkg in &lock.packages {
        versions
            .entry(pkg.name.as_str())
            .or_default()
            .push(pkg.version.as_deref());
    }

    let mut packages = Vec::new();
    let mut edges = Vec::new();

    for pkg in &lock.packages {
        let node = DepNode {
            name: pkg.name.clone(),
            version: pkg.version.clone(),
            ecosystem: Ecosystem::Python,
        };
        let from = node.id();
        packages.push(node);
        for dep in &pkg.dependencies {
            let version = match (&dep.version, versions.get(dep.name.as_str())) {
                (Some(v), _) => Some(v.as_str()),
                (None, Some(vs)) if vs.len() == 1 => vs[0],
                _ => None,
            };
            let to = dep_id(&dep.name, version);
            if !versions.contains_key(dep.name.as_str()) || version.is_none() {
                packages.push(DepNode {
                    name: dep.name.clone(),
                    version: version.map(str::to_string),
                    ecosystem: Ecosystem::Python,
                });
            }
            edges.push((from.clone(), to));
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

    let root = DepNode {
        name: project_name.to_string(),
        version: project_version.map(|s| s.to_string()),
        ecosystem: Ecosystem::Python,
    };
    let root_id = root.id();
    let mut packages = vec![root];
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
                    edges.push((root_id.clone(), name));
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

    let mut seen: BTreeMap<String, DepNode> = BTreeMap::new();
    let mut edges = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((from_raw, to_raw)) = line.split_once(' ') else {
            continue;
        };

        // `module@version` is the full identity; distinct versions stay distinct.
        for raw in [from_raw, to_raw] {
            seen.entry(raw.to_string()).or_insert_with(|| {
                let (name, version) = split_go_module(raw);
                DepNode {
                    name,
                    version,
                    ecosystem: Ecosystem::Go,
                }
            });
        }
        edges.push((from_raw.to_string(), to_raw.to_string()));
    }

    let packages: Vec<DepNode> = seen.into_values().collect();

    Ok((packages, edges))
}

fn split_go_module(s: &str) -> (String, Option<String>) {
    match s.split_once('@') {
        Some((name, ver)) => (name.to_string(), Some(ver.to_string())),
        None => (s.to_string(), None),
    }
}

#[cfg(test)]
mod lockfile_identity_tests {
    //! Distinct versions of one package are distinct nodes, and each edge
    //! points at the version the dependent actually resolves to.

    use super::*;
    use std::collections::BTreeSet;

    type Ver = Option<String>;
    type Nodes = BTreeSet<(String, Ver)>;
    type Edges = BTreeSet<(String, Ver, String, Ver)>;

    fn graph_shape((packages, edges): (Vec<DepNode>, Vec<(String, String)>)) -> (Nodes, Edges) {
        let (graph, _) = build_dep_graph(&packages, &edges);
        let nodes = graph
            .node_weights()
            .map(|n| (n.name.clone(), n.version.clone()))
            .collect();
        let edges = graph
            .edge_indices()
            .map(|e| {
                let (a, b) = graph.edge_endpoints(e).unwrap();
                let (a, b) = (&graph[a], &graph[b]);
                (
                    a.name.clone(),
                    a.version.clone(),
                    b.name.clone(),
                    b.version.clone(),
                )
            })
            .collect();
        (nodes, edges)
    }

    fn v(s: &str) -> Ver {
        Some(s.to_string())
    }

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn npm_lock_keeps_nested_versions_apart() {
        let dir = tempfile::tempdir().unwrap();
        let lock = write(
            dir.path(),
            "package-lock.json",
            r#"{"lockfileVersion": 3, "packages": {
                "": {"name": "root", "dependencies": {"a": "^1", "b": "^2"}},
                "node_modules/a": {"version": "1.0.0", "dependencies": {"b": "^1"}},
                "node_modules/a/node_modules/b": {"version": "1.0.0"},
                "node_modules/b": {"version": "2.0.0"},
                "node_modules/c": {"version": "1.0.0", "dependencies": {"b": "^2"}}
            }}"#,
        );
        let (nodes, edges) = graph_shape(parse_npm_lock(&lock).unwrap());
        assert_eq!(
            nodes,
            BTreeSet::from([
                ("a".into(), v("1.0.0")),
                ("b".into(), v("1.0.0")),
                ("b".into(), v("2.0.0")),
                ("c".into(), v("1.0.0")),
            ])
        );
        assert_eq!(
            edges,
            BTreeSet::from([
                ("a".into(), v("1.0.0"), "b".into(), v("1.0.0")),
                ("c".into(), v("1.0.0"), "b".into(), v("2.0.0")),
            ])
        );
    }

    #[test]
    fn uv_lock_keeps_forked_versions_apart() {
        let dir = tempfile::tempdir().unwrap();
        let lock = write(
            dir.path(),
            "uv.lock",
            r#"version = 1

[[package]]
name = "app"
version = "0.1.0"
source = { editable = "." }
dependencies = [
    { name = "numpy", version = "1.26.4", source = { registry = "https://pypi.org/simple" }, marker = "python_full_version < '3.10'" },
    { name = "numpy", version = "2.1.0", source = { registry = "https://pypi.org/simple" }, marker = "python_full_version >= '3.10'" },
    { name = "pandas" },
]

[[package]]
name = "numpy"
version = "1.26.4"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "numpy"
version = "2.1.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pandas"
version = "2.2.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "numpy", version = "2.1.0", source = { registry = "https://pypi.org/simple" } },
    { name = "tzdata" },
]

[[package]]
name = "tzdata"
version = "2024.1"
source = { registry = "https://pypi.org/simple" }
"#,
        );
        let (nodes, edges) = graph_shape(parse_uv_lock(&lock).unwrap());
        assert_eq!(
            nodes,
            BTreeSet::from([
                ("app".into(), v("0.1.0")),
                ("numpy".into(), v("1.26.4")),
                ("numpy".into(), v("2.1.0")),
                ("pandas".into(), v("2.2.0")),
                ("tzdata".into(), v("2024.1")),
            ])
        );
        assert_eq!(
            edges,
            BTreeSet::from([
                ("app".into(), v("0.1.0"), "numpy".into(), v("1.26.4")),
                ("app".into(), v("0.1.0"), "numpy".into(), v("2.1.0")),
                ("app".into(), v("0.1.0"), "pandas".into(), v("2.2.0")),
                ("pandas".into(), v("2.2.0"), "numpy".into(), v("2.1.0")),
                ("pandas".into(), v("2.2.0"), "tzdata".into(), v("2024.1")),
            ])
        );
    }

    #[test]
    fn manifest_fallbacks_link_root_to_direct_deps() {
        let dir = tempfile::tempdir().unwrap();
        let pkg = write(
            dir.path(),
            "package.json",
            r#"{"name": "web", "version": "1.0.0", "dependencies": {"left-pad": "^1.3.0"}}"#,
        );
        let (_, edges) = graph_shape(parse_npm(&pkg).unwrap());
        assert_eq!(
            edges,
            BTreeSet::from([("web".into(), v("1.0.0"), "left-pad".into(), v("^1.3.0"))])
        );

        let py = write(
            dir.path(),
            "pyproject.toml",
            "[project]\nname = \"svc\"\nversion = \"0.2.0\"\ndependencies = [\"requests>=2\"]\n",
        );
        let (_, edges) = graph_shape(parse_python(&py).unwrap());
        assert_eq!(
            edges,
            BTreeSet::from([("svc".into(), v("0.2.0"), "requests".into(), None)])
        );
    }

    #[test]
    fn go_mod_graph_keeps_module_versions_apart() {
        let dir = tempfile::tempdir().unwrap();
        let graph = write(
            dir.path(),
            "graph.txt",
            "example.com/app golang.org/x/text@v0.3.0\n\
             example.com/app example.com/lib@v1.0.0\n\
             example.com/lib@v1.0.0 golang.org/x/text@v0.1.0\n",
        );
        let (nodes, edges) = graph_shape(parse_go_mod_graph(&graph).unwrap());
        assert_eq!(
            nodes,
            BTreeSet::from([
                ("example.com/app".into(), None),
                ("example.com/lib".into(), v("v1.0.0")),
                ("golang.org/x/text".into(), v("v0.1.0")),
                ("golang.org/x/text".into(), v("v0.3.0")),
            ])
        );
        assert_eq!(
            edges,
            BTreeSet::from([
                (
                    "example.com/app".into(),
                    None,
                    "example.com/lib".into(),
                    v("v1.0.0")
                ),
                (
                    "example.com/app".into(),
                    None,
                    "golang.org/x/text".into(),
                    v("v0.3.0")
                ),
                (
                    "example.com/lib".into(),
                    v("v1.0.0"),
                    "golang.org/x/text".into(),
                    v("v0.1.0")
                ),
            ])
        );
    }
}
