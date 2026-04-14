use anyhow::Result;
use petgraph::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use super::*;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug, Clone)]
pub(crate) struct FilesArgs {
    /// Project root directory or GitHub URL (e.g. https://github.com/owner/repo).
    #[arg(default_value = ".")]
    pub path: String,

    /// Ecosystem (auto-detected if omitted).
    #[arg(long, value_enum)]
    pub ecosystem: Option<Ecosystem>,

    /// Include test files in the graph.
    #[arg(long, default_value_t = false)]
    pub include_tests: bool,

    /// Include benchmark files.
    #[arg(long, default_value_t = false)]
    pub include_benches: bool,

    /// Include example files.
    #[arg(long, default_value_t = false)]
    pub include_examples: bool,

    /// Include build scripts (build.rs, setup.py, etc.).
    #[arg(long, default_value_t = false)]
    pub include_build: bool,

    /// Centrality metric for sorting.
    #[arg(short, long, value_enum, default_value_t = Metric::Pagerank)]
    pub metric: Metric,

    /// Top-N rows.
    #[arg(short = 'n', long, default_value_t = 25)]
    pub top: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,

    /// Aggregate results by directory instead of individual files.
    ///
    /// Useful for large codebases where file-level is too noisy.
    #[arg(long, default_value_t = false)]
    pub directory: bool,

    /// Focus on a specific file: show its imports, dependents, and co-changers.
    ///
    /// Accepts a partial path (e.g. "graph.rs" matches "src/hnsw/graph.rs").
    #[arg(long)]
    pub focus: Option<String>,

    /// Overlay git change history (change frequency + co-change coupling).
    ///
    /// When enabled, combines structural centrality with temporal signals
    /// to produce a hotspot score. Requires a git repository.
    #[arg(long, default_value_t = false)]
    pub git: bool,

    /// Git history window in days (default: 90).
    #[arg(long, default_value_t = 90)]
    pub git_days: u64,
}

// ---------------------------------------------------------------------------
// File classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FileRole {
    /// Library root (lib.rs, __init__.py, index.ts).
    LibRoot,
    /// Binary / CLI entry point (main.rs, __main__.py, main.go).
    BinEntry,
    /// Production source code.
    Source,
    /// Test file.
    Test,
    /// Benchmark file.
    Bench,
    /// Example file.
    Example,
    /// Build script (build.rs, setup.py).
    Build,
}

fn classify_rust_file(path: &Path, project_root: &Path) -> FileRole {
    let rel = path.strip_prefix(project_root).unwrap_or(path);
    let rel_str = rel.to_string_lossy();

    // build.rs at project root
    if rel_str == "build.rs" {
        return FileRole::Build;
    }

    // tests/ directory → integration tests
    if rel_str.starts_with("tests/") || rel_str.starts_with("tests\\") {
        return FileRole::Test;
    }

    // benches/ directory
    if rel_str.starts_with("benches/") || rel_str.starts_with("benches\\") {
        return FileRole::Bench;
    }

    // examples/ directory
    if rel_str.starts_with("examples/") || rel_str.starts_with("examples\\") {
        return FileRole::Example;
    }

    // src/main.rs or src/bin/*.rs
    if rel_str == "src/main.rs" || rel_str.starts_with("src/bin/") {
        return FileRole::BinEntry;
    }

    // src/lib.rs
    if rel_str == "src/lib.rs" {
        return FileRole::LibRoot;
    }

    FileRole::Source
}

fn classify_python_file(path: &Path, project_root: &Path) -> FileRole {
    let rel = path.strip_prefix(project_root).unwrap_or(path);
    let rel_str = rel.to_string_lossy();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    if rel_str.starts_with("tests/")
        || rel_str.starts_with("test/")
        || file_name.starts_with("test_")
        || file_name.ends_with("_test.py")
        || file_name == "conftest.py"
    {
        return FileRole::Test;
    }

    if rel_str.starts_with("benchmarks/") || rel_str.starts_with("bench/") {
        return FileRole::Bench;
    }

    if rel_str.starts_with("examples/") {
        return FileRole::Example;
    }

    if file_name == "setup.py" || file_name == "setup.cfg" {
        return FileRole::Build;
    }

    if file_name == "__main__.py" {
        return FileRole::BinEntry;
    }

    if file_name == "__init__.py" {
        return FileRole::LibRoot;
    }

    FileRole::Source
}

fn classify_js_file(path: &Path, project_root: &Path) -> FileRole {
    let rel = path.strip_prefix(project_root).unwrap_or(path);
    let rel_str = rel.to_string_lossy();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    if file_name.contains(".test.")
        || file_name.contains(".spec.")
        || rel_str.starts_with("__tests__/")
        || rel_str.contains("/__tests__/")
        || rel_str.starts_with("tests/")
        || rel_str.starts_with("test/")
    {
        return FileRole::Test;
    }

    if rel_str.starts_with("examples/") || rel_str.starts_with("example/") {
        return FileRole::Example;
    }

    // Config files
    if file_name.contains("config.") || file_name.starts_with("webpack.") {
        return FileRole::Build;
    }

    if file_name == "index.ts"
        || file_name == "index.js"
        || file_name == "index.tsx"
        || file_name == "index.jsx"
    {
        return FileRole::LibRoot;
    }

    FileRole::Source
}

fn classify_go_file(path: &Path, project_root: &Path) -> FileRole {
    let rel = path.strip_prefix(project_root).unwrap_or(path);
    let rel_str = rel.to_string_lossy();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    // Check bench before test -- _bench_test.go also ends with _test.go.
    if file_name.ends_with("_bench_test.go") {
        return FileRole::Bench;
    }

    if file_name.ends_with("_test.go") {
        return FileRole::Test;
    }

    if rel_str.starts_with("cmd/") && file_name == "main.go" {
        return FileRole::BinEntry;
    }

    if file_name == "main.go" {
        return FileRole::BinEntry;
    }

    if rel_str.starts_with("examples/") || rel_str.starts_with("example/") {
        return FileRole::Example;
    }

    FileRole::Source
}

fn classify_file(path: &Path, project_root: &Path, ecosystem: Ecosystem) -> FileRole {
    match ecosystem {
        Ecosystem::Cargo => classify_rust_file(path, project_root),
        Ecosystem::Python => classify_python_file(path, project_root),
        Ecosystem::Npm => classify_js_file(path, project_root),
        Ecosystem::Go => classify_go_file(path, project_root),
    }
}

fn should_include(role: FileRole, args: &FilesArgs) -> bool {
    match role {
        FileRole::LibRoot | FileRole::BinEntry | FileRole::Source => true,
        FileRole::Test => args.include_tests,
        FileRole::Bench => args.include_benches,
        FileRole::Example => args.include_examples,
        FileRole::Build => args.include_build,
    }
}

// ---------------------------------------------------------------------------
// Ecosystem auto-detection
// ---------------------------------------------------------------------------

pub(crate) fn detect_ecosystem(dir: &Path) -> Option<Ecosystem> {
    // Priority: Cargo > Go > Python > npm.
    // Cargo first because Rust projects often have package.json for tooling.
    // Go before Python because go.mod is unambiguous.
    if dir.join("Cargo.toml").exists() {
        return Some(Ecosystem::Cargo);
    }
    if dir.join("go.mod").exists() {
        return Some(Ecosystem::Go);
    }
    if dir.join("pyproject.toml").exists()
        || dir.join("uv.lock").exists()
        || dir.join("setup.py").exists()
    {
        return Some(Ecosystem::Python);
    }
    if dir.join("package.json").exists() || dir.join("package-lock.json").exists() {
        return Some(Ecosystem::Npm);
    }
    None
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

fn discover_files(root: &Path, ecosystem: Ecosystem) -> Vec<PathBuf> {
    let extensions: &[&str] = match ecosystem {
        Ecosystem::Cargo => &["rs"],
        Ecosystem::Python => &["py"],
        Ecosystem::Npm => &["ts", "tsx", "js", "jsx", "mjs"],
        Ecosystem::Go => &["go"],
    };

    let mut files = Vec::new();
    walk_dir(root, extensions, &mut files);
    files.sort();
    files
}

fn walk_dir(dir: &Path, extensions: &[&str], out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip common non-source directories.
            if matches!(
                name,
                "target"
                    | "node_modules"
                    | ".git"
                    | "__pycache__"
                    | ".mypy_cache"
                    | ".pytest_cache"
                    | "dist"
                    | "build"
                    | ".next"
                    | ".vercel"
                    | ".nuxt"
                    | ".svelte-kit"
                    | ".angular"
                    | "coverage"
                    | "vendor"
                    | ".venv"
                    | "venv"
                    | ".tox"
                    | "archive"
            ) {
                continue;
            }
            walk_dir(&path, extensions, out);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if extensions.contains(&ext) {
                out.push(path);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Import edge: source file → target file
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FileEdge {
    from: PathBuf,
    to: PathBuf,
}

/// Extract external dependency names from source files.
fn extract_external_deps(
    files: &[PathBuf],
    ecosystem: Ecosystem,
    internal_prefixes: &HashSet<String>,
) -> HashMap<PathBuf, Vec<String>> {
    let mut result: HashMap<PathBuf, Vec<String>> = HashMap::new();
    let std_prefixes: HashSet<&str> = match ecosystem {
        Ecosystem::Cargo => ["std", "core", "alloc", "proc_macro", "test"]
            .iter()
            .copied()
            .collect(),
        Ecosystem::Python => {
            // Skip stdlib -- too many to list, focus on third-party.
            HashSet::new()
        }
        Ecosystem::Npm | Ecosystem::Go => HashSet::new(),
    };

    for file in files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let mut deps: HashSet<String> = HashSet::new();

        for line in content.lines() {
            let line = line.trim();
            match ecosystem {
                Ecosystem::Cargo => {
                    // `use foo::...` where foo is not crate/super/self/known
                    let use_part = strip_visibility(line)
                        .strip_prefix("use ")
                        .or_else(|| line.strip_prefix("use "));
                    if let Some(use_part) = use_part {
                        let use_part = use_part.trim_end_matches(';').trim();
                        if use_part.starts_with("crate::")
                            || use_part.starts_with("super::")
                            || use_part.starts_with("self::")
                        {
                            continue;
                        }
                        let first_seg = use_part.split("::").next().unwrap_or("");
                        if !first_seg.is_empty()
                            && !internal_prefixes.contains(first_seg)
                            && !std_prefixes.contains(first_seg)
                        {
                            deps.insert(first_seg.to_string());
                        }
                    }
                }
                Ecosystem::Python => {
                    if let Some(rest) = line.strip_prefix("import ") {
                        let mod_name = rest.split(' ').next().unwrap_or(rest);
                        let top = mod_name.split('.').next().unwrap_or(mod_name);
                        if !top.is_empty()
                            && !top.starts_with('.')
                            && !internal_prefixes.contains(top)
                        {
                            deps.insert(top.to_string());
                        }
                    } else if let Some(rest) = line.strip_prefix("from ") {
                        if let Some((mod_part, _)) = rest.split_once(" import ") {
                            let mod_part = mod_part.trim();
                            if !mod_part.starts_with('.') {
                                let top = mod_part.split('.').next().unwrap_or(mod_part);
                                if !internal_prefixes.contains(top) {
                                    deps.insert(top.to_string());
                                }
                            }
                        }
                    }
                }
                Ecosystem::Npm => {
                    for spec in extract_js_import_specifiers(line) {
                        if !spec.starts_with('.') && !spec.starts_with('@') {
                            let pkg = spec.split('/').next().unwrap_or(&spec);
                            deps.insert(pkg.to_string());
                        } else if spec.starts_with("@") {
                            // Scoped package: @scope/pkg
                            let parts: Vec<&str> = spec.splitn(3, '/').collect();
                            if parts.len() >= 2 {
                                deps.insert(format!("{}/{}", parts[0], parts[1]));
                            }
                        }
                    }
                }
                Ecosystem::Go => {
                    if let Some(import_path) = extract_go_import(line) {
                        if !internal_prefixes
                            .iter()
                            .any(|p| import_path.starts_with(p.as_str()))
                        {
                            // Use the first two segments as the package identifier.
                            let parts: Vec<&str> = import_path.splitn(4, '/').collect();
                            if parts.len() >= 3 {
                                deps.insert(format!("{}/{}/{}", parts[0], parts[1], parts[2]));
                            }
                        }
                    }
                }
            }
        }

        if !deps.is_empty() {
            let mut sorted: Vec<String> = deps.into_iter().collect();
            sorted.sort();
            result.insert(file.clone(), sorted);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Rust import parser
// ---------------------------------------------------------------------------

fn parse_rust_imports(root: &Path, files: &[PathBuf]) -> Vec<FileEdge> {
    // Find all crate roots in the project (handles workspaces with nested crates).
    let crate_roots = find_rust_crate_roots(root);

    // Global maps across all crates.
    let mut mod_to_file: HashMap<String, PathBuf> = HashMap::new();
    let mut file_to_mod: HashMap<PathBuf, String> = HashMap::new();
    let mut file_to_crate: HashMap<PathBuf, String> = HashMap::new();
    let mut known_crates: HashSet<String> = HashSet::new();

    for (crate_dir, crate_name) in &crate_roots {
        known_crates.insert(crate_name.clone());
        let src_dir = crate_dir.join("src");
        for file in files {
            // Only map files that are actually under this crate's src/.
            if !file.starts_with(&src_dir) {
                continue;
            }
            if let Some(mod_path) = rust_file_to_mod_path(file, &src_dir, crate_name) {
                mod_to_file.insert(mod_path.clone(), file.clone());
                file_to_mod.insert(file.clone(), mod_path);
                file_to_crate.insert(file.clone(), crate_name.clone());
            }
        }
    }

    let mut edges = Vec::new();

    for file in files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let this_mod = file_to_mod.get(file).cloned().unwrap_or_default();
        let crate_name = file_to_crate
            .get(file)
            .cloned()
            .unwrap_or_else(|| "crate".to_string());

        let logical_lines = join_rust_logical_lines(&content);

        for line in &logical_lines {
            let line = line.trim();

            if let Some(mod_name) = parse_mod_declaration(line) {
                let child_mod = format!("{}::{}", this_mod, mod_name);
                if let Some(target) = mod_to_file.get(&child_mod) {
                    if target != file {
                        edges.push(FileEdge {
                            from: file.clone(),
                            to: target.clone(),
                        });
                    }
                }
            }

            if let Some(targets) =
                parse_use_statement(line, &this_mod, &crate_name, &known_crates, &mod_to_file)
            {
                for target in targets {
                    if target != *file {
                        edges.push(FileEdge {
                            from: file.clone(),
                            to: target,
                        });
                    }
                }
            }
        }
    }

    edges
}

/// Join multi-line `use` and `mod` statements into single logical lines.
/// Handles patterns like:
/// ```
/// use crate::{
///     foo,
///     bar,
/// };
/// ```
fn join_rust_logical_lines(content: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut accum = String::new();
    let mut in_use = false;
    let mut brace_depth: i32 = 0;

    for line in content.lines() {
        let trimmed = line.trim();

        if in_use {
            accum.push(' ');
            accum.push_str(trimmed);
            brace_depth += trimmed.chars().filter(|c| *c == '{').count() as i32;
            brace_depth -= trimmed.chars().filter(|c| *c == '}').count() as i32;
            if brace_depth <= 0 && trimmed.ends_with(';') {
                result.push(std::mem::take(&mut accum));
                in_use = false;
                brace_depth = 0;
            }
            continue;
        }

        // Detect start of multi-line use/pub use.
        let stripped = strip_visibility(trimmed);
        let is_use_start = stripped.starts_with("use ") || trimmed.starts_with("use ");
        if is_use_start && !trimmed.ends_with(';') {
            in_use = true;
            brace_depth = trimmed.chars().filter(|c| *c == '{').count() as i32;
            brace_depth -= trimmed.chars().filter(|c| *c == '}').count() as i32;
            accum = trimmed.to_string();
            continue;
        }

        result.push(trimmed.to_string());
    }

    // Flush any unterminated accumulator.
    if !accum.is_empty() {
        result.push(accum);
    }

    result
}

/// Join multi-line statements delimited by `(` `)` (Python, JS).
fn join_paren_lines(content: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut accum = String::new();
    let mut depth: i32 = 0;

    for line in content.lines() {
        let trimmed = line.trim();
        if depth > 0 {
            accum.push(' ');
            accum.push_str(trimmed);
            depth += trimmed.chars().filter(|c| *c == '(').count() as i32;
            depth -= trimmed.chars().filter(|c| *c == ')').count() as i32;
            if depth <= 0 {
                result.push(std::mem::take(&mut accum));
                depth = 0;
            }
            continue;
        }

        let opens = trimmed.chars().filter(|c| *c == '(').count() as i32;
        let closes = trimmed.chars().filter(|c| *c == ')').count() as i32;
        if opens > closes {
            depth = opens - closes;
            accum = trimmed.to_string();
            continue;
        }

        result.push(trimmed.to_string());
    }

    if !accum.is_empty() {
        result.push(accum);
    }

    result
}

/// Find all crate roots in a project: (crate_dir, crate_name).
/// Handles single crates, workspaces, and nested crates/ directories.
fn find_rust_crate_roots(root: &Path) -> Vec<(PathBuf, String)> {
    let mut roots = Vec::new();

    // Check if root itself is a crate.
    let root_cargo = root.join("Cargo.toml");
    if root_cargo.exists() {
        let src = root.join("src");
        if src.is_dir() {
            roots.push((root.to_path_buf(), read_rust_crate_name(root)));
        }
    }

    // Walk for nested Cargo.toml files (workspace members).
    walk_for_cargo_tomls(root, &mut roots, 0);

    // Dedup by crate dir.
    roots.sort_by(|a, b| a.0.cmp(&b.0));
    roots.dedup_by(|a, b| a.0 == b.0);

    roots
}

fn walk_for_cargo_tomls(dir: &Path, roots: &mut Vec<(PathBuf, String)>, depth: usize) {
    if depth > 5 {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, "target" | ".git" | "node_modules" | "vendor") {
                continue;
            }
            let cargo_toml = path.join("Cargo.toml");
            let src_dir = path.join("src");
            if cargo_toml.exists() && src_dir.is_dir() {
                roots.push((path.clone(), read_rust_crate_name(&path)));
            }
            walk_for_cargo_tomls(&path, roots, depth + 1);
        }
    }
}

fn read_rust_crate_name(root: &Path) -> String {
    let cargo_toml = root.join("Cargo.toml");
    if let Ok(raw) = std::fs::read_to_string(&cargo_toml) {
        if let Ok(val) = raw.parse::<toml::Value>() {
            // Check [lib] name first, then [package] name.
            if let Some(name) = val
                .get("lib")
                .and_then(|l| l.get("name"))
                .and_then(|n| n.as_str())
            {
                return name.replace('-', "_");
            }
            if let Some(name) = val
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())
            {
                return name.replace('-', "_");
            }
        }
    }
    // Fallback: directory name.
    root.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("crate")
        .replace('-', "_")
}

fn rust_file_to_mod_path(file: &Path, src_dir: &Path, crate_name: &str) -> Option<String> {
    let rel = file.strip_prefix(src_dir).ok()?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");

    // lib.rs or main.rs → "crate" (the crate root module)
    if rel_str == "lib.rs" || rel_str == "main.rs" {
        return Some(crate_name.to_string());
    }

    // foo.rs → "crate::foo"
    // foo/mod.rs → "crate::foo"
    // foo/bar.rs → "crate::foo::bar"
    // foo/bar/mod.rs → "crate::foo::bar"
    let rel_str = if rel_str.ends_with("/mod.rs") {
        rel_str.trim_end_matches("/mod.rs").to_string()
    } else {
        rel_str.trim_end_matches(".rs").to_string()
    };

    // bin/foo.rs → skip (handled separately)
    if rel_str.starts_with("bin/") {
        return Some(format!("bin::{}", rel_str.trim_start_matches("bin/")));
    }

    let mod_path = rel_str.replace('/', "::");
    Some(format!("{}::{}", crate_name, mod_path))
}

fn parse_mod_declaration(line: &str) -> Option<String> {
    // Match: `mod foo;` or `pub mod foo;` or `pub(crate) mod foo;`
    // Skip: `mod tests {`, `mod foo {` (inline modules, not file declarations)
    let line = line.trim();
    if !line.ends_with(';') {
        return None;
    }

    // Strip attributes like #[cfg(feature = "...")] -- these are on preceding lines,
    // but `mod foo;` on its own line is what we care about.
    let stripped = strip_visibility(line);
    let stripped = stripped.trim();

    if let Some(rest) = stripped.strip_prefix("mod ") {
        let name = rest.trim_end_matches(';').trim();
        // Validate it's a simple identifier.
        if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
            return Some(name.to_string());
        }
    }
    None
}

fn strip_visibility(line: &str) -> &str {
    if let Some(rest) = line.strip_prefix("pub(crate) ") {
        return rest;
    }
    if let Some(rest) = line.strip_prefix("pub(super) ") {
        return rest;
    }
    if let Some(rest) = line.strip_prefix("pub(self) ") {
        return rest;
    }
    // pub(in path) -- complex, just strip pub(...) prefix
    if line.starts_with("pub(") {
        if let Some(close) = line.find(") ") {
            return &line[close + 2..];
        }
    }
    if let Some(rest) = line.strip_prefix("pub ") {
        return rest;
    }
    line
}

fn parse_use_statement(
    line: &str,
    current_mod: &str,
    crate_name: &str,
    known_crates: &HashSet<String>,
    mod_to_file: &HashMap<String, PathBuf>,
) -> Option<Vec<PathBuf>> {
    let line = line.trim();
    let use_part = if let Some(rest) = strip_visibility(line).strip_prefix("use ") {
        rest
    } else if let Some(rest) = line.strip_prefix("use ") {
        rest
    } else {
        return None;
    };

    let use_part = use_part.trim_end_matches(';').trim();

    let (base_mod, _rest) = if use_part.starts_with("crate::") {
        let resolved = format!("{}{}", crate_name, &use_part["crate".len()..]);
        (resolved, "")
    } else if use_part.starts_with("super::") {
        // Handle chained super:: (e.g. super::super::foo).
        let mut base = current_mod.to_string();
        let mut rest = use_part;
        while let Some(after) = rest.strip_prefix("super::") {
            base = base
                .rsplit_once("::")
                .map(|(p, _)| p.to_string())
                .unwrap_or_default();
            rest = after;
        }
        let resolved = if rest.is_empty() || base.is_empty() {
            if base.is_empty() {
                rest.to_string()
            } else {
                base
            }
        } else {
            format!("{}::{}", base, rest)
        };
        (resolved, "")
    } else if use_part.starts_with("self::") {
        let relative = use_part.strip_prefix("self::").unwrap_or(use_part);
        (format!("{}::{}", current_mod, relative), "")
    } else {
        let first_seg = use_part.split("::").next().unwrap_or("");
        if known_crates.contains(first_seg) {
            // Cross-crate workspace import: `use other_crate::foo`.
            (use_part.to_string(), "")
        } else {
            // Try as bare sibling module: `use graph::Foo` → `current_mod::graph::Foo`.
            let sibling_mod = format!("{}::{}", current_mod, first_seg);
            let sibling_exists = mod_to_file.contains_key(&sibling_mod)
                || mod_to_file
                    .keys()
                    .any(|k| k.starts_with(&format!("{}::", sibling_mod)));
            if sibling_exists {
                (format!("{}::{}", current_mod, use_part), "")
            } else {
                return None;
            }
        }
    };

    // The base_mod might be "crate::foo::bar::Baz" or "crate::foo::{A, B}".
    // We need to find the longest prefix that matches a known module.
    let mut targets = Vec::new();
    resolve_use_path(&base_mod, mod_to_file, &mut targets);

    if targets.is_empty() {
        None
    } else {
        Some(targets)
    }
}

fn resolve_use_path(path: &str, mod_to_file: &HashMap<String, PathBuf>, out: &mut Vec<PathBuf>) {
    // Handle grouped imports: `crate::foo::{bar, baz}`
    if let Some(brace_start) = path.find('{') {
        let prefix = &path[..brace_start];
        let rest = &path[brace_start + 1..];
        let rest = rest.trim_end_matches('}');
        for item in rest.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let full = format!("{}{}", prefix, item);
            resolve_use_path(&full, mod_to_file, out);
        }
        return;
    }

    // Try progressively shorter prefixes to find the owning module's file.
    let mut path_str = path.to_string();
    // Replace "crate::" with the actual crate name if present.
    // The mod_to_file keys use the crate name, not "crate".
    // Actually, we need to check both forms. The caller already resolved super/self,
    // but "crate::" → we stored as "crate_name::".
    // Let's check if any key starts with the first segment.

    loop {
        if let Some(file) = mod_to_file.get(&path_str) {
            out.push(file.clone());
            return;
        }
        // Strip last segment and try again.
        match path_str.rsplit_once("::") {
            Some((parent, _)) => path_str = parent.to_string(),
            None => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Python import parser
// ---------------------------------------------------------------------------

fn parse_python_imports(root: &Path, files: &[PathBuf]) -> Vec<FileEdge> {
    // Detect the package directory (src/pkg/ or pkg/ or root).
    let (pkg_name, pkg_dir) = detect_python_package(root);

    // Map: module path (dot-separated) → file path.
    let mut mod_to_file: HashMap<String, PathBuf> = HashMap::new();
    let mut file_to_mod: HashMap<PathBuf, String> = HashMap::new();

    for file in files {
        if let Some(mod_path) = python_file_to_mod_path(file, &pkg_dir, &pkg_name) {
            mod_to_file.insert(mod_path.clone(), file.clone());
            file_to_mod.insert(file.clone(), mod_path);
        }
    }

    let mut edges = Vec::new();

    for file in files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let this_mod = file_to_mod.get(file).cloned().unwrap_or_default();

        // Join multi-line imports: `from pkg import (\n  A,\n  B\n)`
        let logical_lines = join_paren_lines(&content);

        for line in &logical_lines {
            let line = line.trim();

            if let Some(targets) =
                parse_python_from_import(line, &this_mod, &pkg_name, &mod_to_file)
            {
                for target in targets {
                    if target != *file {
                        edges.push(FileEdge {
                            from: file.clone(),
                            to: target,
                        });
                    }
                }
                continue;
            }

            if let Some(targets) = parse_python_import(line, &mod_to_file) {
                for target in targets {
                    if target != *file {
                        edges.push(FileEdge {
                            from: file.clone(),
                            to: target,
                        });
                    }
                }
            }
        }
    }

    edges
}

fn detect_python_package(root: &Path) -> (String, PathBuf) {
    // Check src/ layout first.
    let src = root.join("src");
    if src.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&src) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() && p.join("__init__.py").exists() {
                    let name = p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("pkg")
                        .to_string();
                    return (name, p);
                }
            }
        }
    }

    // Check root-level package dirs.
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("__init__.py").exists() {
                let name = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("pkg")
                    .to_string();
                if !matches!(
                    name.as_str(),
                    "tests" | "test" | "docs" | "examples" | "benchmarks"
                ) {
                    return (name, p);
                }
            }
        }
    }

    // Fallback: treat root as the package.
    let name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("pkg")
        .to_string();
    (name, root.to_path_buf())
}

fn python_file_to_mod_path(file: &Path, pkg_dir: &Path, pkg_name: &str) -> Option<String> {
    let rel = file.strip_prefix(pkg_dir).ok()?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");

    if rel_str == "__init__.py" {
        return Some(pkg_name.to_string());
    }

    let rel_str = if rel_str.ends_with("/__init__.py") {
        rel_str.trim_end_matches("/__init__.py").to_string()
    } else {
        rel_str.trim_end_matches(".py").to_string()
    };

    let mod_path = rel_str.replace('/', ".");
    Some(format!("{}.{}", pkg_name, mod_path))
}

fn parse_python_from_import(
    line: &str,
    current_mod: &str,
    pkg_name: &str,
    mod_to_file: &HashMap<String, PathBuf>,
) -> Option<Vec<PathBuf>> {
    // `from .foo import bar` or `from ..foo import bar` or `from pkg.foo import bar`
    let rest = line.strip_prefix("from ")?;
    let (module_part, _) = rest.split_once(" import ")?;
    let module_part = module_part.trim();

    let resolved = if module_part.starts_with('.') {
        // Relative import.
        let dots = module_part.chars().take_while(|c| *c == '.').count();
        let relative = &module_part[dots..];

        // Go up `dots` levels from current module.
        let mut base = current_mod.to_string();
        for _ in 0..dots {
            base = base
                .rsplit_once('.')
                .map(|(p, _)| p.to_string())
                .unwrap_or_default();
        }

        if relative.is_empty() {
            base
        } else {
            format!("{}.{}", base, relative)
        }
    } else if module_part.starts_with(pkg_name) {
        // Absolute import within the package.
        module_part.to_string()
    } else {
        // External package.
        return None;
    };

    let mut targets = Vec::new();
    resolve_python_path(&resolved, mod_to_file, &mut targets);
    if targets.is_empty() {
        None
    } else {
        Some(targets)
    }
}

fn parse_python_import(line: &str, mod_to_file: &HashMap<String, PathBuf>) -> Option<Vec<PathBuf>> {
    // `import pkg.foo.bar` or `import pkg.foo.bar as baz`
    let rest = line.strip_prefix("import ")?;
    // Skip `from ... import ...` (handled separately).
    if line.starts_with("from ") {
        return None;
    }

    let mut targets = Vec::new();
    for part in rest.split(',') {
        let part = part.trim();
        let mod_path = part.split(" as ").next().unwrap_or(part).trim();
        resolve_python_path(mod_path, mod_to_file, &mut targets);
    }

    if targets.is_empty() {
        None
    } else {
        Some(targets)
    }
}

fn resolve_python_path(path: &str, mod_to_file: &HashMap<String, PathBuf>, out: &mut Vec<PathBuf>) {
    let mut path_str = path.to_string();
    loop {
        if let Some(file) = mod_to_file.get(&path_str) {
            out.push(file.clone());
            return;
        }
        match path_str.rsplit_once('.') {
            Some((parent, _)) => path_str = parent.to_string(),
            None => break,
        }
    }
}

// ---------------------------------------------------------------------------
// JS/TS import parser
// ---------------------------------------------------------------------------

fn parse_js_imports(root: &Path, files: &[PathBuf]) -> Vec<FileEdge> {
    let mut edges = Vec::new();

    let file_set: HashSet<PathBuf> = files.iter().cloned().collect();
    let path_aliases = read_tsconfig_paths(root);

    for file in files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let dir = file.parent().unwrap_or(root);

        for line in content.lines() {
            let line = line.trim();

            for spec in extract_js_import_specifiers(line) {
                // Resolve the import to a filesystem path.
                let resolved = if spec.starts_with('.') {
                    resolve_js_import(dir, &spec, &file_set)
                } else if let Some(abs_path) = resolve_alias_to_path(&spec, &path_aliases) {
                    // Alias resolved to absolute path. Try with extensions.
                    let parent = abs_path.parent().unwrap_or(root).to_path_buf();
                    let fname = abs_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    resolve_js_import(&parent, &format!("./{}", fname), &file_set)
                } else {
                    continue;
                };

                if let Some(resolved) = resolved {
                    if resolved != *file {
                        edges.push(FileEdge {
                            from: file.clone(),
                            to: resolved,
                        });
                    }
                }
            }
        }
    }

    edges
}

/// Read path aliases from tsconfig.json / jsconfig.json.
/// Returns Vec<(prefix, replacement_dir)>, e.g. ("@/", "./src/").
fn read_tsconfig_paths(root: &Path) -> Vec<(String, PathBuf)> {
    let mut aliases = Vec::new();

    // Read tsconfig.json/jsconfig.json for path aliases.
    let candidates = ["tsconfig.json", "jsconfig.json"];
    for name in &candidates {
        let path = root.join(name);
        if let Ok(raw) = std::fs::read_to_string(&path) {
            let cleaned: String = raw
                .lines()
                .map(|l| {
                    if let Some(pos) = l.find("//") {
                        &l[..pos]
                    } else {
                        l
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cleaned) {
                aliases = extract_path_aliases(&val, root);
                break;
            }
        }
    }

    // Always discover workspace packages (even without tsconfig).
    discover_workspace_packages(root, &mut aliases);

    // Fallback: @/ → root if nothing else matched.
    if aliases.is_empty() {
        aliases.push(("@/".to_string(), root.to_path_buf()));
    }

    aliases
}

fn extract_path_aliases(tsconfig: &serde_json::Value, root: &Path) -> Vec<(String, PathBuf)> {
    let mut aliases = Vec::new();

    let base_url = tsconfig
        .get("compilerOptions")
        .and_then(|c| c.get("baseUrl"))
        .and_then(|b| b.as_str())
        .unwrap_or(".");
    let base_dir = root.join(base_url);

    if let Some(paths) = tsconfig
        .get("compilerOptions")
        .and_then(|c| c.get("paths"))
        .and_then(|p| p.as_object())
    {
        for (pattern, targets) in paths {
            // Pattern like "@/*" → strip the wildcard.
            let prefix = pattern.trim_end_matches('*');
            if let Some(first_target) = targets.as_array().and_then(|a| a.first()) {
                if let Some(target_str) = first_target.as_str() {
                    let target_path = target_str.trim_end_matches('*');
                    let resolved = base_dir.join(target_path);
                    aliases.push((prefix.to_string(), resolved));
                }
            }
        }
    }

    // Discover npm/yarn/pnpm workspace packages.
    // Maps `@scope/pkg` or `pkg` → the package's directory.
    discover_workspace_packages(root, &mut aliases);

    aliases
}

/// Find all workspace packages by scanning for package.json files in common locations.
fn discover_workspace_packages(root: &Path, aliases: &mut Vec<(String, PathBuf)>) {
    // Read root package.json for workspace patterns.
    let root_pkg = root.join("package.json");
    if !root_pkg.exists() {
        return;
    }
    let raw = match std::fs::read_to_string(&root_pkg) {
        Ok(r) => r,
        Err(_) => return,
    };
    let val: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Workspace dirs: from "workspaces" field (npm/yarn) or common patterns.
    let mut search_dirs: Vec<PathBuf> = Vec::new();
    if let Some(workspaces) = val.get("workspaces") {
        let patterns = if let Some(arr) = workspaces.as_array() {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        } else if let Some(obj) = workspaces.as_object() {
            // yarn workspaces: { "packages": ["packages/*"] }
            obj.get("packages")
                .and_then(|p| p.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        for pattern in patterns {
            let pattern = pattern.trim_end_matches("/*").trim_end_matches("/**");
            let dir = root.join(pattern);
            if dir.is_dir() {
                search_dirs.push(dir);
            }
        }
    }

    // Common fallback patterns.
    if search_dirs.is_empty() {
        for dir_name in ["packages", "apps", "libs", "modules"] {
            let d = root.join(dir_name);
            if d.is_dir() {
                search_dirs.push(d);
            }
        }
    }

    // Scan each workspace dir for package.json files.
    for dir in &search_dirs {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let pkg_json = entry.path().join("package.json");
            if !pkg_json.exists() {
                continue;
            }
            let raw = match std::fs::read_to_string(&pkg_json) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let pkg_val: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(name) = pkg_val.get("name").and_then(|n| n.as_str()) {
                // Map "name/" → package directory.
                // For imports like `@calcom/lib/hooks/useLocale`,
                // the prefix `@calcom/lib/` maps to `packages/lib/`.
                let pkg_dir = entry.path();

                // Check for explicit "main" or "exports" to find src root.
                let src_dir = if pkg_dir.join("src").is_dir() {
                    pkg_dir.join("src")
                } else {
                    pkg_dir.clone()
                };

                aliases.push((format!("{}/", name), src_dir));
            }
        }
    }
}

fn resolve_alias_to_path(spec: &str, aliases: &[(String, PathBuf)]) -> Option<PathBuf> {
    for (prefix, target_dir) in aliases {
        if let Some(rest) = spec.strip_prefix(prefix.as_str()) {
            return Some(target_dir.join(rest));
        }
    }
    None
}

fn extract_js_import_specifiers(line: &str) -> Vec<String> {
    let mut specs = Vec::new();

    // `import ... from '...'` or `import ... from "..."`
    if line.starts_with("import ") || line.starts_with("export ") {
        if let Some(spec) = extract_string_after(line, " from ") {
            specs.push(spec);
        } else if line.starts_with("import '") || line.starts_with("import \"") {
            // Side-effect import: `import './foo'`
            if let Some(spec) = extract_quoted_string(&line["import ".len()..]) {
                specs.push(spec);
            }
        }
    }

    // `require('...')` -- only at statement level (not inside a string).
    // Heuristic: require( must be preceded by start-of-line, `=`, `(`, or whitespace.
    if let Some(pos) = line.find("require(") {
        let before = if pos > 0 {
            line.as_bytes()[pos - 1]
        } else {
            b' '
        };
        if matches!(before, b' ' | b'=' | b'(' | b'\t' | b',') || pos == 0 {
            let after = &line[pos + "require(".len()..];
            if let Some(spec) = extract_quoted_string(after) {
                specs.push(spec);
            }
        }
    }

    specs
}

fn extract_string_after(line: &str, marker: &str) -> Option<String> {
    let pos = line.find(marker)?;
    let after = &line[pos + marker.len()..];
    extract_quoted_string(after)
}

fn extract_quoted_string(s: &str) -> Option<String> {
    let s = s.trim();
    let (quote, rest) = if let Some(rest) = s.strip_prefix('\'') {
        ('\'', rest)
    } else if let Some(rest) = s.strip_prefix('"') {
        ('"', rest)
    } else {
        return None;
    };
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn resolve_js_import(dir: &Path, spec: &str, file_set: &HashSet<PathBuf>) -> Option<PathBuf> {
    let base = dir.join(spec);

    // Try exact path, then with extensions, then as directory/index.
    let extensions = ["", ".ts", ".tsx", ".js", ".jsx", ".mjs"];

    for ext in &extensions {
        let candidate = PathBuf::from(format!("{}{}", base.display(), ext));
        if let Ok(canonical) = candidate.canonicalize() {
            if file_set.contains(&canonical) {
                return Some(canonical);
            }
        }
        // Also try without canonicalize (symlinks, etc.)
        if file_set.contains(&candidate) {
            return Some(candidate);
        }
    }

    // Try as directory: spec/index.{ts,tsx,js,jsx}
    let dir_extensions = ["index.ts", "index.tsx", "index.js", "index.jsx"];
    for idx in &dir_extensions {
        let candidate = base.join(idx);
        if let Ok(canonical) = candidate.canonicalize() {
            if file_set.contains(&canonical) {
                return Some(canonical);
            }
        }
        if file_set.contains(&candidate) {
            return Some(candidate);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Go import parser
// ---------------------------------------------------------------------------

fn parse_go_imports(root: &Path, files: &[PathBuf]) -> Vec<FileEdge> {
    // Go imports are package-level (directory-based).
    let module_name = read_go_module_name(root);

    // Map: package import path → canonical representative file.
    // We pick one file per package to avoid N-edge fan-out.
    // Preference: file named after package dir > doc.go > alphabetically first.
    let mut pkg_to_files: HashMap<String, Vec<PathBuf>> = HashMap::new();

    for file in files {
        if let Some(pkg_path) = go_file_to_pkg_path(file, root, &module_name) {
            pkg_to_files.entry(pkg_path).or_default().push(file.clone());
        }
    }

    let mut pkg_canonical: HashMap<String, PathBuf> = HashMap::new();
    for (pkg, pkg_files) in &mut pkg_to_files {
        pkg_files.sort();
        // Pick canonical: package-name-matching file > first non-doc > first.
        let pkg_leaf = pkg.rsplit('/').next().unwrap_or(pkg);
        let canonical = pkg_files
            .iter()
            .find(|f| {
                f.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s == pkg_leaf)
                    .unwrap_or(false)
            })
            .or_else(|| {
                pkg_files.iter().find(|f| {
                    f.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n != "doc.go")
                        .unwrap_or(true)
                })
            })
            .or(pkg_files.first())
            .cloned();
        if let Some(c) = canonical {
            pkg_canonical.insert(pkg.clone(), c);
        }
    }

    let mut edges = Vec::new();

    for file in files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        for line in content.lines() {
            let line = line.trim();

            if let Some(import_path) = extract_go_import(line) {
                if !import_path.starts_with(&module_name) {
                    continue;
                }

                if let Some(target) = pkg_canonical.get(&import_path) {
                    if target != file {
                        edges.push(FileEdge {
                            from: file.clone(),
                            to: target.clone(),
                        });
                    }
                }
            }
        }
    }

    edges
}

fn read_go_module_name(root: &Path) -> String {
    let go_mod = root.join("go.mod");
    if let Ok(content) = std::fs::read_to_string(&go_mod) {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("module ") {
                return rest.trim().to_string();
            }
        }
    }
    String::new()
}

fn go_file_to_pkg_path(file: &Path, root: &Path, module_name: &str) -> Option<String> {
    let dir = file.parent()?;
    let rel = dir.strip_prefix(root).ok()?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");

    if rel_str.is_empty() {
        Some(module_name.to_string())
    } else {
        Some(format!("{}/{}", module_name, rel_str))
    }
}

fn extract_go_import(line: &str) -> Option<String> {
    // Match: `"some/import/path"` (possibly with alias prefix)
    let line = line.trim();
    // Skip `import (` and `)` lines, and the `import` keyword line.
    if line == "import (" || line == ")" || line == "import" {
        return None;
    }

    // Could be: `import "path"` or `_ "path"` or `alias "path"`
    extract_quoted_string(line).or_else(|| {
        // With alias: `foo "path"`
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() == 2 {
            extract_quoted_string(parts[1])
        } else {
            None
        }
    })
}

// ---------------------------------------------------------------------------
// Output row
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub(crate) struct FileRow {
    pub file: String,
    pub role: FileRole,
    pub in_degree: usize,
    pub out_degree: usize,
    pub dependents: usize,
    pub dependencies: usize,
    pub pagerank: f64,
    pub consumers_pagerank: f64,
    pub betweenness: f64,
    pub orphan: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycle_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commits: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub churn_risk: Option<f64>,
    /// External packages this file imports (ecosystem-level deps).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub external_deps: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct FilesResult {
    pub rows: Vec<FileRow>,
    pub nodes: usize,
    pub edges: usize,
    pub ecosystem: Ecosystem,
    pub orphan_count: usize,
    pub cycles: Vec<Vec<String>>,
    /// Direct edges: (from_file, to_file).
    #[serde(skip)]
    pub direct_edges: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// Core analysis
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Git history analysis
// ---------------------------------------------------------------------------

/// Parse git log to get per-file commit counts within a time window.
fn git_file_commit_counts(root: &Path, days: u64) -> HashMap<String, usize> {
    let since = format!("--since={} days ago", days);
    let out = ProcessCommand::new("git")
        .args(["log", "--name-only", "--pretty=format:", &since])
        .current_dir(root)
        .output();
    let out = match out {
        Ok(o) if o.status.success() => o,
        _ => return HashMap::new(),
    };
    let stdout = String::from_utf8_lossy(&out.stdout);

    let mut counts: HashMap<String, usize> = HashMap::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        *counts.entry(line.to_string()).or_insert(0) += 1;
    }
    counts
}

/// Clone a git URL to a temp directory. Returns the path.
/// Contract a file-level graph into a directory-level graph.
/// Returns: (contracted graph, node labels, file count per directory).
fn contract_to_directories(
    graph: &DiGraph<PathBuf, f64>,
    root: &Path,
) -> (DiGraph<String, f64>, Vec<String>, HashMap<String, usize>) {
    let mut dir_indices: HashMap<String, NodeIndex> = HashMap::new();
    let mut contracted: DiGraph<String, f64> = DiGraph::new();
    let mut file_to_dir_idx: Vec<NodeIndex> = Vec::new();
    let mut file_counts: HashMap<String, usize> = HashMap::new();

    for n in graph.node_indices() {
        let file = graph.nw(n);
        let rel = file.strip_prefix(root).unwrap_or(file);
        let dir = rel
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        let dir = if dir.is_empty() { ".".to_string() } else { dir };

        *file_counts.entry(dir.clone()).or_insert(0) += 1;

        let dir_idx = *dir_indices
            .entry(dir.clone())
            .or_insert_with(|| contracted.add_node(dir));
        file_to_dir_idx.push(dir_idx);
    }

    // Add edges between directories (merge weights).
    for e in graph.edge_references() {
        let from_dir = file_to_dir_idx[e.source().index()];
        let to_dir = file_to_dir_idx[e.target().index()];
        if from_dir == to_dir {
            continue;
        }
        let cur = contracted
            .find_edge(from_dir, to_dir)
            .and_then(|ei| contracted.edge_weight(ei).copied())
            .unwrap_or(0.0);
        contracted.update_edge(from_dir, to_dir, cur + e.weight());
    }

    let labels: Vec<String> = contracted
        .node_indices()
        .map(|n| contracted.nw(n).clone())
        .collect();

    (contracted, labels, file_counts)
}

fn clone_repo_to_temp(url: &str) -> Result<PathBuf> {
    let tmp = std::env::temp_dir().join(format!("pkgrank-{:016x}", fnv1a64(url.as_bytes())));
    if tmp.exists() {
        // Reuse existing clone.
        return Ok(tmp);
    }
    eprintln!("cloning {} → {}", url, tmp.display());
    let out = ProcessCommand::new("git")
        .args(["clone", "--depth", "1", url])
        .arg(&tmp)
        .output()
        .map_err(|e| anyhow::anyhow!("git clone failed: {}", e))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow::anyhow!("git clone failed: {}", stderr.trim()));
    }
    Ok(tmp)
}

fn is_url(s: &str) -> bool {
    s.starts_with("https://") || s.starts_with("http://") || s.starts_with("git@")
}

/// Expand shorthand like "owner/repo" to "https://github.com/owner/repo".
fn expand_uri(s: &str) -> String {
    if is_url(s) || PathBuf::from(s).exists() {
        return s.to_string();
    }
    // owner/repo pattern (exactly one slash, no dots or spaces).
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() == 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && !p.contains('.') && !p.contains(' '))
    {
        return format!("https://github.com/{}/{}", parts[0], parts[1]);
    }
    s.to_string()
}

pub(crate) fn files_analyze(args: &FilesArgs) -> Result<FilesResult> {
    let uri = expand_uri(&args.path);
    let root = if is_url(&uri) {
        clone_repo_to_temp(&uri)?
    } else {
        let p = PathBuf::from(&uri);
        if p.is_file() {
            p.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            p
        }
    };

    let ecosystem = args
        .ecosystem
        .or_else(|| detect_ecosystem(&root))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Could not detect ecosystem in {}. Pass --ecosystem explicitly.",
                root.display()
            )
        })?;

    let all_files = discover_files(&root, ecosystem);
    let mut included_files: Vec<PathBuf> = Vec::new();
    let mut file_roles: HashMap<PathBuf, FileRole> = HashMap::new();

    for file in &all_files {
        let role = classify_file(file, &root, ecosystem);
        if should_include(role, args) {
            included_files.push(file.clone());
            file_roles.insert(file.clone(), role);
        }
    }

    let edges = match ecosystem {
        Ecosystem::Cargo => parse_rust_imports(&root, &included_files),
        Ecosystem::Python => parse_python_imports(&root, &included_files),
        Ecosystem::Npm => parse_js_imports(&root, &included_files),
        Ecosystem::Go => parse_go_imports(&root, &included_files),
    };

    // Extract external dependency names per file.
    let mut internal_prefixes: HashSet<String> = HashSet::new();
    match ecosystem {
        Ecosystem::Cargo => {
            let crate_roots = find_rust_crate_roots(&root);
            for (_, name) in &crate_roots {
                internal_prefixes.insert(name.clone());
            }
            // Also add all known module names (from mod declarations) as internal.
            for file in &included_files {
                if let Ok(content) = std::fs::read_to_string(file) {
                    for line in content.lines() {
                        if let Some(mod_name) = parse_mod_declaration(line.trim()) {
                            internal_prefixes.insert(mod_name);
                        }
                    }
                }
            }
        }
        Ecosystem::Python => {
            let (pkg_name, _) = detect_python_package(&root);
            internal_prefixes.insert(pkg_name);
        }
        Ecosystem::Go => {
            let mod_name = read_go_module_name(&root);
            if !mod_name.is_empty() {
                internal_prefixes.insert(mod_name);
            }
        }
        Ecosystem::Npm => {} // JS doesn't have a simple internal prefix
    }
    let external_deps_map = extract_external_deps(&included_files, ecosystem, &internal_prefixes);

    let mut graph: DiGraph<PathBuf, f64> = DiGraph::new();
    let mut node_map: HashMap<PathBuf, NodeIndex> = HashMap::new();

    for file in &included_files {
        let idx = graph.add_node(file.clone());
        node_map.insert(file.clone(), idx);
    }

    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
    for edge in &edges {
        if let (Some(&from_idx), Some(&to_idx)) = (node_map.get(&edge.from), node_map.get(&edge.to))
        {
            let key = (from_idx.index(), to_idx.index());
            if seen_edges.insert(key) {
                graph.update_edge(from_idx, to_idx, 1.0);
            }
        }
    }

    // Compute labels and centrality.
    // Directory mode: contract to directory-level graph first.
    // File mode: compute directly on the PathBuf graph (avoids copying).
    // Precomputed analysis vectors.
    struct AnalysisVecs {
        labels: Vec<String>,
        pr: Vec<f64>,
        consumers_pr: Vec<f64>,
        bc: Vec<f64>,
        in_degrees: Vec<usize>,
        out_degrees: Vec<usize>,
        transitive_dependents: Vec<usize>,
        transitive_deps: Vec<usize>,
        scc_labels: Vec<usize>,
        direct_edges: Vec<(String, String)>,
        node_count: usize,
        edge_count: usize,
    }

    fn compute_analysis<N: Clone + std::fmt::Debug>(
        g: &DiGraph<N, f64>,
        labels: Vec<String>,
    ) -> AnalysisVecs {
        let pr = pagerank_auto(g);
        let consumers_pr = pagerank_auto(&reverse_graph(g));
        let bc = betweenness_centrality(g);
        let in_degrees: Vec<usize> = g
            .node_indices()
            .map(|n| g.neighbors_directed(n, Direction::Incoming).count())
            .collect();
        let out_degrees: Vec<usize> = g
            .node_indices()
            .map(|n| g.neighbors_directed(n, Direction::Outgoing).count())
            .collect();
        let mut ep: Vec<(usize, usize)> = Vec::new();
        for e in g.edge_references() {
            ep.push((e.source().index(), e.target().index()));
        }
        let (td, tdd) = reachability_counts_edges(g.node_count(), &ep);
        let scc = strongly_connected_components(g);
        let direct_edges: Vec<(String, String)> = g
            .edge_references()
            .map(|e| {
                (
                    labels[e.source().index()].clone(),
                    labels[e.target().index()].clone(),
                )
            })
            .collect();
        AnalysisVecs {
            labels,
            pr,
            consumers_pr,
            bc,
            in_degrees,
            out_degrees,
            transitive_dependents: td,
            transitive_deps: tdd,
            scc_labels: scc,
            direct_edges,
            node_count: g.node_count(),
            edge_count: g.edge_count(),
        }
    }

    let av = if args.directory {
        let (contracted, labels, _counts) = contract_to_directories(&graph, &root);
        compute_analysis(&contracted, labels)
    } else {
        let labels: Vec<String> = graph
            .node_indices()
            .map(|n| {
                graph
                    .nw(n)
                    .strip_prefix(&root)
                    .unwrap_or(graph.nw(n))
                    .to_string_lossy()
                    .to_string()
            })
            .collect();
        compute_analysis(&graph, labels)
    };
    let mut scc_sizes: HashMap<usize, usize> = HashMap::new();
    for &label in &av.scc_labels {
        *scc_sizes.entry(label).or_insert(0) += 1;
    }
    let mut cycles: Vec<Vec<String>> = Vec::new();
    let mut cycle_id_map: HashMap<usize, usize> = HashMap::new();
    {
        let mut cycle_labels: Vec<usize> = scc_sizes
            .iter()
            .filter(|(_, &size)| size > 1)
            .map(|(&label, _)| label)
            .collect();
        cycle_labels.sort();

        for (cycle_idx, &scc_label) in cycle_labels.iter().enumerate() {
            cycle_id_map.insert(scc_label, cycle_idx);
            let members: Vec<String> = (0..av.node_count)
                .filter(|&i| av.scc_labels[i] == scc_label)
                .map(|i| av.labels[i].clone())
                .collect();
            cycles.push(members);
        }
    }

    // Git history (optional).
    let git_counts: HashMap<String, usize> = if args.git {
        git_file_commit_counts(&root, args.git_days)
    } else {
        HashMap::new()
    };
    let max_commits = git_counts.values().copied().max().unwrap_or(1).max(1) as f64;

    let mut rows: Vec<FileRow> = (0..av.node_count)
        .map(|i| {
            let rel = av.labels[i].clone();
            let role = if args.directory {
                FileRole::Source
            } else {
                let full = root.join(&rel);
                file_roles.get(&full).copied().unwrap_or(FileRole::Source)
            };
            let in_degree = av.in_degrees[i];
            let out_degree = av.out_degrees[i];
            let cycle_id = cycle_id_map.get(&av.scc_labels[i]).copied();

            let commits = if args.git {
                if args.directory {
                    Some(
                        git_counts
                            .iter()
                            .filter(|(path, _)| {
                                Path::new(path)
                                    .parent()
                                    .map(|p| {
                                        p.to_string_lossy() == rel
                                            || (rel == "." && p == Path::new(""))
                                    })
                                    .unwrap_or(false)
                            })
                            .map(|(_, &c)| c)
                            .sum::<usize>(),
                    )
                } else {
                    Some(git_counts.get(&rel).copied().unwrap_or(0))
                }
            } else {
                None
            };
            let churn_risk = commits.map(|c| av.pr[i] * (c as f64 / max_commits));
            let ext_deps = if args.directory {
                Vec::new()
            } else {
                let full = root.join(&rel);
                external_deps_map.get(&full).cloned().unwrap_or_default()
            };

            FileRow {
                file: rel,
                role,
                in_degree,
                out_degree,
                dependents: av.transitive_dependents[i],
                dependencies: av.transitive_deps[i],
                pagerank: av.pr[i],
                consumers_pagerank: av.consumers_pr[i],
                betweenness: av.bc[i],
                orphan: in_degree == 0 && out_degree == 0,
                cycle_id,
                commits,
                churn_risk,
                external_deps: ext_deps,
            }
        })
        .collect();

    if args.git && matches!(args.metric, Metric::Pagerank) {
        rows.sort_by(|a, b| {
            b.churn_risk
                .unwrap_or(0.0)
                .total_cmp(&a.churn_risk.unwrap_or(0.0))
        });
    } else {
        rows.sort_by(|a, b| match args.metric {
            Metric::Pagerank => b.pagerank.total_cmp(&a.pagerank),
            Metric::ConsumersPagerank => b.consumers_pagerank.total_cmp(&a.consumers_pagerank),
            Metric::Betweenness => b.betweenness.total_cmp(&a.betweenness),
            Metric::Indegree => b.in_degree.cmp(&a.in_degree),
            Metric::Outdegree => b.out_degree.cmp(&a.out_degree),
        });
    }

    let orphan_count = rows.iter().filter(|r| r.orphan).count();

    Ok(FilesResult {
        nodes: av.node_count,
        edges: av.edge_count,
        ecosystem,
        orphan_count,
        cycles,
        rows,
        direct_edges: av.direct_edges,
    })
}

// ---------------------------------------------------------------------------
// Run + print
// ---------------------------------------------------------------------------

pub(crate) fn run_files(args: &FilesArgs) -> Result<()> {
    let result = files_analyze(args)?;

    let fmt = effective_format(args.format);
    match fmt {
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct Out {
                schema_version: u32,
                ok: bool,
                command: &'static str,
                ecosystem: Ecosystem,
                nodes: usize,
                edges: usize,
                orphan_count: usize,
                cycle_count: usize,
                cycles: Vec<Vec<String>>,
                rows_total: usize,
                rows_returned: usize,
                rows: Vec<FileRow>,
            }
            let rows_total = result.rows.len();
            let rows: Vec<FileRow> = result.rows.iter().take(args.top).cloned().collect();
            let out = Out {
                schema_version: 1,
                ok: true,
                command: "files",
                ecosystem: result.ecosystem,
                nodes: result.nodes,
                edges: result.edges,
                orphan_count: result.orphan_count,
                cycle_count: result.cycles.len(),
                cycles: result.cycles.clone(),
                rows_total,
                rows_returned: rows.len(),
                rows,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Text => {
            let git_label = if args.git {
                format!("  git_days={}", args.git_days)
            } else {
                String::new()
            };
            println!(
                "pkgrank files  ecosystem={}  metric={:?}  include_tests={}{}\n",
                result.ecosystem, args.metric, args.include_tests, git_label
            );
            if args.git {
                println!(
                    "{:>4}  {:>8}  {:>5}  {:>10}  {:>5}  {:>3}  {:>3}  {:<10}  file",
                    "rank", "churn", "comms", "pr", "blast", "in", "out", "role"
                );
            } else {
                println!(
                    "{:>4}  {:>10}  {:>10}  {:>9}  {:>5}  {:>5}  {:>3}  {:>3}  {:<10}  file",
                    "rank", "pr", "cons_pr", "between", "blast", "deps", "in", "out", "role"
                );
            }
            println!("{:\u{2500}<110}", "");
            for (i, r) in result.rows.iter().take(args.top).enumerate() {
                let mut label = format!("{:?}", r.role).to_lowercase();
                if r.cycle_id.is_some() {
                    label = format!("{}*", label);
                }
                if args.git {
                    println!(
                        "{:>4}. {:>8.6} {:>5} {:>10.6} {:>5} {:>3} {:>3}  {:<10}  {}",
                        i + 1,
                        r.churn_risk.unwrap_or(0.0),
                        r.commits.unwrap_or(0),
                        r.pagerank,
                        r.dependents,
                        r.in_degree,
                        r.out_degree,
                        label,
                        r.file
                    );
                } else {
                    println!(
                        "{:>4}. {:>10.6} {:>10.6} {:>9.6} {:>5} {:>5} {:>3} {:>3}  {:<10}  {}",
                        i + 1,
                        r.pagerank,
                        r.consumers_pagerank,
                        r.betweenness,
                        r.dependents,
                        r.dependencies,
                        r.in_degree,
                        r.out_degree,
                        label,
                        r.file
                    );
                }
            }

            // Summary.
            let density = if result.nodes > 1 {
                result.edges as f64 / (result.nodes as f64 * (result.nodes as f64 - 1.0))
            } else {
                0.0
            };
            println!(
                "\n{} files, {} edges, density={:.4}",
                result.nodes, result.edges, density
            );
            println!(
                "{} orphans, {} cycles",
                result.orphan_count,
                result.cycles.len()
            );

            if !result.cycles.is_empty() {
                println!("\ncycles (* marks files in a cycle):");
                for (i, cycle) in result.cycles.iter().enumerate() {
                    let preview: Vec<&str> = cycle.iter().take(5).map(|s| s.as_str()).collect();
                    let suffix = if cycle.len() > 5 {
                        format!(", ... (+{})", cycle.len() - 5)
                    } else {
                        String::new()
                    };
                    println!(
                        "  cycle {}: {} files  [{}{}]",
                        i,
                        cycle.len(),
                        preview.join(", "),
                        suffix
                    );
                }
            }
        }
    }

    // Focus mode: show detailed info about a specific file.
    if let Some(focus) = &args.focus {
        print_focus(focus, &result);
    }

    Ok(())
}

fn print_focus(query: &str, result: &FilesResult) {
    // Match the query against file paths (partial match).
    let matches: Vec<&FileRow> = result
        .rows
        .iter()
        .filter(|r| r.file.contains(query))
        .collect();

    if matches.is_empty() {
        eprintln!("no file matching '{}' found", query);
        return;
    }

    for row in &matches {
        println!("\n{:=<80}", "");
        println!("focus: {}", row.file);
        println!(
            "  role={:?}  pagerank={:.6}  betweenness={:.6}",
            row.role, row.pagerank, row.betweenness
        );
        println!(
            "  in_degree={}  out_degree={}  blast_radius={}  deps={}",
            row.in_degree, row.out_degree, row.dependents, row.dependencies
        );
        if let Some(c) = row.commits {
            println!(
                "  commits={}  churn_risk={:.6}",
                c,
                row.churn_risk.unwrap_or(0.0)
            );
        }
        if let Some(cid) = row.cycle_id {
            println!(
                "  cycle_id={} ({} files)",
                cid,
                result.cycles.get(cid).map(|c| c.len()).unwrap_or(0)
            );
        }

        // Direct imports (this file depends on).
        let imports: Vec<&str> = result
            .direct_edges
            .iter()
            .filter(|(from, _)| from == &row.file)
            .map(|(_, to)| to.as_str())
            .collect();
        if !imports.is_empty() {
            println!("  imports ({}):", imports.len());
            for imp in &imports {
                println!("    -> {}", imp);
            }
        }

        // Direct dependents (files that import this one).
        let dependents: Vec<&str> = result
            .direct_edges
            .iter()
            .filter(|(_, to)| to == &row.file)
            .map(|(from, _)| from.as_str())
            .collect();
        if !dependents.is_empty() {
            println!("  imported by ({}):", dependents.len());
            for dep in dependents.iter().take(15) {
                println!("    <- {}", dep);
            }
            if dependents.len() > 15 {
                println!("    ... (+{} more)", dependents.len() - 15);
            }
        }
    }
}
