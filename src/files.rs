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
    if dir.join("Cargo.toml").exists() {
        return Some(Ecosystem::Cargo);
    }
    if dir.join("package.json").exists() || dir.join("package-lock.json").exists() {
        return Some(Ecosystem::Npm);
    }
    if dir.join("pyproject.toml").exists()
        || dir.join("uv.lock").exists()
        || dir.join("setup.py").exists()
    {
        return Some(Ecosystem::Python);
    }
    if dir.join("go.mod").exists() {
        return Some(Ecosystem::Go);
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

        // Join multi-line use/mod statements into logical lines.
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
        // Cross-crate: check if first segment is a known workspace crate.
        let first_seg = use_part.split("::").next().unwrap_or("");
        if known_crates.contains(first_seg) {
            (use_part.to_string(), "")
        } else {
            return None;
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

        for line in content.lines() {
            let line = line.trim();

            // `from .foo import bar` or `from ..utils import X`
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

            // `import pkg.foo.bar`
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

    // Build a set of known files for resolution.
    let file_set: HashSet<PathBuf> = files.iter().cloned().collect();

    for file in files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let dir = file.parent().unwrap_or(root);

        for line in content.lines() {
            let line = line.trim();

            // ESM: `import ... from './foo'` or `import './foo'`
            // CJS: `require('./foo')`
            // Dynamic: `import('./foo')`
            for spec in extract_js_import_specifiers(line) {
                // Only resolve relative imports (intra-project).
                if !spec.starts_with('.') {
                    continue;
                }

                if let Some(resolved) = resolve_js_import(dir, &spec, &file_set) {
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
    // Detect the module name from go.mod.
    let module_name = read_go_module_name(root);

    // Map: package import path → list of files in that package.
    let mut pkg_to_files: HashMap<String, Vec<PathBuf>> = HashMap::new();
    let mut file_to_pkg: HashMap<PathBuf, String> = HashMap::new();

    for file in files {
        if let Some(pkg_path) = go_file_to_pkg_path(file, root, &module_name) {
            pkg_to_files
                .entry(pkg_path.clone())
                .or_default()
                .push(file.clone());
            file_to_pkg.insert(file.clone(), pkg_path);
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

            // `import "module/pkg/path"` or within import block.
            if let Some(import_path) = extract_go_import(line) {
                // Only intra-module imports.
                if !import_path.starts_with(&module_name) {
                    continue;
                }

                if let Some(target_files) = pkg_to_files.get(&import_path) {
                    for target in target_files {
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

#[derive(Debug, Serialize)]
pub(crate) struct FileRow {
    pub file: String,
    pub role: FileRole,
    pub in_degree: usize,
    pub out_degree: usize,
    pub pagerank: f64,
    pub consumers_pagerank: f64,
    pub betweenness: f64,
}

// ---------------------------------------------------------------------------
// Core analysis
// ---------------------------------------------------------------------------

/// Clone a git URL to a temp directory. Returns the path.
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

pub(crate) fn files_analyze(args: &FilesArgs) -> Result<(Vec<FileRow>, usize, usize, Ecosystem)> {
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

    // Discover and classify files.
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

    // Parse imports.
    let edges = match ecosystem {
        Ecosystem::Cargo => parse_rust_imports(&root, &included_files),
        Ecosystem::Python => parse_python_imports(&root, &included_files),
        Ecosystem::Npm => parse_js_imports(&root, &included_files),
        Ecosystem::Go => parse_go_imports(&root, &included_files),
    };

    // Build graph.
    let mut graph: DiGraph<PathBuf, f64> = DiGraph::new();
    let mut node_map: HashMap<PathBuf, NodeIndex> = HashMap::new();

    for file in &included_files {
        let idx = graph.add_node(file.clone());
        node_map.insert(file.clone(), idx);
    }

    // Dedup edges.
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

    // Centrality.
    let pr = pagerank_auto(&graph);
    let consumers_pr = pagerank_auto(&reverse_graph(&graph));
    let bc = betweenness_centrality(&graph);

    let mut rows: Vec<FileRow> = graph
        .node_indices()
        .map(|n| {
            let file = graph.nw(n);
            let rel = file
                .strip_prefix(&root)
                .unwrap_or(file)
                .to_string_lossy()
                .to_string();
            let role = file_roles.get(file).copied().unwrap_or(FileRole::Source);

            FileRow {
                file: rel,
                role,
                in_degree: graph.neighbors_directed(n, Direction::Incoming).count(),
                out_degree: graph.neighbors_directed(n, Direction::Outgoing).count(),
                pagerank: pr[n.index()],
                consumers_pagerank: consumers_pr[n.index()],
                betweenness: bc[n.index()],
            }
        })
        .collect();

    rows.sort_by(|a, b| match args.metric {
        Metric::Pagerank => b.pagerank.total_cmp(&a.pagerank),
        Metric::ConsumersPagerank => b.consumers_pagerank.total_cmp(&a.consumers_pagerank),
        Metric::Betweenness => b.betweenness.total_cmp(&a.betweenness),
        Metric::Indegree => b.in_degree.cmp(&a.in_degree),
        Metric::Outdegree => b.out_degree.cmp(&a.out_degree),
    });

    let node_count = graph.node_count();
    let edge_count = graph.edge_count();

    Ok((rows, node_count, edge_count, ecosystem))
}

// ---------------------------------------------------------------------------
// Run + print
// ---------------------------------------------------------------------------

pub(crate) fn run_files(args: &FilesArgs) -> Result<()> {
    let (rows, nodes, edges, ecosystem) = files_analyze(args)?;

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
                rows_total: usize,
                rows_returned: usize,
                rows: Vec<FileRow>,
            }
            let rows_total = rows.len();
            let rows: Vec<_> = rows.into_iter().take(args.top).collect();
            let out = Out {
                schema_version: 1,
                ok: true,
                command: "files",
                ecosystem,
                nodes,
                edges,
                rows_total,
                rows_returned: rows.len(),
                rows,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Text => {
            println!(
                "pkgrank files  ecosystem={}  metric={:?}  include_tests={}\n",
                ecosystem, args.metric, args.include_tests
            );
            println!(
                "{:>4}  {:>10}  {:>10}  {:>9}  {:>3}  {:>3}  {:<10}  file",
                "rank", "pr", "cons_pr", "between", "in", "out", "role"
            );
            println!("{:\u{2500}<100}", "");
            for (i, r) in rows.iter().take(args.top).enumerate() {
                println!(
                    "{:>4}. {:>10.6} {:>10.6} {:>9.6} {:>3} {:>3}  {:<10}  {}",
                    i + 1,
                    r.pagerank,
                    r.consumers_pagerank,
                    r.betweenness,
                    r.in_degree,
                    r.out_degree,
                    format!("{:?}", r.role).to_lowercase(),
                    r.file
                );
            }
            println!("\n{} files, {} edges", nodes, edges);
        }
    }

    Ok(())
}
