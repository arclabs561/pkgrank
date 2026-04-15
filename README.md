# pkgrank

Ranks nodes in a dependency graph by structural importance (PageRank, betweenness, degree).

Two axes of analysis:
- **File-level** (`files`): which source files are structural hotspots, forming cycles, or high churn risk? Polyglot (Rust, Python, JS/TS, Go), works on any local path or GitHub URL.
- **Package-level** (`analyze`): which packages in a dependency tree are most central, most depended-on, most risky to change? Supports Cargo, npm, Python, and Go.

## Install

```bash
cargo install pkgrank
pkgrank --help
```

## Quick start

```bash
# File-level: structural hotspots in any project (no toolchain needed)
pkgrank files .
pkgrank files tokio-rs/tokio
pkgrank files https://github.com/fastapi/fastapi

# File-level with git churn risk overlay
pkgrank files . --git

# Focus on a specific file
pkgrank files . --focus main.rs

# Directory-level aggregation for large codebases
pkgrank files astral-sh/ruff --directory

# CI gate: fail on architectural violations
pkgrank files . --fail-on-violation

# What files are affected by a change?
pkgrank files . --affected src/parser.rs

# Package-level: rank dependencies by importance (auto-detects ecosystem)
pkgrank analyze
pkgrank analyze path/to/npm-project
pkgrank analyze path/to/python-project

# Blast radius: what breaks if serde changes?
pkgrank blast-radius serde
pkgrank blast-radius express path/to/npm-project

# Upgrade priority: which outdated deps matter most? (Cargo only)
pkgrank upgrade-priority
```

## File-level analysis (`pkgrank files`)

Analyze the import graph within a project. Static parsing (no toolchain required). Works across Rust, Python, JS/TS, and Go.

```bash
# Any local project (ecosystem auto-detected)
pkgrank files .

# Any GitHub repo via URL or shorthand
pkgrank files tokio-rs/tokio
pkgrank files https://github.com/django/django

# Git churn risk: combine structural centrality with change frequency
pkgrank files . --git

# Focus on one file: see imports, dependents, co-changers, blast radius
pkgrank files . --git --focus lib.rs

# Directory aggregation for large codebases
pkgrank files astral-sh/ruff --directory

# Include test files (excluded by default)
pkgrank files . --include-tests

# Cache results for repeated queries
pkgrank files . --cache

# CI: fail if cycles or layer violations exist
pkgrank files . --fail-on-violation

# Affected files: what breaks if these files change?
pkgrank files . --affected src/graph.rs,src/index.rs
```

Output includes:
- **Structural role**: foundation (high in, low out), hub (high both), consumer (low in, high out), leaf
- **Instability**: `I = out/(in+out)`, 0 = stable provider, 1 = unstable consumer
- **Blast radius**: transitive dependents (how many files break if this one changes)
- **Cycle detection**: Tarjan's SCC, files in cycles marked with `*`
- **Orphan detection**: files with no imports and no dependents
- **Churn risk** (`--git`): structural centrality * change frequency. Files marked `!!` are in the danger zone (central + volatile)
- **Bus factor** (`--git`): unique contributors per file
- **Co-change coupling** (`--git`): files that change together in commits
- **Layer violations**: detects when stable files import from unstable files (Clean Architecture dependency rule)
- **External deps**: which third-party packages each file imports

Cross-project queries via SQLite (auto-enabled):

```bash
pkgrank query hotspots          # highest churn risk files
pkgrank query deps              # most-used external deps
pkgrank query projects          # list all analyzed projects
pkgrank query "files lib.rs"    # search for files by name
pkgrank query compare           # diff between last two snapshots
pkgrank query drift             # centrality changes over time
```

For JS/TS projects, resolves tsconfig.json path aliases (`@/`, etc.) and npm workspace packages (`@scope/pkg`). Detects cross-language seams (PyO3, NAPI) between Rust and Python/JS.

## Package-level analysis (`pkgrank analyze`)

Rank packages in a dependency graph by centrality. Auto-detects ecosystem from directory contents.

```bash
# Auto-detect: finds Cargo.toml, package-lock.json, uv.lock, or go.mod
pkgrank analyze

# Explicit ecosystem override
pkgrank analyze --ecosystem js path/to/project
pkgrank analyze --ecosystem python path/to/project
pkgrank analyze --ecosystem go path/to/project

# Choose metric
pkgrank analyze --metric consumers-pagerank -n 10

# JSON output
pkgrank analyze --format json --json-limit 200
```

**Graph model**: nodes are packages, directed edges are $A \to B$ iff package A depends on package B.

**Interpretation**:
- PageRank on the depends-on graph surfaces **shared dependencies / substrate packages**.
- Consumer PageRank (reversed graph) surfaces **top-level orchestrators / consumers**.

### Blast radius

Show everything that transitively depends on a package:

```bash
pkgrank blast-radius serde
pkgrank blast-radius express path/to/npm-project
pkgrank blast-radius --workspace-only=false -n 20 serde
```

Output is sorted by BFS depth (closest dependents first), then by PageRank within each depth level.

### Upgrade priority (Cargo only)

Combine `cargo outdated` with centrality ranking to prioritize which upgrades matter most:

```bash
pkgrank upgrade-priority -n 15
```

Requires [`cargo-outdated`](https://crates.io/crates/cargo-outdated). Scores each outdated dep by `10*ln(dependents+1) + 1000*pagerank + urgency_bonus`.

### TLC score

The `triage` and `view` commands produce a **TLC (Top-Level Cost) score** for each crate and repo:

- **Blast radius**: `10 * ln(transitive_dependents + 1)`
- **Centrality**: `1000 * pagerank`
- **Boundary complexity**: number of third-party dependencies

Higher TLC = more structurally important and/or more exposed. It is a triage signal, not a quality metric.

## Cargo workspace tools

These subcommands use `cargo metadata` and are specific to Rust/Cargo workspaces.

| Command | What it does |
|---------|-------------|
| `sweep-local` | Run pkgrank across a local multi-repo workspace, write per-repo artifacts |
| `view` | One-shot HTML + JSON snapshot (local sweep + optional crates.io crawl) |
| `triage` | Artifact-backed triage bundle (same payload as MCP `pkgrank_triage`) |
| `cratesio` | Build a crates.io dependency graph and rank it |

### Module-level analysis (Rust only)

`pkgrank modules` shells out to [`cargo-modules`](https://github.com/regexident/cargo-modules) and ranks items by coupling. This is **intra-package** analysis: which modules, types, or traits inside a single crate are the coupling hotspots?

```bash
cargo install cargo-modules
pkgrank modules --manifest-path ../Cargo.toml -p walk --lib -n 25
pkgrank modules-sweep --manifest-path ../Cargo.toml -p walk -p innr --lib
```

CLI defaults include types + traits (functions hidden). MCP defaults are more conservative; use `preset` like `file-api` or `file-full` for the CLI-like view.

## JSON output shape (stable wrapper)

For commands that support `--format json`, the JSON is wrapped for forwards-compatible parsing:

```json
{
  "schema_version": 1,
  "ok": true,
  "command": "analyze|modules|modules-sweep|cratesio",
  "rows": [ /* ... */ ]
}
```

## Auto-JSON when piped

When stdout is not a TTY, output defaults to JSON. This makes pkgrank composable with `jq` without requiring `--format json`.

## MCP stdio server

`pkgrank mcp-stdio` runs an MCP server over stdio for integration with Cursor and other editors.

Toolset selection:
- Default: **slim** (small tool surface)
- `PKGRANK_MCP_TOOLSET=full`: advanced tools (module/type graph, polyglot, files)
- `PKGRANK_MCP_TOOLSET=debug`: internal artifact-inspection tools

## Configurable invariant rules

Cross-axis dependency rules are loaded from `dev_repos_overview.json` (under `--root` at `evals/arch/dev_repos_overview.json`). Add a `forbidden_edges` array:

```json
{
  "axes": { "core": ["libfoo", "libbar"], "apps": ["myapp"] },
  "forbidden_edges": [
    { "from": "core", "to": "apps" }
  ]
}
```

## Tests

- Default test suite is offline/deterministic, uses local targets.
- URL-backed tests (crates.io crawl) require `PKGRANK_E2E_NETWORK=1`.

## Non-goals

- **Security / advisory analysis**: use `cargo audit` or `cargo deny`.
- **Graph visualization**: output is ranked tables and JSON. Use `cargo-depgraph` or Graphviz.
- **Circular dependency breaking**: cycles are detected but no suggestions for breaking them.
- **License compliance**: no license analysis.
- **Build / test / deploy**: pkgrank analyzes structure, not execution.

## Dependencies

- Centrality algorithms delegated to [`graphops`](https://crates.io/crates/graphops) (PageRank / PPR / betweenness / reachability).
