# pkgrank

`pkgrank` ranks nodes in a dependency graph using centrality metrics.

Two modes: **inter-package** (which crates/packages in your dependency tree are most central?) and **intra-project** (which files in your codebase are structurally central, hotspots, or forming cycles?)

Supports Cargo, npm, Python, and Go. Works on local paths or GitHub URLs.

## Install

```bash
cargo install pkgrank
pkgrank --help
```

## Two axes of analysis

pkgrank answers two structurally different questions:

- **Inter-package centrality** (`analyze`, `sweep-local`, `triage`, `cratesio`): which packages in a workspace are most central, most depended-on, most risky to change?
- **Intra-project file centrality** (`files`): which files in your codebase are structurally central? Includes cycle detection, blast radius, churn risk from git history, and orphan detection. Polyglot (Rust/Python/JS/Go), works on any GitHub URL.
- **Intra-package item centrality** (`modules`, `modules-sweep`): which modules or items *inside* a Rust package are the coupling hotspots? (Requires `cargo-modules`.)

Both use the same metrics (PageRank, consumer PageRank, betweenness, degree) applied to different graphs.

## TL;DR

```bash
# Inter-package: rank local crates by importance (PageRank)
pkgrank -n 10

# Inter-package: rank by "who consumes this?" (Consumer PageRank)
pkgrank --metric consumers-pagerank -n 10

# Blast radius: what breaks if serde changes?
pkgrank blast-radius serde --workspace-only=false

# Upgrade priority: which outdated deps matter most?
pkgrank upgrade-priority

# File-level: structural hotspots in any project (no toolchain needed)
pkgrank files .
pkgrank files tokio-rs/tokio
pkgrank files https://github.com/fastapi/fastapi --ecosystem python

# File-level with git churn risk overlay
pkgrank files . --git

# Focus on a specific file
pkgrank files . --focus main.rs

# Directory-level aggregation for large codebases
pkgrank files astral-sh/ruff --directory

# Intra-package: item-level coupling hotspots (Rust only, needs cargo-modules)
pkgrank modules --manifest-path ../Cargo.toml -p walk --lib -n 25

# Polyglot: analyze an npm project
pkgrank polyglot --ecosystem npm path/to/project

# Polyglot: analyze a Python project (uv.lock or pyproject.toml)
pkgrank polyglot --ecosystem python path/to/project
```

## Graph model

- Nodes are Cargo packages (from `cargo metadata`).
- Directed edges are $A \to B$ iff **crate A depends on crate B**.

## Interpretation

- PageRank on the depends-on graph tends to surface **shared dependencies / "substrate" crates**.
- To surface **top-level orchestrators / consumers**, use the "consumer PageRank" (PageRank on the reversed graph).

## Scoring: TLC (Top-Level Cost)

The `triage` and `view` commands produce a **TLC score** for each crate and repo. TLC is a composite heuristic that combines:

- **Blast radius**: `10 * ln(transitive_dependents + 1)` -- how many things break if this changes
- **Centrality**: `1000 * pagerank` -- structural importance in the dependency graph
- **Boundary complexity**: number of third-party dependencies -- surface area exposed to external changes

Higher TLC = more structurally important and/or more exposed. It is a triage signal, not a quality metric.

## Usage (inter-package: local crate graph)

Analyze the current directory (finds `Cargo.toml` if present):

```bash
cargo run -- -n 25
```

Pick the "top-level orchestrators" view:

```bash
cargo run -- --metric consumers-pagerank -n 25
```

Bound JSON output explicitly:

```bash
cargo run -- analyze --format json --json-limit 200
```

Write per-repo artifacts under `evals/pkgrank/` (super-workspace mode):

```bash
cargo run -- sweep-local --root . --out evals/pkgrank --mode workspace-slice -n 10
```

Triage (artifact-backed summary, same payload as MCP `pkgrank_triage`):

```bash
cargo run -- triage --root . --out evals/pkgrank
```

## Blast radius

Show everything that transitively depends on a package:

```bash
pkgrank blast-radius serde --workspace-only=false -n 20
```

Output is sorted by BFS depth (closest dependents first), then by PageRank within each depth level. Useful for answering "what breaks if I upgrade this?" before reviewing Dependabot PRs.

## Upgrade priority

Combine `cargo outdated` with centrality ranking to prioritize which upgrades matter most:

```bash
pkgrank upgrade-priority -n 15
```

Requires [`cargo-outdated`](https://crates.io/crates/cargo-outdated) to be installed. Scores each outdated dep by `10*ln(dependents+1) + 1000*pagerank + urgency_bonus` where urgency is major/minor/patch.

## File-level analysis (`pkgrank files`)

Analyze the import graph within a project. Static parsing (no toolchain required). Works across Rust, Python, JS/TS, and Go.

```bash
# Any local project (ecosystem auto-detected)
pkgrank files .

# Any GitHub repo via URL or shorthand
pkgrank files tokio-rs/tokio
pkgrank files https://github.com/django/django --ecosystem python

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

Cross-project analysis via SQLite (auto-enabled):

```bash
# Query across all previously analyzed projects
pkgrank query hotspots          # highest churn risk files
pkgrank query deps              # most-used external deps
pkgrank query projects          # list all analyzed projects
pkgrank query "files lib.rs"    # search for files by name
pkgrank query compare           # diff between last two snapshots
pkgrank query drift             # centrality changes over time
```

For JS/TS projects, resolves tsconfig.json path aliases (`@/`, etc.) and npm workspace packages (`@scope/pkg`). Detects cross-language seams (PyO3, NAPI) between Rust and Python/JS.

## Polyglot analysis (npm, Python, Go)

Analyze dependency graphs from non-Cargo ecosystems:

```bash
# npm: uses package-lock.json if available, falls back to package.json (direct deps only)
pkgrank polyglot --ecosystem npm path/to/project

# Python: uses uv.lock if available, falls back to pyproject.toml (direct deps only)
pkgrank polyglot --ecosystem python path/to/project

# Go: runs `go mod graph` in the directory (or reads a pre-captured output file)
pkgrank polyglot --ecosystem go path/to/project
```

When only a manifest (no lock file) is available, the graph contains direct dependencies only with no transitive resolution. A note is printed to stderr.

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

`pkgrank analyze --format json` also includes explicit bounding metadata:

- `rows_total`: total rows computed
- `rows_returned`: rows included in `rows`
- `truncated`: whether `rows` was truncated
- `json_limit`: the applied limit (if any)

## Usage (intra-package: module/item graph via cargo-modules)

`pkgrank modules` shells out to [`cargo-modules`](https://github.com/regexident/cargo-modules) and parses its DOT output.

Install once:

```bash
cargo install cargo-modules
```

Defaults are tuned for a "fast, actionable hotspot scan":

- aggregate by **file**
- include **types + traits**
- hide functions / externs / sysroot
- show a few strongest edges
- cache `cargo-modules` DOT output

Note on **CLI vs MCP defaults**:

- The **CLI** `pkgrank modules` defaults include **types + traits** (and hide functions).
- The **MCP** `pkgrank_modules` tool is more conservative by default (hides fns/types/traits unless you opt in via `preset` or `include_*`), because MCP payloads are easy to blow up accidentally.
  - If you want the CLI-like view from MCP, pass a `preset` like `file-api` or `file-full`.

File-level hotspots (explicit, but these are now close to the defaults):

```bash
cargo run -- modules --manifest-path ../Cargo.toml -p walk --lib -n 25
```

Workspace sweep (summary-only):

```bash
cargo run -- modules-sweep --manifest-path ../Cargo.toml -p walk -p innr --lib
```

Use presets when you want a different "view" quickly:

```bash
# Item-level view, more verbose
cargo run -- modules --manifest-path ../Cargo.toml -p walk --lib --preset node-full -n 25
```

Failure semantics:

- Default: **continue on error** and report which packages failed.
- `--fail-fast`: stop on first failure.
- `--continue-on-error=false`: equivalent explicit form.

Caching:

- `modules`/`modules-sweep` cache `cargo modules dependencies` DOT output under `evals/pkgrank/modules_cache/`.
- Use `--cache-refresh` to force regeneration.

## MCP stdio server (Cursor)

`pkgrank mcp-stdio` runs an MCP server over stdio. Stdout is reserved for JSON-RPC frames.

Run:

```bash
cargo run -- mcp-stdio
```

Toolset selection (optional):

- Default: **slim** (small tool surface; "just works" for Cursor)
- Opt-in:
  - `PKGRANK_MCP_TOOLSET=full` to expose advanced tools (e.g. module/type graph centrality)
  - `PKGRANK_MCP_TOOLSET=debug` to also expose internal artifact-inspection tools

Environment (optional):

- `PKGRANK_ROOT`: default root directory for artifact-backed tools
- `PKGRANK_OUT`: default artifacts directory (default `evals/pkgrank`)

Tools (high level):

- Default (Cursor MCP): `pkgrank_view`, `pkgrank_triage`, `pkgrank_analyze`, `pkgrank_repo_detail`, `pkgrank_crate_detail`, `pkgrank_snapshot`, `pkgrank_compare_runs`, `pkgrank_blast_radius`
- Advanced (opt-in: `PKGRANK_MCP_TOOLSET=full`): `pkgrank_status`, `pkgrank_modules`, `pkgrank_modules_sweep`, `pkgrank_upgrade_priority`, `pkgrank_polyglot`, `pkgrank_files`
- Debug (opt-in: `PKGRANK_MCP_TOOLSET=debug`): internal artifact-inspection tools (e.g. TLC tables, invariants list, PPR summaries)

## Analysis caching

Pass `--cache` to cache analysis results under `evals/pkgrank/analysis_cache/`. Subsequent runs with the same parameters read from cache instead of re-running `cargo metadata` + graph computation:

```bash
pkgrank analyze --cache -n 10          # first run: computes + caches
pkgrank analyze --cache -n 10          # second run: reads from cache
pkgrank analyze --cache --cache-refresh # force recompute
```

Cache keys are derived from manifest path, workspace-only flag, dev/build dep inclusion, and feature flags.

The `modules` and `modules-sweep` commands cache `cargo-modules` DOT output separately under `evals/pkgrank/modules_cache/`.

## Auto-JSON when piped

When stdout is not a TTY (piped to another command or redirected to a file), output defaults to JSON instead of text. This makes pkgrank composable with `jq` and other tools without requiring `--format json`.

## Configurable invariant rules

Cross-axis dependency rules are loaded from `dev_repos_overview.json` (under the `--root` directory at `evals/arch/dev_repos_overview.json`). Add a `forbidden_edges` array to define which axis-to-axis dependencies are violations:

```json
{
  "axes": { "core": ["libfoo", "libbar"], "apps": ["myapp"] },
  "forbidden_edges": [
    { "from": "core", "to": "apps" }
  ]
}
```

If no `forbidden_edges` key is present, no invariant violations are reported.

## Tests (E2E targets)

- Default test suite is **offline/deterministic** and uses **local real targets** (the dev super-workspace itself).
- URL-backed tests (crates.io crawl) are **opt-in**:
  - set `PKGRANK_E2E_NETWORK=1` before running tests.

## Invariants (must not drift)

- Edge meaning: $A \to B$ means "A depends on B".
- Dependency kind gating: `--dev` / `--build` control whether those edges exist.
- Workspace restriction: "workspace-only" means nodes/edges restricted to the current Cargo workspace members.

## Non-goals

- **Security / advisory analysis**: no CVE, advisory, or vulnerability integration. Use `cargo audit` or `cargo deny`.
- **Graph visualization**: output is ranked tables and JSON, not rendered graph images. Use `cargo-depgraph` or Graphviz for visual graphs.
- **Circular dependency breaking**: `pkgrank files` detects cycles (via Tarjan's SCC) but does not suggest how to break them.
- **License compliance**: no license analysis or policy enforcement.
- **Build / test / deploy**: pkgrank analyzes structure; it does not execute builds or tests.

## User stories (what this is for)

- **Onboarding / orientation**: "What are the most central crates in this workspace?"
  - Use: `pkgrank analyze` and `pkgrank triage`.
- **Blast radius before upgrading**: "What breaks if I upgrade serde?"
  - Use: `pkgrank blast-radius serde --workspace-only=false`
- **Prioritized upgrades**: "I have 40 outdated deps; which 5 should I fix first?"
  - Use: `pkgrank upgrade-priority -n 5`
- **Dependency slimming**: "Why is this crate so central?"
  - Use: `pkgrank analyze --metric consumers-pagerank`
- **Refactor hotspots inside a crate**: "Which files are the coupling hotspots?"
  - Use: `pkgrank files .` or `pkgrank modules --aggregate file`
- **Quick architectural overview of any repo**: "What does this codebase look like structurally?"
  - Use: `pkgrank files owner/repo`
- **Find hidden coupling**: "Which files change together even without direct imports?"
  - Use: `pkgrank files . --git --focus myfile.rs`
- **Cycle detection**: "Does this project have circular dependencies?"
  - Use: `pkgrank files .` (cycles reported in summary)
- **Polyglot analysis**: "Rank my npm/Python/Go deps by centrality."
  - Use: `pkgrank polyglot --ecosystem npm .`
- **Shareable artifacts**: "Write an HTML snapshot I can point people at."
  - Use: `pkgrank view` / `pkgrank sweep-local`.

## Dependencies / integration notes

- `pkgrank` delegates centrality algorithms to [`graphops`](https://crates.io/crates/graphops) (PageRank / PPR / betweenness / reachability).
