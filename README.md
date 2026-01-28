## pkgrank

`pkgrank` ranks nodes in a Cargo dependency graph using centrality metrics.

Graph model:

- Nodes are Cargo packages (from `cargo metadata`).
- Directed edges are $A \to B$ iff **crate A depends on crate B**.

Interpretation:

- PageRank on the depends-on graph tends to surface **shared dependencies / “substrate” crates**.
- To surface **top-level orchestrators / consumers**, use the “consumer PageRank” (PageRank on the reversed graph).

### Usage (local crate graph)

Analyze the current directory (finds `Cargo.toml` if present):

```bash
cargo run -p pkgrank -- -n 25
```

Pick the “top-level orchestrators” view:

```bash
cargo run -p pkgrank -- --metric consumers-pagerank -n 25
```

Bound JSON output explicitly:

```bash
cargo run -p pkgrank -- analyze --format json --json-limit 200
```

Write per-repo artifacts under `evals/pkgrank/` (super-workspace mode):

```bash
cargo run -p pkgrank -- sweep-local --root . --out evals/pkgrank --mode workspace-slice -n 10
```

Triage (artifact-backed summary, same payload as MCP `pkgrank_triage`):

```bash
cargo run -p pkgrank -- triage --root . --out evals/pkgrank
```

### JSON output shape (stable wrapper)

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

### Usage (module/item graph via cargo-modules)

`pkgrank modules` shells out to [`cargo-modules`](https://github.com/regexident/cargo-modules) and parses its DOT output.

Install once:

```bash
cargo install cargo-modules
```

Defaults are tuned for a “fast, actionable hotspot scan”:

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
cargo run -p pkgrank -- modules --manifest-path ../Cargo.toml -p walk --lib -n 25
```

Workspace sweep (summary-only):

```bash
cargo run -p pkgrank -- modules-sweep --manifest-path ../Cargo.toml -p walk -p innr --lib
```

Use presets when you want a different “view” quickly:

```bash
# Item-level view, more verbose
cargo run -p pkgrank -- modules --manifest-path ../Cargo.toml -p walk --lib --preset node-full -n 25
```

Failure semantics:

- Default: **continue on error** and report which packages failed.
- `--fail-fast`: stop on first failure.
- `--continue-on-error=false`: equivalent explicit form.

Caching:

- `modules`/`modules-sweep` cache `cargo modules dependencies` DOT output under `evals/pkgrank/modules_cache/`.
- Use `--cache-refresh` to force regeneration.

### MCP stdio server (Cursor)

`pkgrank mcp-stdio` runs an MCP server over stdio. Stdout is reserved for JSON-RPC frames.

Run:

```bash
cargo run -p pkgrank -- mcp-stdio
```

Toolset selection (optional):

- Default: **slim** (small tool surface; “just works” for Cursor)
- Opt-in:
  - `PKGRANK_MCP_TOOLSET=full` to expose advanced tools (e.g. module/type graph centrality)
  - `PKGRANK_MCP_TOOLSET=debug` to also expose internal artifact-inspection tools

Environment (optional):

- `PKGRANK_ROOT`: default root directory for artifact-backed tools
- `PKGRANK_OUT`: default artifacts directory (default `evals/pkgrank`)

Tools (high level):

- Default (Cursor MCP): `pkgrank_view`, `pkgrank_triage`, `pkgrank_analyze`, `pkgrank_repo_detail`, `pkgrank_crate_detail`, `pkgrank_snapshot`, `pkgrank_compare_runs`
- Advanced (opt-in: `PKGRANK_MCP_TOOLSET=full`): `pkgrank_status`, `pkgrank_modules`, `pkgrank_modules_sweep`
- Debug (opt-in: `PKGRANK_MCP_TOOLSET=debug`): internal artifact-inspection tools (e.g. TLC tables, invariants list, PPR summaries)

### Tests (E2E targets)

- Default test suite is **offline/deterministic** and uses **local real targets** (the dev super-workspace itself).
- URL-backed tests (crates.io crawl) are **opt-in**:
  - set `PKGRANK_E2E_NETWORK=1` before running tests.

### Invariants (must not drift)

- Edge meaning: $A \to B$ means “A depends on B”.
- Dependency kind gating: `--dev` / `--build` control whether those edges exist.
- Workspace restriction: “workspace-only” means nodes/edges restricted to the current Cargo workspace members.

### User stories (what this is for)

These are the “real” workflows this tool is meant to serve.

- **Onboarding / orientation**: “What are the most central crates in this workspace? Who are the orchestrators?”
  - Use: `pkgrank analyze` (metric `pagerank` vs `consumers-pagerank`) and `pkgrank triage`.
- **Dependency slimming / graph sanity**: “Why is this crate so central / so sticky? What depends on it?”
  - Use: `pkgrank analyze --metric consumers-pagerank` + drill into origins and degrees; optionally generate artifacts via `pkgrank view`.
- **Refactor hotspots inside a crate**: “Which files/modules/items are the coupling hotspots?”
  - Use: `pkgrank modules` with `--aggregate file` (hot files) or `--aggregate node` (hot items).
- **Workspace sweep**: “Run that hotspot scan across a bunch of crates and summarize failures/results.”
  - Use: `pkgrank modules-sweep` (summary-only by default).
- **Shareable artifacts**: “Write an HTML snapshot I can point people at.”
  - Use: `pkgrank view` / `pkgrank sweep-local`.

Evidence pointers (qualitative):

- Community demand for “internal dependency graph visualization” (rust-analyzer crate graph is commonly suggested): `https://www.reddit.com/r/rust/comments/x04zko/visualizing_internal_dependencies_as_a_graph/`
- Cursor’s framing of MCP: “connect to external tools and data sources” (so MCP tends to be used for “triage + drilldown”): `https://cursor.com/docs/context/mcp`
- Cargo’s own `cargo tree` docs emphasize dependency display, reverse-deps (`--invert`), and “duplicates” as a common pain point: `https://doc.rust-lang.org/cargo/commands/cargo-tree.html`

### Dependencies / integration notes

- `pkgrank` computes centralities via the local `graphops/` crate (a path dependency in this workspace).

