## pkgrank

`pkgrank` ranks nodes in a Cargo dependency graph using centrality metrics.

Graph model:

- Nodes are Cargo packages (from `cargo metadata`).
- Directed edges are \(A \to B\) iff **crate A depends on crate B**.

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

Write per-repo artifacts under `evals/pkgrank/` (super-workspace mode):

```bash
cargo run -p pkgrank -- sweep-local --root . --out evals/pkgrank --mode workspace-slice -n 10
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

Environment (optional):

- `PKGRANK_ROOT`: default root directory for artifact-backed tools
- `PKGRANK_OUT`: default artifacts directory (default `evals/pkgrank`)

Tools (high level):

- Artifact-backed: `pkgrank_view`, `pkgrank_status`, `pkgrank_triage`, `pkgrank_repo_detail`, `pkgrank_crate_detail`, `pkgrank_snapshot`, `pkgrank_compare_runs`
- Direct compute: `pkgrank_analyze`, `pkgrank_modules`, `pkgrank_modules_sweep`

### Invariants (must not drift)

- Edge meaning: \(A \to B\) means “A depends on B”.
- Dependency kind gating: `--dev` / `--build` control whether those edges exist.
- Workspace restriction: “workspace-only” means nodes/edges restricted to the current Cargo workspace members.

### Dependencies / integration notes

- `pkgrank` intentionally avoids depending on private sibling repos.
  Centrality operators (PageRank / PPR / reachability / betweenness) come from `walk` (pulled via git).

- For the dev super-workspace, you can still iterate on a local `walk/` checkout without changing
  `pkgrank` by using a patch override:

```toml
# In your *dev super-workspace* root Cargo.toml:
[patch."https://github.com/arclabs561/walk"]
walk = { path = "walk" }
```

