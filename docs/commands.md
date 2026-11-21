# pkgrank commands

This page covers the commands that are useful after the basic file analysis,
package analysis, and `blast-radius` workflows in the README.

## File Analysis

Analyze a local project:

```bash
pkgrank files .
```

Analyze a remote repo:

```bash
pkgrank files tokio-rs/tokio
pkgrank files https://github.com/fastapi/fastapi
pkgrank files gl:inkscape/inkscape
pkgrank files cb:forgejo/forgejo
pkgrank files sh:~sircmpwn/aerc
pkgrank files bb:pypy/pypy
```

Useful options:

```bash
pkgrank files . --git
pkgrank files . --focus src/main.rs
pkgrank files . --directory
pkgrank files . --include-tests
pkgrank files . --cache
pkgrank files . --fail-on-violation
pkgrank files . --affected src/parser.rs
git diff --name-only | pkgrank files . --affected -
```

When output is piped, `files` writes one JSON object. With `--affected`,
`.changed` lists the inputs and `.affected` lists their transitive dependents,
excluding the inputs themselves. An empty result has `.affected_count == 0`
and `.affected == []`. With `--focus`, `.focus` contains the matching file
rows with their `imports` and `imported_by` paths. Both options work together.
For example:

```bash
pkgrank files . --affected src/parser.rs | jq -r '.affected[]'
```

A test in `.affected` has an import path to a changed file. Whether its
assertions catch a bug is a separate question. In `durability`, changing
`src/recordlog.rs` flagged all 38 existing integration test files. Their tests
passed with the bug present. Four new regression cases failed on that revision
and passed with the [torn-tail fix](https://github.com/arclabs561/durability/commit/4bdcdc54287fd69b99a4c027f0f9e4b31e3ee072).

Rows include structural role, PageRank, consumer PageRank, betweenness,
instability, blast radius, cycle membership, external dependencies, and optional
git churn fields.

## Package Analysis

Rank packages in a dependency graph:

```bash
pkgrank
pkgrank path/to/npm-project
pkgrank --ecosystem js path/to/project
pkgrank --ecosystem python path/to/project
pkgrank --ecosystem go path/to/project
pkgrank --metric consumers-pagerank -n 10
pkgrank --format json --json-limit 200
```

The graph edge `A -> B` means package A depends on package B.

- PageRank on the depends-on graph surfaces shared dependencies.
- Consumer PageRank on the reversed graph surfaces top-level consumers.

## Blast Radius

Show packages that transitively depend on a package:

```bash
pkgrank blast-radius my_crate
pkgrank blast-radius serde --workspace-only=false
pkgrank blast-radius express path/to/npm-project
pkgrank blast-radius serde --workspace-only=false -n 20
```

Output is sorted by BFS depth, then by PageRank within each depth.

## Upgrade Priority

For Cargo projects, combine `cargo outdated` with centrality:

```bash
pkgrank upgrade-priority -n 15
pkgrank upgrade-priority --format json | jq '.rows[:5]'
```

Requires `cargo-outdated`.

## Architecture Rules

`pkgrank files` can check layer rules from `.pkgrank.toml`:

```toml
[layers]
domain = ["src/domain/**", "src/models/**"]
infra = ["src/infra/**", "src/db/**"]
api = ["src/api/**", "src/routes/**"]

[[deny]]
from = "domain"
to = "infra"

[[allow]]
from = "domain"
to = ["domain"]
```

`[[deny]]` forbids specific imports. `[[allow]]` makes the listed targets the
only allowed imports for a layer. Same-layer imports are allowed.

Run:

```bash
pkgrank files . --fail-on-violation
```

## Stored Queries

`pkgrank files` stores snapshots to SQLite by default. To make that explicit:

```bash
pkgrank files . --store true
```

Query stored snapshots with:

```bash
pkgrank query hotspots
pkgrank query deps
pkgrank query projects
pkgrank query "files lib.rs"
pkgrank query compare
pkgrank query drift
```

## Workspace Commands

These commands use Cargo metadata and are intended for multi-repo or
multi-crate analysis:

| Command | What it writes |
| --- | --- |
| `sweep-local` | Per-repo artifacts for a directory of repos |
| `view` | HTML and JSON snapshot from local metadata and optional crates.io crawl |
| `triage` | Artifact-backed triage bundle |
| `cratesio` | Bounded crates.io dependency crawl |

The `view` and `triage` commands use a TLC score:

- blast radius: `10 * ln(transitive_dependents + 1)`
- centrality: `1000 * pagerank`
- boundary complexity: number of third-party dependencies

TLC is a triage signal, not a quality metric.

## Module Analysis

`pkgrank modules` shells out to `cargo modules dependencies` and ranks Rust
modules, types, traits, and functions by coupling:

```bash
cargo install cargo-modules
pkgrank modules -p my_crate --lib -n 25
pkgrank modules-sweep -p crate_a -p crate_b --lib
```

## MCP Server

Run the MCP stdio server:

```bash
pkgrank mcp-stdio
```

Toolset selection:

- default: slim tool surface
- `PKGRANK_MCP_TOOLSET=full`: module graph, file analysis, and ecosystem tools
- `PKGRANK_MCP_TOOLSET=debug`: internal artifact-inspection tools
