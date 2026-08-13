# pkgrank

Centrality analysis for dependency graphs and file-level import graphs.

`pkgrank` ranks source files and packages by graph position, so you can see
which files are structural hubs, which packages have large blast radius, and
where dependency cycles exist.

## Install

Prebuilt binaries:

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/arclabs561/pkgrank/releases/latest/download/pkgrank-installer.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://github.com/arclabs561/pkgrank/releases/latest/download/pkgrank-installer.ps1 | iex"
```

Or install from crates.io:

```bash
cargo install pkgrank
```

## Usage

Rank files in the current repo:

```bash
pkgrank files .
```

When stdout is piped, output is JSON:

```json
{
  "command": "files",
  "ecosystem": "rust",
  "nodes": 10,
  "cycle_count": 1,
  "rows": [
    {
      "file": "src/main.rs",
      "structure": "hub",
      "dependents": 8,
      "dependencies": 9,
      "pagerank": 0.34444790840941036
    },
    {
      "file": "src/dep_graph.rs",
      "structure": "foundation",
      "dependents": 9,
      "dependencies": 0,
      "pagerank": 0.13405667652537667
    }
  ]
}
```

Rank packages instead of files:

```bash
pkgrank
pkgrank path/to/project
pkgrank --ecosystem python path/to/project
```

Check blast radius for a package:

```bash
pkgrank blast-radius my_crate
pkgrank blast-radius serde --workspace-only=false
pkgrank blast-radius express path/to/npm-project
```

## Commands

| Command | What it ranks |
| --- | --- |
| `pkgrank files <path-or-repo>` | Source files in a project import graph |
| `pkgrank [path]` | Packages in a dependency graph |
| `pkgrank blast-radius <package>` | Transitive dependents of one package |

`files` supports Rust, Python, JS/TS/Svelte/Vue, and Go. It respects
`.gitignore` through `git ls-files` for git repos, falls back to a filtered walk
for non-git directories, and excludes generated files, fixtures, vendor trees,
and docs by default.

More commands and configuration are documented in
[`docs/commands.md`](docs/commands.md).

## Output

TTY output is text. Piped output is JSON unless `--format text` is set.

The JSON wrapper is stable:

```json
{
  "schema_version": 1,
  "ok": true,
  "command": "files",
  "rows": []
}
```

`pkgrank files` also reports cycle counts, orphan counts, layer violations, and
rule violations.

## Limits

- `pkgrank` analyzes dependency structure, not runtime behavior.
- It detects cycles and layer violations; it does not suggest how to break them.
- It does not perform license or security advisory analysis.
- Go import resolution uses `go list -json` when available and static parsing
  otherwise.

## Tests

```bash
cargo test
```

URL-backed regression tests are opt-in with `PKGRANK_E2E_NETWORK=1`.

## License

MIT OR Apache-2.0
