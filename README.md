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

To try a small graph from an empty directory, create a Python package with
one import. The scan runs without installing Python dependencies.

```sh
mkdir -p demo/sample
printf '[project]\nname = "sample"\nversion = "0.1.0"\n' > demo/pyproject.toml
touch demo/sample/__init__.py
printf 'def hello(): return "hello"\n' > demo/sample/core.py
printf 'from sample.core import hello\n' > demo/sample/cli.py
pkgrank files demo --format json --store false
```

The report has `"nodes": 3` and `"edges": 1`. In `rows`, `sample/cli.py`
has one dependency and `sample/core.py` has one dependent. The empty
`sample/__init__.py` has neither.

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
| `pkgrank mcp-stdio` | Runs an MCP server on stdio (default `stdio` feature) |

`files` supports Rust, Python, JS/TS/Svelte/Vue, and Go. It respects
`.gitignore` through `git ls-files` for git repos, falls back to a filtered walk
for non-git directories, and excludes generated files, fixtures, vendor trees,
and docs by default.

More commands and configuration are documented in
[`docs/commands.md`](docs/commands.md).

## Output

The file rankings, package rankings, and blast-radius commands print text in a
terminal and JSON when piped. Use `--format text` or `--format json` to select
the format explicitly. Their JSON reports include `schema_version`, `ok`,
`command`, and `rows`.

`pkgrank files` also reports cycle counts, orphan counts, layer violations, and
rule violations. It selects ecosystems from manifests at the scan root. In a
Git checkout with no detected root or nested manifest, it uses tracked source
files in the selected scope to identify supported languages. If a
tracked nested project uses another ecosystem, text output gives a command to
scan it from that root. JSON lists its `ecosystem` and relative `manifest` in
`unscanned_projects`. An explicit `--ecosystem` suppresses these hints.

By default it stores each run in a local SQLite database for
`pkgrank query`; pass `--store false` to skip that.

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
