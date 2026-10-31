# Changelog

## [Unreleased]

### Fixed

- `files` on a remote URL refreshes an earlier clone of that URL to the remote's current HEAD (`git fetch` + `reset --hard`) instead of reusing it unchanged, so repeat runs no longer analyze a stale checkout. The analyzed commit is reported as `source_commit` in JSON output and `commit=` in the text header.
- A `.pkgrank.toml` that cannot be read or parsed, or that names an invalid layer glob, now fails `files` with an error naming the file (and the glob). It was previously ignored, which silently disabled rule enforcement, including under `--fail-on-violation`.

### Changed

- `polyglot` and `blast-radius` key npm (`package-lock.json`), uv (`uv.lock`) and Go (`go mod graph`) packages by name and version, as the Cargo graph does by package id. Two versions of one package are now two rows, and each edge points at the version the dependent resolves to (npm: nearest `node_modules`; uv: the dependency's locked version). They were previously merged into one node under the first-seen version. `blast-radius` accepts `name@version`; a bare name that matches several versions is an error listing them.

## [0.5.2] - 2026-10-30

### Fixed

- Python relative imports in a package's `__init__.py` resolve against that package: `from .core import f` in `pkg/__init__.py` now links to `pkg/core.py`. They previously resolved one package level too high, so the edge was dropped or pointed at a sibling of the package.

## [0.5.1] - 2026-06-11

### Fixed

- Repos skipped during the standalone-repos merge (cargo metadata failures) were stderr-only and invisible in artifacts; `merged.skipped.json` receipt now written unconditionally in merged mode and skips fold into `sweep.summary.json`'s failed list.
- `analyze` paths (MCP, upgrade-priority) now route through the merged-metadata dispatcher and work on rootless super-roots; analysis cache bypassed in merged mode (its key derives from a manifest that doesn't exist).
- Top-level repo discovery follows symlinked dirs, consistent with the umbrella level.

### Added

- Merge anomaly reporting: semver-incompatible version unifications and duplicate local member names are warned once and recorded in the receipt.
- Merged metadata memoized per process (was rebuilt 3x per view invocation).

## [0.5.0] - 2026-06-10

### Added

- `view --mode local` and `sweep-local --mode repo-roots` now work when `--root` is a directory of standalone repos with no root `Cargo.toml`: repo roots are auto-discovered (immediate subdirs with a manifest, plus one level under umbrella dirs), per-repo `cargo metadata` is merged into a virtual workspace, and cross-repo version deps are unified by name onto local workspace members.

### Changed

- `sweep-local` prints a one-line receipt (ok/failed/missing counts) instead of exiting silently when discovery finds nothing.

Earlier releases predate this changelog; see git history.
