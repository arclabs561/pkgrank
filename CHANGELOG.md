# Changelog

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
