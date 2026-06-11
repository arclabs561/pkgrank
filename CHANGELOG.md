# Changelog

## [0.5.0] - 2026-06-10

### Added

- `view --mode local` and `sweep-local --mode repo-roots` now work when `--root` is a directory of standalone repos with no root `Cargo.toml`: repo roots are auto-discovered (immediate subdirs with a manifest, plus one level under umbrella dirs), per-repo `cargo metadata` is merged into a virtual workspace, and cross-repo version deps are unified by name onto local workspace members.

### Changed

- `sweep-local` prints a one-line receipt (ok/failed/missing counts) instead of exiting silently when discovery finds nothing.

Earlier releases predate this changelog; see git history.
