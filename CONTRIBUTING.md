# Contributing to pkgrank

Thanks for your interest. pkgrank is centrality analysis for dependency graphs and file-level import graphs.

## Before you start

For non-trivial work (new APIs, features, large refactors), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Rust toolchain: stable, MSRV `1.80`. Use `rustup` to manage.
- Optional: `cargo-nextest` for faster test runs (`cargo install cargo-nextest`).

```
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Style

- Direct, lowercase prose in commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Commit messages: `pkgrank: short lowercase description`. One commit per logical change.
- `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings` must pass before `git add`.

## Testing

- `cargo test --all-features` for the full matrix.
- Test names should describe the property under test, not the function under test.

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes.
- Link the related issue.
- CI must be green before requesting review.

## Releasing

Open [Publish pkgrank](https://github.com/arclabs561/pkgrank/actions/workflows/publish.yml)
and select **Run workflow** on `main`. Leave the box empty for a dry run.
Check it to test the version in `Cargo.toml`, tag that commit, publish the crate,
and build the GitHub Release with its installers. Creating an existing tag fails.
If a job fails after the tag was created, use **Re-run failed jobs** on that run.

Pushing a matching version tag with Git also starts both pipelines.

If the crate is already published and the GitHub Release is missing, open
[Release installers](https://github.com/arclabs561/pkgrank/actions/workflows/release.yml).
Run it from `main` and enter the existing tag (for example, `v0.6.0`).
It checks the tag against the commit recorded in the published crate and builds
all five binary archives and both installers from that commit.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
