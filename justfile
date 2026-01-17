default:
    @just --list

check:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    cargo test

test:
    cargo test

fmt:
    cargo fmt --all
