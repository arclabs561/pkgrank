//! Cargo blast radius must distinguish different versions of one dependency.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::{fs, path::Path, process::Command};

fn write_package(dir: &Path, name: &str, version: &str, dependencies: &str) {
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(dir.join("src/lib.rs"), "").unwrap();
    fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = '{name}'\nversion = '{version}'\nedition = '2021'\n[dependencies]\n{dependencies}"
        ),
    )
    .unwrap();
}

#[test]
fn cargo_blast_radius_requires_a_version_when_ambiguous() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("project");
    write_package(&dir.path().join("v1"), "shared", "1.0.0", "");
    write_package(&dir.path().join("v2"), "shared", "2.0.0", "");
    write_package(
        &workspace.join("one"),
        "one",
        "0.1.0",
        "shared = { path = '../../v1' }",
    );
    write_package(
        &workspace.join("two"),
        "two",
        "0.1.0",
        "shared = { path = '../../v2' }",
    );
    fs::write(
        workspace.join("Cargo.toml"),
        "[workspace]\nmembers = ['one', 'two']\nresolver = '2'",
    )
    .unwrap();
    let blast = |target: &str| {
        Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"))
            .args(["blast-radius", target])
            .arg(&workspace)
            .args(["--workspace-only=false", "--format", "json"])
            .env("CARGO_NET_OFFLINE", "true")
            .assert()
    };
    blast("shared")
        .failure()
        .stderr(predicate::str::contains("shared@1.0.0, shared@2.0.0"));
    blast("shared@1.0.0")
        .success()
        .stdout(predicate::str::contains("\"name\": \"one\""))
        .stdout(predicate::str::contains("\"name\": \"two\"").not());
    blast("shared@2.0.0")
        .success()
        .stdout(predicate::str::contains("\"name\": \"two\""))
        .stdout(predicate::str::contains("\"name\": \"one\"").not());
}
