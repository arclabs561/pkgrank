//! `blast-radius` on a lockfile with two versions of one package.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

fn npm_project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("package-lock.json"),
        r#"{"lockfileVersion": 3, "packages": {
            "": {"name": "root", "dependencies": {"a": "^1", "b": "^2"}},
            "node_modules/a": {"version": "1.0.0", "dependencies": {"b": "^1"}},
            "node_modules/a/node_modules/b": {"version": "1.0.0"},
            "node_modules/b": {"version": "2.0.0"},
            "node_modules/c": {"version": "1.0.0", "dependencies": {"b": "^2"}}
        }}"#,
    )
    .unwrap();
    dir
}

fn blast(dir: &tempfile::TempDir, package: &str) -> assert_cmd::assert::Assert {
    Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"))
        .args(["blast-radius", package])
        .arg(dir.path())
        .args(["--ecosystem", "npm", "--format", "json"])
        .assert()
}

#[test]
fn versioned_target_reports_only_its_own_dependents() {
    let dir = npm_project();
    blast(&dir, "b@1.0.0")
        .success()
        .stdout(predicate::str::contains("\"found\": true"))
        .stdout(predicate::str::contains(
            "\"total_transitive_dependents\": 1",
        ))
        .stdout(predicate::str::contains("\"name\": \"a\""))
        .stdout(predicate::str::contains("\"name\": \"c\"").not());
}

#[test]
fn unversioned_name_resolves_when_unique() {
    let dir = npm_project();
    blast(&dir, "a")
        .success()
        .stdout(predicate::str::contains("\"found\": true"));
}

#[test]
fn ambiguous_name_asks_for_a_version() {
    let dir = npm_project();
    blast(&dir, "b")
        .failure()
        .stderr(predicate::str::contains("b@1.0.0"))
        .stderr(predicate::str::contains("b@2.0.0"));
}
