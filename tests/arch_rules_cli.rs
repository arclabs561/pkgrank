//! A malformed `.pkgrank.toml` must fail the run instead of silently
//! disabling rule enforcement.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

fn project(rules: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/lib.rs"), "mod a;\n").unwrap();
    std::fs::write(dir.path().join("src/a.rs"), "pub fn f() {}\n").unwrap();
    std::fs::write(dir.path().join(".pkgrank.toml"), rules).unwrap();
    dir
}

fn run(dir: &tempfile::TempDir, format: &str) -> assert_cmd::assert::Assert {
    Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"))
        .args(["files", "--store", "false", "--format", format])
        .arg(dir.path())
        .assert()
}

#[test]
fn unparseable_rules_file_fails_the_run() {
    let dir = project("layers = [not toml\n");
    for format in ["json", "text"] {
        run(&dir, format)
            .failure()
            .stderr(predicate::str::contains(".pkgrank.toml"));
    }
}

#[test]
fn invalid_layer_glob_fails_the_run() {
    let dir = project("[layers]\ncore = [\"src/[a.rs\"]\n");
    for format in ["json", "text"] {
        run(&dir, format)
            .failure()
            .stderr(predicate::str::contains(".pkgrank.toml"))
            .stderr(predicate::str::contains("src/[a.rs"));
    }
}

#[test]
fn valid_rules_file_still_runs() {
    let dir = project("[layers]\ncore = [\"src/*.rs\"]\n");
    run(&dir, "json").success();
}
