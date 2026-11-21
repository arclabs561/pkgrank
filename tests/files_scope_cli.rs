use assert_cmd::prelude::*;
use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};

fn write(root: &Path, path: &str, content: &str) {
    let path = root.join(path);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn git(root: &Path, args: &[&str]) {
    assert!(Command::new("git")
        .current_dir(root)
        .args(args)
        .status()
        .unwrap()
        .success());
}

fn scan(root: &Path, options: &[&str]) -> assert_cmd::assert::Assert {
    Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"))
        .current_dir(root)
        .args(["files", ".", "--store", "false"])
        .args(options)
        .assert()
}

fn report(root: &Path, options: &[&str]) -> Value {
    serde_json::from_slice(&scan(root, options).success().get_output().stdout).unwrap()
}

#[test]
fn reports_only_outer_tracked_projects_in_unscanned_ecosystems() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    git(root, &["init", "-q"]);
    write(root, "pyproject.toml", "");
    write(root, "main.py", "print(1)");
    for path in [
        "rust/Cargo.toml",
        "rust/member/Cargo.toml",
        "rust/src/lib.rs",
        "rust/ui/package.json",
        "rust/ui/package-lock.json",
        "cli/go.mod",
        "tools/pyproject.toml",
        "testdata/go.mod",
        "vendor/skip/go.mod",
        "fixtures/skip/Cargo.toml",
        "docs/site/deno.json",
        "other/fuzz/Cargo.toml",
    ] {
        write(root, path, "");
    }
    git(root, &["add", "."]);
    write(root, "untracked/Cargo.toml", "");
    let data = report(root, &["--format", "json"]);
    assert_eq!(data["nodes"], 1);
    assert_eq!(
        data["unscanned_projects"],
        json!([
            {"ecosystem":"go","manifest":"cli/go.mod"},
            {"ecosystem":"rust","manifest":"rust/Cargo.toml"},
            {"ecosystem":"js","manifest":"rust/ui/package.json"},
        ])
    );
    let forced = report(root, &["--format", "json", "--ecosystem", "python"]);
    assert!(forced.get("unscanned_projects").is_none());
    let nested = report(&root.join("rust"), &["--format", "json"]);
    assert_eq!(nested["nodes"], 1);
    assert_eq!(
        nested["unscanned_projects"],
        json!([{"ecosystem":"js","manifest":"ui/package.json"}])
    );
    let text = scan(root, &["--format", "text"]).success();
    assert!(String::from_utf8_lossy(&text.get_output().stdout).contains("pkgrank files './rust'"));
}

#[test]
fn cached_graph_reports_new_and_deleted_manifests_without_source_changes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    git(root, &["init", "-q"]);
    write(root, "pyproject.toml", "");
    write(root, "main.py", "print(1)");
    git(root, &["add", "."]);
    let options = ["--format", "json", "--cache"];
    assert!(report(root, &options).get("unscanned_projects").is_none());
    write(root, "team's crate/Cargo.toml", "");
    git(root, &["add", "team's crate/Cargo.toml"]);
    assert_eq!(
        report(root, &options)["unscanned_projects"][0]["manifest"],
        "team's crate/Cargo.toml"
    );
    let text = scan(root, &["--format", "text", "--cache"]).success();
    assert!(String::from_utf8_lossy(&text.get_output().stdout)
        .contains("pkgrank files './team'\\''s crate'"));
    fs::remove_file(root.join("team's crate/Cargo.toml")).unwrap();
    assert!(report(root, &options).get("unscanned_projects").is_none());
}

#[test]
fn an_empty_root_points_to_its_nested_project() {
    let tmp = tempfile::tempdir().unwrap();
    git(tmp.path(), &["init", "-q"]);
    write(tmp.path(), "nested/deno.json", "{}");
    git(tmp.path(), &["add", "."]);
    let out = scan(tmp.path(), &["--format", "text"]).failure();
    assert!(String::from_utf8_lossy(&out.get_output().stderr).contains("pkgrank files './nested'"));
}
