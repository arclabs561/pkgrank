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

#[test]
fn linking_a_workspace_preserves_its_package_and_dependents() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = serde_json::json!({"version": "1.0.0", "dependencies": {"leaf": "1.0.0"}});
    let installed = serde_json::json!({
        "": {"name": "root"},
        "node_modules/app": {"version": "1.0.0", "dependencies": {"bridge": "1.0.0"}},
        "node_modules/bridge": bridge,
        "node_modules/leaf": {"version": "1.0.0"}
    });

    // npm omits the name on a workspace entry if it matches the directory.
    for target in [
        None,
        Some("packages/bridge"),
        Some("packages/implementation"),
    ] {
        let mut packages = installed.clone();
        if let Some(target) = target {
            packages["node_modules/bridge"] = serde_json::json!({"link": true, "resolved": target});
            packages[target] = bridge.clone();
            if target == "packages/implementation" {
                packages[target]["name"] = "bridge".into();
            }
        }
        std::fs::write(
            dir.path().join("package-lock.json"),
            serde_json::to_vec(&serde_json::json!({"lockfileVersion": 3, "packages": packages}))
                .unwrap(),
        )
        .unwrap();
        let ranked = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"))
            .args(["--ecosystem", "npm", "--format", "json"])
            .arg(dir.path())
            .assert()
            .success();
        let graph: serde_json::Value = serde_json::from_slice(&ranked.get_output().stdout).unwrap();
        assert_eq!(graph["rows_total"], 3, "target={target:?}");

        let out = blast(&dir, "leaf").success();
        let report: serde_json::Value = serde_json::from_slice(&out.get_output().stdout).unwrap();
        let reached: Vec<_> = report["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                (
                    r["name"].as_str().unwrap(),
                    r["bfs_depth"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(reached, [("bridge", 1), ("app", 2)], "target={target:?}");
        let out = blast(&dir, "bridge@1.0.0").success();
        let report: serde_json::Value = serde_json::from_slice(&out.get_output().stdout).unwrap();
        assert_eq!(report["total_transitive_dependents"], 1);
        assert_eq!(report["rows"][0]["name"], "app");
    }
}
