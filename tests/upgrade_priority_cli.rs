#![cfg(unix)]

use assert_cmd::cargo::cargo_bin_cmd;
use serde_json::{json, Value};
use std::{env, fs, os::unix::fs::PermissionsExt, path::Path};

fn package(dir: &Path, name: &str, version: &str, deps: &str) {
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(dir.join("src/lib.rs"), "").unwrap();
    fs::write(dir.join("Cargo.toml"), format!("[workspace]\n[package]\nname='{name}'\nversion='{version}'\nedition='2021'\n[dependencies]\n{deps}")).unwrap();
}

#[test]
fn ranks_resolved_versions_and_rejects_incomplete_member_output() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    package(&dir.join("v1"), "common", "1.0.0", "");
    package(&dir.join("v2"), "common", "2.0.0", "");
    package(
        &dir.join("middle"),
        "middle",
        "0.1.0",
        "common = { path = '../v2' }",
    );
    package(
        &dir.join("app"),
        "app",
        "0.1.0",
        "common = { path = '../v1' }\nmiddle = { path = '../middle' }",
    );
    // Cargo resolves the actual packages. Only the available upgrade versions are fixtures.
    let shim = dir.join("cargo-outdated");
    fs::write(&shim, "#!/bin/sh\ncat \"$UPGRADE_FIXTURE\"\n").unwrap();
    fs::set_permissions(&shim, fs::Permissions::from_mode(0o755)).unwrap();
    let fixture = dir.join("out.json");
    let path = env::join_paths(
        std::iter::once(dir.to_path_buf()).chain(env::split_paths(&env::var_os("PATH").unwrap())),
    )
    .unwrap();
    let cmd = || {
        let mut c = cargo_bin_cmd!("pkgrank");
        c.env("PATH", &path)
            .env("UPGRADE_FIXTURE", &fixture)
            .env("CARGO_NET_OFFLINE", "true")
            .args(["upgrade-priority", "--format", "json"])
            .arg(dir.join("app"));
        c
    };
    let dep = |name: &str, project: &str, compat: &str, latest: &str| json!({"name":name, "project":project, "compat":compat, "latest":latest, "kind":"Normal"});
    let first = json!({"crate_name":"app","dependencies":[
        dep("common","1.0.0","---","1.1.0"),dep("middle->common","2.0.0","2.0.1","Removed"),
        dep("missing","3.0.0","---","3.1.0"),dep("gone","4.0.0","---","Removed")
    ]});
    let second =
        json!({"crate_name":"middle","dependencies":[dep("common","2.0.0","---","2.0.1")]});
    fs::write(&fixture, format!("{first}\n{second}\n")).unwrap();
    let out = cmd().assert().success().get_output().stdout.clone();
    let report: Value = serde_json::from_slice(&out).unwrap();
    let rows = report["rows"].as_array().unwrap();
    assert_eq!(report["outdated_count"], 5);
    assert_eq!(rows.len(), 3);
    let row = |v| rows.iter().find(|r| r["current_version"] == v).unwrap();
    assert_eq!(row("1.0.0")["transitive_dependents"], 1);
    assert_eq!(row("2.0.0")["transitive_dependents"], 2);
    assert_eq!(row("2.0.0")["name"], "common");
    assert_eq!(row("2.0.0")["target_version"], "2.0.1");
    assert_eq!(row("2.0.0")["upgrade_urgency"], "patch");
    assert_eq!(row("2.0.0")["reported_via"].as_array().unwrap().len(), 2);
    assert_eq!(rows.last().unwrap()["graph_match"], "outside_graph");
    let restricted = cmd()
        .args(["--workspace-only", "true"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let restricted: Value = serde_json::from_slice(&restricted).unwrap();
    assert!(restricted["rows"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["graph_match"] == "outside_graph"));
    for bad in [String::new(), format!("{first}\n{{")] {
        fs::write(&fixture, bad).unwrap();
        cmd().assert().failure();
    }
}
