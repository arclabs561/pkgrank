//! `view --mode local` against a super-workspace root: a directory of
//! standalone repos with no root Cargo.toml (the dev/ layout since the root
//! workspace was deleted). The merged path must discover repo roots, run
//! per-repo `cargo metadata`, unify cross-repo copies of first-party crates
//! by name, and still emit the same artifacts as the single-workspace path.

use std::fs;
use std::path::Path;
use std::process::Command;

fn write(path: &Path, content: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn lib_manifest(name: &str) -> String {
    format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n")
}

#[test]
fn view_local_merges_standalone_repo_roots() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // Repo 1: a plain library crate.
    write(&root.join("liba/Cargo.toml"), &lib_manifest("liba"));
    write(&root.join("liba/src/lib.rs"), "pub fn a() {}\n");

    // Repo 2: depends on liba by version. The vendored copy keeps the test
    // hermetic (no registry lookup), while still resolving to a package id
    // that differs from liba's own workspace-member id -- which is exactly
    // the cross-repo aliasing the merge has to unify.
    write(
        &root.join("appb/Cargo.toml"),
        "[package]\nname = \"appb\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
         [dependencies]\nliba = { version = \"0.1\", path = \"vendor/liba\" }\n",
    );
    write(&root.join("appb/src/lib.rs"), "pub fn b() { liba::a() }\n");
    write(
        &root.join("appb/vendor/liba/Cargo.toml"),
        &lib_manifest("liba"),
    );
    write(&root.join("appb/vendor/liba/src/lib.rs"), "pub fn a() {}\n");

    // Repo 3: lives one level down inside an umbrella dir (no Cargo.toml at
    // the umbrella level), like `infra/<repo>` layouts.
    write(&root.join("um/libu/Cargo.toml"), &lib_manifest("libu"));
    write(&root.join("um/libu/src/lib.rs"), "pub fn u() {}\n");

    let out_dir = root.join("evals/pkgrank");
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
    cmd.args([
        "view",
        "--root",
        root.to_str().unwrap(),
        "--out",
        out_dir.to_str().unwrap(),
        "--mode",
        "local",
    ]);
    let out = cmd.output().expect("spawn pkgrank");
    assert!(
        out.status.success(),
        "pkgrank view failed (status={:?})\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    // tlc.crates.json: one row per first-party crate. The vendored liba copy
    // must be unified onto the liba repo's crate, not show up as a duplicate.
    let tlc: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out_dir.join("tlc.crates.json")).unwrap())
            .unwrap();
    let rows = tlc.as_array().unwrap();
    let names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    assert_eq!(
        names.iter().filter(|n| **n == "liba").count(),
        1,
        "vendored liba copy was not unified: {names:?}"
    );
    assert!(names.contains(&"appb"), "appb missing from tlc: {names:?}");
    assert!(
        names.contains(&"libu"),
        "umbrella repo libu missing from tlc: {names:?}"
    );

    let liba = rows.iter().find(|r| r["name"] == "liba").unwrap();
    assert_eq!(
        liba["transitive_dependents"].as_u64(),
        Some(1),
        "appb -> liba cross-repo edge missing"
    );
    assert_eq!(liba["origin"].as_str(), Some("workspace_member"));

    // local.first_party.adj.json: the cross-repo edge in adjacency form.
    let adj: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(out_dir.join("local.first_party.adj.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(adj["appb"], serde_json::json!(["liba"]));
}

/// Merge anomalies must reach artifacts, not just stderr: a repo whose
/// `cargo metadata` fails lands in `merged.skipped.json` (and the sweep
/// summary's failed list), and a name-only unification across
/// semver-incompatible versions is recorded as a version mismatch.
#[test]
fn view_local_records_merge_anomalies() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // Local liba is 0.1.0; the vendored copy consumed by appb is 0.9.9.
    write(&root.join("liba/Cargo.toml"), &lib_manifest("liba"));
    write(&root.join("liba/src/lib.rs"), "pub fn a() {}\n");
    write(
        &root.join("appb/Cargo.toml"),
        "[package]\nname = \"appb\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
         [dependencies]\nliba = { version = \"0.9\", path = \"vendor/liba\" }\n",
    );
    write(&root.join("appb/src/lib.rs"), "pub fn b() { liba::a() }\n");
    write(
        &root.join("appb/vendor/liba/Cargo.toml"),
        "[package]\nname = \"liba\"\nversion = \"0.9.9\"\nedition = \"2021\"\n",
    );
    write(&root.join("appb/vendor/liba/src/lib.rs"), "pub fn a() {}\n");

    // A deliberately-broken repo: the workspace lists a member that does not
    // exist, so `cargo metadata` fails for this repo only.
    write(
        &root.join("broken/Cargo.toml"),
        "[workspace]\nmembers = [\"does-not-exist\"]\n",
    );

    let out_dir = root.join("evals/pkgrank");
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
    cmd.args([
        "view",
        "--root",
        root.to_str().unwrap(),
        "--out",
        out_dir.to_str().unwrap(),
        "--mode",
        "local",
    ]);
    let out = cmd.output().expect("spawn pkgrank");
    assert!(
        out.status.success(),
        "pkgrank view failed (status={:?})\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    // The cross-repo edge still unifies despite the version gap.
    let tlc: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out_dir.join("tlc.crates.json")).unwrap())
            .unwrap();
    let rows = tlc.as_array().unwrap();
    let names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    assert_eq!(
        names.iter().filter(|n| **n == "liba").count(),
        1,
        "vendored liba copy was not unified: {names:?}"
    );
    let liba = rows.iter().find(|r| r["name"] == "liba").unwrap();
    assert_eq!(
        liba["transitive_dependents"].as_u64(),
        Some(1),
        "appb -> liba cross-repo edge missing despite unification"
    );

    // merged.skipped.json: the broken repo and the version mismatch.
    let receipt: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out_dir.join("merged.skipped.json")).unwrap())
            .unwrap();
    let skipped = receipt["skipped"].as_array().unwrap();
    assert_eq!(skipped.len(), 1, "expected one skipped repo: {receipt}");
    assert_eq!(skipped[0]["repo"], "broken");
    assert!(
        !skipped[0]["error"].as_str().unwrap().is_empty(),
        "skip entry has no error text: {receipt}"
    );
    let mismatches = receipt["version_mismatches"].as_array().unwrap();
    assert!(
        mismatches.iter().any(|m| m["name"] == "liba"
            && m["canonical_version"] == "0.1.0"
            && m["aliased_version"] == "0.9.9"
            && m["repo"] == "appb"),
        "liba version mismatch missing from receipt: {receipt}"
    );

    // sweep.summary.json folds the skipped repo into its failed list.
    let sweep: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out_dir.join("sweep.summary.json")).unwrap())
            .unwrap();
    assert!(
        sweep["failed"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["repo"] == "broken"),
        "broken repo missing from sweep.summary.json failed list: {sweep}"
    );

    // One-line summary on stderr points at the receipt.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("1 skipped (see"),
        "missing merge-skip summary line on stderr:\n{stderr}"
    );
}
