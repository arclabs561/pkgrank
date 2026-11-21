use assert_cmd::Command;
use serde_json::{json, Value};
use std::fs;

fn project() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[package]\nname='demo'\nversion='0.1.0'\nedition='2021'\n",
    )
    .unwrap();
    fs::write(root.path().join("src/lib.rs"), "mod leaf;\n").unwrap();
    fs::write(root.path().join("src/leaf.rs"), "pub fn value() {}\n").unwrap();
    root
}

fn files(root: &tempfile::TempDir) -> Command {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
    cmd.arg("files").arg(root.path()).args(["--store", "false"]);
    cmd
}

#[test]
fn piped_affected_and_focus_share_one_document_even_when_rank_rows_are_limited() {
    let root = project();
    let result = files(&root)
        .args(["--affected", "-", "--focus", "leaf", "-n", "0"])
        .write_stdin(" src/leaf.rs \n\n")
        .assert()
        .success();
    let doc: Value = serde_json::from_slice(&result.get_output().stdout).unwrap();
    assert_eq!(doc["schema_version"], 1);
    assert_eq!(doc["rows"], json!([]));
    assert_eq!(doc["rows_total"], 2);
    assert_eq!(doc["changed"], json!(["src/leaf.rs"]));
    assert_eq!(doc["affected"], json!(["src/lib.rs"]));
    assert_eq!(doc["affected_count"], 1);
    assert_eq!(doc["focus"][0]["file"], "src/leaf.rs");
    assert_eq!(doc["focus"][0]["imports"], json!([]));
    assert_eq!(doc["focus"][0]["imported_by"], json!(["src/lib.rs"]));
}

#[test]
fn empty_and_unrequested_details_are_distinguishable() {
    let root = project();
    let empty = files(&root)
        .args([
            "--format",
            "json",
            "--affected",
            "src/lib.rs",
            "--focus",
            "unknown",
        ])
        .assert()
        .success();
    let doc: Value = serde_json::from_slice(&empty.get_output().stdout).unwrap();
    assert_eq!(doc["affected_count"], 0);
    assert_eq!(doc["affected"], json!([]));
    assert_eq!(doc["focus"], json!([]));

    let plain = files(&root).args(["--format", "json"]).assert().success();
    let doc: Value = serde_json::from_slice(&plain.get_output().stdout).unwrap();
    for key in ["affected", "affected_count", "changed", "focus"] {
        assert!(doc.get(key).is_none(), "{key}");
    }
}

#[test]
fn explicit_text_retains_focus_and_affected_sections() {
    let root = project();
    let result = files(&root)
        .args([
            "--format",
            "text",
            "--affected",
            "src/leaf.rs",
            "--focus",
            "leaf",
        ])
        .assert()
        .success();
    let stdout = std::str::from_utf8(&result.get_output().stdout).unwrap();
    assert!(stdout.contains("focus: src/leaf.rs"));
    assert!(stdout.contains("affected by [src/leaf.rs] (1 files):"));
}
