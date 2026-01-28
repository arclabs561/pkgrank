use assert_cmd::cargo::cargo_bin_cmd;

fn write_file(path: &std::path::Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, contents).unwrap();
}

#[test]
fn triage_cli_reads_artifacts_without_refresh() {
    let mut root = std::env::temp_dir();
    root.push(format!("pkgrank-triage-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    let out_rel = std::path::PathBuf::from("evals/pkgrank");
    let out_dir = root.join(&out_rel);

    // Minimal valid artifacts.
    write_file(
        &out_dir.join("tlc.crates.json"),
        r#"[{
  "repo":"pkgrank",
  "axis":"tekne",
  "name":"pkgrank",
  "manifest_path":"pkgrank/Cargo.toml",
  "origin":"workspace_member",
  "pagerank":0.1,
  "betweenness":0.0,
  "transitive_dependents":1,
  "transitive_dependencies":2,
  "third_party_deps":3,
  "score":1.0,
  "why":"test",
  "repo_git_commits_30d":null,
  "repo_git_days_since_last_commit":null
}]"#,
    );
    write_file(
        &out_dir.join("tlc.repos.json"),
        r#"[{
  "repo":"pkgrank",
  "axis":"tekne",
  "deps_pagerank":0.2,
  "consumers_pagerank":0.3,
  "transitive_dependents":1,
  "transitive_dependencies":2,
  "third_party_deps":3,
  "violation_weight":0,
  "score":1.0,
  "why":"test",
  "git_commits_30d":null,
  "git_days_since_last_commit":null
}]"#,
    );
    write_file(
        &out_dir.join("ecosystem.invariants.violations.json"),
        r#"[]"#,
    );
    write_file(&out_dir.join("ppr.aggregate.json"), r#"[["pkgrank",0.5]]"#);

    let out = cargo_bin_cmd!("pkgrank")
        .args([
            "triage",
            "--root",
            root.to_string_lossy().as_ref(),
            "--out",
            out_rel.to_string_lossy().as_ref(),
            "--refresh-if-missing=false",
            "--format",
            "json",
            "--limit",
            "5",
            "--ppr-top",
            "3",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json stdout");
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "triage");
    assert!(v.get("result").is_some());
    assert!(v.get("summary_text").and_then(|x| x.as_str()).is_some());
}
