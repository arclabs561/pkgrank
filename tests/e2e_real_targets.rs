use std::process::Command;

fn parse_json_stdout(cmd: &mut Command) -> serde_json::Value {
    let out = cmd.output().expect("spawn pkgrank");
    assert!(
        out.status.success(),
        "pkgrank failed (status={:?})\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    serde_json::from_slice(&out.stdout).expect("stdout JSON parse")
}

#[test]
fn analyze_on_real_workspace_root_cargo_toml() {
    // This is the "real" target: the dev super-workspace root Cargo.toml.
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
    cmd.args([
        "--format",
        "json",
        "--workspace-only=true",
        "--metric",
        "pagerank",
        "-n",
        "5",
        "..",
    ]);
    let v = parse_json_stdout(&mut cmd);
    assert_eq!(v.get("schema_version").and_then(|x| x.as_u64()), Some(1));
    assert_eq!(v.get("ok").and_then(|x| x.as_bool()), Some(true));
    assert_eq!(v.get("command").and_then(|x| x.as_str()), Some("analyze"));
    assert!(v.get("metric").and_then(|x| x.as_str()).is_some());
    assert!(v.get("sorted_by").and_then(|x| x.as_str()).is_some());
    assert!(v.get("convergence").is_some());
    assert!(v.get("rows").and_then(|x| x.as_array()).is_some());
}

#[test]
fn analyze_consumers_pagerank_on_lattix_core_is_nonempty() {
    // Use-case: "what are the orchestrators / top-level consumers?"
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
    cmd.args([
        "--format",
        "json",
        "--workspace-only=false",
        "--metric",
        "consumers-pagerank",
        "-n",
        "10",
        "../lattix/core/lattix-core",
    ]);
    let v = parse_json_stdout(&mut cmd);
    let rows = v
        .get("rows")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(!rows.is_empty());
}

#[test]
fn cratesio_crawl_is_opt_in_network() {
    // Use-case: real URL-backed crawl. This is intentionally opt-in for determinism.
    if std::env::var("PKGRANK_E2E_NETWORK").ok().as_deref() != Some("1") {
        return;
    }

    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
    cmd.args([
        "cratesio", "--format", "json", "--seed", "serde", "--depth", "1",
    ]);
    let v = parse_json_stdout(&mut cmd);
    // We don't assert exact rows (network freshness), just that it's well-formed and non-empty.
    assert_eq!(v.get("schema_version").and_then(|x| x.as_u64()), Some(1));
    assert_eq!(v.get("ok").and_then(|x| x.as_bool()), Some(true));
    assert_eq!(v.get("command").and_then(|x| x.as_str()), Some("cratesio"));
    assert!(v.get("rows").and_then(|x| x.as_array()).is_some());
}
