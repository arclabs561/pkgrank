use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

#[test]
fn pkgrank_analyze_json_is_versioned_envelope() {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
    cmd.args(["--format", "json", "-n", "1"]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"schema_version\""))
        .stdout(predicate::str::contains("\"command\": \"analyze\""))
        .stdout(predicate::str::contains("\"metric\""))
        .stdout(predicate::str::contains("\"sorted_by\""))
        .stdout(predicate::str::contains("\"convergence\""))
        .stdout(predicate::str::contains("\"rows\""));
}

#[test]
fn pkgrank_stats_go_to_stderr_and_do_not_break_json_stdout() {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
    cmd.args(["--format", "json", "--stats", "-n", "1"]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"schema_version\""))
        .stderr(predicate::str::contains("\"type\":\"pkgrank_stats\""));
}
