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
        .stdout(predicate::str::contains("\"rows\""));
}
