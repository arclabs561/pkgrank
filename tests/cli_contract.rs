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

#[cfg(target_os = "linux")] // Isolate the SQLite store through XDG_DATA_HOME.
#[test]
fn query_displays_unicode_project_paths() {
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    let project = temp.path().join(format!("{}ab", "界".repeat(41)));
    std::fs::create_dir(&project).unwrap();
    std::fs::write(project.join("pyproject.toml"), "[project]\nname = 'demo'\n").unwrap();
    std::fs::write(project.join("demo.py"), "def example(): pass\n").unwrap();

    let command = || {
        let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"));
        cmd.env("XDG_DATA_HOME", &data);
        cmd
    };
    command()
        .arg("files")
        .arg(&project)
        .args(["--store", "true"])
        .assert()
        .success();
    for (view, width) in [("files", 30), ("projects", 40)] {
        let suffix = format!("{}ab", "界".repeat(width - 2));
        command()
            .args(["query", view])
            .assert()
            .success()
            .stdout(predicate::str::contains(suffix));
    }
}
