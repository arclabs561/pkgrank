//! Exercise manifest selection independently of the Go toolchain version on PATH.
#![cfg(unix)]

use assert_cmd::prelude::*;
use std::{env, fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn go_manifest_directory_and_saved_graph_report_the_same_impact() {
    let temp = tempfile::tempdir().unwrap();
    let module = temp.path().join("app");
    fs::create_dir(&module).unwrap();
    fs::write(module.join("go.mod"), "module example.org/app\n").unwrap();
    // Another manifest in the working directory must not override the explicit input.
    fs::write(module.join("Cargo.toml"), "").unwrap();
    let dump = "example.org/app example.org/bridge@v1.0.0\nexample.org/bridge@v1.0.0 example.org/leaf@v1.0.0\n";
    fs::write(module.join("graph.txt"), dump).unwrap();

    let go = temp.path().join("go");
    fs::write(
        &go,
        "#!/bin/sh\n[ \"$*\" = 'mod graph' ] && [ -f go.mod ] && cat graph.txt\n",
    )
    .unwrap();
    fs::set_permissions(&go, fs::Permissions::from_mode(0o755)).unwrap();
    let mut search_path = vec![temp.path().to_path_buf()];
    search_path.extend(env::split_paths(&env::var_os("PATH").unwrap()));
    let search_path = env::join_paths(search_path).unwrap();
    let report = |cwd: &std::path::Path, args: &[&str]| {
        let checked = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"))
            .current_dir(cwd)
            .env("PATH", &search_path)
            .args(args)
            .assert()
            .success();
        serde_json::from_slice::<serde_json::Value>(&checked.get_output().stdout).unwrap()
    };
    let expected = report(
        &module,
        &[
            "blast-radius",
            "example.org/leaf",
            "graph.txt",
            "--ecosystem",
            "go",
        ],
    );
    assert_eq!(expected["total_transitive_dependents"], 2);
    assert_eq!(expected["rows"][0]["name"], "example.org/bridge");
    assert_eq!(expected["rows"][0]["bfs_depth"], 1);
    assert_eq!(expected["rows"][1]["name"], "example.org/app");
    assert_eq!(expected["rows"][1]["bfs_depth"], 2);
    assert_eq!(
        report(&module, &["blast-radius", "example.org/leaf", "go.mod"]),
        expected
    );
    assert_eq!(
        report(
            temp.path(),
            &["blast-radius", "example.org/leaf", "app/go.mod"]
        ),
        expected
    );
    assert_eq!(
        report(
            &module,
            &["blast-radius", "example.org/leaf", ".", "--ecosystem", "go"]
        ),
        expected
    );
    assert_eq!(
        report(&module, &["go.mod"]),
        report(&module, &["--ecosystem", "go", "graph.txt"])
    );
}
