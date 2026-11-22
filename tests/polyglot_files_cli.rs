use assert_cmd::prelude::*;
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path, process::Command};

fn files(root: &Path, flags: &[&str]) -> BTreeMap<String, Value> {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("pkgrank"))
        .arg("files")
        .arg(root)
        .args(["--format", "json", "--store", "false"])
        .args(flags)
        .assert()
        .success();
    let report: Value = serde_json::from_slice(&output.get_output().stdout).unwrap();
    report["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            (
                row["file"].as_str().unwrap().replace('\\', "/"),
                row.clone(),
            )
        })
        .collect()
}

#[test]
fn mixed_project_uses_each_languages_roles_and_imports() {
    let project = tempfile::tempdir().unwrap();
    let root = project.path();
    for (path, source) in [
        ("Cargo.toml", "[package]\nname='demo'\nversion='0.1.0'\n"),
        ("go.mod", "module example.org/demo\n"),
        ("pyproject.toml", "[project]\nname='demo'\n"),
        ("package.json", "{}"),
        ("src/lib.rs", "mod requests;\nuse anyhow::Result;\n"),
        ("src/requests.rs", ""),
        ("main.go", "package main\nimport \"github.com/acme/lib\"\n"),
        ("app_test.go", ""),
        ("app_bench_test.go", ""),
        ("pkg/__init__.py", "import requests\nimport os\n"),
        ("pkg/test_math.py", ""),
        ("ui/index.ts", "import react from 'react';\n"),
        ("ui/math.spec.ts", ""),
        ("vite.config.ts", ""),
    ] {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, source).unwrap();
    }

    let production = files(root, &["--cache"]);
    assert_eq!(production.len(), 5);
    assert_eq!(
        production["src/lib.rs"]["external_deps"],
        serde_json::json!(["anyhow"])
    );
    assert_eq!(production["main.go"]["role"], "bin_entry");
    assert_eq!(
        production["main.go"]["external_deps"],
        serde_json::json!(["github.com/acme/lib"])
    );
    assert_eq!(production["pkg/__init__.py"]["role"], "lib_root");
    // `requests` is an internal Rust module but an external Python package.
    assert_eq!(
        production["pkg/__init__.py"]["external_deps"],
        serde_json::json!(["requests"])
    );
    assert_eq!(production["ui/index.ts"]["role"], "lib_root");
    assert_eq!(
        production["ui/index.ts"]["external_deps"],
        serde_json::json!(["react"])
    );
    assert_eq!(production, files(root, &["--cache"]));

    let tests = files(root, &["--cache", "--include-tests"]);
    assert_eq!(tests.len(), 8);
    for file in ["app_test.go", "pkg/test_math.py", "ui/math.spec.ts"] {
        assert_eq!(tests[file]["role"], "test");
    }

    let all = files(root, &["--cache", "--include-all"]);
    assert_eq!(all.len(), 10);
    assert_eq!(all["app_bench_test.go"]["role"], "bench");
    assert_eq!(all["vite.config.ts"]["role"], "build");

    let python = files(root, &["--ecosystem", "python"]);
    assert_eq!(python.len(), 1);
    assert_eq!(
        python["pkg/__init__.py"]["external_deps"],
        production["pkg/__init__.py"]["external_deps"]
    );
}

#[test]
fn changing_the_metric_with_a_warm_cache_returns_the_new_top_file() {
    let project = tempfile::tempdir().unwrap();
    let root = project.path();
    fs::create_dir(root.join("src")).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname='demo'\nversion='0.1.0'\n",
    )
    .unwrap();
    fs::write(root.join("src/lib.rs"), "mod leaf;\n").unwrap();
    fs::write(root.join("src/leaf.rs"), "").unwrap();

    let inbound = files(root, &["--cache", "--metric", "indegree", "--top", "1"]);
    assert!(inbound.contains_key("src/leaf.rs"));
    let outbound = files(root, &["--cache", "--metric", "outdegree", "--top", "1"]);
    assert!(outbound.contains_key("src/lib.rs"));
}

#[test]
fn rust_roles_are_relative_to_the_nearest_package() {
    let project = tempfile::tempdir().unwrap();
    let root = project.path();
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers=['crates/*']\n",
    )
    .unwrap();
    for path in [
        "crates/a/src/lib.rs",
        "crates/a/src/main.rs",
        "crates/a/src/bin/report.rs",
        "crates/a/src/tests/helpers.rs",
        "crates/a/src/examples/method.rs",
        "crates/a/tests/tensor/ops/mod.rs",
        "crates/a/benches/latency.rs",
        "crates/a/examples/demo.rs",
        "crates/a/build.rs",
        "tools/tests/example.rs",
    ] {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "").unwrap();
    }

    let manifest = root.join("crates/a/Cargo.toml");
    fs::write(&manifest, "[package]\nname='a'\nversion='0.1.0'\n").unwrap();
    let production = files(root, &["--cache"]);
    assert_eq!(production.len(), 6);
    assert_eq!(production["crates/a/src/lib.rs"]["role"], "lib_root");
    for path in ["crates/a/src/main.rs", "crates/a/src/bin/report.rs"] {
        assert_eq!(production[path]["role"], "bin_entry");
    }
    for path in [
        "crates/a/src/tests/helpers.rs",
        "crates/a/src/examples/method.rs",
        "tools/tests/example.rs",
    ] {
        assert_eq!(production[path]["role"], "source");
    }

    let tests = files(root, &["--include-tests"]);
    assert_eq!(tests.len(), 7);
    assert_eq!(tests["crates/a/tests/tensor/ops/mod.rs"]["role"], "test");
    let all = files(root, &["--include-all"]);
    assert_eq!(all.len(), 10);
    assert_eq!(all["crates/a/examples/demo.rs"]["role"], "example");
    assert_eq!(all["crates/a/benches/latency.rs"]["role"], "bench");
    assert_eq!(all["crates/a/build.rs"]["role"], "build");

    // Changing only the manifest changes ownership, even with the cache warm.
    fs::write(&manifest, "[workspace]\n").unwrap();
    let changed = files(root, &["--cache"]);
    assert_eq!(
        changed["crates/a/tests/tensor/ops/mod.rs"]["role"],
        "source"
    );
}

#[test]
fn cached_files_follow_changed_imports_even_when_source_timestamps_are_preserved() {
    let project = tempfile::tempdir().unwrap();
    let root = project.path();
    fs::write(root.join("pyproject.toml"), "[project]\nname = 'app'\n").unwrap();
    let source = root.join("app.py");
    fs::write(&source, "import first\n").unwrap();
    fs::write(root.join("first.py"), "").unwrap();
    fs::write(root.join("other.py"), "").unwrap();
    let original_time = source.metadata().unwrap().modified().unwrap();
    let warm = files(root, &["--cache"]);
    assert_eq!(warm["first.py"]["in_degree"], 1);

    fs::write(&source, "import other\n").unwrap();
    fs::File::options()
        .write(true)
        .open(&source)
        .unwrap()
        .set_modified(original_time)
        .unwrap();
    let cached = files(root, &["--cache"]);
    assert_eq!(cached["other.py"]["in_degree"], 1);
    assert_eq!(cached, files(root, &[]));
}
