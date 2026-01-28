//! Smoke test for `pkgrank mcp-stdio`.
//!
//! This starts a child process running `pkgrank mcp-stdio` and calls a few tools.
//! It’s meant to validate the MCP surface without relying on Cursor as the client.

#[cfg(not(feature = "stdio"))]
fn main() {
    eprintln!("mcp_smoke requires `--features stdio` (or default features enabled)");
}

#[cfg(feature = "stdio")]
use rmcp::{
    model::CallToolRequestParam,
    service::ServiceExt,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
#[cfg(feature = "stdio")]
// keep serde_json in scope for json! macro usage
use serde_json as _;
#[cfg(feature = "stdio")]
use serde_json::Value;
#[cfg(feature = "stdio")]
use std::path::{Path, PathBuf};
#[cfg(feature = "stdio")]
use tokio::process::Command;

#[cfg(feature = "stdio")]
fn tool_names(tools: &rmcp::model::ListToolsResult) -> Vec<String> {
    tools.tools.iter().map(|t| t.name.to_string()).collect()
}

#[cfg(feature = "stdio")]
fn assert_toolset(toolset: &str, names: &[String]) -> anyhow::Result<()> {
    let mut missing = Vec::new();
    for required in [
        "pkgrank_view",
        "pkgrank_triage",
        "pkgrank_analyze",
        "pkgrank_repo_detail",
        "pkgrank_crate_detail",
        "pkgrank_snapshot",
        "pkgrank_compare_runs",
    ] {
        if !names.iter().any(|n| n == required) {
            missing.push(required);
        }
    }
    anyhow::ensure!(
        missing.is_empty(),
        "[{toolset}] missing expected baseline tools: {:?} (got {:?})",
        missing,
        names
    );

    let has = |n: &str| names.iter().any(|x| x == n);
    match toolset {
        "slim" => {
            for forbidden in [
                "pkgrank_status",
                "pkgrank_modules",
                "pkgrank_modules_sweep",
                "pkgrank_tlc_crates",
                "pkgrank_tlc_repos",
                "pkgrank_invariants",
                "pkgrank_ppr_summary",
            ] {
                anyhow::ensure!(
                    !has(forbidden),
                    "[slim] should not advertise {forbidden} (got {:?})",
                    names
                );
            }
        }
        "full" => {
            for required in ["pkgrank_status", "pkgrank_modules", "pkgrank_modules_sweep"] {
                anyhow::ensure!(has(required), "[full] missing {required} (got {:?})", names);
            }
            for forbidden in [
                "pkgrank_tlc_crates",
                "pkgrank_tlc_repos",
                "pkgrank_invariants",
                "pkgrank_ppr_summary",
            ] {
                anyhow::ensure!(
                    !has(forbidden),
                    "[full] should not advertise debug-only {forbidden} (got {:?})",
                    names
                );
            }
        }
        "debug" => {
            for required in [
                "pkgrank_status",
                "pkgrank_modules",
                "pkgrank_modules_sweep",
                "pkgrank_tlc_crates",
                "pkgrank_tlc_repos",
                "pkgrank_invariants",
                "pkgrank_ppr_summary",
            ] {
                anyhow::ensure!(
                    has(required),
                    "[debug] missing {required} (got {:?})",
                    names
                );
            }
        }
        other => anyhow::bail!("unknown toolset {other}"),
    }
    Ok(())
}

#[cfg(feature = "stdio")]
fn extract_wrapped_json(tool: &str, call: &rmcp::model::CallToolResult) -> anyhow::Result<Value> {
    let texts: Vec<&str> = call
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
        .collect();
    anyhow::ensure!(
        texts.len() == 1,
        "{tool}: expected exactly one text payload, got {}",
        texts.len()
    );
    let v: Value = serde_json::from_str(texts[0])?;
    anyhow::ensure!(
        v.get("schema_version").and_then(|x| x.as_u64()) == Some(1),
        "{tool}: missing/invalid schema_version"
    );
    anyhow::ensure!(
        v.get("ok").and_then(|x| x.as_bool()) == Some(true),
        "{tool}: ok != true"
    );
    anyhow::ensure!(
        v.get("tool").and_then(|x| x.as_str()) == Some(tool),
        "{tool}: tool field mismatch"
    );
    anyhow::ensure!(v.get("result").is_some(), "{tool}: missing result");
    Ok(v)
}

#[cfg(feature = "stdio")]
async fn spawn_and_check(bin: &Path, root: &Path, toolset: &str) -> anyhow::Result<()> {
    eprintln!(
        "spawning: {} mcp-stdio (toolset={})",
        bin.display(),
        toolset
    );
    let service = ()
        .serve(TokioChildProcess::new(Command::new(bin).configure(
            |cmd| {
                cmd.arg("mcp-stdio");
                // Always set explicitly to avoid inheriting user shell env.
                cmd.env("PKGRANK_MCP_TOOLSET", toolset);
            },
        ))?)
        .await?;

    let tools = service.list_tools(Default::default()).await?;
    let names = tool_names(&tools);
    assert_toolset(toolset, &names)?;

    // Quick wrapper-schema checks on a couple core tools.
    let view = service
        .call_tool(CallToolRequestParam {
            name: "pkgrank_view".into(),
            arguments: Some(
                serde_json::json!({"root": root.display().to_string(), "mode": "local"})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        })
        .await?;
    let _ = extract_wrapped_json("pkgrank_view", &view)?;

    let triage = service
        .call_tool(CallToolRequestParam {
            name: "pkgrank_triage".into(),
            arguments: Some(
                serde_json::json!({"root": root.display().to_string(), "limit": 5, "ppr_top": 5})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        })
        .await?;
    let triage_v = extract_wrapped_json("pkgrank_triage", &triage)?;
    anyhow::ensure!(
        triage_v
            .get("summary_text")
            .and_then(|x| x.as_str())
            .is_some(),
        "pkgrank_triage: expected summary_text"
    );

    // Debug-only tool should work in debug toolset.
    if toolset == "debug" {
        let inv = service
            .call_tool(CallToolRequestParam {
                name: "pkgrank_invariants".into(),
                arguments: Some(
                    serde_json::json!({"root": root.display().to_string()})
                        .as_object()
                        .cloned()
                        .unwrap_or_default(),
                ),
            })
            .await?;
        let _ = extract_wrapped_json("pkgrank_invariants", &inv)?;
    }

    service.cancel().await?;
    Ok(())
}

#[cfg(feature = "stdio")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pkgrank_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev_root = pkgrank_root.parent().unwrap().to_path_buf();
    let bin = pkgrank_root.join("target/debug/pkgrank");
    // Ensure the server binary exists *and* matches current sources/features.
    // (The example itself can compile without rebuilding the pkgrank binary.)
    let status = Command::new("cargo")
        .current_dir(&pkgrank_root)
        .args(["build", "--features", "stdio"])
        .status()
        .await?;
    anyhow::ensure!(status.success(), "cargo build --features stdio failed");
    anyhow::ensure!(bin.exists(), "expected pkgrank binary at {}", bin.display());

    // Validate toolsets + schema wrapper contract.
    spawn_and_check(&bin, &dev_root, "slim").await?;
    spawn_and_check(&bin, &dev_root, "full").await?;
    spawn_and_check(&bin, &dev_root, "debug").await?;
    Ok(())
}
