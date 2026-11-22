//! Smoke test for `pkgrank mcp-stdio`.
//!
//! This starts a child process running `pkgrank mcp-stdio` and calls a few tools.
//! It’s meant to validate the MCP surface without relying on Cursor as the client.

use anyhow::Context;
use cargo_metadata::Message;
use rmcp::{
    model::CallToolRequestParams,
    service::ServiceExt,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

fn tool_names(tools: &rmcp::model::ListToolsResult) -> Vec<String> {
    tools.tools.iter().map(|t| t.name.to_string()).collect()
}

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

async fn spawn_and_check(bin: &Path, root: &Path, out: &Path, toolset: &str) -> anyhow::Result<()> {
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
        .call_tool(
            CallToolRequestParams::new("pkgrank_view").with_arguments(
                serde_json::json!({
                    "root": root.display().to_string(),
                    "out": out.display().to_string(),
                    "mode": "local"
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        )
        .await?;
    let _ = extract_wrapped_json("pkgrank_view", &view)?;

    let analyze = service
        .call_tool(
            CallToolRequestParams::new("pkgrank_analyze").with_arguments(
                serde_json::json!({"path": root.display().to_string(), "top": 5})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        )
        .await?;
    let _ = extract_wrapped_json("pkgrank_analyze", &analyze)?;

    // Debug-only tool should work in debug toolset.
    if toolset == "debug" {
        let tlc = service
            .call_tool(
                CallToolRequestParams::new("pkgrank_tlc_crates").with_arguments(
                    serde_json::json!({
                        "root": root.display().to_string(),
                        "out": out.display().to_string(),
                        "limit": 5
                    })
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
                ),
            )
            .await?;
        let _ = extract_wrapped_json("pkgrank_tlc_crates", &tlc)?;
    }

    service.cancel().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pkgrank_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // An example build does not compile the server. Ask Cargo for the executable
    // it produced so target-directory overrides and platform suffixes work.
    let build = Command::new("cargo")
        .current_dir(&pkgrank_root)
        .args([
            "build",
            "--bin",
            "pkgrank",
            "--features",
            "stdio",
            "--message-format=json-render-diagnostics",
        ])
        .stderr(Stdio::inherit())
        .output()
        .await?;
    anyhow::ensure!(build.status.success(), "could not build the MCP server");
    let bin = Message::parse_stream(build.stdout.as_slice())
        .find_map(|message| match message.ok()? {
            Message::CompilerArtifact(artifact) if artifact.target.name == "pkgrank" => {
                artifact.executable.map(|path| path.into_std_path_buf())
            }
            _ => None,
        })
        .context("Cargo did not report the pkgrank executable")?;

    // Validate toolsets + schema wrapper contract.
    let out = tempfile::tempdir()?;
    spawn_and_check(&bin, &pkgrank_root, out.path(), "slim").await?;
    spawn_and_check(&bin, &pkgrank_root, out.path(), "full").await?;
    spawn_and_check(&bin, &pkgrank_root, out.path(), "debug").await?;
    Ok(())
}
