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
use std::path::PathBuf;
#[cfg(feature = "stdio")]
use tokio::process::Command;

#[cfg(feature = "stdio")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let bin = root.join("target/debug/pkgrank");
    eprintln!("spawning: {} mcp-stdio", bin.display());

    let service = ()
        .serve(TokioChildProcess::new(Command::new(&bin).configure(|cmd| {
            cmd.arg("mcp-stdio");
        }))?)
        .await?;

    let info = service.peer_info();
    println!("peer_info: {:#?}", info);

    let tools = service.list_tools(Default::default()).await?;
    println!("tools: {:#?}", tools);

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
    println!("pkgrank_view: {:#?}", view);

    let triage = service
        .call_tool(CallToolRequestParam {
            name: "pkgrank_triage".into(),
            arguments: Some(
                serde_json::json!({"root": root.display().to_string(), "limit": 8, "ppr_top": 8})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        })
        .await?;
    println!("pkgrank_triage: {:#?}", triage);

    // Snapshot the current artifacts to a run dir (A).
    let snap_a = service
        .call_tool(CallToolRequestParam {
            name: "pkgrank_snapshot".into(),
            arguments: Some(
                serde_json::json!({"root": root.display().to_string(), "label": "smoke-a"})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        })
        .await?;
    println!("pkgrank_snapshot(smoke-a): {:#?}", snap_a);

    // Re-run view to simulate a new run (should be deterministic if nothing changed).
    let view2 = service
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
    println!("pkgrank_view (2): {:#?}", view2);

    // Snapshot (B).
    let snap_b = service
        .call_tool(CallToolRequestParam {
            name: "pkgrank_snapshot".into(),
            arguments: Some(
                serde_json::json!({"root": root.display().to_string(), "label": "smoke-b"})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        })
        .await?;
    println!("pkgrank_snapshot(smoke-b): {:#?}", snap_b);

    // Compare snapshots.
    let compare = service
        .call_tool(CallToolRequestParam {
            name: "pkgrank_compare_runs".into(),
            arguments: Some(
                serde_json::json!({
                    "root": root.display().to_string(),
                    "old_out": "evals/pkgrank/runs/smoke-a",
                    "new_out": "evals/pkgrank/runs/smoke-b",
                    "limit": 10
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ),
        })
        .await?;
    println!("pkgrank_compare_runs: {:#?}", compare);

    let repo_detail = service
        .call_tool(CallToolRequestParam {
            name: "pkgrank_repo_detail".into(),
            arguments: Some(
                serde_json::json!({"root": root.display().to_string(), "repo": "hop"})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        })
        .await?;
    println!("pkgrank_repo_detail(hop): {:#?}", repo_detail);

    let crate_detail = service
        .call_tool(CallToolRequestParam {
            name: "pkgrank_crate_detail".into(),
            arguments: Some(
                serde_json::json!({"root": root.display().to_string(), "crate": "hop-core"})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        })
        .await?;
    println!("pkgrank_crate_detail(hop-core): {:#?}", crate_detail);

    service.cancel().await?;
    Ok(())
}

