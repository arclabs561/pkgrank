//! MCP stdio server implementation for pkgrank.

pub(crate) use rmcp::ErrorData as McpError;
use rmcp::{
    handler::server::router::tool::ToolRouter as RmcpToolRouter,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
    ServiceExt,
};
use schemars::JsonSchema;
use serde::Deserialize;
pub(crate) use tokio::process::Command as TokioCommand;
pub(crate) use tokio::time::timeout;

use super::*;

// ---------------------------------------------------------------------------
// run_mcp_stdio (entry point)
// ---------------------------------------------------------------------------

pub fn run_mcp_stdio() -> anyhow::Result<()> {
    // IMPORTANT: stdout is reserved for MCP JSON-RPC frames.
    // If you need diagnostics, use stderr.
    let rt = tokio::runtime::Runtime::new().context("failed to build tokio runtime")?;
    rt.block_on(async {
        let toolset_raw =
            std::env::var("PKGRANK_MCP_TOOLSET").unwrap_or_else(|_| "slim".to_string());
        let toolset = toolset_raw.trim().to_ascii_lowercase();

        // Default should "just work": keep tool surface small unless explicitly opted-in.
        match toolset.as_str() {
            "" | "slim" | "minimal" => {
                let service = PkgrankStdioMcpSlim::new();
                let running = service
                    .serve(stdio())
                    .await
                    .context("failed to start stdio MCP server (toolset=slim)")?;
                let _ = running
                    .waiting()
                    .await
                    .context("stdio MCP server task join failed (toolset=slim)")?;
            }
            "full" => {
                let service = PkgrankStdioMcpFull::new();
                let running = service
                    .serve(stdio())
                    .await
                    .context("failed to start stdio MCP server (toolset=full)")?;
                let _ = running
                    .waiting()
                    .await
                    .context("stdio MCP server task join failed (toolset=full)")?;
            }
            "debug" => {
                let service = PkgrankStdioMcpDebug::new();
                let running = service
                    .serve(stdio())
                    .await
                    .context("failed to start stdio MCP server (toolset=debug)")?;
                let _ = running
                    .waiting()
                    .await
                    .context("stdio MCP server task join failed (toolset=debug)")?;
            }
            other => {
                eprintln!(
                    "warning: PKGRANK_MCP_TOOLSET must be one of: slim, full, debug (got {other}); defaulting to slim"
                );
                let service = PkgrankStdioMcpSlim::new();
                let running = service
                    .serve(stdio())
                    .await
                    .context("failed to start stdio MCP server (toolset=slim)")?;
                let _ = running
                    .waiting()
                    .await
                    .context("stdio MCP server task join failed (toolset=slim)")?;
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared MCP param structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankCompareRunsArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    pub root: Option<String>,
    /// "new" artifact directory (relative to root if not absolute).
    pub new_out: String,
    /// "old" artifact directory (relative to root if not absolute).
    pub old_out: String,
    /// Limit number of deltas returned.
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankSnapshotArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    pub root: Option<String>,
    /// Current artifact directory to snapshot (relative to root if not absolute).
    #[serde(default)]
    pub out: Option<String>,
    /// Destination directory (relative to root if not absolute). Defaults to `<out>/runs/<label>`.
    #[serde(default)]
    pub dest: Option<String>,
    /// Label used for default dest.
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankTriageArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    pub root: Option<String>,
    /// Artifact directory (relative to root if not absolute).
    #[serde(default)]
    pub out: Option<String>,
    /// If required artifacts are missing, re-run `pkgrank_view` first.
    #[serde(default)]
    pub refresh_if_missing: Option<bool>,
    /// View mode used for refresh (local/cratesio/both). Defaults to "local".
    #[serde(default)]
    pub mode: Option<String>,
    /// Mark artifacts stale if older than this many minutes. Defaults to 60.
    #[serde(default)]
    pub stale_minutes: Option<u64>,
    /// Limit returned rows.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Optional axis filter for TLC tables.
    #[serde(default)]
    pub axis: Option<String>,
    /// Include PPR aggregate top-k.
    #[serde(default)]
    pub ppr_top: Option<usize>,

    /// Summarize READMEs (bounded by the *_top_* limits below). Default: false.
    #[serde(default)]
    pub summarize_readmes: Option<bool>,
    /// Number of top repos (post-filter) to summarize. Default: 0.
    #[serde(default)]
    pub summarize_repos_top: Option<usize>,
    /// Number of top crates (post-filter) to summarize. Default: 0.
    #[serde(default)]
    pub summarize_crates_top: Option<usize>,
    /// Max chars of README fed into LLM (default: 12000).
    #[serde(default)]
    pub llm_input_max_chars: Option<usize>,
    /// Timeout seconds for the LLM command (default: 30).
    #[serde(default)]
    pub llm_timeout_secs: Option<u64>,
    /// Cache summaries under `<out>/readme_ai_cache/` (default: true).
    #[serde(default)]
    pub llm_cache: Option<bool>,
    /// Include raw LLM output in triage results (default: false).
    #[serde(default)]
    pub llm_include_raw: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankRepoDetailArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    pub root: Option<String>,
    /// Artifact directory (relative to root if not absolute).
    #[serde(default)]
    pub out: Option<String>,
    /// Repo name (e.g. "anno", "hop", "innr").
    pub repo: String,
    /// Include README snippet (default: false).
    #[serde(default)]
    pub include_readme: Option<bool>,
    /// Max README snippet chars (default: 4000).
    #[serde(default)]
    pub readme_max_chars: Option<usize>,
    /// Summarize README via configured local LLM (default: false).
    #[serde(default)]
    pub summarize_readme: Option<bool>,
    /// Max chars of README fed into LLM (default: 12000).
    #[serde(default)]
    pub llm_input_max_chars: Option<usize>,
    /// Timeout seconds for the LLM command (default: 30).
    #[serde(default)]
    pub llm_timeout_secs: Option<u64>,
    /// Cache summaries under `<out>/readme_ai_cache/` (default: true).
    #[serde(default)]
    pub llm_cache: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankCrateDetailArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    pub root: Option<String>,
    /// Artifact directory (relative to root if not absolute).
    #[serde(default)]
    pub out: Option<String>,
    /// Crate name (Cargo package name).
    #[serde(rename = "crate")]
    pub krate: String,
    /// Include README snippet (default: false).
    #[serde(default)]
    pub include_readme: Option<bool>,
    /// Max README snippet chars (default: 4000).
    #[serde(default)]
    pub readme_max_chars: Option<usize>,
    /// Summarize README via configured local LLM (default: false).
    #[serde(default)]
    pub summarize_readme: Option<bool>,
    /// Max chars of README fed into LLM (default: 12000).
    #[serde(default)]
    pub llm_input_max_chars: Option<usize>,
    /// Timeout seconds for the LLM command (default: 30).
    #[serde(default)]
    pub llm_timeout_secs: Option<u64>,
    /// Cache summaries under `<out>/readme_ai_cache/` (default: true).
    #[serde(default)]
    pub llm_cache: Option<bool>,
}

// ---------------------------------------------------------------------------
// MCP-specific param structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankViewToolArgs {
    /// Root directory containing the dev super-workspace.
    #[serde(default)]
    pub root: Option<String>,
    /// Output directory for artifacts (relative to root if not absolute).
    #[serde(default)]
    pub out: Option<String>,
    /// Mode: "local", "cratesio", or "both".
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankAnalyzeToolArgs {
    /// Path to a Cargo.toml or directory containing one.
    #[serde(default)]
    pub path: Option<String>,
    /// Metric to sort by: pagerank, consumers-pagerank, indegree, outdegree, betweenness.
    #[serde(default)]
    pub metric: Option<String>,
    #[serde(default)]
    pub workspace_only: Option<bool>,
    #[serde(default)]
    pub dev: Option<bool>,
    #[serde(default)]
    pub build: Option<bool>,
    #[serde(default)]
    pub top: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankModulesToolArgs {
    /// Path to a Cargo manifest to analyze.
    #[serde(default)]
    pub manifest_path: Option<String>,
    /// Package to analyze (same meaning as `cargo modules -p ...`).
    #[serde(default)]
    pub package: Option<String>,
    /// Analyze the package library target.
    #[serde(default)]
    pub lib: Option<bool>,
    /// Analyze the named binary target.
    #[serde(default)]
    pub bin: Option<String>,
    /// Analyze with `#[cfg(test)]` enabled.
    #[serde(default)]
    pub cfg_test: Option<bool>,
    /// Preset: "file-full", "file-api", "node-full", "node-api".
    #[serde(default)]
    pub preset: Option<String>,
    /// Centrality metric to sort by (same enum as CLI).
    #[serde(default)]
    pub metric: Option<String>,
    /// Aggregate: "node", "module", or "file".
    #[serde(default)]
    pub aggregate: Option<String>,
    /// Edge kind: "uses", "owns", or "both".
    #[serde(default)]
    pub edge_kind: Option<String>,
    /// Max rows to return (after sorting).
    #[serde(default)]
    pub top: Option<usize>,
    /// Include functions in graph.
    #[serde(default)]
    pub include_fns: Option<bool>,
    /// Include types in graph.
    #[serde(default)]
    pub include_types: Option<bool>,
    /// Include traits in graph.
    #[serde(default)]
    pub include_traits: Option<bool>,
    /// Include externs in graph.
    #[serde(default)]
    pub include_externs: Option<bool>,
    /// Include sysroot crates in graph.
    #[serde(default)]
    pub include_sysroot: Option<bool>,
    /// Cache cargo-modules DOT output on disk.
    #[serde(default)]
    pub cache: Option<bool>,
    /// Force refresh cache.
    #[serde(default)]
    pub cache_refresh: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankModulesSweepToolArgs {
    /// Path to a Cargo manifest (workspace root or a single crate).
    #[serde(default)]
    pub manifest_path: Option<String>,
    /// Packages to analyze (repeatable).
    #[serde(default)]
    pub packages: Option<Vec<String>>,
    /// Analyze all workspace member packages under the manifest.
    #[serde(default)]
    pub all_packages: Option<bool>,
    /// Analyze the package library target.
    #[serde(default)]
    pub lib: Option<bool>,
    /// Analyze the named binary target (applies to all packages).
    #[serde(default)]
    pub bin: Option<String>,
    /// Analyze with `#[cfg(test)]` enabled.
    #[serde(default)]
    pub cfg_test: Option<bool>,
    /// Preset: "file-full", "file-api", "node-full", "node-api".
    #[serde(default)]
    pub preset: Option<String>,
    /// Centrality metric to sort by (same enum as CLI).
    #[serde(default)]
    pub metric: Option<String>,
    /// Max rows per package to return (after sorting).
    #[serde(default)]
    pub top: Option<usize>,
    /// Continue if a package analysis fails.
    #[serde(default)]
    pub continue_on_error: Option<bool>,
    /// Convenience alias: stop on first error.
    #[serde(default)]
    pub fail_fast: Option<bool>,
    /// Cache cargo-modules DOT output on disk.
    #[serde(default)]
    pub cache: Option<bool>,
    /// Force refresh cache.
    #[serde(default)]
    pub cache_refresh: Option<bool>,

    /// Include per-package `rows` (can be large). Default: false.
    #[serde(default)]
    pub include_rows: Option<bool>,
    /// Include per-package `top_edges` (can be large). Default: false.
    #[serde(default)]
    pub include_top_edges: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankTlcCratesToolArgs {
    #[serde(default)]
    pub root: Option<String>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub axis: Option<String>,
    #[serde(default)]
    pub repo: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankTlcReposToolArgs {
    #[serde(default)]
    pub root: Option<String>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub axis: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankInvariantsToolArgs {
    #[serde(default)]
    pub root: Option<String>,
    #[serde(default)]
    pub out: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankPprSummaryToolArgs {
    #[serde(default)]
    pub root: Option<String>,
    #[serde(default)]
    pub out: Option<String>,
}

// ---------------------------------------------------------------------------
// New feature MCP param structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankBlastRadiusToolArgs {
    /// Package name to analyze (e.g. "serde").
    pub package: String,
    /// Path to Cargo.toml or directory.
    #[serde(default)]
    pub path: Option<String>,
    /// Include dev-dependencies.
    #[serde(default)]
    pub dev: Option<bool>,
    /// Include build-dependencies.
    #[serde(default)]
    pub build: Option<bool>,
    /// Restrict to workspace members.
    #[serde(default)]
    pub workspace_only: Option<bool>,
    /// Max rows to return.
    #[serde(default)]
    pub top: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankUpgradePriorityToolArgs {
    /// Path to Cargo.toml or directory.
    #[serde(default)]
    pub path: Option<String>,
    /// Include dev-dependencies in graph analysis.
    #[serde(default)]
    pub dev: Option<bool>,
    /// Include build-dependencies.
    #[serde(default)]
    pub build: Option<bool>,
    /// Restrict graph to workspace members.
    #[serde(default)]
    pub workspace_only: Option<bool>,
    /// Max rows to return.
    #[serde(default)]
    pub top: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankPolyglotToolArgs {
    /// Ecosystem: "npm", "python", or "go".
    pub ecosystem: String,
    /// Path to lock file or directory.
    #[serde(default)]
    pub path: Option<String>,
    /// Centrality metric: "pagerank", "consumers-pagerank", "betweenness", "indegree", "outdegree".
    #[serde(default)]
    pub metric: Option<String>,
    /// Max rows to return.
    #[serde(default)]
    pub top: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PkgrankFilesToolArgs {
    /// Project path or GitHub URL (e.g. "owner/repo" or "https://github.com/owner/repo").
    #[serde(default)]
    pub path: Option<String>,
    /// Ecosystem: "cargo", "npm", "python", "go". Auto-detected if omitted.
    #[serde(default)]
    pub ecosystem: Option<String>,
    /// Include test files in the graph.
    #[serde(default)]
    pub include_tests: Option<bool>,
    /// Centrality metric: "pagerank", "consumers-pagerank", "betweenness", "indegree", "outdegree".
    #[serde(default)]
    pub metric: Option<String>,
    /// Max rows to return.
    #[serde(default)]
    pub top: Option<usize>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn p_display(root: &Path, p: Option<&PathBuf>) -> Option<String> {
    p.map(|p| {
        p.strip_prefix(root)
            .ok()
            .map(|x| x.display().to_string())
            .unwrap_or_else(|| p.display().to_string())
    })
}

pub(crate) fn mcp_ok(
    tool: &str,
    result: serde_json::Value,
    summary_text: Option<String>,
) -> Result<CallToolResult, McpError> {
    let payload = serde_json::json!({
        "schema_version": 1,
        "ok": true,
        "tool": tool,
        "summary_text": summary_text,
        "result": result,
    });
    Ok(CallToolResult::success(vec![Content::text(
        payload.to_string(),
    )]))
}

// ---------------------------------------------------------------------------
// Server structs
// ---------------------------------------------------------------------------

// --- MCP stdio server (Cursor integration) ---
//
// Mirrors `threadlog mcp-stdio`:
// - keep stdout clean (transport)
// - return JSON payloads as Content::text
// - keep tool surface small and stable

#[derive(Clone)]
pub(crate) struct PkgrankStdioMcpFull {
    tool_router: RmcpToolRouter<Self>,
}

// Toolset selection is runtime via `PKGRANK_MCP_TOOLSET` (default: slim).

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct PkgrankStdioMcpDebug {
    tool_router: RmcpToolRouter<Self>,
}

#[derive(Clone)]
#[allow(dead_code)] // constructed when `mcp-debug` is ON
pub(crate) struct PkgrankStdioMcpSlim {
    tool_router: RmcpToolRouter<Self>,
}

// ---------------------------------------------------------------------------
// PkgrankStdioMcpDebug impl
// ---------------------------------------------------------------------------

#[tool_router]
impl PkgrankStdioMcpDebug {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    // Debug toolset: include the full surface plus internal artifact inspection tools.
    // Delegate "full" tools to `PkgrankStdioMcpFull` to avoid duplicating logic.

    #[tool(description = "Check pkgrank artifact status (what exists, where, and basic metadata)")]
    async fn pkgrank_status(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_status(params).await
    }

    #[tool(description = "Generate pkgrank HTML/JSON artifacts (pkgrank view)")]
    async fn pkgrank_view(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_view(params).await
    }

    #[tool(
        description = "Triage bundle: top TLC crates/repos + invariants + PPR top-k (artifact-backed)"
    )]
    async fn pkgrank_triage(
        &self,
        params: Parameters<PkgrankTriageArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpSlim::new().pkgrank_triage(params).await
    }

    #[tool(description = "Get repo details from artifacts")]
    async fn pkgrank_repo_detail(
        &self,
        params: Parameters<PkgrankRepoDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_repo_detail(params).await
    }

    #[tool(description = "Get crate details from artifacts")]
    async fn pkgrank_crate_detail(
        &self,
        params: Parameters<PkgrankCrateDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new()
            .pkgrank_crate_detail(params)
            .await
    }

    #[tool(description = "Snapshot pkgrank artifacts into a new directory (copy selected files)")]
    async fn pkgrank_snapshot(
        &self,
        params: Parameters<PkgrankSnapshotArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_snapshot(params).await
    }

    #[tool(description = "Compare two pkgrank artifact directories (TLC crates/repos deltas)")]
    async fn pkgrank_compare_runs(
        &self,
        params: Parameters<PkgrankCompareRunsArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new()
            .pkgrank_compare_runs(params)
            .await
    }

    #[tool(description = "Compute local crate ranking (pkgrank analyze)")]
    async fn pkgrank_analyze(
        &self,
        params: Parameters<PkgrankAnalyzeToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_analyze(params).await
    }

    #[tool(
        description = "Compute internal module/item centrality (pkgrank modules; cargo-modules-backed)"
    )]
    async fn pkgrank_modules(
        &self,
        params: Parameters<PkgrankModulesToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new().pkgrank_modules(params).await
    }

    #[tool(
        description = "Compute internal module/item centrality across multiple packages (pkgrank modules-sweep)"
    )]
    async fn pkgrank_modules_sweep(
        &self,
        params: Parameters<PkgrankModulesSweepToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new()
            .pkgrank_modules_sweep(params)
            .await
    }

    // --- Internal artifact inspection tools (debug-only) ---
    // Keep these typed and schema-versioned; they are intentionally not in the default toolset.

    #[tool(description = "Return top TLC crates (artifact-backed) with optional filters (debug)")]
    async fn pkgrank_tlc_crates(
        &self,
        params: Parameters<PkgrankTlcCratesToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let limit = params.0.limit.unwrap_or(25).min(500);
        let axis = params.0.axis.clone();
        let repo = params.0.repo.clone();

        let path = PkgrankStdioMcpFull::artifact_path(&root, &out, "tlc.crates.json");
        let rows: Vec<TlcCrateRow> = PkgrankStdioMcpFull::read_json_file(&path)?;
        let mut out_rows = Vec::new();
        for r in rows {
            if let Some(ax) = axis.as_ref() {
                if &r.axis != ax {
                    continue;
                }
            }
            if let Some(rr) = repo.as_ref() {
                if &r.repo != rr {
                    continue;
                }
            }
            out_rows.push(r);
            if out_rows.len() >= limit {
                break;
            }
        }
        let result = serde_json::json!({
            "source": path.display().to_string(),
            "rows": out_rows,
        });
        mcp_ok("pkgrank_tlc_crates", result, None)
    }

    #[tool(description = "Return top TLC repos (artifact-backed) with optional filters (debug)")]
    async fn pkgrank_tlc_repos(
        &self,
        params: Parameters<PkgrankTlcReposToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let limit = params.0.limit.unwrap_or(25).min(500);
        let axis = params.0.axis.clone();

        let path = PkgrankStdioMcpFull::artifact_path(&root, &out, "tlc.repos.json");
        let rows: Vec<TlcRepoRow> = PkgrankStdioMcpFull::read_json_file(&path)?;
        let mut out_rows = Vec::new();
        for r in rows {
            if let Some(ax) = axis.as_ref() {
                if &r.axis != ax {
                    continue;
                }
            }
            out_rows.push(r);
            if out_rows.len() >= limit {
                break;
            }
        }
        let result = serde_json::json!({
            "source": path.display().to_string(),
            "rows": out_rows,
        });
        mcp_ok("pkgrank_tlc_repos", result, None)
    }

    #[tool(description = "List invariant violations (artifact-backed) (debug)")]
    async fn pkgrank_invariants(
        &self,
        params: Parameters<PkgrankInvariantsToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let path =
            PkgrankStdioMcpFull::artifact_path(&root, &out, "ecosystem.invariants.violations.json");
        let rows: Vec<RepoInvariantViolation> = PkgrankStdioMcpFull::read_json_file(&path)?;
        let result = serde_json::json!({
            "source": path.display().to_string(),
            "violations": rows,
        });
        mcp_ok("pkgrank_invariants", result, None)
    }

    #[tool(description = "Return PPR summaries (artifact-backed) (debug)")]
    async fn pkgrank_ppr_summary(
        &self,
        params: Parameters<PkgrankPprSummaryToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let entry_path = PkgrankStdioMcpFull::artifact_path(&root, &out, "ppr.entrypoints.json");
        let agg_path = PkgrankStdioMcpFull::artifact_path(&root, &out, "ppr.aggregate.json");
        let entry: serde_json::Value = PkgrankStdioMcpFull::read_json_file(&entry_path)?;
        let agg: serde_json::Value = PkgrankStdioMcpFull::read_json_file(&agg_path)?;
        let result = serde_json::json!({
            "entrypoints_source": entry_path.display().to_string(),
            "aggregate_source": agg_path.display().to_string(),
            "entrypoints": entry,
            "aggregate": agg,
        });
        mcp_ok("pkgrank_ppr_summary", result, None)
    }
}

#[tool_handler]
impl rmcp::ServerHandler for PkgrankStdioMcpDebug {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Tools for ranking crates/repos in the local Cargo workspace (pkgrank).\n\n\
                 Toolsets:\n\
                 - default: PKGRANK_MCP_TOOLSET=slim (small surface)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=full (includes module/type graph tools)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=debug (full + internal artifact inspection tools)"
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// PkgrankStdioMcpFull impl
// ---------------------------------------------------------------------------

#[tool_router]
impl PkgrankStdioMcpFull {
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    pub(crate) fn default_root(params_root: Option<&str>) -> PathBuf {
        if let Some(r) = params_root {
            return PathBuf::from(r);
        }
        if let Ok(r) = std::env::var("PKGRANK_ROOT") {
            if !r.trim().is_empty() {
                return PathBuf::from(r);
            }
        }
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }

    pub(crate) fn default_out(params_out: Option<&str>) -> PathBuf {
        if let Some(o) = params_out {
            return PathBuf::from(o);
        }
        if let Ok(o) = std::env::var("PKGRANK_OUT") {
            if !o.trim().is_empty() {
                return PathBuf::from(o);
            }
        }
        PathBuf::from("evals/pkgrank")
    }

    pub(crate) fn artifact_path(root: &Path, out: &Path, rel: &str) -> PathBuf {
        // If `out` is absolute, do not join with root.
        if out.is_absolute() {
            out.join(rel)
        } else {
            root.join(out).join(rel)
        }
    }

    pub(crate) fn ensure_dir(path: &Path) -> Result<(), McpError> {
        std::fs::create_dir_all(path).map_err(|e| {
            McpError::internal_error(
                format!("failed to create dir {}: {}", path.display(), e),
                None,
            )
        })
    }

    pub(crate) fn read_json_file<T: serde::de::DeserializeOwned>(
        path: &Path,
    ) -> Result<T, McpError> {
        let raw = fs::read_to_string(path).map_err(|e| {
            McpError::internal_error(format!("failed to read {}: {}", path.display(), e), None)
        })?;
        serde_json::from_str::<T>(&raw).map_err(|e| {
            McpError::internal_error(format!("failed to parse {}: {}", path.display(), e), None)
        })
    }

    #[tool(
        description = "Triage bundle: top TLC crates/repos + invariants + PPR top-k (artifact-backed)"
    )]
    async fn pkgrank_triage(
        &self,
        params: Parameters<PkgrankTriageArgs>,
    ) -> Result<CallToolResult, McpError> {
        // Keep triage in all toolsets; delegate to the slim implementation.
        PkgrankStdioMcpSlim::new().pkgrank_triage(params).await
    }

    #[tool(description = "Check pkgrank artifact status (what exists, where, and basic metadata)")]
    async fn pkgrank_status(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());
        let out_dir = if out.is_absolute() {
            out.clone()
        } else {
            root.join(&out)
        };

        let files = [
            "pkgrank_overview.html",
            "by_repo.summary.json",
            "root.workspace_only.json",
            "ecosystem.repo_graph.rows.json",
            "ecosystem.axes.pagerank.json",
            "ecosystem.invariants.violations.json",
            "tlc.crates.json",
            "tlc.repos.json",
            "ppr.entrypoints.json",
            "ppr.aggregate.json",
        ];

        let mut status = Vec::new();
        for f in files {
            let p = out_dir.join(f);
            let meta = std::fs::metadata(&p).ok();
            status.push(serde_json::json!({
                "path": p.display().to_string(),
                "exists": meta.is_some(),
                "size_bytes": meta.as_ref().map(|m| m.len()),
                "mtime_epoch_secs": meta.and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs()),
            }));
        }

        let payload = serde_json::json!({
            "root": root.display().to_string(),
            "out_dir": out_dir.display().to_string(),
            "files": status,
        });
        mcp_ok("pkgrank_status", payload, None)
    }

    #[tool(description = "Generate pkgrank HTML/JSON artifacts (pkgrank view)")]
    async fn pkgrank_view(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());
        let mode = match params.0.mode.as_deref().unwrap_or("local") {
            "local" => ViewMode::Local,
            "cratesio" => ViewMode::CratesIo,
            "both" => ViewMode::Both,
            other => {
                return Err(McpError::invalid_params(
                    format!("mode must be one of: local, cratesio, both (got {other})"),
                    None,
                ));
            }
        };

        let args = ViewArgs {
            root,
            out,
            mode,
            local_top: 10,
            cratesio_depth: 2,
            cratesio_dev: false,
            cratesio_build: false,
            cratesio_optional: false,
            quiet: true,
        };

        run_view(&args).map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let html_path = Self::artifact_path(&args.root, &args.out, "pkgrank_overview.html");

        let payload = serde_json::json!({
            "root": args.root.display().to_string(),
            "out_dir": if args.out.is_absolute() { args.out.display().to_string() } else { args.root.join(&args.out).display().to_string() },
            "html_path": html_path.display().to_string(),
        });
        mcp_ok("pkgrank_view", payload, None)
    }

    #[tool(description = "Get repo details from artifacts")]
    async fn pkgrank_repo_detail(
        &self,
        params: Parameters<PkgrankRepoDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());
        let include_readme = params.0.include_readme.unwrap_or(false);
        let summarize_readme = params.0.summarize_readme.unwrap_or(false);
        let llm_input_max_chars = params.0.llm_input_max_chars.unwrap_or(12_000).min(80_000);
        let llm_timeout_secs = params.0.llm_timeout_secs.unwrap_or(30).min(600);
        let llm_cache = params.0.llm_cache.unwrap_or(true);
        let readme_max_chars = params.0.readme_max_chars.unwrap_or(4000).min(50_000);
        let mut payload = repo_detail_payload(
            &root,
            &out,
            &params.0.repo,
            include_readme,
            readme_max_chars,
        )?;

        let readme_path = find_readme_for_repo(&root, &params.0.repo);
        let ai = maybe_add_readme_llm_summary(
            &root,
            &out,
            "repo",
            &params.0.repo,
            readme_path.as_ref(),
            summarize_readme,
            llm_input_max_chars,
            llm_timeout_secs,
            llm_cache,
        )
        .await?;
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("readme_ai".to_string(), ai);
        }
        mcp_ok("pkgrank_repo_detail", payload, None)
    }

    #[tool(description = "Get crate details from artifacts")]
    async fn pkgrank_crate_detail(
        &self,
        params: Parameters<PkgrankCrateDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());
        let include_readme = params.0.include_readme.unwrap_or(false);
        let summarize_readme = params.0.summarize_readme.unwrap_or(false);
        let llm_input_max_chars = params.0.llm_input_max_chars.unwrap_or(12_000).min(80_000);
        let llm_timeout_secs = params.0.llm_timeout_secs.unwrap_or(30).min(600);
        let llm_cache = params.0.llm_cache.unwrap_or(true);
        let readme_max_chars = params.0.readme_max_chars.unwrap_or(4000).min(50_000);
        let mut payload = crate_detail_payload(
            &root,
            &out,
            &params.0.krate,
            include_readme,
            readme_max_chars,
        )?;

        // Use TLC row (manifest_path) to locate README if present.
        let readme_path = payload
            .get("tlc_row")
            .and_then(|v| v.get("manifest_path"))
            .and_then(|v| v.as_str())
            .and_then(|mp| find_readme_for_manifest(&root, mp));

        let ai = maybe_add_readme_llm_summary(
            &root,
            &out,
            "crate",
            &params.0.krate,
            readme_path.as_ref(),
            summarize_readme,
            llm_input_max_chars,
            llm_timeout_secs,
            llm_cache,
        )
        .await?;
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("readme_ai".to_string(), ai);
        }
        mcp_ok("pkgrank_crate_detail", payload, None)
    }

    // NOTE: internal artifact inspection tools live in `PkgrankStdioMcpDebug`.

    #[tool(description = "Snapshot pkgrank artifacts into a new directory (copy selected files)")]
    async fn pkgrank_snapshot(
        &self,
        params: Parameters<PkgrankSnapshotArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let out = Self::default_out(params.0.out.as_deref());

        let label = params.0.label.clone().unwrap_or_else(|| {
            format!(
                "run-{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            )
        });

        let dest = if let Some(d) = params.0.dest.as_deref() {
            PathBuf::from(d)
        } else {
            // Default: `<out>/runs/<label>`
            out.join("runs").join(&label)
        };

        let src_dir = if out.is_absolute() {
            out.clone()
        } else {
            root.join(&out)
        };
        let dst_dir = if dest.is_absolute() {
            dest.clone()
        } else {
            root.join(&dest)
        };
        Self::ensure_dir(&dst_dir)?;

        let files = [
            "pkgrank_overview.html",
            "by_repo.summary.json",
            "root.workspace_only.json",
            "ecosystem.repo_graph.rows.json",
            "ecosystem.axes.pagerank.json",
            "ecosystem.invariants.violations.json",
            "tlc.crates.json",
            "tlc.repos.json",
            "ppr.entrypoints.json",
            "ppr.aggregate.json",
        ];

        let mut copied = Vec::new();
        let mut missing = Vec::new();
        for f in files {
            let src = src_dir.join(f);
            let dst = dst_dir.join(f);
            if !src.exists() {
                missing.push(src.display().to_string());
                continue;
            }
            if let Some(parent) = dst.parent() {
                Self::ensure_dir(parent)?;
            }
            std::fs::copy(&src, &dst).map_err(|e| {
                McpError::internal_error(
                    format!(
                        "failed to copy {} -> {}: {}",
                        src.display(),
                        dst.display(),
                        e
                    ),
                    None,
                )
            })?;
            copied.push(serde_json::json!({
                "src": src.display().to_string(),
                "dst": dst.display().to_string(),
            }));
        }

        let payload = serde_json::json!({
            "root": root.display().to_string(),
            "source_dir": src_dir.display().to_string(),
            "dest_dir": dst_dir.display().to_string(),
            "label": label,
            "copied": copied,
            "missing": missing,
        });
        mcp_ok("pkgrank_snapshot", payload, None)
    }

    #[tool(description = "Compare two pkgrank artifact directories (TLC crates/repos deltas)")]
    async fn pkgrank_compare_runs(
        &self,
        params: Parameters<PkgrankCompareRunsArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = Self::default_root(params.0.root.as_deref());
        let new_out = PathBuf::from(&params.0.new_out);
        let old_out = PathBuf::from(&params.0.old_out);
        let limit = params.0.limit.unwrap_or(25).min(200);

        let new_crates_path = Self::artifact_path(&root, &new_out, "tlc.crates.json");
        let old_crates_path = Self::artifact_path(&root, &old_out, "tlc.crates.json");
        let new_repos_path = Self::artifact_path(&root, &new_out, "tlc.repos.json");
        let old_repos_path = Self::artifact_path(&root, &old_out, "tlc.repos.json");

        let new_crates: Vec<TlcCrateRow> = Self::read_json_file(&new_crates_path)?;
        let old_crates: Vec<TlcCrateRow> = Self::read_json_file(&old_crates_path)?;
        let new_repos: Vec<TlcRepoRow> = Self::read_json_file(&new_repos_path)?;
        let old_repos: Vec<TlcRepoRow> = Self::read_json_file(&old_repos_path)?;

        let mut old_crate_rank: HashMap<String, usize> = HashMap::new();
        for (i, r) in old_crates.iter().enumerate() {
            old_crate_rank.insert(r.name.clone(), i + 1);
        }
        let mut old_repo_rank: HashMap<String, usize> = HashMap::new();
        for (i, r) in old_repos.iter().enumerate() {
            old_repo_rank.insert(r.repo.clone(), i + 1);
        }

        let mut crate_deltas = Vec::new();
        for (i, r) in new_crates.iter().enumerate() {
            let new_rank = i + 1;
            let old_rank = old_crate_rank.get(&r.name).copied();
            let delta_rank = old_rank.map(|o| (o as i64) - (new_rank as i64));
            crate_deltas.push(serde_json::json!({
                "crate": r.name,
                "repo": r.repo,
                "axis": r.axis,
                "new_rank": new_rank,
                "old_rank": old_rank,
                "delta_rank": delta_rank,
                "new_score": r.score,
            }));
        }
        crate_deltas.sort_by(|a, b| {
            let da = a
                .get("delta_rank")
                .and_then(|x| x.as_i64())
                .unwrap_or(0)
                .abs();
            let db = b
                .get("delta_rank")
                .and_then(|x| x.as_i64())
                .unwrap_or(0)
                .abs();
            db.cmp(&da)
        });

        let mut repo_deltas = Vec::new();
        for (i, r) in new_repos.iter().enumerate() {
            let new_rank = i + 1;
            let old_rank = old_repo_rank.get(&r.repo).copied();
            let delta_rank = old_rank.map(|o| (o as i64) - (new_rank as i64));
            repo_deltas.push(serde_json::json!({
                "repo": r.repo,
                "axis": r.axis,
                "new_rank": new_rank,
                "old_rank": old_rank,
                "delta_rank": delta_rank,
                "new_score": r.score,
            }));
        }
        repo_deltas.sort_by(|a, b| {
            let da = a
                .get("delta_rank")
                .and_then(|x| x.as_i64())
                .unwrap_or(0)
                .abs();
            let db = b
                .get("delta_rank")
                .and_then(|x| x.as_i64())
                .unwrap_or(0)
                .abs();
            db.cmp(&da)
        });

        let payload = serde_json::json!({
            "root": root.display().to_string(),
            "new_out": new_out.display().to_string(),
            "old_out": old_out.display().to_string(),
            "paths": {
                "new_tlc_crates": new_crates_path.display().to_string(),
                "old_tlc_crates": old_crates_path.display().to_string(),
                "new_tlc_repos": new_repos_path.display().to_string(),
                "old_tlc_repos": old_repos_path.display().to_string(),
            },
            "crate_rank_deltas": crate_deltas.into_iter().take(limit).collect::<Vec<_>>(),
            "repo_rank_deltas": repo_deltas.into_iter().take(limit).collect::<Vec<_>>(),
        });
        mcp_ok("pkgrank_compare_runs", payload, None)
    }

    #[tool(description = "Compute local crate ranking (pkgrank analyze)")]
    async fn pkgrank_analyze(
        &self,
        params: Parameters<PkgrankAnalyzeToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let path = params.0.path.clone().unwrap_or_else(|| ".".to_string());
        let metric = params
            .0
            .metric
            .as_deref()
            .map(parse_metric)
            .transpose()?
            .unwrap_or(Metric::Pagerank);
        let analyze = AnalyzeArgs {
            ecosystem: None,
            path: PathBuf::from(path),
            metric,
            top: params.0.top.unwrap_or(25),
            dev: params.0.dev.unwrap_or(false),
            build: params.0.build.unwrap_or(false),
            workspace_only: params.0.workspace_only.unwrap_or(true),
            all_features: false,
            no_default_features: false,
            features: None,
            format: OutputFormat::Json,
            stats: false,
            json_limit: None,
            cache: false,
            cache_refresh: false,
        };
        let (rows, convergence) = analyze_rows_with_convergence(&analyze)
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let rows_total = rows.len();
        let rows_returned = rows_total.min(analyze.top);
        let rows = rows.into_iter().take(analyze.top).collect::<Vec<_>>();
        let truncated = rows_returned < rows_total;
        let payload = serde_json::json!({
            "rows": rows,
            "rows_total": rows_total,
            "rows_returned": rows_returned,
            "truncated": truncated,
            "limit": analyze.top,
            "convergence": convergence,
            "metric": format!("{:?}", metric),
            "sorted_by": format!("{:?}", metric),
        });
        mcp_ok("pkgrank_analyze", payload, None)
    }

    #[tool(
        description = "Compute internal module/item centrality (pkgrank modules; cargo-modules-backed)"
    )]
    async fn pkgrank_modules(
        &self,
        params: Parameters<PkgrankModulesToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let mut args = modules_args_from_tool_params(&params.0)?;
        let (rows, nodes, edges, aggregate_label, top_edges) = match run_modules_core(&args) {
            Ok(v) => v,
            Err(e) => {
                // cargo-modules errors if a package has both --lib and --bin targets and neither
                // is explicitly selected. For MCP UX, auto-fallback to --lib when it looks safe.
                let msg = format!("{:#}", e);
                if !args.lib && args.bin.is_none() && msg.contains("Multiple targets present") {
                    args.lib = true;
                    run_modules_core(&args)
                        .map_err(|e2| McpError::internal_error(format!("{:#}", e2), None))?
                } else {
                    return Err(McpError::internal_error(msg, None));
                }
            }
        };
        let rows_total = rows.len();
        let rows_returned = rows_total.min(args.top);
        let rows = rows.into_iter().take(args.top).collect::<Vec<_>>();
        let truncated = rows_returned < rows_total;
        let payload = serde_json::json!({
            "effective": {
                "manifest_path": args.manifest_path.display().to_string(),
                "package": args.package,
                "target": if args.lib {
                    "lib".to_string()
                } else if let Some(bin) = args.bin.as_ref() {
                    format!("bin={bin}")
                } else {
                    "default".to_string()
                },
                "cfg_test": args.cfg_test,
                "preset": match args.preset {
                    ModulesPreset::None => None,
                    p => Some(format!("{p:?}")),
                },
                "metric": format!("{:?}", args.metric),
                "edge_kind": format!("{:?}", args.edge_kind),
                "aggregate": format!("{:?}", args.aggregate),
                "top": args.top,
                "include": {
                    "externs": args.include_externs,
                    "sysroot": args.include_sysroot,
                    "fns": args.include_fns,
                    "types": args.include_types,
                    "traits": args.include_traits,
                },
                "cache": args.cache,
                "cache_refresh": args.cache_refresh,
            },
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "aggregate_label": aggregate_label,
            },
            "rows": rows,
            "rows_total": rows_total,
            "rows_returned": rows_returned,
            "truncated": truncated,
            "limit": args.top,
            "top_edges": top_edges,
        });
        mcp_ok("pkgrank_modules", payload, None)
    }

    #[tool(
        description = "Compute internal module/item centrality across multiple packages (pkgrank modules-sweep)"
    )]
    async fn pkgrank_modules_sweep(
        &self,
        params: Parameters<PkgrankModulesSweepToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let args = modules_sweep_args_from_tool_params(&params.0)?;
        let include_rows = params.0.include_rows.unwrap_or(false);
        let include_top_edges = params.0.include_top_edges.unwrap_or(false);
        let payload = modules_sweep_payload(&args, include_rows, include_top_edges)
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        mcp_ok("pkgrank_modules_sweep", payload, None)
    }

    #[tool(
        description = "Show all packages that transitively depend on a given package (blast radius)"
    )]
    async fn pkgrank_blast_radius(
        &self,
        params: Parameters<PkgrankBlastRadiusToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let args = BlastRadiusArgs {
            package: params.0.package.clone(),
            path: PathBuf::from(params.0.path.as_deref().unwrap_or(".")),
            ecosystem: None,
            dev: params.0.dev.unwrap_or(false),
            build: params.0.build.unwrap_or(false),
            workspace_only: params.0.workspace_only.unwrap_or(true),
            format: OutputFormat::Json,
            top: params.0.top.unwrap_or(50),
            cache: false,
            cache_refresh: false,
        };
        let result = compute_blast_radius(&args)
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let payload = serde_json::to_value(&result)
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        mcp_ok("pkgrank_blast_radius", payload, None)
    }

    #[tool(description = "Combine cargo outdated with centrality ranking to prioritize upgrades")]
    async fn pkgrank_upgrade_priority(
        &self,
        params: Parameters<PkgrankUpgradePriorityToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let args = UpgradePriorityArgs {
            path: PathBuf::from(params.0.path.as_deref().unwrap_or(".")),
            dev: params.0.dev.unwrap_or(false),
            build: params.0.build.unwrap_or(false),
            workspace_only: params.0.workspace_only.unwrap_or(true),
            top: params.0.top.unwrap_or(20),
            format: OutputFormat::Json,
            cache: false,
            cache_refresh: false,
        };
        let (outdated_count, rows) = compute_upgrade_priority(&args)
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let payload = serde_json::json!({
            "outdated_count": outdated_count,
            "rows": rows,
        });
        mcp_ok("pkgrank_upgrade_priority", payload, None)
    }

    #[tool(
        description = "Analyze a non-Cargo lock file (npm package-lock.json, Python uv.lock, Go go.mod)"
    )]
    async fn pkgrank_polyglot(
        &self,
        params: Parameters<PkgrankPolyglotToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let ecosystem = match params.0.ecosystem.as_str() {
            "js" | "npm" => dep_graph::Ecosystem::Js,
            "python" => dep_graph::Ecosystem::Python,
            "go" => dep_graph::Ecosystem::Go,
            other => {
                return Err(McpError::invalid_params(
                    format!("ecosystem must be one of: js, python, go (got {other})"),
                    None,
                ));
            }
        };
        let metric = params
            .0
            .metric
            .as_deref()
            .map(parse_metric)
            .transpose()?
            .unwrap_or(Metric::Pagerank);
        let top = params.0.top.unwrap_or(25);
        let path = PathBuf::from(params.0.path.as_deref().unwrap_or("."));
        let (_graph, _map, rows) = polyglot_analyze(ecosystem, &path, metric)
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let rows_total = rows.len();
        let rows: Vec<_> = rows.into_iter().take(top).collect();
        let payload = serde_json::json!({
            "ecosystem": ecosystem,
            "rows_total": rows_total,
            "rows_returned": rows.len(),
            "rows": rows,
        });
        mcp_ok("pkgrank_polyglot", payload, None)
    }

    #[tool(
        description = "Analyze file-level import graph within a project. Static parsing (no toolchain required). Supports local paths, GitHub URLs, and owner/repo shorthand. Returns centrality scores, cycles, orphans, and blast radius per file."
    )]
    async fn pkgrank_files(
        &self,
        params: Parameters<PkgrankFilesToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        let ecosystem = params
            .0
            .ecosystem
            .as_deref()
            .map(|e| match e {
                "rust" | "cargo" => Ok(dep_graph::Ecosystem::Rust),
                "js" | "npm" => Ok(dep_graph::Ecosystem::Js),
                "python" => Ok(dep_graph::Ecosystem::Python),
                "go" => Ok(dep_graph::Ecosystem::Go),
                other => Err(McpError::invalid_params(
                    format!("ecosystem must be one of: rust, js, python, go (got {other})"),
                    None,
                )),
            })
            .transpose()?;
        let metric = params
            .0
            .metric
            .as_deref()
            .map(parse_metric)
            .transpose()?
            .unwrap_or(Metric::Pagerank);
        let top = params.0.top.unwrap_or(25);
        let args = FilesArgs {
            path: params.0.path.unwrap_or_else(|| ".".to_string()),
            ecosystem,
            include_tests: params.0.include_tests.unwrap_or(false),
            include_all: false,
            metric,
            top,
            format: OutputFormat::Json,
            directory: false,
            cache: true,
            focus: None,
            git: false,
            git_days: 90,
            store: false,
        };
        let result =
            files_analyze(&args).map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let rows_total = result.rows.len();
        let rows: Vec<_> = result.rows.into_iter().take(top).collect();
        let payload = serde_json::json!({
            "ecosystem": result.ecosystem,
            "nodes": result.nodes,
            "edges": result.edges,
            "orphan_count": result.orphan_count,
            "cycle_count": result.cycles.len(),
            "cycles": result.cycles,
            "rows_total": rows_total,
            "rows_returned": rows.len(),
            "rows": rows,
        });
        mcp_ok("pkgrank_files", payload, None)
    }
}

#[tool_handler]
impl rmcp::ServerHandler for PkgrankStdioMcpFull {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Tools for ranking crates/repos in the local Cargo workspace (pkgrank).\n\n\
                 Toolsets:\n\
                 - default: PKGRANK_MCP_TOOLSET=slim (small surface)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=full (includes module/type graph tools)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=debug (internal artifact inspection tools)"
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// PkgrankStdioMcpSlim impl
// ---------------------------------------------------------------------------

// Slim stdio MCP server: smaller tool surface for day-to-day use.
#[tool_router]
impl PkgrankStdioMcpSlim {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Triage bundle: top TLC crates/repos + invariants + PPR top-k (artifact-backed)"
    )]
    async fn pkgrank_triage(
        &self,
        params: Parameters<PkgrankTriageArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let mode = match params.0.mode.as_deref().unwrap_or("local") {
            "local" => ViewMode::Local,
            "cratesio" => ViewMode::CratesIo,
            "both" => ViewMode::Both,
            other => {
                return Err(McpError::invalid_params(
                    format!("mode must be one of: local, cratesio, both (got {other})"),
                    None,
                ));
            }
        };

        // Reuse the CLI triage implementation by mapping MCP params into CLI args.
        let cli = TriageCliArgs {
            root,
            out,
            refresh_if_missing: params.0.refresh_if_missing.unwrap_or(true),
            mode,
            stale_minutes: params.0.stale_minutes.unwrap_or(60),
            limit: params.0.limit.unwrap_or(15),
            axis: params.0.axis.clone(),
            ppr_top: params.0.ppr_top.unwrap_or(12),
            summarize_readmes: params.0.summarize_readmes.unwrap_or(false),
            summarize_repos_top: params.0.summarize_repos_top.unwrap_or(0),
            summarize_crates_top: params.0.summarize_crates_top.unwrap_or(0),
            llm_input_max_chars: params.0.llm_input_max_chars.unwrap_or(12_000),
            llm_timeout_secs: params.0.llm_timeout_secs.unwrap_or(30),
            llm_cache: params.0.llm_cache.unwrap_or(true),
            llm_include_raw: params.0.llm_include_raw.unwrap_or(false),
            format: OutputFormat::Json,
        };

        let (payload, summary) = triage_payload_from_cli(&cli)
            .await
            .map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        mcp_ok("pkgrank_triage", payload, Some(summary))
    }

    #[tool(description = "Generate pkgrank artifacts (pkgrank view)")]
    async fn pkgrank_view(
        &self,
        params: Parameters<PkgrankViewToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        // Delegate to full implementation logic by constructing args locally (copy, not reuse trait objects).
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let mode = match params.0.mode.as_deref().unwrap_or("local") {
            "local" => ViewMode::Local,
            "cratesio" => ViewMode::CratesIo,
            "both" => ViewMode::Both,
            other => {
                return Err(McpError::invalid_params(
                    format!("mode must be one of: local, cratesio, both (got {other})"),
                    None,
                ));
            }
        };

        let args = ViewArgs {
            root,
            out,
            mode,
            local_top: 10,
            cratesio_depth: 2,
            cratesio_dev: false,
            cratesio_build: false,
            cratesio_optional: false,
            quiet: true,
        };

        run_view(&args).map_err(|e| McpError::internal_error(format!("{:#}", e), None))?;
        let html_path =
            PkgrankStdioMcpFull::artifact_path(&args.root, &args.out, "pkgrank_overview.html");
        let payload = serde_json::json!({
            "root": args.root.display().to_string(),
            "out_dir": if args.out.is_absolute() { args.out.display().to_string() } else { args.root.join(&args.out).display().to_string() },
            "html_path": html_path.display().to_string(),
        });
        mcp_ok("pkgrank_view", payload, None)
    }

    #[tool(description = "Compute local crate ranking (pkgrank analyze)")]
    async fn pkgrank_analyze(
        &self,
        params: Parameters<PkgrankAnalyzeToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull {
            tool_router: PkgrankStdioMcpFull::tool_router(),
        }
        .pkgrank_analyze(params)
        .await
    }

    #[tool(description = "Get repo details from artifacts")]
    async fn pkgrank_repo_detail(
        &self,
        params: Parameters<PkgrankRepoDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let include_readme = params.0.include_readme.unwrap_or(false);
        let readme_max_chars = params.0.readme_max_chars.unwrap_or(4000).min(50_000);
        let payload = repo_detail_payload(
            &root,
            &out,
            &params.0.repo,
            include_readme,
            readme_max_chars,
        )?;
        mcp_ok("pkgrank_repo_detail", payload, None)
    }

    #[tool(description = "Get crate details from artifacts")]
    async fn pkgrank_crate_detail(
        &self,
        params: Parameters<PkgrankCrateDetailArgs>,
    ) -> Result<CallToolResult, McpError> {
        let root = PkgrankStdioMcpFull::default_root(params.0.root.as_deref());
        let out = PkgrankStdioMcpFull::default_out(params.0.out.as_deref());
        let include_readme = params.0.include_readme.unwrap_or(false);
        let readme_max_chars = params.0.readme_max_chars.unwrap_or(4000).min(50_000);
        let payload = crate_detail_payload(
            &root,
            &out,
            &params.0.krate,
            include_readme,
            readme_max_chars,
        )?;
        mcp_ok("pkgrank_crate_detail", payload, None)
    }

    #[tool(description = "Snapshot pkgrank artifacts (copy selected files)")]
    async fn pkgrank_snapshot(
        &self,
        params: Parameters<PkgrankSnapshotArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull {
            tool_router: PkgrankStdioMcpFull::tool_router(),
        }
        .pkgrank_snapshot(params)
        .await
    }

    #[tool(description = "Compare two artifact directories (TLC deltas)")]
    async fn pkgrank_compare_runs(
        &self,
        params: Parameters<PkgrankCompareRunsArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull {
            tool_router: PkgrankStdioMcpFull::tool_router(),
        }
        .pkgrank_compare_runs(params)
        .await
    }

    #[tool(
        description = "Show all packages that transitively depend on a given package (blast radius)"
    )]
    async fn pkgrank_blast_radius(
        &self,
        params: Parameters<PkgrankBlastRadiusToolArgs>,
    ) -> Result<CallToolResult, McpError> {
        PkgrankStdioMcpFull::new()
            .pkgrank_blast_radius(params)
            .await
    }
}

#[tool_handler]
impl rmcp::ServerHandler for PkgrankStdioMcpSlim {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Tools for ranking crates/repos in the local Cargo workspace (pkgrank).\n\n\
                 Toolsets:\n\
                 - default: PKGRANK_MCP_TOOLSET=slim (small surface)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=full (includes module/type graph tools)\n\
                 - opt-in:  PKGRANK_MCP_TOOLSET=debug (internal artifact inspection tools)"
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
