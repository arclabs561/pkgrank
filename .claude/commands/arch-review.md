# /arch-review -- Architectural review of pkgrank

Audit the structural design: module decomposition, graph model correctness, MCP surface, CLI contract, and JSON stability. Read-only -- does not modify code.

## Procedure

### 0. Read prior arch reports

```bash
eza --sort=modified -r .claude/reports/arch-*.md 2>/dev/null | head -3
```

### 1. Module structure assessment

Current structure:

```
src/
├── main.rs      (7500+ lines -- CLI, MCP, graph, triage, sweep, HTML, JSON)
└── graphops.rs   (500 lines -- PageRank, PPR, betweenness, reachability)
```

This is the primary structural debt. Assess:
- Which logical subsystems could be extracted into modules?
- Recommended decomposition (at minimum): `cli.rs`, `graph.rs`, `mcp.rs`, `triage.rs`, `sweep.rs`, `html.rs`
- What data structures are shared across subsystems? (these become the module boundary interfaces)
- Would extraction enable unit testing of subsystems that are currently only E2E-tested?

### 2. Graph model invariants

The declared invariants (from README):

```
Edge meaning: A → B means "A depends on B"
Dependency kind gating: --dev / --build control edge inclusion
Workspace restriction: "workspace-only" restricts to workspace members
```

Verify in code:
- Edge direction is consistent everywhere (graph construction, PageRank interpretation, triage reports)
- `consumers-pagerank` correctly reverses the graph (not re-labels edges)
- `--workspace-only` filtering is applied before graph algorithms, not after (ordering matters for PageRank)

### 3. PageRank correctness

`graphops.rs` implements:
- Unweighted PageRank
- Weighted PageRank
- Personalized PageRank (PPR)
- Reachability counts
- Betweenness centrality (Brandes)

Audit:
- **Dangling node handling**: nodes with no outgoing edges should distribute mass uniformly (teleport). Verify the dangling sum is computed correctly.
- **Convergence**: check that the L1 diff is computed before the swap (not after).
- **Weighted PR**: edge weights must be non-negative. The checked variant validates this -- verify the unchecked variant documents the assumption.
- **PPR personalization**: verify the personalization vector is normalized.
- **Betweenness normalization**: `1/((n-1)(n-2))` is correct for directed graphs. Verify.

### 4. MCP tool surface

pkgrank exposes an MCP stdio server with tiered toolsets:

```
Default (slim): pkgrank_view, pkgrank_triage, pkgrank_analyze, pkgrank_repo_detail,
                pkgrank_crate_detail, pkgrank_snapshot, pkgrank_compare_runs
Full:           + pkgrank_status, pkgrank_modules, pkgrank_modules_sweep
Debug:          + internal artifact-inspection tools
```

Audit:
- **Tool boundedness**: every tool must have explicit limits (`-n`, `--limit`, pagination). Flag unbounded tools.
- **Read-only tools**: verify none of the "view"/"triage"/"analyze" tools write files as a side effect.
- **Write tools**: `sweep-local`, `view` write artifacts. Verify they write to controlled paths only.
- **Error discipline**: tool errors must include what failed, why, and next move. Grep for generic "failed" messages.

### 5. CLI contract stability

```bash
cargo run -- --help 2>&1
```

Check:
- **Subcommands**: analyze (default), modules, modules-sweep, sweep-local, triage, cratesio, mcp-stdio
- **Global flags**: `--format`, `--stats`, `-n`, `--workspace-only`, `--metric`
- **Exit codes**: 0 for success, non-zero for errors. Verify error paths use proper exit codes (not panic).

### 6. JSON envelope stability

The JSON wrapper is a contract:

```json
{
  "schema_version": 1,
  "ok": true,
  "command": "...",
  "rows": [...]
}
```

Audit:
- Is `schema_version` bumped when the schema changes?
- Are new fields added additively (not removing old ones)?
- Is `truncated` reliable (not just "we think we truncated")?
- Do all subcommands use the same envelope, or do some bypass it?

### 7. Artifact system

pkgrank writes artifacts to `evals/pkgrank/`:

```bash
ls evals/pkgrank/ 2>/dev/null
```

Audit:
- Are artifact paths deterministic (same input -> same path)?
- Is the cache (`modules_cache/`) invalidated correctly?
- Are artifacts gitignored?

### 8. Error handling

```bash
rg 'unwrap\(\)|panic!\(|expect\(|anyhow!' --type rust src/ -n
```

pkgrank uses `anyhow` for error handling. Check:
- Are errors propagated with context (`with_context`)?
- Are there raw `unwrap()` calls in non-test code that could panic on user input?
- Does the MCP server handle errors gracefully (return error response, not crash)?

### 9. Self-containment audit

A public repo must build and test standalone:
- No `path = "../sibling"` deps in Cargo.toml
- No tests that require sibling repos by default (super-workspace tests are gated)
- No hardcoded paths to `~/Documents/dev/` or similar

```bash
rg '\.\./|~/Documents|/Users/' --type rust src/ tests/ -n
```

Flag any remaining cross-repo references in default (non-gated) code paths.

### 10. Write the report

Save to `.claude/reports/arch-YYYY-MM-DD.md`. Structure:

1. **Module structure**: current state, decomposition recommendations
2. **Graph model**: invariant verification results
3. **Algorithm correctness**: PageRank/PPR/betweenness audit findings
4. **MCP surface**: boundedness, side effects, error discipline
5. **CLI contract**: stability, exit codes, help text
6. **JSON stability**: envelope compliance, schema versioning
7. **Error handling**: unwrap census, anyhow context coverage
8. **Self-containment**: cross-repo reference scan
9. **Actionable items**: ordered by impact

## What this is NOT

- Not a QA check (that's `/qa`)
- Not a performance analysis
- Not a monolith decomposition plan (though findings here inform one)

This answers: "is pkgrank's architecture sound, self-contained, and contract-stable?"
