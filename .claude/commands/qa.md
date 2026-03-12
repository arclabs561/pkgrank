# /qa -- Quality audit of pkgrank

Run a comprehensive quality pass: build, lint, test, E2E contract tests, MCP smoke, and structural checks. Produces a timestamped report in `.claude/reports/`.

## Execution strategy

- **Stop early on build failure**: if `cargo check --all-features` fails, the QA is blocked. Report and stop.
- **Capture exact output**: save command output to temp files so findings are reliable. Don't eyeball scrollback.
- **Read previous reports first**: comparison with prior runs catches regressions.
- **Full output**: read all diagnostic output. Do not truncate or pipe through head/tail.
- **Self-contained tests only**: default test suite must pass with NO sibling repos. Tests requiring the super-workspace are gated behind `PKGRANK_E2E_SUPERWORKSPACE=1`.

## Report convention

Write to `.claude/reports/qa-YYYY-MM-DD.md` (globally gitignored via `~/.gitignore_global`). For same-day reruns, append `-v2`, `-v3`.

## Procedure

### 0. Read prior QA reports

Check for prior reports in order: `.claude/reports/`, `qa/reports/`, `.qa/reports/`, `.claude/` root (flat files like `audit-report.md`). Read the most recent found. If reports exist in old locations, move them to `.claude/reports/` with dated names before proceeding.

```bash
eza --sort=modified -r .claude/reports/qa-*.md qa/reports/qa-*.md 2>/dev/null | head -3
```

Read the most recent 1-2 reports if they exist. Note open issues to watch for.

### 1. Build check

```bash
cargo check --all-features 2>&1
```

If this fails, stop and report. Everything else depends on compilation.

### 2. Format check

```bash
cargo fmt -- --check
```

### 3. Clippy

```bash
cargo clippy --all-targets --all-features -- -D warnings 2>&1
```

### 4. Tests (self-contained)

```bash
cargo test 2>&1
```

All tests must pass without sibling repos, without network, without env vars. Record: total tests, pass/fail count.

Critical test contracts:
- **JSON envelope**: `schema_version`, `ok`, `command`, `rows` fields present
- **stderr/stdout separation**: stats go to stderr, JSON to stdout
- **Self-analysis**: pkgrank can analyze its own dependency graph
- **Triage from artifacts**: synthetic artifact read-back works

### 5. Graph algorithm unit tests

```bash
cargo test graphops 2>&1
```

These test PageRank, PPR, betweenness centrality, and reachability on known-answer graphs. Verify all pass and check that edge cases are covered:
- Empty graph
- Single node
- Chain, cycle, star, diamond topologies

### 6. Optional: super-workspace E2E

```bash
PKGRANK_E2E_SUPERWORKSPACE=1 cargo test analyze_on_super_workspace_root 2>&1
```

Only run this when testing in the dev super-workspace. Skip in CI or standalone checkouts.

### 7. Optional: network E2E

```bash
PKGRANK_E2E_NETWORK=1 cargo test cratesio_crawl 2>&1
```

Tests crates.io crawl. Only run when network is available and you want to verify crawl still works.

### 8. Doc compilation

```bash
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features 2>&1
```

Note: pkgrank is a binary crate, so doc surface is minimal. `graphops` module docs should compile clean.

### 9. Structural checks

#### 9a. Monolith assessment

`src/main.rs` is the known structural debt (7500+ lines). Track its line count:

```bash
wc -l src/main.rs src/graphops.rs
```

If it has grown since last QA, note the delta. Eventually this should be decomposed into modules.

#### 9b. Error handling

```bash
rg 'unwrap\(\)|panic!\(' --type rust src/ -n
```

`graphops.rs` uses a stringly-typed `Error(String)`. This is acceptable for an internal module but should be noted.

#### 9c. MCP tool surface

Verify the MCP toolset descriptions match actual behavior:

```bash
rg 'tool_name|description.*=' --type rust src/main.rs -n | head -30
```

Check: tool names are stable, descriptions are accurate, no tools are accidentally exposed in the default (slim) toolset.

#### 9d. JSON output stability

The JSON envelope is a contract. Verify these fields exist in the output:

```bash
cargo run -- --format json -n 1 . 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print([k for k in d.keys()])"
```

Expected keys: `schema_version`, `ok`, `command`, `metric`, `sorted_by`, `convergence`, `rows`, `rows_total`, `rows_returned`, `truncated`, `json_limit`.

### 10. Pre-publish gate (when applicable)

pkgrank is currently `publish = false`. When/if that changes:

#### 10a. Version coherence

- README says "not on crates.io" -- update if publishing
- Remove `publish = false` from Cargo.toml

#### 10b. Dependency check

```bash
cargo tree --edges no-dev --prefix none 2>&1 | sort -u
```

No path deps should be in the publish manifest.

#### 10c. Self-containment check

All default tests must pass in a fresh clone with no sibling repos:

```bash
git clone https://github.com/arclabs561/pkgrank /tmp/pkgrank-test
cd /tmp/pkgrank-test && cargo test
```

### 11. Write the report

Save to `.claude/reports/qa-YYYY-MM-DD.md`. Structure:

1. **Test conditions**: date, commit SHA, rustc version
2. **Check results**: pass/fail for each check (fmt, clippy, tests, graphops, doc)
3. **Test matrix**: self-contained vs super-workspace vs network results
4. **Structural assessment**: main.rs line count, monolith status
5. **Contract checks**: JSON envelope, MCP tools, stderr/stdout
6. **Bug table**: concrete issues found with file:line references
7. **Comparison with prior run**: regressions, improvements
8. **Actionable items**: specific things worth fixing, ordered by impact

## What this is NOT

- Not a performance benchmark
- Not an auto-fixer
- Not an architectural decomposition plan (that's `/arch-review`)

This answers: "is pkgrank healthy, correct, and self-contained?"
