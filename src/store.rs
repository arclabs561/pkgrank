use anyhow::Result;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::files::FilesResult;

const SCHEMA_VERSION: i32 = 1;

/// Open or create the pkgrank SQLite database.
pub(crate) fn open_db(db_path: &Path) -> Result<Connection> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
    migrate(&conn)?;
    Ok(conn)
}

/// Default database path: ~/.local/share/pkgrank/pkgrank.db
pub(crate) fn default_db_path() -> PathBuf {
    let base = dirs_next::data_dir()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".local/share"))
        })
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("pkgrank").join("pkgrank.db")
}

fn migrate(conn: &Connection) -> Result<()> {
    let version: i32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;

    if version < 1 {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                ecosystem TEXT NOT NULL,
                last_analyzed INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                analyzed_at INTEGER NOT NULL,
                git_rev TEXT,
                node_count INTEGER,
                edge_count INTEGER,
                cycle_count INTEGER,
                orphan_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                snapshot_id INTEGER REFERENCES snapshots(id),
                path TEXT NOT NULL,
                role TEXT NOT NULL,
                pagerank REAL,
                consumers_pagerank REAL,
                betweenness REAL,
                in_degree INTEGER,
                out_degree INTEGER,
                dependents INTEGER,
                dependencies INTEGER,
                commits INTEGER,
                churn_risk REAL
            );

            CREATE TABLE IF NOT EXISTS external_deps (
                file_id INTEGER REFERENCES files(id),
                package_name TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_files_snapshot ON files(snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
            CREATE INDEX IF NOT EXISTS idx_snapshots_project ON snapshots(project_id);
            ",
        )?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    }

    Ok(())
}

/// Store a FilesResult as a new snapshot.
/// Get the current git HEAD rev for a directory.
pub(crate) fn git_head_rev(dir: &Path) -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(dir)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

pub(crate) fn store_snapshot(
    conn: &Connection,
    project_path: &str,
    result: &FilesResult,
) -> Result<i64> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let ecosystem = format!("{}", result.ecosystem);

    // Upsert project.
    conn.execute(
        "INSERT INTO projects (path, ecosystem, last_analyzed) VALUES (?1, ?2, ?3)
         ON CONFLICT(path) DO UPDATE SET ecosystem=?2, last_analyzed=?3",
        params![project_path, ecosystem, now],
    )?;
    let project_id: i64 = conn.query_row(
        "SELECT id FROM projects WHERE path = ?1",
        params![project_path],
        |row| row.get(0),
    )?;

    // Detect git revision.
    let git_rev = git_head_rev(Path::new(project_path));

    // Insert snapshot.
    conn.execute(
        "INSERT INTO snapshots (project_id, analyzed_at, git_rev, node_count, edge_count, cycle_count, orphan_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            project_id,
            now,
            git_rev,
            result.nodes,
            result.edges,
            result.cycles.len(),
            result.orphan_count,
        ],
    )?;
    let snapshot_id = conn.last_insert_rowid();

    // Insert files.
    let mut file_stmt = conn.prepare(
        "INSERT INTO files (snapshot_id, path, role, pagerank, consumers_pagerank, betweenness,
         in_degree, out_degree, dependents, dependencies, commits, churn_risk)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
    )?;
    let mut dep_stmt =
        conn.prepare("INSERT INTO external_deps (file_id, package_name) VALUES (?1, ?2)")?;

    for row in &result.rows {
        file_stmt.execute(params![
            snapshot_id,
            row.file,
            format!("{:?}", row.role).to_lowercase(),
            row.pagerank,
            row.consumers_pagerank,
            row.betweenness,
            row.in_degree,
            row.out_degree,
            row.dependents,
            row.dependencies,
            row.commits,
            row.churn_risk,
        ])?;
        let file_id = conn.last_insert_rowid();

        for dep in &row.external_deps {
            dep_stmt.execute(params![file_id, dep])?;
        }
    }

    Ok(snapshot_id)
}

/// Query: top files by churn risk across all projects.
pub(crate) fn query_top_churn(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<(String, String, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT p.path, f.path, f.churn_risk
         FROM files f
         JOIN snapshots s ON f.snapshot_id = s.id
         JOIN projects p ON s.project_id = p.id
         WHERE f.churn_risk IS NOT NULL AND f.churn_risk > 0
         AND s.id = (SELECT MAX(s2.id) FROM snapshots s2 WHERE s2.project_id = p.id)
         ORDER BY f.churn_risk DESC
         LIMIT ?1",
    )?;
    let rows = stmt
        .query_map(params![limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

/// Query: centrality drift -- files whose PageRank changed most between last two snapshots.
pub(crate) fn query_drift(
    conn: &Connection,
    project_path: &str,
    limit: usize,
) -> Result<Vec<(String, f64, f64, f64)>> {
    let mut stmt = conn.prepare(
        "WITH latest AS (
            SELECT id FROM snapshots WHERE project_id = (
                SELECT id FROM projects WHERE path = ?1
            ) ORDER BY analyzed_at DESC, id DESC LIMIT 1
        ), prev AS (
            SELECT id FROM snapshots WHERE project_id = (
                SELECT id FROM projects WHERE path = ?1
            ) ORDER BY analyzed_at DESC, id DESC LIMIT 1 OFFSET 1
        )
        SELECT f2.path, f1.pagerank as prev_pr, f2.pagerank as curr_pr,
               (f2.pagerank - f1.pagerank) as delta
        FROM files f2
        JOIN latest ON f2.snapshot_id = latest.id
        LEFT JOIN files f1 ON f1.path = f2.path
            AND f1.snapshot_id = (SELECT id FROM prev)
        WHERE f1.pagerank IS NOT NULL
        ORDER BY ABS(f2.pagerank - f1.pagerank) DESC
        LIMIT ?2",
    )?;
    let rows = stmt
        .query_map(params![project_path, limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

/// Query: compare two snapshots of a project and show what changed.
pub(crate) fn query_compare(
    conn: &Connection,
    project_path: &str,
) -> Result<(Option<CompareResult>, Option<String>, Option<String>)> {
    // Get last two snapshot IDs.
    let mut stmt = conn.prepare(
        "SELECT id, git_rev FROM snapshots WHERE project_id = (
            SELECT id FROM projects WHERE path = ?1
        ) ORDER BY analyzed_at DESC, id DESC LIMIT 2",
    )?;
    let snaps: Vec<(i64, Option<String>)> = stmt
        .query_map(params![project_path], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(|r| r.ok())
        .collect();

    if snaps.len() < 2 {
        return Ok((None, None, None));
    }

    let (curr_id, curr_rev) = &snaps[0];
    let (prev_id, prev_rev) = &snaps[1];

    // Get file metrics for both snapshots.
    let mut file_stmt = conn.prepare(
        "SELECT path, pagerank, in_degree, dependents FROM files WHERE snapshot_id = ?1",
    )?;

    let curr: HashMap<String, (f64, i64, i64)> = file_stmt
        .query_map(params![curr_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                (row.get(1)?, row.get(2)?, row.get(3)?),
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let prev: HashMap<String, (f64, i64, i64)> = file_stmt
        .query_map(params![prev_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                (row.get(1)?, row.get(2)?, row.get(3)?),
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    // Find files that were added, removed, or changed PageRank.
    let mut added: Vec<String> = Vec::new();
    let mut removed: Vec<String> = Vec::new();
    let mut changed: Vec<(String, f64, f64)> = Vec::new(); // (file, prev_pr, curr_pr)

    for (file, (pr, _, _)) in &curr {
        if let Some((prev_pr, _, _)) = prev.get(file) {
            let delta = (pr - prev_pr).abs();
            if delta > 1e-6 {
                changed.push((file.clone(), *prev_pr, *pr));
            }
        } else {
            added.push(file.clone());
        }
    }
    for file in prev.keys() {
        if !curr.contains_key(file) {
            removed.push(file.clone());
        }
    }

    changed.sort_by(|a, b| (b.2 - b.1).abs().total_cmp(&(a.2 - a.1).abs()));

    Ok((
        Some(CompareResult {
            added,
            removed,
            changed,
            curr_files: curr.len(),
            prev_files: prev.len(),
        }),
        curr_rev.clone(),
        prev_rev.clone(),
    ))
}

#[derive(Debug)]
pub(crate) struct CompareResult {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<(String, f64, f64)>, // (file, prev_pr, curr_pr)
    pub curr_files: usize,
    pub prev_files: usize,
}

/// Query: most-used external dependencies across all projects.
pub(crate) fn query_top_deps(conn: &Connection, limit: usize) -> Result<Vec<(String, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT ed.package_name, COUNT(DISTINCT s.project_id) as project_count
         FROM external_deps ed
         JOIN files f ON ed.file_id = f.id
         JOIN snapshots s ON f.snapshot_id = s.id
         WHERE s.id = (SELECT MAX(s2.id) FROM snapshots s2 WHERE s2.project_id = s.project_id)
         GROUP BY ed.package_name
         ORDER BY project_count DESC
         LIMIT ?1",
    )?;
    let rows = stmt
        .query_map(params![limit], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}
