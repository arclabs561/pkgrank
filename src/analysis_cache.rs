use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{dep_graph::fnv1a64, Row};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AnalysisCacheEntry {
    pub schema_version: u32,
    pub rows: Vec<Row>,
    pub convergence: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct AnalysisCacheMeta {
    pub generated_at_unix: i64,
    pub manifest_path: String,
    pub key_hex: String,
    pub workspace_only: bool,
    pub dev: bool,
    pub build: bool,
}

pub(crate) fn analysis_cache_key(
    manifest_path: &Path,
    workspace_only: bool,
    dev: bool,
    build: bool,
    all_features: bool,
    no_default_features: bool,
    features: Option<&str>,
) -> u64 {
    let material = format!(
        "manifest={}\nworkspace_only={}\ndev={}\nbuild={}\nall_features={}\nno_default_features={}\nfeatures={:?}",
        manifest_path.display(),
        workspace_only,
        dev,
        build,
        all_features,
        no_default_features,
        features,
    );
    fnv1a64(material.as_bytes())
}

fn cache_path(cache_root: &Path, key: u64, ext: &str) -> std::path::PathBuf {
    cache_root.join(format!("analyze_{key:016x}.{ext}"))
}

pub(crate) fn analysis_cache_read(cache_root: &Path, key: u64) -> Option<AnalysisCacheEntry> {
    let path = cache_path(cache_root, key, "json");
    let raw = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&raw).ok()
}

pub(crate) fn analysis_cache_write(
    cache_root: &Path,
    key: u64,
    entry: &AnalysisCacheEntry,
    manifest_path: &Path,
    workspace_only: bool,
    dev: bool,
    build: bool,
) {
    let _ = fs::create_dir_all(cache_root);
    let data_path = cache_path(cache_root, key, "json");
    let meta_path = cache_path(cache_root, key, "meta.json");

    if let Ok(json) = serde_json::to_string_pretty(entry) {
        let _ = fs::write(&data_path, json);
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let meta = AnalysisCacheMeta {
        generated_at_unix: now,
        manifest_path: manifest_path.display().to_string(),
        key_hex: format!("{key:016x}"),
        workspace_only,
        dev,
        build,
    };
    if let Ok(json) = serde_json::to_string_pretty(&meta) {
        let _ = fs::write(&meta_path, json);
    }
}
