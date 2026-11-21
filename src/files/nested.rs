use super::{excluded_source_dir, Ecosystem, ECOSYSTEM_MARKERS};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct UnscannedProject {
    ecosystem: Ecosystem,
    manifest: PathBuf,
}

impl std::fmt::Display for UnscannedProject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dir = self.manifest.parent().unwrap_or(Path::new("."));
        // Prefix with ./ to disambiguate local directories from owner/repo.
        let arg = format!("./{}", dir.display());
        let quoted = arg.replace('\'', "'\\''");
        write!(
            f,
            "unscanned {} project at {} (from the scan root: pkgrank files '{quoted}')",
            self.ecosystem,
            self.manifest.display()
        )
    }
}

pub(super) fn unscanned_projects(root: &Path, covered: &[Ecosystem]) -> Vec<UnscannedProject> {
    let mut missing = HashMap::new();
    for (ecosystem, markers) in ECOSYSTEM_MARKERS {
        if !covered.contains(ecosystem) {
            for (priority, marker) in markers.iter().enumerate() {
                missing.insert(*marker, (*ecosystem, priority));
            }
        }
    }
    if missing.is_empty() {
        return Vec::new();
    }

    let Ok(out) = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z", "--cached", "--"])
        .args(missing.keys().map(|name| format!(":(glob)**/{name}")))
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for bytes in out.stdout.split(|&b| b == 0) {
        let Ok(path) = std::str::from_utf8(bytes) else {
            continue;
        };
        let manifest = Path::new(path);
        let Some(parent) = manifest.parent() else {
            continue;
        };
        if parent.as_os_str().is_empty()
            || parent.iter().any(|part| {
                let name = part.to_str().unwrap_or("");
                excluded_source_dir(name)
                    || matches!(name, "test" | "tests" | "example" | "examples" | "fuzz")
            })
            || !root.join(manifest).is_file()
        {
            continue;
        }
        let name = manifest
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if let Some(&(ecosystem, priority)) = missing.get(name) {
            candidates.push((
                parent,
                priority,
                UnscannedProject {
                    ecosystem,
                    manifest: manifest.into(),
                },
            ));
        }
    }
    // An outer project already discovers same-language sources below it.
    // Prefer its manifest to its lockfile when both describe the same root.
    candidates.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    let mut projects = Vec::new();
    let mut roots = HashSet::new();
    for (dir, _, project) in candidates {
        if dir
            .ancestors()
            .all(|ancestor| !roots.contains(&(project.ecosystem, ancestor)))
        {
            roots.insert((project.ecosystem, dir));
            projects.push(project);
        }
    }
    projects
}
