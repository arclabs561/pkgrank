use clap::ValueEnum;
use petgraph::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Ecosystem {
    #[value(alias = "cargo")]
    Rust,
    #[value(alias = "npm")]
    Js,
    Python,
    Go,
}

impl std::fmt::Display for Ecosystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rust => write!(f, "rust"),
            Self::Js => write!(f, "js"),
            Self::Python => write!(f, "python"),
            Self::Go => write!(f, "go"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(crate) struct DepNode {
    pub name: String,
    pub version: Option<String>,
    pub ecosystem: Ecosystem,
}

impl DepNode {
    /// Graph identity: `name@version`, or `name` when the version is unknown.
    /// Two versions of one package are two nodes, as with Cargo `PackageId`s.
    pub(crate) fn id(&self) -> String {
        dep_id(&self.name, self.version.as_deref())
    }
}

pub(crate) fn dep_id(name: &str, version: Option<&str>) -> String {
    match version {
        Some(v) => format!("{name}@{v}"),
        None => name.to_string(),
    }
}

pub(crate) type DepGraph = DiGraph<DepNode, f64>;
pub(crate) type DepNodeMap = HashMap<String, NodeIndex>;

/// Build a generic dependency graph from a list of packages and edges.
/// Edges are `(from_id, to_id)` where ids match `DepNode::id`; packages with
/// the same id are merged into one node.
pub(crate) fn build_dep_graph(
    packages: &[DepNode],
    edges: &[(String, String)],
) -> (DepGraph, DepNodeMap) {
    let mut graph = DepGraph::new();
    let mut map = DepNodeMap::new();

    for pkg in packages {
        map.entry(pkg.id())
            .or_insert_with(|| graph.add_node(pkg.clone()));
    }

    for (from, to) in edges {
        if let (Some(&fi), Some(&ti)) = (map.get(from), map.get(to)) {
            graph.update_edge(fi, ti, 1.0);
        }
    }

    (graph, map)
}

/// FNV-1a 64-bit hash (deterministic, non-cryptographic).
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
