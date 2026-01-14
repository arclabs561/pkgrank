//! pkgrank (Rust) - Cargo dependency graph centrality analysis
//!
//! Computes PageRank and other centrality metrics over Cargo dependency graphs.

use cargo_metadata::{MetadataCommand, PackageId};
use clap::{Parser, ValueEnum};
use petgraph::prelude::*;
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(name = "pkgrank")]
#[command(about = "Cargo dependency graph centrality analysis")]
struct Args {
    /// Path to Cargo.toml or directory
    #[arg(default_value = ".")]
    path: String,

    /// Centrality metric
    #[arg(short, long, value_enum, default_value = "pagerank")]
    metric: Metric,

    /// Number of top packages to show
    #[arg(short = 'n', long, default_value = "10")]
    top: usize,

    /// Include dev-dependencies
    #[arg(long)]
    dev: bool,

    /// Include build-dependencies
    #[arg(long)]
    build: bool,

    /// Show only workspace members
    #[arg(long)]
    workspace_only: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Metric {
    Pagerank,
    Indegree,
    Outdegree,
    Betweenness,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let manifest_path = if args.path.ends_with("Cargo.toml") {
        args.path.clone()
    } else {
        format!("{}/Cargo.toml", args.path)
    };

    let metadata = MetadataCommand::new()
        .manifest_path(&manifest_path)
        .exec()?;

    let mut graph: DiGraph<&str, ()> = DiGraph::new();
    let mut node_map: HashMap<&PackageId, NodeIndex> = HashMap::new();

    for pkg in &metadata.packages {
        let idx = graph.add_node(&pkg.name);
        node_map.insert(&pkg.id, idx);
    }

    for pkg in &metadata.packages {
        let pkg_idx = node_map[&pkg.id];
        for dep in &pkg.dependencies {
            if let Some(dep_pkg) = metadata.packages.iter().find(|p| p.name == dep.name) {
                let include = match dep.kind {
                    cargo_metadata::DependencyKind::Normal => true,
                    cargo_metadata::DependencyKind::Development => args.dev,
                    cargo_metadata::DependencyKind::Build => args.build,
                    _ => false,
                };
                if include {
                    let dep_idx = node_map[&dep_pkg.id];
                    graph.add_edge(pkg_idx, dep_idx, ());
                }
            }
        }
    }

    let scores: Vec<(&str, f64)> = match args.metric {
        Metric::Pagerank => pagerank(&graph),
        Metric::Indegree => degree_centrality(&graph, Direction::Incoming),
        Metric::Outdegree => degree_centrality(&graph, Direction::Outgoing),
        Metric::Betweenness => betweenness_centrality(&graph),
    };

    let workspace_members: std::collections::HashSet<_> = metadata
        .workspace_members
        .iter()
        .filter_map(|id| metadata.packages.iter().find(|p| &p.id == id))
        .map(|p| p.name.as_str())
        .collect();

    let mut filtered: Vec<_> = scores
        .into_iter()
        .filter(|(name, _)| !args.workspace_only || workspace_members.contains(name))
        .collect();

    filtered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top {} by {:?}:", args.top, args.metric);
    println!("{:─<50}", "");
    for (i, (name, score)) in filtered.iter().take(args.top).enumerate() {
        println!("{:3}. {:40} {:.6}", i + 1, name, score);
    }
    println!("\n{} nodes, {} edges", graph.node_count(), graph.edge_count());

    Ok(())
}

fn pagerank<'a>(graph: &'a DiGraph<&'a str, ()>) -> Vec<(&'a str, f64)> {
    let n = graph.node_count();
    if n == 0 { return vec![]; }

    let damping = 0.85;
    let mut scores: Vec<f64> = vec![1.0 / n as f64; n];
    let mut new_scores = vec![0.0; n];

    for _ in 0..100 {
        let mut diff = 0.0;
        for node in graph.node_indices() {
            let mut sum = 0.0;
            for neighbor in graph.neighbors_directed(node, Direction::Incoming) {
                let out_deg = graph.neighbors_directed(neighbor, Direction::Outgoing).count() as f64;
                if out_deg > 0.0 { sum += scores[neighbor.index()] / out_deg; }
            }
            new_scores[node.index()] = (1.0 - damping) / n as f64 + damping * sum;
            diff += (new_scores[node.index()] - scores[node.index()]).abs();
        }
        std::mem::swap(&mut scores, &mut new_scores);
        if diff < 1e-8 { break; }
    }

    graph.node_indices().map(|i| (*graph.node_weight(i).unwrap(), scores[i.index()])).collect()
}

fn degree_centrality<'a>(graph: &'a DiGraph<&'a str, ()>, dir: Direction) -> Vec<(&'a str, f64)> {
    let n = graph.node_count() as f64;
    if n <= 1.0 {
        return graph.node_indices().map(|i| (*graph.node_weight(i).unwrap(), 0.0)).collect();
    }
    graph.node_indices().map(|i| {
        let deg = graph.neighbors_directed(i, dir).count() as f64 / (n - 1.0);
        (*graph.node_weight(i).unwrap(), deg)
    }).collect()
}

fn betweenness_centrality<'a>(graph: &'a DiGraph<&'a str, ()>) -> Vec<(&'a str, f64)> {
    let n = graph.node_count();
    if n <= 2 {
        return graph.node_indices().map(|i| (*graph.node_weight(i).unwrap(), 0.0)).collect();
    }

    let mut betweenness = vec![0.0; n];

    for s in graph.node_indices() {
        let mut stack = Vec::new();
        let mut pred: Vec<Vec<NodeIndex>> = vec![vec![]; n];
        let mut sigma = vec![0.0; n];
        let mut dist: Vec<i32> = vec![-1; n];

        sigma[s.index()] = 1.0;
        dist[s.index()] = 0;

        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for w in graph.neighbors_directed(v, Direction::Outgoing) {
                if dist[w.index()] < 0 {
                    dist[w.index()] = dist[v.index()] + 1;
                    queue.push_back(w);
                }
                if dist[w.index()] == dist[v.index()] + 1 {
                    sigma[w.index()] += sigma[v.index()];
                    pred[w.index()].push(v);
                }
            }
        }

        let mut delta = vec![0.0; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w.index()] {
                delta[v.index()] += (sigma[v.index()] / sigma[w.index()]) * (1.0 + delta[w.index()]);
            }
            if w != s { betweenness[w.index()] += delta[w.index()]; }
        }
    }

    let norm = if n > 2 { 2.0 / ((n - 1) * (n - 2)) as f64 } else { 1.0 };
    graph.node_indices().map(|i| (*graph.node_weight(i).unwrap(), betweenness[i.index()] * norm)).collect()
}
