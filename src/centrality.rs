use petgraph::prelude::*;

// Use the Tekne L1.5 `walk` crate for PageRank/PPR/reachability.
pub use walk::PageRankConfig;
pub use walk::personalized_pagerank;
pub use walk::reachability_counts_edges;

/// PageRank over a directed graph. Returns one score per NodeIndex, ordered by index.
///
/// Uses `walk`'s PageRank implementation; chooses unweighted vs weighted based on edge weights.
pub fn pagerank<N>(graph: &DiGraph<N, f64>) -> Vec<f64> {
    let is_unweighted = graph.edge_weights().all(|w| (*w - 1.0).abs() < 1e-12);
    let cfg = PageRankConfig::default();
    if is_unweighted {
        walk::pagerank(graph, cfg)
    } else {
        walk::pagerank_weighted(graph, cfg)
    }
}

/// Reverse a graph while preserving node order (so NodeIndex::index() aligns).
pub fn reverse_graph<N: Clone>(graph: &DiGraph<N, f64>) -> DiGraph<N, f64> {
    let mut rev: DiGraph<N, f64> = DiGraph::new();
    let mut idx_map: Vec<NodeIndex> = Vec::with_capacity(graph.node_count());
    for n in graph.node_indices() {
        idx_map.push(rev.add_node(graph.node_weight(n).expect("node weight").clone()));
    }
    for e in graph.edge_references() {
        let u = e.source().index();
        let v = e.target().index();
        let w = *e.weight();
        rev.update_edge(idx_map[v], idx_map[u], w);
    }
    rev
}

