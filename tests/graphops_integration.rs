//! Integration tests for graphops algorithms as used by pkgrank.
//!
//! These tests verify the graphops crate produces correct results for the
//! graph shapes pkgrank relies on (chains, diamonds, cycles, stars, etc.).

use graphops::{
    betweenness_centrality, pagerank, pagerank_weighted, pagerank_weighted_checked,
    personalized_pagerank, reachability_counts_edges, PageRankConfig,
};
use petgraph::prelude::*;

/// Helper: build a DiGraph<&str, f64> from labeled edges (uniform weight 1.0).
fn graph_from_edges(
    labels: &[&'static str],
    edges: &[(usize, usize)],
) -> DiGraph<&'static str, f64> {
    let mut g = DiGraph::new();
    let nodes: Vec<_> = labels.iter().map(|l| g.add_node(*l)).collect();
    for &(u, v) in edges {
        g.add_edge(nodes[u], nodes[v], 1.0);
    }
    g
}

/// Helper: build a DiGraph<&str, f64> from labeled edges with explicit weights.
fn graph_from_weighted_edges(
    labels: &[&'static str],
    edges: &[(usize, usize, f64)],
) -> DiGraph<&'static str, f64> {
    let mut g = DiGraph::new();
    let nodes: Vec<_> = labels.iter().map(|l| g.add_node(*l)).collect();
    for &(u, v, w) in edges {
        g.add_edge(nodes[u], nodes[v], w);
    }
    g
}

// ---- pagerank ----

#[test]
fn pagerank_empty_graph() {
    let g: DiGraph<(), f64> = DiGraph::new();
    let scores = pagerank(&g, PageRankConfig::default());
    assert!(scores.is_empty());
}

#[test]
fn pagerank_single_node() {
    let mut g: DiGraph<(), f64> = DiGraph::new();
    g.add_node(());
    let scores = pagerank(&g, PageRankConfig::default());
    assert_eq!(scores.len(), 1);
    assert!(
        (scores[0] - 1.0).abs() < 1e-6,
        "lone node should have score 1.0"
    );
}

#[test]
fn pagerank_chain_last_node_highest() {
    let g = graph_from_edges(&["A", "B", "C", "D"], &[(0, 1), (1, 2), (2, 3)]);
    let scores = pagerank(&g, PageRankConfig::default());
    assert_eq!(scores.len(), 4);
    let max_idx = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(max_idx, 3, "sink node D should have highest PageRank");
}

#[test]
fn pagerank_cycle_uniform() {
    let g = graph_from_edges(&["A", "B", "C"], &[(0, 1), (1, 2), (2, 0)]);
    let scores = pagerank(&g, PageRankConfig::default());
    let expected = 1.0 / 3.0;
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (s - expected).abs() < 1e-6,
            "node {i}: expected ~{expected}, got {s}"
        );
    }
}

#[test]
fn pagerank_scores_sum_to_one() {
    let g = graph_from_edges(&["A", "B", "C", "D"], &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let scores = pagerank(&g, PageRankConfig::default());
    let total: f64 = scores.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "PageRank scores should sum to 1.0, got {total}"
    );
}

// ---- betweenness_centrality ----

#[test]
fn betweenness_path_graph_center_highest() {
    let g = graph_from_edges(
        &["0", "1", "2", "3", "4"],
        &[(0, 1), (1, 2), (2, 3), (3, 4)],
    );
    let bc = betweenness_centrality(&g);
    assert_eq!(bc.len(), 5);
    assert!(bc[0].abs() < 1e-9, "endpoint 0 should have 0 betweenness");
    assert!(bc[4].abs() < 1e-9, "endpoint 4 should have 0 betweenness");
    let max_idx = bc
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(max_idx, 2, "center node should have highest betweenness");
    assert!(
        (bc[1] - bc[3]).abs() < 1e-9,
        "nodes 1 and 3 should have equal betweenness"
    );
}

#[test]
fn betweenness_star_center_highest() {
    let g = graph_from_edges(&["C", "L1", "L2", "L3"], &[(0, 1), (0, 2), (0, 3)]);
    let bc = betweenness_centrality(&g);
    for (i, &b) in bc.iter().enumerate() {
        assert!(
            b.abs() < 1e-9,
            "node {i}: expected 0 betweenness in out-star, got {b}"
        );
    }
}

#[test]
fn betweenness_two_nodes_is_zero() {
    let g = graph_from_edges(&["A", "B"], &[(0, 1)]);
    let bc = betweenness_centrality(&g);
    assert_eq!(bc, vec![0.0; 2]);
}

// ---- reachability_counts_edges ----

#[test]
fn reachability_chain() {
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let (dependents, dependencies) = reachability_counts_edges(4, &edges);

    assert_eq!(dependencies[0], 3);
    assert_eq!(dependencies[1], 2);
    assert_eq!(dependencies[2], 1);
    assert_eq!(dependencies[3], 0);

    assert_eq!(dependents[0], 0);
    assert_eq!(dependents[1], 1);
    assert_eq!(dependents[2], 2);
    assert_eq!(dependents[3], 3);
}

#[test]
fn reachability_diamond() {
    let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
    let (dependents, dependencies) = reachability_counts_edges(4, &edges);

    assert_eq!(dependencies[0], 3);
    assert_eq!(dependencies[1], 1);
    assert_eq!(dependencies[2], 1);
    assert_eq!(dependencies[3], 0);

    assert_eq!(dependents[0], 0);
    assert_eq!(dependents[1], 1);
    assert_eq!(dependents[2], 1);
    assert_eq!(dependents[3], 3);
}

#[test]
fn reachability_cycle() {
    let edges = vec![(0, 1), (1, 2), (2, 0)];
    let (dependents, dependencies) = reachability_counts_edges(3, &edges);
    for i in 0..3 {
        assert_eq!(dependencies[i], 2, "node {i} dependencies");
        assert_eq!(dependents[i], 2, "node {i} dependents");
    }
}

#[test]
fn reachability_out_of_bounds_edges_ignored() {
    let edges = vec![(0, 1), (1, 99)];
    let (dependents, dependencies) = reachability_counts_edges(3, &edges);
    assert_eq!(dependencies[0], 1);
    assert_eq!(dependencies[1], 0);
    assert_eq!(dependents[1], 1);
}

// ---- weighted pagerank ----

#[test]
fn weighted_pagerank_uniform_weights_match_unweighted() {
    let g = graph_from_edges(&["A", "B", "C", "D"], &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let cfg = PageRankConfig::default();
    let unweighted = pagerank(&g, cfg);
    let weighted = pagerank_weighted(&g, cfg);
    assert_eq!(unweighted.len(), weighted.len());
    for (i, (u, w)) in unweighted.iter().zip(weighted.iter()).enumerate() {
        assert!(
            (u - w).abs() < 1e-10,
            "node {i}: unweighted={u}, weighted={w} -- should match for uniform weights"
        );
    }
}

#[test]
fn weighted_pagerank_heavy_edge_concentrates_rank() {
    let g_heavy = graph_from_weighted_edges(&["A", "B", "C"], &[(0, 1, 100.0), (0, 2, 1.0)]);
    let scores_heavy = pagerank_weighted(&g_heavy, PageRankConfig::default());
    assert!(
        scores_heavy[1] > scores_heavy[2],
        "B (heavy edge) should have higher rank than C: B={}, C={}",
        scores_heavy[1],
        scores_heavy[2]
    );

    let g_uniform = graph_from_weighted_edges(&["A", "B", "C"], &[(0, 1, 1.0), (0, 2, 1.0)]);
    let scores_uniform = pagerank_weighted(&g_uniform, PageRankConfig::default());
    assert!(
        (scores_uniform[1] - scores_uniform[2]).abs() < 1e-6,
        "uniform weights: B and C should be equal: B={}, C={}",
        scores_uniform[1],
        scores_uniform[2]
    );

    assert!(
        scores_heavy[1] > scores_uniform[1],
        "heavy-weighted B should exceed uniform B: heavy={}, uniform={}",
        scores_heavy[1],
        scores_uniform[1]
    );
}

#[test]
fn weighted_pagerank_scores_sum_to_one() {
    let g = graph_from_weighted_edges(
        &["A", "B", "C", "D"],
        &[(0, 1, 3.0), (0, 2, 1.0), (1, 3, 2.0), (2, 3, 5.0)],
    );
    let scores = pagerank_weighted(&g, PageRankConfig::default());
    let total: f64 = scores.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "weighted PageRank scores should sum to 1.0, got {total}"
    );
}

#[test]
fn weighted_pagerank_empty_graph() {
    let g: DiGraph<(), f64> = DiGraph::new();
    let scores = pagerank_weighted(&g, PageRankConfig::default());
    assert!(scores.is_empty());
}

#[test]
fn weighted_pagerank_single_node() {
    let mut g: DiGraph<&str, f64> = DiGraph::new();
    g.add_node("A");
    let scores = pagerank_weighted(&g, PageRankConfig::default());
    assert_eq!(scores.len(), 1);
    assert!(
        (scores[0] - 1.0).abs() < 1e-6,
        "lone node should have score 1.0, got {}",
        scores[0]
    );
}

#[test]
fn weighted_pagerank_checked_rejects_negative_weight() {
    let g = graph_from_weighted_edges(&["A", "B"], &[(0, 1, -1.0)]);
    let result = pagerank_weighted_checked(&g, PageRankConfig::default());
    assert!(result.is_err(), "negative edge weight should be rejected");
}

#[test]
fn weighted_pagerank_checked_rejects_nan_weight() {
    let g = graph_from_weighted_edges(&["A", "B"], &[(0, 1, f64::NAN)]);
    let result = pagerank_weighted_checked(&g, PageRankConfig::default());
    assert!(result.is_err(), "NaN edge weight should be rejected");
}

#[test]
fn weighted_pagerank_cycle_uniform_weights() {
    let g =
        graph_from_weighted_edges(&["A", "B", "C"], &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
    let scores = pagerank_weighted(&g, PageRankConfig::default());
    let expected = 1.0 / 3.0;
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (s - expected).abs() < 1e-6,
            "node {i}: expected ~{expected}, got {s}"
        );
    }
}

// ---- personalized pagerank ----

#[test]
fn ppr_personalized_node_gets_highest_score() {
    let g = graph_from_edges(&["A", "B", "C", "D"], &[(0, 1), (1, 2), (2, 3)]);
    let pers = vec![0.0, 0.0, 0.0, 1.0];
    let scores = personalized_pagerank(&g, PageRankConfig::default(), &pers);
    let max_idx = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        max_idx, 3,
        "node D (personalized) should have the highest PPR score"
    );
}

#[test]
fn ppr_star_center_vs_leaf_personalization() {
    let g = graph_from_edges(&["center", "L1", "L2", "L3"], &[(0, 1), (0, 2), (0, 3)]);
    let cfg = PageRankConfig::default();

    let pers_center = vec![1.0, 0.0, 0.0, 0.0];
    let scores_center = personalized_pagerank(&g, cfg, &pers_center);
    assert!(
        scores_center[0] > scores_center[1],
        "center should score higher than any leaf when personalized to center"
    );
    assert!(
        (scores_center[1] - scores_center[2]).abs() < 1e-6
            && (scores_center[2] - scores_center[3]).abs() < 1e-6,
        "leaves should have equal scores by symmetry: {:?}",
        &scores_center[1..4]
    );

    let pers_leaf = vec![0.0, 1.0, 0.0, 0.0];
    let scores_leaf = personalized_pagerank(&g, cfg, &pers_leaf);
    assert!(
        scores_leaf[1] > scores_leaf[2],
        "leaf1 should score higher than leaf2 when personalized to leaf1"
    );
    assert!(
        scores_leaf[1] > scores_leaf[3],
        "leaf1 should score higher than leaf3 when personalized to leaf1"
    );
}

#[test]
fn ppr_scores_sum_to_one() {
    let g = graph_from_edges(&["A", "B", "C", "D"], &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let pers = vec![0.5, 0.5, 0.0, 0.0];
    let scores = personalized_pagerank(&g, PageRankConfig::default(), &pers);
    let total: f64 = scores.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "PPR scores should sum to 1.0, got {total}"
    );
}

#[test]
fn ppr_uniform_personalization_matches_standard_pagerank() {
    let g = graph_from_edges(&["A", "B", "C", "D"], &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let cfg = PageRankConfig::default();
    let standard = pagerank(&g, cfg);
    let pers = vec![0.25, 0.25, 0.25, 0.25];
    let ppr = personalized_pagerank(&g, cfg, &pers);
    for (i, (s, p)) in standard.iter().zip(ppr.iter()).enumerate() {
        assert!(
            (s - p).abs() < 1e-6,
            "node {i}: standard={s}, PPR(uniform)={p} -- should match"
        );
    }
}

#[test]
fn ppr_empty_graph() {
    let g: DiGraph<(), f64> = DiGraph::new();
    let scores = personalized_pagerank(&g, PageRankConfig::default(), &[]);
    assert!(scores.is_empty());
}

#[test]
fn ppr_zero_personalization_falls_back_to_uniform() {
    let g = graph_from_edges(&["A", "B", "C"], &[(0, 1), (1, 2), (2, 0)]);
    let cfg = PageRankConfig::default();
    let standard = pagerank(&g, cfg);
    let pers = vec![0.0, 0.0, 0.0];
    let ppr = personalized_pagerank(&g, cfg, &pers);
    for (i, (s, p)) in standard.iter().zip(ppr.iter()).enumerate() {
        assert!(
            (s - p).abs() < 1e-6,
            "node {i}: standard={s}, PPR(zero)={p} -- should match for zero personalization"
        );
    }
}
