//! Local centrality algorithms.
//!
//! `pkgrank` started life depending on a sibling crate (`graphops/`) for graph algorithms.
//! Since this repo is meant to be a standalone, public tool, we keep a small subset here to
//! avoid cross-repo path dependencies.
//!
//! Scope:
//! - PageRank (unweighted + weighted) with an optional “checked” validation pass.
//! - Personalized PageRank (PPR) for simple entrypoint heuristics.
//! - Reachability counts (transitive closure sizes) for “blast radius” style metrics.
//! - Betweenness centrality (Brandes; directed, unweighted).

use petgraph::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Error(String);

impl Error {
    fn invalid_parameter(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct PageRankRun {
    pub scores: Vec<f64>,
    pub iterations: usize,
    pub diff_l1: f64,
    pub converged: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct PageRankConfig {
    pub damping: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl PageRankConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.damping.is_finite() {
            return Err(Error::invalid_parameter("damping must be finite"));
        }
        if !(0.0..=1.0).contains(&self.damping) {
            return Err(Error::invalid_parameter("damping must be in [0,1]"));
        }
        if self.max_iterations == 0 {
            return Err(Error::invalid_parameter("max_iterations must be > 0"));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(Error::invalid_parameter("tolerance must be finite and > 0"));
        }
        Ok(())
    }
}

/// Checked PageRank (unweighted).
pub fn pagerank_checked<N>(graph: &DiGraph<N, f64>, config: PageRankConfig) -> Result<Vec<f64>> {
    config.validate()?;
    Ok(pagerank(graph, config))
}

pub fn pagerank<N>(graph: &DiGraph<N, f64>, config: PageRankConfig) -> Vec<f64> {
    pagerank_run(graph, config).scores
}

/// PageRank with convergence reporting (unweighted: split mass evenly over outgoing edges).
pub fn pagerank_run<N>(graph: &DiGraph<N, f64>, config: PageRankConfig) -> PageRankRun {
    let n = graph.node_count();
    if n == 0 {
        return PageRankRun {
            scores: Vec::new(),
            iterations: 0,
            diff_l1: 0.0,
            converged: true,
        };
    }

    let n_f64 = n as f64;
    let mut scores = vec![1.0 / n_f64; n];
    let mut new_scores = vec![0.0; n];

    let out_degrees: Vec<usize> = (0..n)
        .map(|i| {
            graph
                .neighbors_directed(NodeIndex::new(i), Direction::Outgoing)
                .count()
        })
        .collect();

    let mut iters = 0usize;
    let mut last_diff = f64::INFINITY;
    let mut converged = false;
    for _ in 0..config.max_iterations {
        iters += 1;

        let dangling_sum: f64 = out_degrees
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)| scores[i])
            .sum();
        let dangling_contrib = config.damping * dangling_sum / n_f64;
        let teleport = (1.0 - config.damping) / n_f64;
        new_scores.fill(teleport + dangling_contrib);

        for u in 0..n {
            let deg = out_degrees[u];
            if deg == 0 {
                continue;
            }
            let share = config.damping * scores[u] / deg as f64;
            for v in graph.neighbors_directed(NodeIndex::new(u), Direction::Outgoing) {
                new_scores[v.index()] += share;
            }
        }

        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();
        last_diff = diff;
        std::mem::swap(&mut scores, &mut new_scores);
        if diff < config.tolerance {
            converged = true;
            break;
        }
    }

    PageRankRun {
        scores,
        iterations: iters,
        diff_l1: last_diff,
        converged,
    }
}

pub fn pagerank_checked_run<N>(
    graph: &DiGraph<N, f64>,
    config: PageRankConfig,
) -> Result<PageRankRun> {
    config.validate()?;
    Ok(pagerank_run(graph, config))
}

/// Weighted PageRank (outgoing mass split proportional to non-negative edge weights).
pub fn pagerank_weighted<N>(graph: &DiGraph<N, f64>, config: PageRankConfig) -> Vec<f64> {
    pagerank_weighted_run(graph, config).scores
}

pub fn pagerank_weighted_run<N>(graph: &DiGraph<N, f64>, config: PageRankConfig) -> PageRankRun {
    let n = graph.node_count();
    if n == 0 {
        return PageRankRun {
            scores: Vec::new(),
            iterations: 0,
            diff_l1: 0.0,
            converged: true,
        };
    }

    let n_f64 = n as f64;
    let mut scores = vec![1.0 / n_f64; n];
    let mut new_scores = vec![0.0; n];

    // Precompute outgoing edges for each node once.
    let outgoing: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|u| {
            graph
                .edges_directed(NodeIndex::new(u), Direction::Outgoing)
                .map(|e| (e.target().index(), (*e.weight()).max(0.0)))
                .collect()
        })
        .collect();
    let out_wsum: Vec<f64> = outgoing
        .iter()
        .map(|edges| edges.iter().map(|&(_v, w)| w).sum())
        .collect();

    let mut iters = 0usize;
    let mut last_diff = f64::INFINITY;
    let mut converged = false;
    for _ in 0..config.max_iterations {
        iters += 1;

        let dangling_sum: f64 = out_wsum
            .iter()
            .enumerate()
            .filter(|(_, &ws)| ws == 0.0)
            .map(|(i, _)| scores[i])
            .sum();

        let dangling_contrib = config.damping * dangling_sum / n_f64;
        let teleport = (1.0 - config.damping) / n_f64;
        new_scores.fill(teleport + dangling_contrib);

        for u in 0..n {
            let ws = out_wsum[u];
            if ws <= 0.0 {
                continue;
            }
            for &(v, w) in &outgoing[u] {
                if w > 0.0 {
                    new_scores[v] += config.damping * scores[u] * (w / ws);
                }
            }
        }

        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();
        last_diff = diff;
        std::mem::swap(&mut scores, &mut new_scores);
        if diff < config.tolerance {
            converged = true;
            break;
        }
    }

    PageRankRun {
        scores,
        iterations: iters,
        diff_l1: last_diff,
        converged,
    }
}

pub fn pagerank_weighted_checked<N>(
    graph: &DiGraph<N, f64>,
    config: PageRankConfig,
) -> Result<Vec<f64>> {
    pagerank_weighted_checked_run(graph, config).map(|r| r.scores)
}

pub fn pagerank_weighted_checked_run<N>(
    graph: &DiGraph<N, f64>,
    config: PageRankConfig,
) -> Result<PageRankRun> {
    config.validate()?;
    for e in graph.edge_references() {
        let w = *e.weight();
        if !w.is_finite() {
            return Err(Error::invalid_parameter("edge weights must be finite"));
        }
        if w < 0.0 {
            return Err(Error::invalid_parameter(
                "edge weights must be non-negative",
            ));
        }
    }
    Ok(pagerank_weighted_run(graph, config))
}

/// Personalized PageRank (PPR).
pub fn personalized_pagerank<N>(
    graph: &DiGraph<N, f64>,
    config: PageRankConfig,
    personalization: &[f64],
) -> Vec<f64> {
    personalized_pagerank_run(graph, config, personalization).scores
}

fn personalized_pagerank_run<N>(
    graph: &DiGraph<N, f64>,
    config: PageRankConfig,
    personalization: &[f64],
) -> PageRankRun {
    let n = graph.node_count();
    if n == 0 {
        return PageRankRun {
            scores: Vec::new(),
            iterations: 0,
            diff_l1: 0.0,
            converged: true,
        };
    }

    let p_sum: f64 = personalization.iter().sum();
    let p_vec: Vec<f64> = if p_sum > 0.0 {
        personalization.iter().map(|&x| x / p_sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    let mut scores = p_vec.clone();
    let mut new_scores = vec![0.0; n];
    let out_degrees: Vec<usize> = (0..n)
        .map(|i| {
            graph
                .neighbors_directed(NodeIndex::new(i), Direction::Outgoing)
                .count()
        })
        .collect();

    let mut iters = 0usize;
    let mut last_diff = f64::INFINITY;
    let mut converged = false;
    for _ in 0..config.max_iterations {
        iters += 1;

        let dangling_sum: f64 = out_degrees
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)| scores[i])
            .sum();

        for i in 0..n {
            new_scores[i] =
                (1.0 - config.damping) * p_vec[i] + config.damping * dangling_sum * p_vec[i];
        }

        for u in 0..n {
            let deg = out_degrees[u];
            if deg == 0 {
                continue;
            }
            let share = config.damping * scores[u] / deg as f64;
            for v in graph.neighbors_directed(NodeIndex::new(u), Direction::Outgoing) {
                new_scores[v.index()] += share;
            }
        }

        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();
        last_diff = diff;
        std::mem::swap(&mut scores, &mut new_scores);
        if diff < config.tolerance {
            converged = true;
            break;
        }
    }

    PageRankRun {
        scores,
        iterations: iters,
        diff_l1: last_diff,
        converged,
    }
}

/// Count transitive reachability for each node in a directed graph.
///
/// Returns `(dependents, dependencies)` where:
/// - `dependencies[u]` is the number of distinct nodes reachable from `u` following `u -> v`
/// - `dependents[u]` is the number of distinct nodes that can reach `u` (reachability in the
///   reversed graph)
pub fn reachability_counts_edges(n: usize, edges: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut fwd: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u >= n || v >= n {
            continue;
        }
        fwd[u].push(v);
        rev[v].push(u);
    }

    let mut dependencies = vec![0usize; n];
    let mut dependents = vec![0usize; n];

    // One visited buffer reused for all BFS runs.
    let mut visited: Vec<u32> = vec![0u32; n];
    let mut stamp: u32 = 0;
    let mut q: Vec<usize> = Vec::new();

    for start in 0..n {
        // Forward reachability (dependencies)
        stamp = stamp.wrapping_add(1);
        q.clear();
        visited[start] = stamp;
        q.push(start);
        let mut head = 0usize;
        let mut count = 0usize;
        while head < q.len() {
            let cur = q[head];
            head += 1;
            for &nx in &fwd[cur] {
                if visited[nx] != stamp {
                    visited[nx] = stamp;
                    q.push(nx);
                    count += 1;
                }
            }
        }
        dependencies[start] = count;

        // Reverse reachability (dependents)
        stamp = stamp.wrapping_add(1);
        q.clear();
        visited[start] = stamp;
        q.push(start);
        let mut head = 0usize;
        let mut count = 0usize;
        while head < q.len() {
            let cur = q[head];
            head += 1;
            for &nx in &rev[cur] {
                if visited[nx] != stamp {
                    visited[nx] = stamp;
                    q.push(nx);
                    count += 1;
                }
            }
        }
        dependents[start] = count;
    }

    (dependents, dependencies)
}

/// Betweenness centrality (Brandes) for directed, unweighted graphs.
///
/// Returns one score per `NodeIndex`, ordered by index.
pub fn betweenness_centrality<N, E, Ix>(graph: &petgraph::Graph<N, E, Directed, Ix>) -> Vec<f64>
where
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n <= 2 {
        return vec![0.0; n];
    }

    let mut betweenness = vec![0.0; n];

    for s in graph.node_indices() {
        let mut stack: Vec<NodeIndex<Ix>> = Vec::new();
        let mut pred: Vec<Vec<NodeIndex<Ix>>> = vec![vec![]; n];
        let mut sigma = vec![0.0f64; n];
        let mut dist: Vec<i32> = vec![-1; n];

        sigma[s.index()] = 1.0;
        dist[s.index()] = 0;

        let mut queue: std::collections::VecDeque<NodeIndex<Ix>> =
            std::collections::VecDeque::new();
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

        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w.index()] {
                let sigma_w = sigma[w.index()];
                if sigma_w > 0.0 {
                    delta[v.index()] += (sigma[v.index()] / sigma_w) * (1.0 + delta[w.index()]);
                }
            }
            if w != s {
                betweenness[w.index()] += delta[w.index()];
            }
        }
    }

    let norm = 1.0 / ((n - 1) * (n - 2)) as f64;
    for b in &mut betweenness {
        *b *= norm;
    }
    betweenness
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a DiGraph<&str, f64> from labeled edges.
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
        // Chain: A -> B -> C -> D. All mass flows toward D (the sink),
        // so D should have the highest score.
        let g = graph_from_edges(&["A", "B", "C", "D"], &[(0, 1), (1, 2), (2, 3)]);
        let scores = pagerank(&g, PageRankConfig::default());
        assert_eq!(scores.len(), 4);
        // D (sink) should have the highest score.
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
        // Cycle: A -> B -> C -> A. By symmetry all scores should be equal.
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
        // Path: 0 -> 1 -> 2 -> 3 -> 4. Node 2 is on the most shortest paths
        // and should have the highest betweenness centrality.
        let g = graph_from_edges(
            &["0", "1", "2", "3", "4"],
            &[(0, 1), (1, 2), (2, 3), (3, 4)],
        );
        let bc = betweenness_centrality(&g);
        assert_eq!(bc.len(), 5);
        // Endpoints (0, 4) should have 0 betweenness (never on a shortest path between others).
        assert!(bc[0].abs() < 1e-9, "endpoint 0 should have 0 betweenness");
        assert!(bc[4].abs() < 1e-9, "endpoint 4 should have 0 betweenness");
        // Center node 2 should have the highest value.
        let max_idx = bc
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2, "center node should have highest betweenness");
        // Symmetry: bc[1] == bc[3] (equidistant from center in a directed path).
        assert!(
            (bc[1] - bc[3]).abs() < 1e-9,
            "nodes 1 and 3 should have equal betweenness"
        );
    }

    #[test]
    fn betweenness_star_center_highest() {
        // Star: center 0 -> {1, 2, 3}. No shortest paths pass through 0 in a
        // directed sense (leaves are sinks), so betweenness is 0 for all.
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
        // With n <= 2 the function returns zeros (no pair (s,t) with s!=t!=v).
        let g = graph_from_edges(&["A", "B"], &[(0, 1)]);
        let bc = betweenness_centrality(&g);
        assert_eq!(bc, vec![0.0; 2]);
    }

    // ---- reachability_counts_edges ----

    #[test]
    fn reachability_chain() {
        // 0 -> 1 -> 2 -> 3
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let (dependents, dependencies) = reachability_counts_edges(4, &edges);

        // dependencies[i] = how many nodes reachable from i
        assert_eq!(dependencies[0], 3); // 0 reaches 1,2,3
        assert_eq!(dependencies[1], 2); // 1 reaches 2,3
        assert_eq!(dependencies[2], 1); // 2 reaches 3
        assert_eq!(dependencies[3], 0); // 3 reaches nobody

        // dependents[i] = how many nodes can reach i
        assert_eq!(dependents[0], 0); // nobody reaches 0
        assert_eq!(dependents[1], 1); // 0 reaches 1
        assert_eq!(dependents[2], 2); // 0,1 reach 2
        assert_eq!(dependents[3], 3); // 0,1,2 reach 3
    }

    #[test]
    fn reachability_diamond() {
        // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
        let (dependents, dependencies) = reachability_counts_edges(4, &edges);

        assert_eq!(dependencies[0], 3); // 0 reaches 1,2,3
        assert_eq!(dependencies[1], 1); // 1 reaches 3
        assert_eq!(dependencies[2], 1); // 2 reaches 3
        assert_eq!(dependencies[3], 0); // 3 is a sink

        assert_eq!(dependents[0], 0);
        assert_eq!(dependents[1], 1); // 0
        assert_eq!(dependents[2], 1); // 0
        assert_eq!(dependents[3], 3); // 0,1,2
    }

    #[test]
    fn reachability_cycle() {
        // Cycle: 0 -> 1 -> 2 -> 0. Every node reaches every other node.
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let (dependents, dependencies) = reachability_counts_edges(3, &edges);
        for i in 0..3 {
            assert_eq!(dependencies[i], 2, "node {i} dependencies");
            assert_eq!(dependents[i], 2, "node {i} dependents");
        }
    }

    #[test]
    fn reachability_out_of_bounds_edges_ignored() {
        // Edges referencing nodes >= n should be silently skipped.
        let edges = vec![(0, 1), (1, 99)];
        let (dependents, dependencies) = reachability_counts_edges(3, &edges);
        assert_eq!(dependencies[0], 1); // 0 reaches 1
        assert_eq!(dependencies[1], 0); // edge to 99 ignored
        assert_eq!(dependents[1], 1); // 0 reaches 1
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

    // ---- weighted pagerank ----

    #[test]
    fn weighted_pagerank_uniform_weights_match_unweighted() {
        // When all edge weights are equal (1.0), weighted PageRank should produce
        // the same scores as unweighted PageRank.
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
        // A -> B (weight 100), A -> C (weight 1).
        // B should receive more rank than C because A's outgoing mass
        // is split proportionally to edge weight.
        //
        // Also compare against uniform weights (A -> B = 1, A -> C = 1) to
        // confirm the heavy weight shifts mass toward B.
        let g_heavy = graph_from_weighted_edges(&["A", "B", "C"], &[(0, 1, 100.0), (0, 2, 1.0)]);
        let scores_heavy = pagerank_weighted(&g_heavy, PageRankConfig::default());
        assert!(
            scores_heavy[1] > scores_heavy[2],
            "B (heavy edge) should have higher rank than C: B={}, C={}",
            scores_heavy[1],
            scores_heavy[2]
        );

        // With uniform weights, B and C should be equal (symmetric sinks).
        let g_uniform = graph_from_weighted_edges(&["A", "B", "C"], &[(0, 1, 1.0), (0, 2, 1.0)]);
        let scores_uniform = pagerank_weighted(&g_uniform, PageRankConfig::default());
        assert!(
            (scores_uniform[1] - scores_uniform[2]).abs() < 1e-6,
            "uniform weights: B and C should be equal: B={}, C={}",
            scores_uniform[1],
            scores_uniform[2]
        );

        // The heavy-weight B should get more rank than the uniform-weight B.
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
        // Cycle A -> B -> C -> A with equal weights: scores should be equal.
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
        // Chain: A -> B -> C -> D. Personalize to D (the sink).
        // D should get the highest score because teleport always goes to D.
        let g = graph_from_edges(&["A", "B", "C", "D"], &[(0, 1), (1, 2), (2, 3)]);
        let pers = vec![0.0, 0.0, 0.0, 1.0]; // all teleport mass to D
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
        // Star: center(0) -> leaf1(1), center(0) -> leaf2(2), center(0) -> leaf3(3).
        //
        // When personalized to center, center gets a high score because teleport
        // always returns to it. When personalized to leaf1, leaf1 gets a higher
        // score relative to the other leaves.
        let g = graph_from_edges(&["center", "L1", "L2", "L3"], &[(0, 1), (0, 2), (0, 3)]);
        let cfg = PageRankConfig::default();

        // Personalize to center.
        let pers_center = vec![1.0, 0.0, 0.0, 0.0];
        let scores_center = personalized_pagerank(&g, cfg, &pers_center);
        assert!(
            scores_center[0] > scores_center[1],
            "center should score higher than any leaf when personalized to center"
        );
        // All leaves should be equal by symmetry.
        assert!(
            (scores_center[1] - scores_center[2]).abs() < 1e-6
                && (scores_center[2] - scores_center[3]).abs() < 1e-6,
            "leaves should have equal scores by symmetry: {:?}",
            &scores_center[1..4]
        );

        // Personalize to leaf1.
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
        // When personalization is uniform, PPR should produce the same result
        // as standard (unweighted) PageRank.
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
        // When all personalization values are 0, the implementation normalizes
        // to uniform, so it should behave like standard PageRank.
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
}
