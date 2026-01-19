use petgraph::prelude::*;

/// PageRank configuration.
///
/// Kept intentionally small and stable: this crate is a CLI/tool, not a research sandbox.
#[derive(Debug, Clone, Copy)]
pub struct PageRankConfig {
    /// Damping factor (usually 0.85).
    pub damping: f64,
    /// Convergence tolerance on L1 delta.
    pub tol: f64,
    /// Max power iterations.
    pub max_iters: usize,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            tol: 1e-12,
            max_iters: 200,
        }
    }
}

/// PageRank over a directed graph. Returns one score per NodeIndex, ordered by index.
///
/// Chooses unweighted vs weighted based on edge weights (all \(w \approx 1\) → unweighted).
pub fn pagerank<N>(graph: &DiGraph<N, f64>) -> Vec<f64> {
    let is_unweighted = graph.edge_weights().all(|w| (*w - 1.0).abs() < 1e-12);
    let cfg = PageRankConfig::default();
    if is_unweighted {
        pagerank_unweighted(graph, cfg)
    } else {
        pagerank_weighted(graph, cfg)
    }
}

/// Personalized PageRank over a directed graph.
///
/// `personalization` is a per-node probability distribution (will be normalized if non-zero).
pub fn personalized_pagerank<N>(
    graph: &DiGraph<N, f64>,
    cfg: PageRankConfig,
    personalization: &[f64],
) -> Vec<f64> {
    let n = graph.node_count();
    if n == 0 {
        return Vec::new();
    }
    if personalization.len() != n {
        // In tool code, panic is acceptable: caller bug (internal plumbing).
        panic!(
            "personalization length {} != node_count {}",
            personalization.len(),
            n
        );
    }

    let mut p = personalization.to_vec();
    let sum: f64 = p.iter().copied().sum();
    if sum <= 0.0 {
        // Fallback: uniform teleport.
        p.fill(1.0 / n as f64);
    } else {
        for x in &mut p {
            *x /= sum;
        }
    }

    // Weighted transition probabilities.
    let mut out_wsum = vec![0.0f64; n];
    for e in graph.edge_references() {
        out_wsum[e.source().index()] += (*e.weight()).max(0.0);
    }

    let d = cfg.damping;
    let mut pr = vec![1.0 / n as f64; n];
    let mut next = vec![0.0f64; n];
    for _ in 0..cfg.max_iters {
        next.fill(0.0);

        // Distribute rank mass along edges.
        for e in graph.edge_references() {
            let u = e.source().index();
            let v = e.target().index();
            let w = (*e.weight()).max(0.0);
            let denom = out_wsum[u];
            if denom > 0.0 && w > 0.0 {
                next[v] += pr[u] * (w / denom);
            }
        }

        // Dangling mass: redistribute according to personalization.
        let mut dangling = 0.0;
        for (u, &ws) in out_wsum.iter().enumerate() {
            if ws <= 0.0 {
                dangling += pr[u];
            }
        }
        if dangling > 0.0 {
            for i in 0..n {
                next[i] += dangling * p[i];
            }
        }

        // Apply damping + teleport (personalization).
        let base = 1.0 - d;
        let mut delta = 0.0;
        for i in 0..n {
            let v = base * p[i] + d * next[i];
            delta += (v - pr[i]).abs();
            pr[i] = v;
        }
        if delta <= cfg.tol {
            break;
        }
    }

    pr
}

/// Reachability counts for a directed graph expressed as edge pairs.
///
/// Returns `(dependents, dependencies)` where:
/// - `dependencies[u]` = number of nodes reachable from `u` following edges \(u \to v\)
/// - `dependents[u]`   = number of nodes that can reach `u` (i.e., reachable in the reversed graph)
///
/// This is \(O(n(n+m))\) via BFS-per-node with a stamp-based visited array.
pub fn reachability_counts_edges(n: usize, edges: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u >= n || v >= n {
            continue;
        }
        adj[u].push(v);
        rev[v].push(u);
    }

    fn bfs_counts(adj: &[Vec<usize>]) -> Vec<usize> {
        let n = adj.len();
        let mut counts = vec![0usize; n];
        let mut seen = vec![0u32; n];
        let mut epoch: u32 = 1;
        let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for s in 0..n {
            if epoch == u32::MAX {
                seen.fill(0);
                epoch = 1;
            }
            epoch += 1;
            seen[s] = epoch;
            q.clear();
            q.push_back(s);
            let mut c = 0usize;
            while let Some(u) = q.pop_front() {
                for &v in &adj[u] {
                    if seen[v] == epoch {
                        continue;
                    }
                    seen[v] = epoch;
                    c += 1;
                    q.push_back(v);
                }
            }
            counts[s] = c;
        }
        counts
    }

    let deps = bfs_counts(&adj);
    let dependents = bfs_counts(&rev);
    (dependents, deps)
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

/// Betweenness centrality (Brandes) for directed, unweighted graphs.
///
/// Returns one score per NodeIndex, ordered by index.
pub fn betweenness_centrality<N>(graph: &DiGraph<N, f64>) -> Vec<f64> {
    let n = graph.node_count();
    if n <= 2 {
        return vec![0.0; n];
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
                // sigma[w] can be 0 for disconnected nodes; guard division.
                if sigma[w.index()] > 0.0 {
                    delta[v.index()] +=
                        (sigma[v.index()] / sigma[w.index()]) * (1.0 + delta[w.index()]);
                }
            }
            if w != s {
                betweenness[w.index()] += delta[w.index()];
            }
        }
    }

    // Directed normalization to [0,1] for connected-ish graphs.
    let norm = 1.0 / ((n - 1) * (n - 2)) as f64;
    for b in &mut betweenness {
        *b *= norm;
    }
    betweenness
}

fn pagerank_unweighted<N>(graph: &DiGraph<N, f64>, cfg: PageRankConfig) -> Vec<f64> {
    // Treat every edge as weight 1.
    let n = graph.node_count();
    if n == 0 {
        return Vec::new();
    }

    let mut out_deg = vec![0usize; n];
    for e in graph.edge_references() {
        out_deg[e.source().index()] += 1;
    }

    let d = cfg.damping;
    let mut pr = vec![1.0 / n as f64; n];
    let mut next = vec![0.0f64; n];
    for _ in 0..cfg.max_iters {
        next.fill(0.0);

        for e in graph.edge_references() {
            let u = e.source().index();
            let v = e.target().index();
            let deg = out_deg[u];
            if deg > 0 {
                next[v] += pr[u] / deg as f64;
            }
        }

        let mut dangling = 0.0;
        for (u, &deg) in out_deg.iter().enumerate() {
            if deg == 0 {
                dangling += pr[u];
            }
        }
        if dangling > 0.0 {
            let add = dangling / n as f64;
            for i in 0..n {
                next[i] += add;
            }
        }

        let base = (1.0 - d) / n as f64;
        let mut delta = 0.0;
        for i in 0..n {
            let v = base + d * next[i];
            delta += (v - pr[i]).abs();
            pr[i] = v;
        }
        if delta <= cfg.tol {
            break;
        }
    }
    pr
}

fn pagerank_weighted<N>(graph: &DiGraph<N, f64>, cfg: PageRankConfig) -> Vec<f64> {
    let n = graph.node_count();
    if n == 0 {
        return Vec::new();
    }
    let mut out_wsum = vec![0.0f64; n];
    for e in graph.edge_references() {
        out_wsum[e.source().index()] += (*e.weight()).max(0.0);
    }

    let d = cfg.damping;
    let mut pr = vec![1.0 / n as f64; n];
    let mut next = vec![0.0f64; n];
    for _ in 0..cfg.max_iters {
        next.fill(0.0);

        for e in graph.edge_references() {
            let u = e.source().index();
            let v = e.target().index();
            let w = (*e.weight()).max(0.0);
            let denom = out_wsum[u];
            if denom > 0.0 && w > 0.0 {
                next[v] += pr[u] * (w / denom);
            }
        }

        let mut dangling = 0.0;
        for (u, &ws) in out_wsum.iter().enumerate() {
            if ws <= 0.0 {
                dangling += pr[u];
            }
        }
        if dangling > 0.0 {
            let add = dangling / n as f64;
            for i in 0..n {
                next[i] += add;
            }
        }

        let base = (1.0 - d) / n as f64;
        let mut delta = 0.0;
        for i in 0..n {
            let v = base + d * next[i];
            delta += (v - pr[i]).abs();
            pr[i] = v;
        }
        if delta <= cfg.tol {
            break;
        }
    }
    pr
}
