package graph

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/pkg/errors"
	"github.com/rs/zerolog/log"
	"github.com/samber/lo"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/network"
	"gonum.org/v1/gonum/graph/simple"
)

type execError struct {
	Command string
	Args    []string
	Stdout  string
	Stderr  string
	Err     error
}

func (e execError) Error() string {
	msg := fmt.Sprintf("failed to run cmd '%v': %v", e.Command, e.Err)
	if e.Stderr != "" {
		msg = fmt.Sprintf("%s: %s", msg, e.Stderr)
	}
	return msg
}

type doExecMode int

const (
	execQuiet doExecMode = iota
	execPipeCombined
)

func (m doExecMode) String() string {
	switch m {
	case execQuiet:
		return "ExecQuiet"
	case execPipeCombined:
		return "ExecPipeCombined"
	default:
		return fmt.Sprintf("ExecInvalidMode(%d)", m)
	}
}

func doExec(
	mode doExecMode,
	dir string,
	envs map[string]string,
	name string,
	args ...string,
) (_ string, err error) {
	start := time.Now()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	envSlice := lo.MapToSlice(envs, func(k, v string) string { return fmt.Sprintf("%s=%s", k, v) })
	cmd.Env = append(os.Environ(), envSlice...)
	var bufStderr bytes.Buffer
	var bufStdout bytes.Buffer
	var stdout io.Writer = &bufStdout
	var stderr io.Writer = &bufStderr
	switch mode {
	case execQuiet:
	case execPipeCombined:
		stdout = io.MultiWriter(&bufStdout, os.Stdout)
		stderr = io.MultiWriter(&bufStderr, os.Stderr)
	default:
		panic(fmt.Errorf("unknown mode %v", mode))
	}
	cmd.Stderr = stderr
	cmd.Stdout = stdout
	defer func() {
		log.Debug().
			Err(err).
			Str("dir", dir).
			Strs("env", envSlice).
			Stringer("dur", time.Since(start).Round(time.Microsecond)).
			Stringer("cmd", cmd).
			Stringer("mode", mode).
			Msg("exec")
	}()
	b, err := cmd.Output()
	if err != nil {
		return "", execError{
			Command: fmt.Sprintf("%v", cmd),
			Stdout:  bufStdout.String(),
			Stderr:  bufStderr.String(),
			Err:     err,
		}
	}
	out := strings.TrimSpace(string(b))
	return out, nil
}

func TransitiveEdges(pkg string) ([]*DirectedEdge, error) {
	target := reModVersion.ReplaceAllString(pkg, "")
	log := log.With().Str("pkg", pkg).Str("target", target).Logger()
	log.Debug().Msg("listing packages")
	dir, err := os.MkdirTemp("", "*-pkgrank")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp dir: %w", err)
	}
	log.Debug().Str("dir", dir).Msg("using temp dir")
	const rootPkg = "pkgrank"
	if _, err := doExec(execQuiet, dir, nil, "go", "mod", "init", rootPkg); err != nil {
		return nil, err
	}
	if _, err := doExec(execQuiet, dir, nil, "go", "get", pkg); err != nil {
		return nil, err
	}
	mainContent := fmt.Sprintf("package main \n import _ \"%s\"", target)
	mainFile := filepath.Join(dir, "main.go")
	if err := os.WriteFile(mainFile, []byte(mainContent), 0644); err != nil {
		return nil, err
	}
	if _, err := doExec(execQuiet, dir, nil, "go", "fmt", mainFile); err != nil {
		return nil, err
	}
	if _, err := doExec(execQuiet, dir, nil, "go", "mod", "tidy"); err != nil {
		fmt.Println("FAILED", dir)
		return nil, err
	}
	envs := map[string]string{
		"DEPGRAPH_ROOT_PKG": target,
		"LOG_LEVEL":         "info",
		"LOG_FORMAT":        "console",
	}
	out, err := doExec(execPipeCombined, dir, envs, "depgraph", ".")
	if err != nil {
		return nil, err
	}
	scanner := bufio.NewScanner(strings.NewReader(out))
	var edges []*DirectedEdge
	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		src, dst := parts[0], parts[1]
		edges = append(edges, NewDirectedEdge("", src, dst))
	}
	return edges, nil
}

// https://github.com/golang/go/blob/master/src/cmd/go/internal/load/pkg.go
// https://github.com/kisielk/godepgraph/blob/master/main.go
// https://en.wikipedia.org/wiki/Centrality#PageRank_centrality
// https://github.com/golang/go/wiki/Modules#quick-start
// https://dave.cheney.net/2014/09/14/go-list-your-swiss-army-knife

var reModVersion = regexp.MustCompile(`(@\w+)$`)

type ImportGraph struct {
	g          *simple.WeightedDirectedGraph
	idToImport map[int64]string
	importToID map[string]int64
}

func NewImportGraph() *ImportGraph {
	return &ImportGraph{
		g:          simple.NewWeightedDirectedGraph(0, 0),
		idToImport: make(map[int64]string),
		importToID: make(map[string]int64),
	}
}

// Len returns the number of nodes in the graph.
func (g *ImportGraph) Len() int {
	return g.g.Nodes().Len()
}

// CentralityMeasure is a method of measuring the centrality of nodes.
type CentralityMeasure string

// Available centrality measures.
const (
	InvalidCentrality  CentralityMeasure = "invalid"
	PageRankCentrality CentralityMeasure = "pagerank"
)

// NewCentralityMeasure returns a new CentralityMeasure from the given raw
// string. An error is returned, if no such
func NewCentralityMeasure(s string) (CentralityMeasure, error) {
	switch s {
	case "pagerank":
		return PageRankCentrality, nil
	default:
		return InvalidCentrality, errors.Errorf("unsupported centrality measure: %s", s)
	}
}

// Centrality returns the a sorted slice of the most important packages in an
// import graph, with the most important listed first. A corresponding slice of
// importances is also returned.
func (g *ImportGraph) Centrality() ([]string, []float64) {
	if g.Len() == 0 {
		return nil, nil
	}
	centrality := network.PageRank(g.g, 0.85, 0.0001)
	type sortable struct {
		imp   string
		score float64
	}
	var sorted []sortable
	for id, score := range centrality {
		sorted = append(sorted, sortable{
			imp:   g.idToImport[id],
			score: score,
		})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].score > sorted[j].score
	})
	imps := make([]string, 0, len(centrality))
	scores := make([]float64, 0, len(centrality))
	for _, s := range sorted {
		imps = append(imps, s.imp)
		scores = append(scores, s.score)
	}
	return imps, scores
}

// UpdateEdge increases the weight on a directed edge between two imports in
// the graph, or creates a new one with weight 1.0 if one already doesn't
// exist. If nodes coressponding to the imports don't already exist, then they
// are created.
func (g *ImportGraph) UpdateEdge(imp1, imp2 string) {
	n1, n2 := g.AddNode(imp1), g.AddNode(imp2)
	we := g.g.WeightedEdge(n1.ID(), n2.ID())
	if we == nil {
		we = g.g.NewWeightedEdge(n1, n2, 1)
	} else {
		// Note that this case won't occur if we only loop over the
		// unique set of package imports, since imp1 is listed
		// uniquely. But it can occur if we iterate over imports
		// duplicately such as by file, or additionally including test
		// imports.
		we = g.g.NewWeightedEdge(n1, n2, we.Weight()+1)
	}
	g.g.SetWeightedEdge(we)

}

// AddNode idempotently returns a node representing the given import in the
// graph. If the import already has a node in the graph, then that existing
// node is returned. Otherwise, a new node is added and returned.
func (g *ImportGraph) AddNode(imp string) graph.Node {
	if id, ok := g.importToID[imp]; ok {
		return g.g.Node(id)
	}
	n := g.g.NewNode()
	g.g.AddNode(n)
	g.importToID[imp] = n.ID()
	g.idToImport[n.ID()] = imp
	return n
}
