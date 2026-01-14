// Package depgraph defines an Analyzer that constructs a graph of dependencies
// between containers (e.g. package A imports package B).
package depgraph

import (
	"fmt"
	"os"

	"github.com/arclabs561/pkgrank/graph"
	"github.com/rs/zerolog/log"
	"golang.org/x/tools/go/analysis"
)

var Analyzer = &analysis.Analyzer{
	Name:             "depgraph",
	Doc:              "construct a graph of dependencies between containers",
	FactTypes:        []analysis.Fact{(*graphFact)(nil)},
	Run:              run,
	RunDespiteErrors: true,
}

type graphFact struct {
	graph.Graph
}

func (f graphFact) AFact() {}

var rootPkg = os.Getenv("DEPGRAPH_ROOT_PKG")

func init() {
	if rootPkg == "" {
		panic("DEPGRAPH_ROOT_PKG not set")
	}
}

// Run is the runner for an analysis pass
func run(pass *analysis.Pass) (interface{}, error) {
	log := log.With().Str("pkg", pass.Pkg.Path()).Str("name", pass.Pkg.Name()).Logger()
	log.Info().Msg("running pass over package")
	ok := pass.ImportPackageFact(pass.Pkg, (*graphFact)(nil))
	if ok {
		log.Info().Msg("already visited package")
		return nil, nil
	}
	f := graphFact{Graph: graph.Graph{
		Container:       pass.Pkg.Path(),
		AddedContainers: map[string]struct{}{pass.Pkg.Path(): {}},
		Nodes:           nil,
		Edges:           nil,
	}}
	for _, dep := range pass.Pkg.Imports() {
		log.Info().Str("dep", dep.Path()).Msg("adding dependency")
		f.Graph.AddEdge(graph.NewDirectedEdge(pass.Pkg.Path(), pass.Pkg.Path(), dep.Path()))
		var g graphFact
		if pass.ImportPackageFact(dep, &g) {
			overlap := f.Graph.Add(g.Graph)
			log.Info().Int("graphOrder", g.Graph.Order()).
				Int("graphSize", g.Graph.Size()).
				Str("dep", dep.Path()).
				Int("overlap", overlap).
				Msg("imported dependecy's package fact")
		} else {
			// This is a bug in the analysis driver, whose document
			// requires that packages are visited in dependency
			// topological order.
			log.Fatal().Str("pkg", dep.Path()).Msg("failed to import package fact")
		}
	}
	pass.ExportPackageFact(&f)
	log.Info().Int("graphOrder", f.Graph.Order()).
		Int("graphSize", f.Graph.Size()).
		Int("deps", len(pass.Pkg.Imports())).
		Msg("exported package fact")
	if pass.Pkg.Path() == rootPkg {
		log.Info().Msg("writing graph")
		for _, edge := range f.Graph.Edges {
			edge, ok := edge.(*graph.DirectedEdge)
			if !ok {
				panic(fmt.Sprintf("unsupport edge type: %T", edge))
			}
			fmt.Println(edge.Src, edge.Dst)
		}
	}
	return nil, nil
}
