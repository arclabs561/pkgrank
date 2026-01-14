package graph_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/arclabs561/pkgrank/graph"
	"github.com/arclabs561/pkgrank/shared"
)

func TestGraphFactAdd(t *testing.T) {
	shared.SetGlobalLogger()
	f := graph.Graph{}
	f.AddEdge(graph.NewDirectedEdge("", "A", "B"))
	f.AddEdge(graph.NewDirectedEdge("", "B", "C"))
	g := graph.Graph{}
	g.AddEdge(graph.NewDirectedEdge("", "A", "B"))
	g.AddEdge(graph.NewDirectedEdge("", "A", "C"))
	f.Add(g)

	assertEqual(t, f.Size(), 3)
	assertEqual(t, f.Edges[graph.EdgeKeyFrom(":A->B")].Weight(), 2.0)
	assertEqual(t, f.Edges[graph.EdgeKeyFrom(":B->C")].Weight(), 1.0)
	assertEqual(t, f.Edges[graph.EdgeKeyFrom(":A->C")].Weight(), 1.0)
}

func assertEqual(t *testing.T, got any, want any) {
	t.Helper()
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf(
			"\nwant: %+v\ngot: %+v\ndiff: (-want +got)\n%s",
			want,
			got,
			diff,
		)
	}
}
