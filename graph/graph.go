package graph

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"github.com/rs/zerolog/log"
)

var _ map[NodeKey]struct{}

type NodeKey struct {
	ID string
}

func (k NodeKey) String() string {
	return k.ID
}

type Node struct {
	NodeKey
	Data *NodeData
}

type NodeData struct{}

var _ map[EdgeKey]struct{}

type EdgeKey struct {
	container string
	id        string
}

func (k EdgeKey) String() string {
	return fmt.Sprintf("%s:%s", k.container, k.id)
}

func EdgeKeyFrom(s string) EdgeKey {
	parts := strings.SplitN(s, ":", 2)
	if len(parts) != 2 {
		panic(fmt.Sprintf("invalid edge key: %q", s))
	}
	return EdgeKey{container: parts[0], id: parts[1]}
}

type Edge interface {
	fmt.Stringer
	Key() EdgeKey
	EdgeType() EdgeType
	// The nodes that this edge connects.
	Nodes() []NodeKey
	Weight() float64
	Valid() error
}

type EdgeType int

const (
	EdgeTypeBase EdgeType = iota
	EdgeTypeDirected
	EdgeTypeUndirected
	EdgeTypeHyper
)

var (
	_ Edge = (*BaseEdge)(nil)
	_ Edge = (*DirectedEdge)(nil)
	_ Edge = (*UndirectedEdge)(nil)
	_ Edge = (*HyperEdge)(nil)
)

type BaseEdge struct {
	EdgeKey    EdgeKey
	EdgeWeight float64
}

func (e BaseEdge) String() string {
	return fmt.Sprintf("%v", e.EdgeKey)
}

func (e BaseEdge) Key() EdgeKey {
	return e.EdgeKey
}

func (e BaseEdge) Nodes() []NodeKey {
	return nil
}

func (e BaseEdge) Weight() float64 {
	return e.EdgeWeight
}

func (e BaseEdge) Valid() error {
	return nil
}

type DirectedEdge struct {
	BaseEdge
	Src NodeKey
	Dst NodeKey
}

func NewDirectedEdge(container string, srcID, dstID string) *DirectedEdge {
	src := NodeKey{ID: srcID}
	dst := NodeKey{ID: dstID}
	return &DirectedEdge{
		BaseEdge: BaseEdge{
			EdgeKey: EdgeKey{
				id:        fmt.Sprintf("%v->%v", src, dst),
				container: container,
			},
			EdgeWeight: 1,
		},
		Src: src,
		Dst: dst,
	}
}

func (e DirectedEdge) String() string {
	return fmt.Sprintf("%v", e.EdgeKey)
}

func (e DirectedEdge) Key() EdgeKey {
	return e.EdgeKey
}

func (e DirectedEdge) Nodes() []NodeKey {
	return []NodeKey{e.Src, e.Dst}
}

func (e DirectedEdge) Weight() float64 {
	return e.EdgeWeight
}

func (e DirectedEdge) Valid() error {
	if err := e.BaseEdge.Valid(); err != nil {
		return fmt.Errorf("invalid base edge: %w", err)
	}
	if e.Src.ID == "" {
		return fmt.Errorf("invalid src: %+v", e.Src)
	}
	if e.Dst.ID == "" {
		return fmt.Errorf("invalid dst: %+v", e.Dst)
	}
	return nil
}

type UndirectedEdge struct {
	BaseEdge
	Left  NodeKey
	Right NodeKey
}

func NewUndirectedEdge(container string, leftID, rightID string) *UndirectedEdge {
	left := NodeKey{ID: leftID}
	right := NodeKey{ID: rightID}
	return &UndirectedEdge{
		BaseEdge: BaseEdge{
			EdgeKey: EdgeKey{
				id:        fmt.Sprintf("%v~%v", left, right),
				container: container,
			},
		},
		Left:  left,
		Right: right,
	}
}

func (e UndirectedEdge) String() string {
	return fmt.Sprintf("%v", e.EdgeKey)
}

func (e UndirectedEdge) Key() EdgeKey {
	return e.EdgeKey
}

func (e UndirectedEdge) Nodes() []NodeKey {
	return []NodeKey{e.Left, e.Right}
}

func (e UndirectedEdge) Weight() float64 {
	return e.EdgeWeight
}

type HyperEdge struct {
	BaseEdge
	UnorderedSet []NodeKey
}

func NewHyperEdge(container string, ids ...string) *HyperEdge {
	sort.Strings(ids)
	keys := make([]NodeKey, len(ids))
	for i, id := range ids {
		keys[i] = NodeKey{ID: id}
	}
	return &HyperEdge{
		BaseEdge: BaseEdge{
			EdgeKey: EdgeKey{
				id:        strings.Join(ids, ","),
				container: container,
			},
		},
		UnorderedSet: keys,
	}
}

func (e HyperEdge) String() string {
	return fmt.Sprintf("%v", e.EdgeKey)
}

func (e HyperEdge) Key() EdgeKey {
	return e.EdgeKey
}

func (e HyperEdge) Nodes() []NodeKey {
	return e.UnorderedSet
}

func (e HyperEdge) Weight() float64 {
	return e.EdgeWeight
}

func (e HyperEdge) Valid() error {
	// if err := e.BaseEdge.Valid(); err != nil {
	// 	return fmt.Errorf("invalid base edge: %w", err)
	// }
	if len(e.Nodes()) == 0 {
		return fmt.Errorf("hyperedge must have at least one node")
	}
	return nil
}

func (e *BaseEdge) EdgeType() EdgeType       { return EdgeTypeBase }
func (e *DirectedEdge) EdgeType() EdgeType   { return EdgeTypeDirected }
func (e *UndirectedEdge) EdgeType() EdgeType { return EdgeTypeUndirected }
func (e *HyperEdge) EdgeType() EdgeType      { return EdgeTypeHyper }

type Graph struct {
	// The Container name for which this graph primarily
	// represents.
	Container string
	// Containers that were added to this graph via the Add method. May not
	// include this Container of this Graph
	AddedContainers map[string]struct{}
	// Nodes contained in this graph.
	Nodes map[NodeKey]Node
	// Every key's Nodes() must return nodes that are present in the Nodes
	// map. The converse is not always true. In particular, whevner there
	// are isolated, edgless nodes.
	Edges map[EdgeKey]Edge
}

// Order returns the number of nodes in the graph.
func (f Graph) Order() int {
	return len(f.Nodes)
}

// Size returns the number of edges in the graph.
func (f Graph) Size() int {
	return len(f.Edges)
}

func (f Graph) String() string {
	var buf bytes.Buffer
	buf.WriteString("\n")
	type item struct {
		name   string
		weight float64
	}
	var sorted []item
	for _, edge := range f.Edges {
		switch edge := edge.(type) {
		case *DirectedEdge:
			sorted = append(sorted, item{
				name:   edge.String(),
				weight: edge.Weight(),
			})
		default:
			log.Fatal().Msgf("unimplemented: %#v", edge)
		}
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].weight != sorted[j].weight {
			return sorted[i].weight > sorted[j].weight
		}
		if sorted[i].name != sorted[j].name {
			return sorted[i].name < sorted[j].name
		}
		return false
	})
	for _, item := range sorted {
		fmt.Fprintf(&buf, "      %v: %g\n", item.name, item.weight)
	}
	return buf.String()
}

func (f *Graph) Add(other Graph, opts ...AddEdgeOptions) int {
	var keep map[string]struct{}
	for container := range other.AddedContainers {
		log := log.With().Str("container", container).Logger()
		if _, ok := f.AddedContainers[container]; ok {
			log.Trace().Msgf("skipping already added container")
			continue
		}
		log.Trace().Msgf("keeping new container")
		if keep == nil {
			keep = make(map[string]struct{})
		}
		keep[container] = struct{}{}
		f.AddedContainers[container] = struct{}{}
	}
	if len(keep) == 0 && len(other.AddedContainers) > 0 {
		log.Debug().Msgf("no new containers to keep")
		return -1
	}
	if len(other.AddedContainers) > 0 {
		log.Trace().Str("keep", fmt.Sprintf("%v", keep)).Msgf("keeping %d containers", len(keep))
	}
	// Otherwise, even if no kept added containers, then we are adding a bare
	// graphFact, and we should keep it.
	overlap := 0
	for _, edge := range other.Edges {
		log := log.With().Stringer("edge", edge).Logger()
		if _, ok := keep[edge.Key().container]; !ok && len(other.AddedContainers) > 0 {
			overlap++
			log.Trace().Msgf("skipping already added edge")
			continue
		}
		_ = f.AddEdge(edge, opts...)
		// log.Fatal().Stringer("edgeKey", edge.Key()).Msg("edge already exists")
	}
	return overlap
}

type AddEdgeOptions struct {
	// Merges toAdd into prev, only modifying prev. Only called if
	// edge already previously existed.
	MergeFunc func(prev Edge, toAdd Edge)
}

var DefaultAddEdgeOptions = AddEdgeOptions{
	MergeFunc: func(prev Edge, edge Edge) {
		switch edge := edge.(type) {
		case *DirectedEdge:
			edge.EdgeWeight += prev.Weight()
		default:
			log.Fatal().Msgf("unimplemented: %#v", edge)
		}
	},
}

func (f *Graph) AddEdge(edge Edge, opts ...AddEdgeOptions) bool {
	if edge.EdgeType() == EdgeTypeBase {
		log.Error().Msgf("cannot add base edges: %+v", edge)
		return false
	}
	if err := edge.Valid(); err != nil {
		log.Error().Err(err).Msgf("not adding invalid edge: %+v", edge)
	}
	log := log.With().Str("edgeKey", edge.Key().String()).Logger()
	opt := DefaultAddEdgeOptions
	if len(opts) > 0 {
		// FIXME, merge opts
		opt = opts[0]
	}
	if f.Edges == nil {
		f.Edges = make(map[EdgeKey]Edge)
	}
	prev, ok := f.Edges[edge.Key()]
	if ok {
		if prev.EdgeType() != edge.EdgeType() {
			log.Fatal().Msgf("cannot add edges of different types: prev=%#v, edge=%#v", prev, edge)
		}
		opt.MergeFunc(prev, edge)
	}
	f.Edges[edge.Key()] = edge
	log.Trace().Bool("prev", ok).Msgf("added edge")
	return ok
}
