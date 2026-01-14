package main

import (
	"github.com/arclabs561/pkgrank/analyzers/depgraph"
	"github.com/arclabs561/pkgrank/shared"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() {
	shared.SetGlobalLogger()
	singlechecker.Main(depgraph.Analyzer)
}
