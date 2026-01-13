package main

import (
	"github.com/henrywallace/pkgrank/analyzers/modver"
	"github.com/henrywallace/pkgrank/shared"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() {
	shared.SetGlobalLogger()
	singlechecker.Main(modver.Analyzer)
}
