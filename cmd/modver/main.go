package main

import (
	"github.com/arclabs561/pkgrank/analyzers/modver"
	"github.com/arclabs561/pkgrank/shared"
	"golang.org/x/tools/go/analysis/singlechecker"
)

func main() {
	shared.SetGlobalLogger()
	singlechecker.Main(modver.Analyzer)
}
