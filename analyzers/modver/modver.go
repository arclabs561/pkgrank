package modver

import (
	"fmt"
	"go/token"
	"os"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/analysis"
)

var Analyzer = &analysis.Analyzer{
	Name:      "moduleVersion",
	Doc:       "finds the module version of its Pass.Pkg.Path()",
	FactTypes: []analysis.Fact{(*ModVerFact)(nil)},
	Run:       run,
	Requires:  []*analysis.Analyzer{},
}

type ModVerFact struct{}

func (f ModVerFact) AFact() {}

func run(pass *analysis.Pass) (interface{}, error) {
	// rootPath := pass.Pkg.GoFiles[0]
	// dir := filepath.Dir(rootPath)

	// // Traverse up until we find go.mod or hit the filesystem root.
	// for {
	// 	gomod := filepath.Join(dir, "go.mod")
	// 	if _, err := os.Stat(gomod); err == nil {
	// 		version, err := getModuleVersionFromGoMod(gomod, pass.Pkg.Path())
	// 		if err != nil {
	// 			return nil, err
	// 		}
	// 		// Return the version as the result of the analyzer.
	// 		return version, nil
	// 	}
	// 	parent := filepath.Dir(dir)
	// 	if parent == dir {
	// 		return nil, errors.New("go.mod not found")
	// 	}
	// 	dir = parent
	// }

	fmt.Println("\n", pass.Pkg.Path())
	pass.Fset.Iterate(func(f *token.File) bool {
		fmt.Println(f.Name())
		return false
	})

	return nil, nil
}

func getModuleVersionFromGoMod(file string, pkgPath string) (string, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return "", err
	}

	modFile, err := modfile.Parse("go.mod", data, nil)
	if err != nil {
		return "", err
	}

	// Check if the pkgPath corresponds to the main module
	if modFile.Module.Mod.Path == pkgPath {
		return modFile.Module.Mod.Version, nil
	}

	// Check if the pkgPath corresponds to a dependency
	for _, require := range modFile.Require {
		if require.Mod.Path == pkgPath {
			return require.Mod.Version, nil
		}
	}

	return "", fmt.Errorf("package path %s not found in go.mod", pkgPath)
}
