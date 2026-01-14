package cmd

import (
	"fmt"
	"os"

	"github.com/arclabs561/pkgrank/pkg"
	"github.com/spf13/cobra"
)

// Execute executes the root command.
func Execute() {
	rootCmd.Execute()
}

var rootCmd = &cobra.Command{
	Use:          "pkgrank",
	Short:        "Discover the graph centrality of Go packages.",
	RunE:         runRoot,
	SilenceUsage: true,
}

func init() {
	rootCmd.Flags().StringP("prefix", "p", "",
		"filter imports with filter, no filter if empty")
	rootCmd.Flags().IntP("num", "n", 16,
		"top number of packages to show, all if non-positive.")
	rootCmd.Flags().Bool("pkg", false,
		"whether to iterate over package imports instead of go files.")
}

func runRoot(cmd *cobra.Command, args []string) error {
	root := os.Args[1]
	prefix, _ := cmd.Flags().GetString("prefix")
	num, _ := cmd.Flags().GetInt("num")
	iterPkg, _ := cmd.Flags().GetBool("pkg")

	pkgs, err := pkg.ListPackages(root)
	if err != nil {
		return err
	}
	g, err := pkg.BuildGraph(pkgs, prefix, iterPkg)
	if err != nil {
		return err
	}
	imps, scores := g.Centrality()
	for i, imp := range imps {
		if i > 0 && i > num {
			break
		}
		fmt.Printf("%.6f %s\n", scores[i], imp)
	}
	return nil
}
