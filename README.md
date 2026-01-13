pkgrank
-------

Discover important Go package dependencies with graph centrality scores on the
package import DAG.

[![Go Report Card](https://goreportcard.com/badge/gojp/goreportcard)](https://goreportcard.com/report/gojp/goreportcard)

Example usage:
```sh
$ go get -u github.com/arclabs561/pkgrank
$ pkgrank crypto/...
0.046325 unsafe
0.042954 io
0.040090 hash
0.037120 errors
0.033212 crypto/internal/subtle
0.030669 strconv
0.028647 math/big
0.028464 crypto
0.026904 sync
0.024728 crypto/subtle
0.024202 internal/cpu
0.022593 crypto/cipher
0.019348 runtime
0.018613 crypto/internal/randutil
0.017637 encoding/asn1
0.017253 time
0.017235 encoding/binary
```

Use `--prefix` to filter package imports with the given prefix, or `--pkg` to
iterate over imports by package, instead of by file.

Current implementation is naive by calling out to `go list` subprocesses. And
currently only pagerank centrality is implemented. Nodes are packages, and
edges are directed graph of A >- imports -> B. Edge weights are the number of
times that package A imports package B, which may be multiple if files are
iterated over instead of packages with `--pkg`.

Dual-licensed under MIT or the [UNLICENSE](http://unlicense.org).
