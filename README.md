# pkgrank

Compute centrality scores over a Go package import graph.

## Example

```bash
go install github.com/arclabs561/pkgrank@latest
pkgrank crypto/...
```

## Notes

- `--prefix` filters imports by prefix.
- `--pkg` aggregates by package instead of by file.

## License

Dual-licensed under MIT or the UNLICENSE.
