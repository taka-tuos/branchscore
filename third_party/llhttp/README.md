# Vendored llhttp

The HTTP server uses the generated C parser from llhttp 9.3.1 as a private,
static library. The generated files are checked in so a normal branchscore
build does not require `libllhttp-dev`, pkg-config, Node.js, npm, or network
access.

The generator inputs are retained under `upstream/`:

- TypeScript grammar and generator sources from llhttp tag `v9.3.1`
  (`9b7e0619e66cfc584cc20d97425444245568d6d9`)
- `package.json` and `package-lock.json` for reproducible npm dependencies
- native C helper sources and the upstream MIT license

`generated/` was produced by `npm run build` from those inputs and matches the
official `release/v9.3.1` output. To regenerate after changing the grammar,
run this from the repository root:

```sh
cmake --build build --target branchscore-llhttp-regenerate
```

The target installs npm dependencies temporarily, runs the pinned generator,
copies `build/c/llhttp.c` and `build/llhttp.h` into `generated/`, and removes
its temporary `node_modules` and build directory. The regular CMake build does
not invoke this target.
