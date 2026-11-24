# Genesis4 for WebAssembly

A single-process build of Genesis4 that runs in a browser Web Worker or in
Node, with a callback that streams diagnostics while the run is in progress.

## Requirements

- Docker (the build runs in the `emscripten/emsdk:7.0.1` image)
- `curl`, for the one-time download of HDF5 and FFTW
- Node 20 or later, for the test

## Build

```sh
./wasm/build.sh
```

The first run downloads a prebuilt wasm HDF5 (from `usnistgov/libhdf5-wasm`)
and the FFTW source, and builds FFTW. Later runs recompile only the sources
that changed; a changed header recompiles everything.

Everything lands in `wasm/build/`, which is ignored by git:

| File | Use |
|---|---|
| `genesis4.mjs`, `genesis4.wasm` | ES module for browsers, workers and Node, with an in-memory file system |
| `genesis4-node.js`, `genesis4-node.wasm` | Node command line on the host file system: `node wasm/build/genesis4-node.js Example1.in` |

### Clean rebuild

Do this after switching branches or rebasing, since objects of source files
that no longer exist would otherwise still be linked:

```sh
rm -rf wasm/build/obj && ./wasm/build.sh
```

Delete all of `wasm/build/` to also refetch HDF5 and rebuild FFTW.

Set `EMSDK_IMAGE` to try another Emscripten version.

## Test

```sh
node wasm/test/run-example1.mjs
```

This runs Example1 in the module and checks that every step reaches the live
callback. `node --test wasm/test/deck.test.mjs` covers the deck rewriting used
by the demo sliders. To compare against a native run, give it an output directory and use
a Python with `h5py`:

```sh
node wasm/test/run-example1.mjs /tmp/wasm-out
python wasm/test/compare.py native/Example1.out.h5 /tmp/wasm-out/Example1.out.h5 /tmp/wasm-out/live.json
```

## Browser demo

The build assembles a self-contained copy of the page, the module and the
example decks in `wasm/build/demo/`, which any static file server can host:

```sh
python3 -m http.server 8765 -d wasm/build/demo
```

Then open <http://127.0.0.1:8765/>. After editing `wasm/demo/`, run
`./wasm/build.sh` again to refresh the copy. The page runs each simulation in
a fresh worker (`demo/worker.js`) and plots power, bunching and the transverse
intensity as the steps arrive.

The sliders rewrite single values in the deck shown under "Input deck"
(`demo/deck.js`); the text stays the source of truth and can still be edited by
hand. Releasing a slider starts a new run, and the previous run stays on the
plots as a dashed line. Add a slider by extending `controls` in
`demo/index.html`.

### On GitHub

The `Documentation` workflow (`.github/workflows/gh-pages.yml`) builds the
module, runs the tests, and publishes the demo next to the manual at `demo/` on
GitHub Pages. HDF5 and FFTW are cached between runs. The demo is also uploaded
as the `genesis4-wasm-demo` artifact on every run, including pull requests.

The page links to the commit it was built from; set `G4_REPO_URL` to point
that link at a fork.

## Using the module

```js
import createModule from "./build/genesis4.mjs";

const g4 = await createModule({
    print: (line) => console.log(line),
    onG4Step: (step) => { /* see below */ },
});
g4.FS.writeFile("run.in", inputText);
g4.FS.writeFile("run.lat", latticeText);
const exitCode = g4.callMain(["run.in"]);
const h5 = g4.FS.readFile("rootname.out.h5");
```

Use one module instance per run: Genesis keeps global state. `callMain` blocks
until the run ends, so call it from a worker; terminating the worker cancels
the run.

### Live diagnostics

Setting `onG4Step` turns the hook on. It is called after every output step
with:

| Field | Content |
|---|---|
| `iz`, `nz` | index of this output step and the total number |
| `ns` | number of slices |
| `z` | position in the undulator, in m |
| `data` | `Float64Array` of length `ns` per record, keyed like the output file: `Field/power`, `Beam/bunching`, ... |
| `field` | `null`, or `{slice, ngrid, gridmax, data}` with `data` the complex field of one slice as interleaved (re, im) doubles on a grid spanning ±`gridmax` m |

Options on the module object:

| Option | Effect |
|---|---|
| `g4Keys` | a `Set` of record names; only these are copied out |
| `g4FieldSlice` | slice to send as `field`; `-1` picks the slice with the highest power; unset sends none |
| `g4FieldEvery` | send `field` only every this many steps (default 1) |

## How the build differs from the native one

- **MPI** is `mpi/mpi.h`, a one-rank stub. The parallel HDF5 calls become
  no-ops through the force-included `mpi/h5mpio_stub.h`.
- **FFTW** plans with `FFTW_ESTIMATE` (`mpi/fftw_estimate.h`): measuring plans
  takes minutes without a cycle counter.
- **Sources** are the `genesis13` list in `CMakeLists.txt`, compiled directly
  with `em++`; CMake itself is not used. The diagnostic plugin interface is
  left out because it needs `dlopen`.
- **Single-threaded**, with at most 4 GB of memory.

The only source changes are `src/Core/DiagnosticLive.cpp` and its call sites,
all under `#ifdef __EMSCRIPTEN__`.
