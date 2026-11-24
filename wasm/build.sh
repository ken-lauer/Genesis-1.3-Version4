#!/bin/bash
# Build genesis4 for WebAssembly with the emscripten/emsdk Docker image.
# Outputs: wasm/build/genesis4.{mjs,wasm} (browser/worker/node module) and
# wasm/build/genesis4-node.{js,wasm} (node CLI on the host file system).
set -euo pipefail
EMSDK_IMAGE=${EMSDK_IMAGE:-emscripten/emsdk:7.0.1}
HDF5_URL=https://github.com/usnistgov/libhdf5-wasm/releases/download/v0.6.0_4.0.23/HDF5-1.14.6-Emscripten.tar.gz
FFTW_URL=https://www.fftw.org/fftw-3.3.10.tar.gz

if [[ ${1:-} != --in-container ]]; then
    root=$(cd "$(dirname "$0")/.." && pwd)
    build=$root/wasm/build
    mkdir -p "$build/hdf5" "$build/include" "$build/obj"
    if [[ ! -f $build/hdf5/lib/libhdf5.a ]]; then
        curl -sL "$HDF5_URL" | tar xz -C "$build/hdf5"
    fi
    if [[ ! -d $build/fftw-src ]]; then
        mkdir -p "$build/fftw-src"
        curl -sL "$FFTW_URL" | tar xz -C "$build/fftw-src" --strip-components=1
    fi
    # the container cannot resolve a worktree's git dir, so stamp the version here
    v() { grep -oE "_GENESIS_VERSION_$1 \"[a-z0-9]+\"" "$root/git-hash.cmake" | cut -d'"' -f2; }
    sed -e "s/@_GENESIS_VERSION_MAJOR@/$(v MAJOR)/" -e "s/@_GENESIS_VERSION_MINOR@/$(v MINOR)/" \
        -e "s/@_GENESIS_VERSION_REV@/$(v REV)/" -e "s/@_GENESIS_VERSION_BETA@/$(v BETA)/" \
        -e "s/@_user_name@/emscripten/" -e "s/@_config_time@/$(date -u '+%Y-%m-%d %H:%M:%S [UTC]')/" \
        -e "s/@GIT_HASH@/$(git -C "$root" log -1 --pretty=format:%H)/" \
        "$root/include/version.h.in" > "$build/include/version.h"
    exec docker run --rm -v "$root":/src "$EMSDK_IMAGE" /src/wasm/build.sh --in-container
fi

B=/src/wasm/build
if [[ ! -f $B/fftw/lib/libfftw3.a ]]; then
    (
        cd $B/fftw-src
        CFLAGS="-O3 -msimd128" emconfigure ./configure --prefix=$B/fftw --disable-fortran \
            --disable-shared --enable-static --disable-doc > $B/fftw-configure.log 2>&1
        emmake make -j"$(nproc)" install > $B/fftw-make.log 2>&1
    )
fi

# FFTW=1 changes class layouts in the headers, so it must reach every file
export CXXFLAGS="-O3 -std=c++17 -fopenmp-simd -fno-math-errno -msimd128 -fwasm-exceptions -w \
    -DOMPI_SKIP_MPICXX -DFFTW=1 -I/src/wasm/mpi -I/src/include -I$B/include -I$B/hdf5/include \
    -I$B/fftw/include -include /src/wasm/mpi/h5mpio_stub.h \
    -include /src/wasm/mpi/fftw_estimate.h"
if [[ $(cat $B/obj/.flags 2>/dev/null) != "$CXXFLAGS" ]]; then
    rm -f $B/obj/*.o
    echo "$CXXFLAGS" > $B/obj/.flags
fi
compile() {
    obj=/src/wasm/build/obj/$(basename "${1%.cpp}").o
    [[ $obj -nt $1 && -z $(find /src/include /src/wasm/mpi -newer "$obj" -print -quit) ]] && return 0
    extra=""
    # matches the native build: the ADI solver is bit-exact only without FMA contraction
    [[ $1 == *FieldSolverADI.cpp ]] && extra="-ffp-contract=off"
    em++ $CXXFLAGS $extra -c "$1" -o "$obj"
}
export -f compile
# the plugin interface needs dlopen
ls /src/src/*/*.cpp | grep -vE "RegPlugin|DiagnosticHook" | xargs -P "$(nproc)" -I{} bash -c 'compile {}'

LIBS="$B/fftw/lib/libfftw3.a $B/hdf5/lib/libhdf5.a $B/hdf5/lib/libszaec.a $B/hdf5/lib/libaec.a $B/hdf5/lib/libzlib-static.a"
LINK="-O3 -fwasm-exceptions -msimd128 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=5MB"
em++ $LINK $B/obj/*.o $LIBS -sNODERAWFS=1 -sENVIRONMENT=node -o $B/genesis4-node.js
em++ $LINK $B/obj/*.o $LIBS -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node \
    -sINVOKE_RUN=0 -sEXIT_RUNTIME=0 -sEXPORTED_RUNTIME_METHODS=callMain,FS -o $B/genesis4.mjs
ls -l $B/genesis4*
