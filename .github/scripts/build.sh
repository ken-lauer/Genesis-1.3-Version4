#!/bin/bash

# for test of diagnostic plugin interface (DPI), available only on Linux
ARG_DPI=""
if [ ! -z "$WITH_DPI" ] ; then
	ARG_DPI="-DUSE_DPI=1"
fi

set -xe

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $ARG_DPI -DCMAKE_CXX_FLAGS="-Wall -Wextra $CMAKE_CXX_FLAGS"
make -C build

# unit test of the vectorized particle push, and the check that every marked
# SIMD loop was vectorized by this compiler
(cd build && ctest --output-on-failure)
make -C build vec-audit

ls -la build/
