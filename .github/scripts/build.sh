#!/bin/bash

set -xe

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wall -Wextra $CMAKE_CXX_FLAGS"

make -j $(($(nproc) - 1)) -C build

ls -la build/genesis4

