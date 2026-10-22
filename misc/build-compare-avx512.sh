#!/usr/bin/env bash
# Build the three binaries to compare on a Linux AVX-512 machine:
#   build-master/genesis4   baseline (master)
#   build-exact/genesis4    readable-simd, ADI bit-identical to master (default)
#   build-ulp/genesis4      readable-simd, -DG4_ADI_FAST=ON (last-ulp differences)
# Usage: misc/build-compare-avx512.sh [extra cmake args, e.g. -DCMAKE_CXX_COMPILER=g++-13]
# All three use the same flags: Release, -march=native (AVX-512 on such a node),
# 512-bit vectors preferred (GCC/clang default to 256-bit on many Intel parts).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FLAGS="-march=native -mprefer-vector-width=512"

build() { # src builddir extra...
    cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$FLAGS" "${@:3}"
    cmake --build "$2" -j "$(nproc)"
}

# baseline from a detached worktree of master
[ -d "$ROOT/.claude/worktrees/master-cmp" ] || git -C "$ROOT" worktree add --detach "$ROOT/.claude/worktrees/master-cmp" master
build "$ROOT/.claude/worktrees/master-cmp" "$ROOT/build-master" "$@"

build "$ROOT" "$ROOT/build-exact" -DG4_ADI_FAST=OFF "$@"
build "$ROOT" "$ROOT/build-ulp"   -DG4_ADI_FAST=ON  "$@"

cmake --build "$ROOT/build-exact" --target vec-audit   # marked loops must vectorize here too
ls -la "$ROOT"/build-{master,exact,ulp}/genesis4
