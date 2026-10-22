#!/usr/bin/env bash
# Build the three binaries to compare on a Linux AVX-512 machine:
#   build-master/genesis4   baseline (the origin/master commit this branch is based on)
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

# baseline: the upstream commit this branch is based on (a local `master`
# branch may be stale, and a different shot-noise seeding scheme makes SASE
# results incomparable)
git -C "$ROOT" fetch -q origin master
BASE=$(git -C "$ROOT" merge-base HEAD origin/master)
WT="$ROOT/.claude/worktrees/master-cmp"
if [ -d "$WT" ]; then git -C "$WT" checkout -q --detach "$BASE"; else git -C "$ROOT" worktree add -q --detach "$WT" "$BASE"; fi
echo "baseline: $(git -C "$WT" log -1 --format='%h %s')"
build "$WT" "$ROOT/build-master" "$@"

build "$ROOT" "$ROOT/build-exact" -DG4_ADI_FAST=OFF "$@"
build "$ROOT" "$ROOT/build-ulp"   -DG4_ADI_FAST=ON  "$@"

cmake --build "$ROOT/build-exact" --target vec-audit   # marked loops must vectorize here too
ls -la "$ROOT"/build-{master,exact,ulp}/genesis4
