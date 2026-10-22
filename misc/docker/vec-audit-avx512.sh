#!/usr/bin/env bash
# Build genesis4 for AVX-512 with GCC in a Linux container (compile only on a
# non-x86 host) and check that the marked SIMD loops vectorize there:
#   misc/docker/vec-audit-avx512.sh [extra cmake args]
# Prints the vec-audit result and, per hot object file, how many instructions
# use zmm (512-bit) registers.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
IMAGE=genesis4-avx512-build
docker build --platform linux/amd64 -t $IMAGE -f "$HERE/Dockerfile.avx512" "$HERE" > /dev/null
docker run --rm --platform linux/amd64 -v "$ROOT":/src:ro $IMAGE bash -c '
  set -e
  cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=x86-64-v4 -mprefer-vector-width=512" '"$*"' > /build/cmake.log 2>&1 || { tail -30 /build/cmake.log; exit 1; }
  cmake --build /build --target genesis13 -j "$(nproc)" > /build/build.log 2>&1 || { grep -E "error" /build/build.log | head -20; exit 1; }
  echo "== vec-audit (GCC $(g++ -dumpversion), -march=x86-64-v4)"
  cmake --build /build --target vec-audit 2>&1 | grep -E "marked loop|vec-audit:" || true
  echo "== zmm instructions per hot object"
  for o in BeamSolver TrackBeam Field Diagnostic DiagnosticUser FieldSolverADI; do
    f=$(find /build -name "$o.cpp.o" | head -1)
    printf "%-16s zmm: %6d   ymm: %6d   xmm: %6d\n" "$o" \
      "$(objdump -d "$f" | grep -c zmm)" "$(objdump -d "$f" | grep -c ymm)" "$(objdump -d "$f" | grep -c xmm)"
  done
'
