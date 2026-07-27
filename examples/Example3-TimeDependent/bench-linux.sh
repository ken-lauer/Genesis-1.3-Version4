#!/usr/bin/env bash
# Perf-instrumented benchmark of the ADI field solver (Linux).
#
# Usage:
#   ./bench-linux.sh /path/to/genesis4 [label]
#
# The binary and inputs are copied to local /tmp and the benchmark runs
# there, so a network filesystem (home/scratch) can't distort the timing.
# Results are copied back to ./bench-out/<label>/. Run once per binary
# (baseline, xsimd) and compare the perf-stat.txt / perf-top.txt files.
#
# Build the binary with:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_CXX_FLAGS="-march=native" -DENABLE_PROFILING=ON
#   cmake --build build -j
#
# -march=native is REQUIRED for xsimd to use AVX-512; without it x86 builds
# fall back to SSE2 (2-wide, same as NEON). Use identical flags for the
# baseline build so the comparison is fair.
#
# If perf complains about permissions:  sudo sysctl kernel.perf_event_paranoid=1

set -euo pipefail

BIN=$(readlink -f "$1")
LABEL=${2:-run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$HERE/bench-out/$LABEL
mkdir -p "$OUT"

TMP=$(mktemp -d /tmp/genesis4-bench.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
cp "$BIN" "$TMP/genesis4"
cp "$HERE/example3-bench.in" "$HERE/Example3.lat" "$TMP/"
cd "$TMP"

echo "== perf stat: IPC + cache hierarchy =="
perf stat -ddd -o perf-stat.txt -- ./genesis4 example3-bench.in > run-stat.log
grep -E "elapsed|insn per cycle|L1-dcache|LLC|dTLB" perf-stat.txt || tail -n 25 perf-stat.txt

# Intel only; comment out elsewhere. Answers "bound on what?" directly.
if perf stat -M TopdownL1 -o topdown.txt -- true 2>/dev/null; then
    echo "== topdown L1 =="
    perf stat -M TopdownL1 -o topdown.txt -- ./genesis4 example3-bench.in > run-topdown.log
    tail -n 15 topdown.txt
fi

echo "== perf record: callgraph profile =="
perf record -F 997 --call-graph fp -o perf.data -- ./genesis4 example3-bench.in > run-record.log
perf report --stdio --no-children -i perf.data 2>/dev/null | head -40 | tee perf-top.txt

# keep the binary next to perf.data: the recorded path is the /tmp copy,
# which is deleted on exit (perf's build-id cache usually covers this, but
# --symfs works even when it doesn't)
cp -r "$TMP"/{perf-stat.txt,perf.data,perf-top.txt,run-*.log,genesis4} "$OUT/"
[ -f "$TMP/topdown.txt" ] && cp "$TMP/topdown.txt" "$OUT/"

cat <<EOF

Results in $OUT
  perf-stat.txt   counters: IPC, L1d/LLC miss rates, TLB
  topdown.txt     frontend/backend/memory-bound split (Intel)
  perf.data       explore interactively:
      perf report -i $OUT/perf.data
      perf annotate -i $OUT/perf.data tridagx   # per-instruction hotspots
EOF
