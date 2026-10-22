#!/usr/bin/env python3
"""
Check that every loop marked G4_SIMD_LOOP / G4_SIMD_LOOP_WITH is vectorized.

    vec-audit.py [--verbose] <build-dir> <source>...

Each source is recompiled with the exact command recorded in the build
directory's compile_commands.json plus the compiler's vectorization remarks
(clang: -Rpass=loop-vectorize -Rpass-missed=loop-vectorize; gcc:
-fopt-info-vec-optimized-missed). A marked loop (the `for` on the line after
the macro) must be reported as vectorized, in the sources and in the headers
they include.
"""
import json
import os
import re
import shlex
import subprocess
import sys

args = [a for a in sys.argv[1:] if a != "--verbose"]
verbose = "--verbose" in sys.argv
build_dir = args[0]
sources = [os.path.abspath(s) for s in args[1:]]
with open(os.path.join(build_dir, "compile_commands.json")) as f:
    commands = {os.path.abspath(e["file"]): e for e in json.load(f)}

vectorized = set()
missed = set()
pattern = re.compile(r"^([^:\n]+):(\d+):\d+: (?:remark|optimized|missed): (.*)$", re.M)
for src in sources:
    entry = commands.get(src)
    if entry is None:
        print(f"vec-audit: {src} not in compile_commands.json")
        sys.exit(2)
    argv = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
    version = subprocess.run([argv[0], "--version"], capture_output=True, text=True).stdout
    compiler_id = "clang" if "clang" in version.lower() else "gcc"
    remark_flags = (["-fopt-info-vec-optimized-missed"] if compiler_id == "gcc"
                    else ["-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize"])
    if "-o" in argv:
        argv[argv.index("-o") + 1] = os.devnull
    argv = [a for a in argv if not a.startswith("-MF") and a not in ("-MD", "-MT")]
    argv = [a for i, a in enumerate(argv) if not (i > 0 and argv[i - 1] in ("-MT", "-MF"))]
    result = subprocess.run(argv + remark_flags, cwd=entry["directory"], capture_output=True, text=True)
    if verbose:
        print(f"{src}: {compiler_id}, rc={result.returncode}, {len(result.stderr.splitlines())} remark lines")
        if result.returncode != 0:
            print(result.stderr[:2000])
    for m in pattern.finditer(result.stderr):
        key = (os.path.basename(m.group(1)), int(m.group(2)))
        text = m.group(3)
        if "vectorized loop" in text or "loop vectorized" in text:
            vectorized.add(key)
        elif "not vectorized" in text or "couldn't vectorize" in text or "not vectorized:" in text:
            missed.add(key)

# every marked loop in the sources and in the headers they include
headers = set()
for src in sources:
    for line in open(src):
        m = re.match(r'\s*#include "([^"]+)"', line)
        if m:
            cand = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(src))), "include", m.group(1))
            if os.path.exists(cand):
                headers.add(os.path.abspath(cand))

failures = 0
checked = 0
for path in sorted(sources + list(headers)):
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if "G4_SIMD_LOOP" in stripped and not stripped.startswith(("#", "//", "*")):
            # clang reports the loop at the pragma line, gcc at the `for` line or
            # at the first statement of the body: accept any line of the loop
            depth = 0
            end = i + 1
            for j in range(i + 1, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth <= 0 and "{" in "".join(lines[i + 1:j + 1]):
                    end = j
                    break
            keys = {(os.path.basename(path), line_no) for line_no in range(i + 1, end + 2)}
            checked += 1
            if keys & vectorized:
                continue
            failures += 1
            why = "reported as not vectorized" if keys & missed else "no vectorization remark"
            print(f"{path}:{i + 2}: marked loop {why}")
print(f"vec-audit: {checked} marked loops, {failures} not vectorized")
sys.exit(1 if failures else 0)
