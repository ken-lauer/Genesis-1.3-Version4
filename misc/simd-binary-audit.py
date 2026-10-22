#!/usr/bin/env python3
"""
Audit a built genesis4 binary for vectorization of the hot kernels.

    misc/simd-binary-audit.py path/to/genesis4

Disassembles the hot functions (objdump) and counts double-precision
arithmetic per vector width: packed 512/256/128-bit and scalar. On an
AVX-512 build the particle and field kernels should be dominated by packed
512-bit (zmm) instructions; a kernel with no packed instructions, or only
256-bit ones, did not vectorize the way it should. The counts are static
(instructions in the code, not executed), so scalar instructions next to
packed ones are normal: loop remainders, per-particle gathers/scatters and
per-slice bookkeeping stay scalar, and packed% is only a rough indication.

Also prints the compiler flags recorded in the debug info (GCC records them
with -g) and whether this machine's CPU supports AVX-512.
"""
import collections
import pathlib
import re
import subprocess
import sys

# kernel -> substrings of the demangled function names that implement it
KERNELS = {
    "RK4 push":          ["beamsolver::rungeKutta<", "beamsolver::pushSlice", "BeamSolver::advance("],
    "transverse track":  ["TrackBeam::track(", "applyQuad("],
    "source deposition": ["Field::constructSource("],
    "beam diagnostics":  ["DiagBeam::getValues(", "DiagBeamUser::getValues("],
    "field diagnostics": ["DiagField::getValues("],
    "ADI field solver":  ["FieldSolverADI::halfStepX(", "FieldSolverADI::halfStepY("],
}
# must be inlined into the vectorized loops: a standalone copy that is called
# from a hot function means the call kept a loop scalar
MUST_INLINE = ["sincos_poly(", "beamsolver::ode(", "cmul("]

if len(sys.argv) != 2:
    sys.exit(__doc__)
binary = sys.argv[1]
asm = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", binary],
                     check=True, capture_output=True, text=True).stdout

header = re.compile(r"^[0-9a-f]+ <(.*)>:$")
insn = re.compile(r"^\s*[0-9a-f]+:\s+([a-z][a-z0-9.]*)\s*(.*)$")
functions = collections.defaultdict(list)   # name -> [(mnemonic, operands)]
current = None
for line in asm.splitlines():
    m = header.match(line)
    if m:
        current = m.group(1)
        continue
    m = insn.match(line)
    if m and current:
        functions[current].append((m.group(1), m.group(2)))

x86 = any("%xmm" in ops or "%rip" in ops for f in list(functions.values())[:50] for _, ops in f)


def classify(mnemonic, operands):
    """Return '512', '256', '128' for packed double arithmetic, 'scalar', or None."""
    if x86:
        if mnemonic.startswith(("vmov", "mov", "vbroadcast", "vextract", "vinsert", "vperm2")):
            return None
        if mnemonic.endswith("pd"):
            if "zmm" in operands:
                return "512"
            if "ymm" in operands:
                return "256"
            return "128"
        if mnemonic.endswith("sd") and "cvt" not in mnemonic:
            return "scalar"
        return None
    # AArch64: packed doubles are .2d (on the operands for GNU objdump,
    # on the mnemonic for Apple's), scalar doubles d<n>
    if not mnemonic.startswith("f") or mnemonic.startswith(("fmov", "fcvt")):
        return None
    if ".2d" in operands or mnemonic.endswith(".2d"):
        return "128"
    if re.search(r"\bd\d+\b", operands):
        return "scalar"
    return None


def counts(names):
    c = collections.Counter()
    for name in names:
        for mnemonic, operands in functions[name]:
            k = classify(mnemonic, operands)
            if k:
                c[k] += 1
    return c


widths = ["512", "256", "128"] if x86 else ["128"]
widest = widths[0]
print(f"binary: {binary}  ({'x86-64' if x86 else 'AArch64'})")
print(f"{'kernel':18} {'functions':>9} " + " ".join(f"{'pk' + w:>7}" for w in widths)
      + f" {'scalar':>7} {'packed%':>8}  verdict")
problems = 0
for kernel, patterns in KERNELS.items():
    names = [n for n in functions if any(p in n for p in patterns)]
    c = counts(names)
    packed = sum(c[w] for w in widths)
    total = packed + c["scalar"]
    share = 100. * packed / total if total else 0.
    if not names:
        verdict = "not found"
    elif c[widest] > 0:
        verdict = f"ok ({widest}-bit)"
    elif packed == 0:
        verdict = "SCALAR"
    else:
        verdict = f"narrower than {widest}-bit" + (" (missing -mprefer-vector-width=512?)" if x86 else "")
    if not verdict.startswith("ok"):
        problems += 1
    print(f"{kernel:18} {len(names):9d} " + " ".join(f"{c[w]:7d}" for w in widths)
          + f" {c['scalar']:7d} {share:7.0f}%  {verdict}")

standalone = sorted({n for n in functions for p in MUST_INLINE if p in n})
if standalone:
    problems += 1
    print("\nnot inlined (a call inside a SIMD loop keeps it scalar):")
    for n in standalone:
        print(f"  {n[:110]}")

# compiler flags recorded in the debug info (GCC: DW_AT_producer)
try:
    info = subprocess.run(["readelf", "--debug-dump=info", binary],
                          capture_output=True, text=True).stdout
    producers = sorted({m.group(1).strip() for m in re.finditer(r"DW_AT_producer\s*:\s*(?:\(.*?\):\s*)?(.*)", info)
                        if "-O" in m.group(1)})
    if producers:
        print("\ncompiler flags recorded in the binary:")
        for p in producers[:3]:
            print(f"  {p[:200]}")
except FileNotFoundError:
    pass

cpuinfo = pathlib.Path("/proc/cpuinfo")
if cpuinfo.exists():
    flags = cpuinfo.read_text()
    print(f"\nthis CPU: avx512f {'yes' if ' avx512f' in flags else 'NO'}, avx2 {'yes' if ' avx2' in flags else 'no'}")

print(f"\n{problems} issue(s)" if problems else "\nall hot kernels vectorized")
sys.exit(1 if problems else 0)
