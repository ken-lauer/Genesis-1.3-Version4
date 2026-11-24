"""
Compare a wasm run against a native reference.

Usage: `python compare.py reference.out.h5 wasm.out.h5 [live.json]`
"""

import json
import sys

import h5py
import numpy as np


def max_rel_diff(x: np.ndarray, y: np.ndarray) -> float:
    """
    Largest absolute difference, relative to the largest reference magnitude.
    """
    scale = np.max(np.abs(x))
    return 0.0 if scale == 0 else float(np.max(np.abs(x - y)) / scale)


def main(reference: str, candidate: str, live: str | None = None) -> None:
    ref, new = h5py.File(reference, "r"), h5py.File(candidate, "r")
    names: list[str] = []
    ref.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    numeric = [n for n in names if not n.startswith("Meta") and ref[n].dtype.kind in "fiu"]
    missing = [n for n in numeric if n not in new]
    diffs = sorted(
        ((max_rel_diff(ref[n][()].astype(float), new[n][()].astype(float)), n)
         for n in numeric if n not in missing and ref[n].shape == new[n].shape),
        reverse=True,
    )
    print(f"{len(diffs)} datasets compared, {sum(d == 0 for d, _ in diffs)} bit-identical")
    for d, n in diffs[:5]:
        print(f"  {d:.3e}  {n}")
    print("missing:", missing)

    if live:
        with open(live) as fp:
            trace = json.load(fp)
        worst = max(
            (max_rel_diff(new[k][()].astype(float).ravel(), np.asarray(v)), k)
            for k, v in trace.items() if k in new
        )
        print(f"live trace vs wasm output file, worst of {len(trace) - 1} keys: {worst[0]:.3e} ({worst[1]})")


if __name__ == "__main__":
    main(*sys.argv[1:])
