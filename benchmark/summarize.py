#!/usr/bin/env python3
"""Summarize a benchmark sweep: per-code hit counts and overall coverage.

Reads <root>/<CODE>/<CODE>_pipeline_summary.json for every code in --codes and
writes a CSV with the same columns as benchmark/results.csv. A code is covered if
any of its lattice-free cells matches the experimental structure, or, when none
does, if a cell of the frozen-lattice fallback pass matches (--adaptive-relax).

    python benchmark/summarize.py --root outputs/benchmark --out my_results.csv
    python benchmark/summarize.py --root outputs/benchmark --compare benchmark/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = [
    "code", "hall_number", "zprime", "smiles", "n_cells", "n_hit_cells", "max_sr_percent",
    "first_hit_cell", "n_fallback_cells", "n_fallback_hit_cells", "outcome",
]


def _is_fallback(rec: dict) -> bool:
    return "adaptive_fixed_lattice" in str(rec.get("workdir", "")) or rec.get("relax_lattice") is False


def summarize_code(root: Path, code: str) -> dict | None:
    path = root / code / f"{code}_pipeline_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    recs = summary.get("conf_qrs", [])
    primary = [r for r in recs if not _is_fallback(r)]
    fallback = [r for r in recs if _is_fallback(r)]
    hits = [i for i, r in enumerate(primary) if (r.get("success_rate") or 0) > 0]
    fb_hits = [r for r in fallback if (r.get("success_rate") or 0) > 0]
    if hits:
        outcome = "covered"
    elif fb_hits:
        outcome = "covered_frozen_lattice"
    else:
        outcome = "miss"
    max_sr = max([(r.get("success_rate") or 0) for r in primary], default=0.0)
    return {
        "code": code,
        "hall_number": summary.get("hall_number"),
        "zprime": summary.get("zprime"),
        "smiles": summary.get("smiles"),
        "n_cells": len(primary),
        "n_hit_cells": len(hits),
        "max_sr_percent": f"{max_sr:.4g}",
        "first_hit_cell": (hits[0] + 1) if hits else "",
        "n_fallback_cells": len(fallback),
        "n_fallback_hit_cells": len(fb_hits),
        "outcome": outcome,
    }


def read_codes(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Output directory passed to run_code.sh")
    ap.add_argument("--codes", type=Path, default=HERE / "codes.txt")
    ap.add_argument("--out", type=Path, default=None, help="Write the per-code table here")
    ap.add_argument("--compare", type=Path, default=None, help="Reference table, e.g. benchmark/results.csv")
    args = ap.parse_args()

    codes = read_codes(args.codes)
    rows, missing = [], []
    for code in codes:
        row = summarize_code(args.root, code)
        (rows.append(row) if row else missing.append(code))

    n_free = sum(r["outcome"] == "covered" for r in rows)
    n_fb = sum(r["outcome"] == "covered_frozen_lattice" for r in rows)
    print(f"codes with results: {len(rows)}/{len(codes)}")
    print(f"covered, lattice free:     {n_free}")
    print(f"covered, frozen pass only: {n_fb}")
    print(f"coverage:                  {n_free + n_fb}/{len(rows)}")
    print("misses:", " ".join(r["code"] for r in rows if r["outcome"] == "miss") or "none")
    if missing:
        print(f"no summary yet ({len(missing)}):", " ".join(missing))

    if args.out:
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")

    if args.compare:
        ref = {r["code"]: r for r in csv.DictReader(args.compare.open())}
        changed = [
            (r["code"], ref[r["code"]]["outcome"], r["outcome"], ref[r["code"]]["n_hit_cells"], r["n_hit_cells"])
            for r in rows if r["code"] in ref and ref[r["code"]]["outcome"] != r["outcome"]
        ]
        deltas = [abs(int(r["n_hit_cells"]) - int(ref[r["code"]]["n_hit_cells"])) for r in rows if r["code"] in ref]
        print(f"vs {args.compare.name}: {len(changed)} outcome changes; "
              f"mean |Δ hit cells| = {sum(deltas) / len(deltas):.1f}" if deltas else "no overlap")
        for code, a, b, ha, hb in changed:
            print(f"  {code}: {a} ({ha}) -> {b} ({hb})")


if __name__ == "__main__":
    main()
