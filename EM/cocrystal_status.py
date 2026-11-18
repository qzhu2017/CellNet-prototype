#!/usr/bin/env python3
"""Progress and results so far of the blind cocrystal arrays (scripts/run_cocrystal.py).

For every system: array tasks finished / total, structures scored, the closest structure to
the experiment (largest site displacement, H removed) and the number of matches below
0.3 / 0.5 / 0.7, with the run that produced the closest one. Reads each task's
<tag>/rematch_exp_0.7/summary.json, written by the task after its search ended.

    python EM/cocrystal_status.py               # all systems in EM_cocrystal/
    python EM/cocrystal_status.py --detail RENPUV
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "EM_cocrystal"))
    ap.add_argument("--detail", default=None, help="print every finished task of this system")
    args = ap.parse_args()
    base = Path(args.dir)
    print(f"{'system':8s} {'tasks':>9s} {'structs':>8s} {'<0.3':>5s} {'<0.5':>5s} {'<0.7':>5s} "
          f"{'closest':>8s}  closest run")
    n_found = n_any = 0
    for plan_f in sorted(base.glob("*/cocrystal_plan.json")):
        pl = json.loads(plan_f.read_text())
        tasks = pl.get("tasks", [])
        rows = []
        for tk in tasks:
            f = plan_f.parent / tk["tag"] / "rematch_exp_0.7" / "summary.json"
            if f.exists():
                s = json.loads(f.read_text())
                g = pl["groups"][tk["run"]]
                rows.append((tk, g, s))
        n_str = sum(s["n_structures"] for _, _, s in rows)
        cnt = {c: sum(s["by_cutoff"].get(c, {}).get("n_matched", 0) for _, _, s in rows) for c in ("0.3", "0.5", "0.7")}
        best = min((r for r in rows if r[2]["closest"]), key=lambda r: r[2]["closest"]["max_disp"], default=None)
        where = (f"{best[1]['symbol']} {best[1]['wyckoff']} Z'={best[1]['zprime_fu']:g}" if best else "")
        close = f"{best[2]['closest']['max_disp']:.3f}" if best else "-"
        n_any += bool(rows)
        n_found += cnt["0.3"] > 0
        print(f"{pl['code']:8s} {len(rows):4d}/{len(tasks):<4d} {n_str:8d} {cnt['0.3']:5d} {cnt['0.5']:5d} "
              f"{cnt['0.7']:5d} {close:>8s}  {where}")
        if args.detail == pl["code"]:
            for tk, g, s in sorted(rows, key=lambda r: (r[2]["closest"] or {}).get("max_disp", 9)):
                c = s["closest"]
                print(f"    {tk['tag']:32s} {s['n_structures']:6d} structs  "
                      f"<0.3 {s['by_cutoff'].get('0.3', {}).get('n_matched', 0):4d}  "
                      f"closest {c['max_disp']:.3f}" if c else f"    {tk['tag']:32s} nothing mapped")
    print(f"\nfound (< 0.3) in {n_found} of {n_any} systems with results")


if __name__ == "__main__":
    main()
