"""Re-check every relaxed structure of a code against experiment with a looser cutoff.

Same test as the pipeline (pymatgen StructureMatcher, H removed, match when the largest
normalized site displacement is below the cutoff), but with stol and the cutoff raised to
--cutoff (default 0.5; counts at 0.3-0.7 are also reported). Scans conf_qrs_*/ and adaptive_*/conf_qrs_*/ *all.cif.

Writes <dir>/<CODE>/rematch_<cutoff>/: matched.cif (lowest energy first), all_mapped.csv
(every structure pymatgen could map, closest first) and summary.json.

    python EM/rematch.py MTNANL OCHTET --dir EM --nproc 48
    python EM/rematch.py OCHTET --ref OCHTET01 --ref-db EM/HEM.db   # vs another polymorph
                                                                     # -> rematch_OCHTET01_<cutoff>/
"""

import argparse
import json
import re
import signal
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Structure  # noqa: E402
from pymatgen.io.cif import CifParser  # noqa: E402

from cellnet.lattice_conf_pipeline import _load_pyxtal_from_db  # noqa: E402

_REF = _MATCHER = None
_TIMEOUT = 10


_DROP = ["H"]


def _init(ref_dict, cutoff, timeout=10, drop=("H",)):
    global _REF, _MATCHER, _TIMEOUT, _DROP
    _TIMEOUT = timeout
    _DROP = list(drop)
    warnings.filterwarnings("ignore")
    _REF = Structure.from_dict(ref_dict)
    _MATCHER = StructureMatcher(ltol=0.3, stol=cutoff, angle_tol=5.0)


_ACTIVE = False


def _timeout(*_):
    if _ACTIVE:  # ignore an alarm that arrives after the comparison finished
        raise TimeoutError


def _compare(item):
    global _ACTIVE
    path, block = item
    label = block.splitlines()[0][5:]
    m = re.search(r"-e(-?[0-9.]+)", label)
    e = -abs(float(m.group(1))) if m else float("nan")
    res = None
    signal.signal(signal.SIGALRM, _timeout)
    try:
        _ACTIVE = True
        signal.alarm(_TIMEOUT)  # PyXtal uses 10 s; raise it on busy nodes (--timeout)
        try:
            s = CifParser.from_str(block).parse_structures(primitive=False)[0]
            s.remove_species(_DROP)
            res = _MATCHER.get_rms_dist(_REF, s)
        except Exception:  # noqa: BLE001  (includes TimeoutError)
            res = None
    except TimeoutError:  # raised while leaving the inner block
        res = None
    finally:
        _ACTIVE = False
        signal.alarm(0)
    return (res[0], res[1], e, label, path) if res else (None, None, e, label, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="+")
    ap.add_argument("--dir", default="EM")
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--ref", help="compare against this refcode instead of CODE")
    ap.add_argument("--ref-db", help="database holding --ref (default: <dir>/<CODE>/<CODE>.db)")
    ap.add_argument("--ref-cif", help="compare against this CIF instead (e.g. the experiment relaxed "
                    "with the run's force field, EM/relax_ref.py); needs --label")
    ap.add_argument("--label", help="output folder tag: rematch_<label>_<cutoff>/")
    ap.add_argument("--timeout", type=int, default=60,
                    help="seconds per comparison before it counts as no match (default 60; PyXtal uses 10, "
                    "which drops real matches on a busy node)")
    ap.add_argument("--drop", default="H", help="elements removed before matching (default H; 'H,O' "
                    "matches the C/N skeleton, so a rotated nitro group does not count)")
    args = ap.parse_args()
    drop = [e for e in args.drop.split(",") if e]

    for code in args.codes:
        d = ROOT / args.dir / code
        ref_code = args.ref or code
        ref_db = ROOT / args.ref_db if args.ref_db else d / f"{code}.db"
        if args.ref_cif:
            ref = CifParser(args.ref_cif).parse_structures(primitive=False)[0]
            ref_code = args.ref_cif
        else:
            ref = _load_pyxtal_from_db(ref_code, db_path=ref_db).to_pymatgen()
        ref.remove_species(drop)
        files = sorted(d.glob("conf_qrs_*/*all.cif")) + sorted(d.glob("adaptive_*/conf_qrs_*/*all.cif"))
        items, blocks = [], {}
        for f in files:
            rel = str(f.relative_to(d))
            for b in f.read_text().split("\ndata_"):
                b = b.strip()
                if b:
                    b = b if b.startswith("data_") else "data_" + b
                    items.append((rel, b))
        with Pool(args.nproc, initializer=_init, initargs=(ref.as_dict(), args.cutoff, args.timeout, drop)) as pool:
            res = pool.map(_compare, items, chunksize=64)
        energies = sorted(r[2] for r in res if r[2] == r[2])  # drop NaN (unlabelled)
        hits = sorted((r for r in res if r[1] is not None and r[1] < args.cutoff), key=lambda r: r[2])
        for (rel, b), r in zip(items, res):
            blocks[(rel, r[3])] = b
        tag = args.label or args.ref
        out = d / (f"rematch_{tag}_{args.cutoff:g}" if tag else f"rematch_{args.cutoff:g}")
        out.mkdir(exist_ok=True)
        (out / "matched.cif").write_text("\n\n".join(blocks[(h[4], h[3])] for h in hits) + ("\n" if hits else ""))
        closest = min((r for r in res if r[1] is not None), key=lambda r: r[1], default=None)
        summary = {
            "code": code, "ref": ref_code, "cutoff": args.cutoff, "n_structures": len(res), "n_matched": len(hits),
            "n_boxes_matched": len({h[4] for h in hits}),
            "closest": {"max_disp": closest[1], "rms": closest[0], "energy": closest[2],
                        "label": closest[3], "file": closest[4]} if closest else None,
            "lowest_energy_match": {"max_disp": hits[0][1], "rms": hits[0][0], "energy": hits[0][2],
                                    "rank": sum(e < hits[0][2] - 1e-6 for e in energies) + 1,
                                    "gap_to_lowest": hits[0][2] - energies[0],
                                    "label": hits[0][3], "file": hits[0][4]} if hits else None,
        }
        mapped = sorted((r for r in res if r[1] is not None), key=lambda r: r[1])
        with open(out / "all_mapped.csv", "w") as f:
            f.write("max_disp,rms,energy,file,label\n")
            f.writelines(f"{r[1]:.4f},{r[0]:.4f},{r[2]:.3f},{r[4]},{r[3]}\n" for r in mapped)
        summary["by_cutoff"] = {}
        for c in (0.3, 0.4, 0.5, 0.6, 0.7):
            if c > args.cutoff + 1e-9:
                continue
            h = sorted((r for r in mapped if r[1] < c), key=lambda r: r[2])
            summary["by_cutoff"][f"{c:g}"] = {
                "n_matched": len(h), "n_boxes": len({r[4] for r in h}),
                "lowest_energy_match": {"energy": h[0][2], "max_disp": h[0][1],
                                        "rank": sum(e < h[0][2] - 1e-6 for e in energies) + 1,
                                        "gap_to_lowest": h[0][2] - energies[0],
                                        "label": h[0][3], "file": h[0][4]} if h else None}
            print(f"    < {c:g}: {len(h):6d} structures in {len({r[4] for r in h}):3d} box files"
                  + (f", lowest-E match {h[0][2]:.2f} (rank {summary['by_cutoff'][f'{c:g}']['lowest_energy_match']['rank']}, "
                     f"+{h[0][2] - energies[0]:.2f}), max_disp {h[0][1]:.3f}" if h else ""), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        lm = summary["lowest_energy_match"]
        print(f"{code:16s} {len(res):7d} structures, {len(hits):5d} match < {args.cutoff} "
              f"in {summary['n_boxes_matched']} box files"
              + (f"; best E {lm['energy']:.2f} (rank {lm['rank']}, +{lm['gap_to_lowest']:.2f}), "
                 f"max_disp {lm['max_disp']:.3f}" if lm else "")
              + (f"; closest max_disp {closest[1]:.3f}" if closest else "; nothing mapped"), flush=True)


if __name__ == "__main__":
    main()
