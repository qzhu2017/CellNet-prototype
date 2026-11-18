#!/usr/bin/env python3
"""Collect CSD crystals with a given Z′ (default 2) as CellNet training data.

Standalone: needs only the CSD Python API (``ccdc``, CSD licence) and, preferably, RDKit
and tqdm. Run it on any machine with the CSD installed; copy the output CSV to the cluster.

Selection = the SPaDe-CSP Z′ = 1 set the current model was trained on
(SPaDe-CSP notebook/01_dataset.ipynb), with Z′ = 2 instead of 1:
  organic, 3D coordinates, not polymeric, not a powder study, R < 10 %,
  not a CSD Communication, no solvent, space group 1-230
plus one kind of molecule only: both independent molecules have the same canonical SMILES
(two copies of one molecule; cocrystals, salts and fragments are rejected).
Only the elements CellNet encodes (C H N O S P F Cl Br I, as in the Z′ = 1 set).
Then the SPaDe limits (MW <= 1500, cell edges 2-50 Å, angles 60-120°, density <= 3,
Z <= 32), and redeterminations of the same crystal are merged (lowest R kept).
Output columns match SPaDe's crystal-info_CSD_filtered.csv; SMILES and MW are for ONE
molecule.

QUICK START (local terminal, in the Python that has ccdc):

  python collect_csd_zprime.py probe                      # ~2 min: speed + yield on 5,000 entries
  python collect_csd_zprime.py run --workers 8            # whole CSD -> csd_zprime2/
  python collect_csd_zprime.py run --workers 8 --max-rows 20000 --no-disorder

  Interrupted? Run the same `run` command again: finished chunks are skipped.
  Result: csd_zprime2/crystal-info_CSD_Z2_filtered.csv (+ merge_report.txt)

On the cluster afterwards:
  python scripts/build_dataset.py --crystal-info crystal-info_CSD_Z2_filtered.csv --out-dir datasets/csd-z2
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

COLUMNS = [
    "refcode", "SMILES", "MW", "sg_symbol", "sg_number", "Z-prime", "Z-value",
    "a", "b", "c", "alpha", "beta", "gamma", "density", "temperature",
    # extra columns (ignored by build_dataset.py; for filtering and splitting by molecule)
    "inchikey", "r_factor", "has_disorder", "n_components",
]

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:  # still works with the CSD's own SMILES
    HAVE_RDKIT = False


def _canonical(smiles: str):
    """Canonical SMILES and InChIKey without stereo marks (RDKit).

    Stereo is dropped, as in the training set (none of its 135,837 SMILES has @), so
    the two molecules of a Z′ = 2 crystal that are mirror images (R + S) count as the same
    compound. Without RDKit the CSD SMILES is used with @ marks removed and no InChIKey.
    """
    if not smiles:
        return None, ""
    if not HAVE_RDKIT:
        return smiles.replace("@", ""), ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, ""
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)  # twice, as in SPaDe
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=False), Chem.MolToInchiKey(mol)


def _passes_entry_filters(entry, zprime: float) -> bool:
    sg_num = entry.crystal.spacegroup_number_and_setting[0]
    return (
        1 <= sg_num <= 230
        and abs(float(entry.crystal.z_prime) - zprime) < 1e-3
        and entry.has_3d_structure is True
        and entry.is_organic is True
        and entry.is_polymeric is False
        and entry.is_powder_study is False
        and entry.r_factor is not None
        and entry.r_factor < 10
        and entry.publication[1].name != "CSD Communication"
        and entry.solvent is None
    )


def _row(entry, zprime: float):
    """Return (row or None, reason)."""
    if not _passes_entry_filters(entry, zprime):
        return None, "filters"
    comps = list(entry.molecule.components)
    canon = [_canonical(c.smiles) for c in comps]
    smiles = {s for s, _ in canon}
    n_expected = int(round(zprime)) if zprime >= 1 else 1
    if None in smiles or len(smiles) != 1 or len(comps) != n_expected:
        return None, "mixed"
    smi, inchikey = canon[0]
    crys = entry.crystal
    return {
        "refcode": crys.identifier,
        "SMILES": smi,
        "MW": comps[0].molecular_weight,
        "sg_symbol": crys.spacegroup_symbol,
        "sg_number": crys.spacegroup_number_and_setting[0],
        "Z-prime": crys.z_prime,
        "Z-value": crys.z_value,
        "a": crys.cell_lengths[0], "b": crys.cell_lengths[1], "c": crys.cell_lengths[2],
        "alpha": crys.cell_angles[0], "beta": crys.cell_angles[1], "gamma": crys.cell_angles[2],
        "density": crys.calculated_density,
        "temperature": entry.temperature,
        "inchikey": inchikey,
        "r_factor": entry.r_factor,
        "has_disorder": entry.has_disorder,
        "n_components": len(comps),
    }, "kept"


def _scan_chunk(task):
    """Worker: scan entries [start, stop) into part_<start>.csv; marks it done at the end."""
    start, stop, zprime, part = task
    from ccdc import io  # each process opens its own reader

    reader = io.EntryReader("CSD")
    counts = {"kept": 0, "filters": 0, "mixed": 0, "error": 0}
    tmp = part.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for i in range(start, stop):
            try:
                row, why = _row(reader[i], zprime)
            except Exception:
                counts["error"] += 1
                continue
            counts[why] += 1
            if row is not None:
                w.writerow(row)
    tmp.rename(part)  # a part file exists only when its chunk finished
    return start, stop, counts


def _n_entries() -> int:
    from ccdc import io

    return len(io.EntryReader("CSD"))


def probe(args) -> None:
    """Scan a small slice to show speed and yield before the full run."""
    from ccdc import io

    reader = io.EntryReader("CSD")
    n = len(reader)
    rng = random.Random(0)
    idx = sorted(rng.sample(range(n), min(args.n, n)))
    counts = {"kept": 0, "filters": 0, "mixed": 0, "error": 0}
    examples = []
    t0 = time.time()
    for i in idx:
        try:
            row, why = _row(reader[i], args.zprime)
        except Exception:
            counts["error"] += 1
            continue
        counts[why] += 1
        if row is not None and len(examples) < 5:
            examples.append(row)
    dt = time.time() - t0
    rate = len(idx) / dt
    kept_frac = counts["kept"] / len(idx)
    print(f"CSD entries: {n:,}   RDKit: {'yes' if HAVE_RDKIT else 'NO (CSD SMILES used, no InChIKey)'}")
    print(f"Probe: {len(idx):,} random entries in {dt:.0f} s ({rate:.0f} entries/s on 1 core)")
    print(f"  kept {counts['kept']}, rejected by filters {counts['filters']}, "
          f"rejected as cocrystal/salt/fragment {counts['mixed']}, unreadable {counts['error']}")
    print(f"Estimate for the whole CSD: ~{kept_frac * n:,.0f} Z′ = {args.zprime:g} entries before "
          f"cell limits and de-duplication; ~{n / rate / 3600:.1f} h on 1 core "
          f"(divide by --workers)")
    for r in examples:
        print(f"  e.g. {r['refcode']}: {r['sg_symbol']}  Z′ {r['Z-prime']}  {r['SMILES'][:60]}")


def run(args) -> None:
    out_dir = Path(args.out_dir or f"csd_zprime{args.zprime:g}")
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    n = args.stop or _n_entries()
    tasks = []
    for start in range(args.start, n, args.chunk):
        stop = min(start + args.chunk, n)
        part = parts_dir / f"part_{start:08d}_{stop:08d}.csv"
        if not part.exists():
            tasks.append((start, stop, args.zprime, part))
    total_chunks = len(range(args.start, n, args.chunk))
    print(f"{n:,} CSD entries in {total_chunks} chunks of {args.chunk:,}; "
          f"{total_chunks - len(tasks)} already done, {len(tasks)} to scan with {args.workers} workers")
    if not HAVE_RDKIT:
        print("Note: RDKit not found; using CSD SMILES and no InChIKey (de-duplication uses SMILES).")

    t0 = time.time()
    totals = {"kept": 0, "filters": 0, "mixed": 0, "error": 0}
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for k, (start, stop, counts) in enumerate(pool.imap_unordered(_scan_chunk, tasks), 1):
                for key in totals:
                    totals[key] += counts[key]
                el = time.time() - t0
                eta = el / k * (len(tasks) - k)
                print(f"[{k}/{len(tasks)}] entries {start:,}-{stop:,}: kept {counts['kept']}  "
                      f"(total kept {totals['kept']:,}; {el / 60:.0f} min elapsed, ~{eta / 60:.0f} min left)",
                      flush=True)
    parts = sorted(parts_dir.glob("part_*.csv"))
    out = out_dir / f"crystal-info_CSD_Z{args.zprime:g}_filtered.csv"
    report = merge_files(parts, out, max_rows=args.max_rows, max_z=args.max_z,
                         no_disorder=args.no_disorder, exclude=args.exclude_inchikeys, seed=args.seed,
                         allowed_elements=None if args.any_elements else frozenset(TRAINING_ELEMENTS))
    (out_dir / "merge_report.txt").write_text(report + "\n")
    print(report)


def _f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


TRAINING_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def _elements(smiles: str) -> set:
    """Element symbols in a SMILES (RDKit, unsanitised); {"?"} if it cannot be read."""
    if HAVE_RDKIT:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        return {a.GetSymbol() for a in mol.GetAtoms()} if mol is not None else {"?"}
    import re

    return set(re.findall(r"Cl|Br|Si|Se|Te|As|Li|Na|Mg|Ca|Sr|Ba|[A-Z][a-z]?", smiles.replace("c", "C")
                          .replace("n", "N").replace("o", "O").replace("s", "S").replace("p", "P")))


def merge_files(parts, out, *, max_rows=None, max_z=32, no_disorder=False, exclude=None, seed=42,
                allowed_elements=frozenset(TRAINING_ELEMENTS)) -> str:
    rows = []
    for p in parts:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    n_raw = len(rows)

    # Scans run without RDKit leave InChIKey empty and keep the CSD SMILES. Fill both in
    # here when RDKit is available (e.g. re-merge on another machine), so that
    # de-duplication and the InChIKey exclusion list work.
    filled = 0
    if HAVE_RDKIT:
        for r in rows:
            if not r.get("inchikey"):
                smi, key = _canonical(r["SMILES"])
                if smi:
                    r["SMILES"], r["inchikey"] = smi, key
                    filled += bool(key)
    missing_keys = sum(1 for r in rows if not r.get("inchikey"))
    if exclude and missing_keys and not HAVE_RDKIT:
        print(f"WARNING: {missing_keys:,} rows have no InChIKey (RDKit missing), so the exclusion "
              f"list cannot be applied to them. Re-run `merge` where RDKit is installed.")

    # Same elements as the Z′ = 1 training set (CellNet's graph encodes only these; any
    # other atom would become an all-zero element vector). This also removes the s-block
    # organometallics the CSD "organic" flag lets through.
    n_before_el = len(rows)
    if allowed_elements:
        rows = [r for r in rows if _elements(r["SMILES"]) <= allowed_elements]
    n_elements = len(rows)
    if HAVE_RDKIT:  # with RDKit, an empty InChIKey means InChI failed (metal complexes etc.)
        rows = [r for r in rows if r.get("inchikey")]
    n_keyed = len(rows)

    def limits(r):  # SPaDe post-filter; Z limit raised for Z′ = 2 (Z = Z′ × multiplicity)
        vals = [_f(r, k) for k in ("MW", "Z-value", "a", "b", "c", "alpha", "beta", "gamma", "density")]
        if None in vals:
            return False
        mw, z, a, b, c, al, be, ga, rho = vals
        return (mw <= 1500 and z <= max_z and all(2 <= x <= 50 for x in (a, b, c))
                and all(60 <= x <= 120 for x in (al, be, ga)) and rho <= 3)

    rows = [r for r in rows if limits(r)]
    n_limits = len(rows)
    if no_disorder:
        rows = [r for r in rows if str(r.get("has_disorder")) not in ("True", "1")]
    n_disorder = len(rows)

    # Redeterminations (ABCDEF, ABCDEF01, ...) of one crystal: same molecule, same space
    # group, cell lengths within 2 %. Keep the lowest R factor. Polymorphs are kept.
    rows.sort(key=lambda r: (_f(r, "r_factor") or 99.0))
    kept, seen = [], {}
    for r in rows:
        key = (r.get("inchikey") or r["SMILES"], r["sg_number"])
        abc = sorted(_f(r, k) for k in ("a", "b", "c"))
        if any(all(abs(x - y) <= 0.02 * y for x, y in zip(abc, o)) for o in seen.get(key, [])):
            continue
        seen.setdefault(key, []).append(abc)
        kept.append(r)
    n_unique = len(kept)

    if exclude:
        # one InChIKey per line; text after "#" is a comment
        bad = {line.split("#", 1)[0].strip() for line in open(exclude)} - {""}
        kept = [r for r in kept if (r.get("inchikey") or r["SMILES"]) not in bad]
    n_excl = len(kept)

    if max_rows and len(kept) > max_rows:
        random.Random(seed).shuffle(kept)
        kept = kept[:max_rows]

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[""] + COLUMNS)  # leading index column, as SPaDe
        w.writeheader()
        for i, r in enumerate(kept):
            w.writerow({"": i, **{k: r.get(k, "") for k in COLUMNS}})

    from collections import Counter

    sg = Counter(r["sg_symbol"] for r in kept).most_common(8)
    n_mol = len({r.get("inchikey") or r["SMILES"] for r in kept})
    return (
        f"raw {n_raw:,} -> elements {n_elements:,} -> InChIKey ok {n_keyed:,} -> "
        f"cell/MW limits {n_limits:,} -> disorder filter {n_disorder:,} -> "
        f"redeterminations merged {n_unique:,} -> exclusions {n_excl:,} -> written {len(kept):,}\n"
        f"distinct molecules: {n_mol:,}" + (f"   (InChIKeys filled in by RDKit: {filled:,})" if filled else "") + "\n"
        f"most common space groups: " + ", ".join(f"{s} {c}" for s, c in sg) + f"\n-> {out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="speed and yield on a random sample (run this first)")
    p.add_argument("--zprime", type=float, default=2.0)
    p.add_argument("--n", type=int, default=5000, help="number of random entries to test")

    r = sub.add_parser("run", help="scan the whole CSD in parallel, then merge")
    r.add_argument("--zprime", type=float, default=2.0)
    r.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    r.add_argument("--out-dir", default=None, help="default: csd_zprime<Z′>")
    r.add_argument("--chunk", type=int, default=20000, help="entries per chunk (resume unit)")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--stop", type=int, default=None, help="stop index (default: whole CSD)")
    r.add_argument("--max-rows", type=int, default=None, help="random cap after de-duplication, e.g. 20000")
    r.add_argument("--max-z", type=float, default=32, help="Z limit (SPaDe used 16 for Z′ = 1)")
    r.add_argument("--no-disorder", action="store_true", help="drop entries flagged as disordered")
    r.add_argument("--exclude-inchikeys", default=None, help="file of InChIKeys to leave out (benchmark targets)")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--any-elements", action="store_true",
                   help="keep elements other than C H N O S P F Cl Br I (CellNet cannot encode them)")

    m = sub.add_parser("merge", help="re-merge existing part files with other settings (no rescan)")
    m.add_argument("parts", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--max-rows", type=int, default=None)
    m.add_argument("--max-z", type=float, default=32)
    m.add_argument("--no-disorder", action="store_true")
    m.add_argument("--exclude-inchikeys", default=None)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--any-elements", action="store_true",
                   help="keep elements other than C H N O S P F Cl Br I (CellNet cannot encode them)")

    args = ap.parse_args()
    if args.cmd == "probe":
        probe(args)
    elif args.cmd == "run":
        run(args)
    else:
        print(merge_files(args.parts, args.out, max_rows=args.max_rows, max_z=args.max_z,
                          no_disorder=args.no_disorder, exclude=args.exclude_inchikeys, seed=args.seed,
                          allowed_elements=None if args.any_elements else frozenset(TRAINING_ELEMENTS)))


if __name__ == "__main__":
    main()
