#!/usr/bin/env python3
"""Build leak-free training tables for joint Z′ = 1 + Z′ = 2 CellNet training.

Molecules are compared by InChIKey (stereo ignored), so every crystal form and every
way of writing the SMILES counts as the same molecule.

  1. Test targets (--exclude, e.g. datasets/benchmark_exclude_inchikeys.txt) are removed
     from ALL training rows, Z′ = 1 and Z′ = 2.
  2. Z′ = 2 training rows whose molecule is in the Z′ = 1 test split are removed, so the
     Z′ = 1 test stays comparable with the current model.
  3. Test splits are not changed (they are not trained on).

Writes:
  <z2-dir>/spade_train.csv        cleaned Z′ = 2 train (in place)
  <out-dir>/train.csv             Z′ = 1 train (cleaned) + Z′ = 2 train (cleaned)
  <out-dir>/test.csv              Z′ = 1 test + Z′ = 2 test
  <out-dir>/build_report.txt      counts and the final leak check

  python scripts/build_joint_dataset.py
"""
from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _key(smiles: str) -> str:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def read(path: Path):
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def write(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z1-dir", default=str(ROOT / "datasets/spade-csp"))
    ap.add_argument("--z2-dir", default=str(ROOT / "datasets/csd-z2"))
    ap.add_argument("--out-dir", default=str(ROOT / "datasets/joint-z12"))
    ap.add_argument("--exclude", default=str(ROOT / "datasets/benchmark_exclude_inchikeys.txt"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    z1, z2, out = Path(args.z1_dir), Path(args.z2_dir), Path(args.out_dir)
    f1, z1_train = read(z1 / "spade_train.csv")
    _, z1_test = read(z1 / "spade_test.csv")
    f2, z2_train = read(z2 / "spade_train.csv")
    _, z2_test = read(z2 / "spade_test.csv")
    assert f1 == f2, f"column mismatch: {f1} vs {f2}"
    targets = {line.split("#", 1)[0].strip() for line in open(args.exclude)} - {""}

    tables = {"z1_train": z1_train, "z1_test": z1_test, "z2_train": z2_train, "z2_test": z2_test}
    smiles = sorted({r["smiles"] for rows in tables.values() for r in rows})
    with Pool(args.workers) as pool:
        keys = dict(zip(smiles, pool.map(_key, smiles, chunksize=500)))
    unkeyed = sum(1 for s in smiles if not keys[s])

    def key(r):
        return keys[r["smiles"]]

    z1_test_keys = {key(r) for r in z1_test} - {""}
    rep = [f"molecules keyed: {len(smiles):,} (no InChIKey: {unkeyed}); test targets: {len(targets)}"]

    def clean(name, rows, drop_z1_test):
        tgt = [r for r in rows if key(r) in targets]
        ovl = [r for r in rows if drop_z1_test and key(r) not in targets and key(r) in z1_test_keys]
        drop = {id(r) for r in tgt + ovl}
        kept = [r for r in rows if id(r) not in drop]
        codes = sorted({r["refcode"] for r in tgt})
        rep.append(f"{name}: {len(rows):,} -> {len(kept):,}  (test-target rows removed {len(tgt)}"
                   + (f": {', '.join(codes)}" if codes else "")
                   + (f"; molecules in Z′=1 test removed {len(ovl)}" if drop_z1_test else "") + ")")
        return kept

    z1_train_c = clean("Z′=1 train", z1_train, drop_z1_test=False)
    z2_train_c = clean("Z′=2 train", z2_train, drop_z1_test=True)

    write(z2 / "spade_train.csv", f2, z2_train_c)
    write(out / "train.csv", f1, z1_train_c + z2_train_c)
    write(out / "test.csv", f1, z1_test + z2_test)

    # Final check: re-read what was written and look for any test-target molecule.
    _, written = read(out / "train.csv")
    leaks = [r["refcode"] for r in written if keys.get(r["smiles"]) in targets]
    _, z2w = read(z2 / "spade_train.csv")
    overlap_left = sum(1 for r in z2w if keys.get(r["smiles"]) in z1_test_keys)
    rep.append(f"joint train {len(written):,} rows (Z′=1 {len(z1_train_c):,} + Z′=2 {len(z2_train_c):,}); "
               f"joint test {len(z1_test) + len(z2_test):,} rows (Z′=1 {len(z1_test):,} + Z′=2 {len(z2_test):,})")
    rep.append(f"CHECK test-target rows left in training: {len(leaks)}; "
               f"Z′=2 train molecules still in Z′=1 test: {overlap_left}")
    text = "\n".join(rep)
    (out / "build_report.txt").write_text(text + "\n")
    print(text)
    if leaks or overlap_left:
        raise SystemExit("LEAK CHECK FAILED")


if __name__ == "__main__":
    main()
