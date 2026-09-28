#!/usr/bin/env python3
"""List Z' <= 1 single-component CSD codes from datasets/test.db.

Eligibility uses the pipeline Z' after special-site → subgroup, not only the
CSD `row.Zprime` value. Structures that become Z' > 1 in that representation
are excluded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyxtal
from pyxtal.db import database

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellnet.graph import smiles_to_graph  # noqa: E402
from cellnet.lattice_conf_pipeline import load_crystal_from_db  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_LOCAL_DB = ROOT / "datasets" / "test.db"
# Fall back to the copy of test.db that PyXtal ships in pyxtal/database.
DEFAULT_DB = _LOCAL_DB if _LOCAL_DB.is_file() else Path(pyxtal.__file__).resolve().parent / "database" / "test.db"
DEFAULT_STATS = ROOT / "checkpoints/cellnet_flow" / "stats.json"


def _hall_vocab(stats_path: Path) -> set[int]:
    import json

    stats = json.loads(stats_path.read_text())
    return {int(key) for key in stats["hall_to_idx"]}


def select_codes(
    db_path: Path,
    *,
    stats_path: Path | None = DEFAULT_STATS,
) -> tuple[list[dict], list[dict]]:
    db = database(str(db_path))
    vocab = _hall_vocab(stats_path) if stats_path and stats_path.is_file() else None
    selected = []
    excluded = []
    for code in db.codes:
        row = db.get_row(code)
        smiles = str(getattr(row, "mol_smi", "") or "")
        zprime = float(getattr(row, "Zprime", float("nan")))
        n_comp = 0 if not smiles else smiles.count(".") + 1
        if zprime != zprime or zprime > 1.0 + 1e-6 or n_comp != 1:
            continue
        record = {
            "code": code,
            "zprime": zprime,
            "Z": getattr(row, "Z", None),
            "spg_num": getattr(row, "spg_num", None),
            "mol_smi": smiles,
        }
        try:
            info = load_crystal_from_db(db_path, code)
        except Exception as exc:
            record["reason"] = f"db_load:{exc}"
            excluded.append(record)
            continue
        record["hall_number"] = info.hall_number
        record["space_group_number"] = info.space_group_number
        record["pipeline_zprime"] = float(info.zprime)
        # Search Z′ after special-site → subgroup. CSD Z′≤1 can become Z′=2.
        if float(info.zprime) > 1.0 + 1e-6:
            record["reason"] = f"pipeline_zprime_{info.zprime:g}"
            excluded.append(record)
            continue
        if smiles_to_graph(info.smiles) is None:
            record["reason"] = "invalid_smiles_graph"
            excluded.append(record)
            continue
        if vocab is not None and info.hall_number not in vocab:
            record["reason"] = f"hall_{info.hall_number}_not_in_flow_vocab"
            excluded.append(record)
            continue
        selected.append(record)
    selected.sort(key=lambda item: item["code"])
    excluded.sort(key=lambda item: item["code"])
    return selected, excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--excluded-out", type=Path, default=None)
    args = parser.parse_args()
    rows, excluded = select_codes(args.db, stats_path=args.stats)
    codes = [row["code"] for row in rows]
    text = "\n".join(codes) + ("\n" if codes else "")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if args.excluded_out is not None:
        args.excluded_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{row['code']}\t{row.get('reason', 'excluded')}"
            for row in excluded
        ]
        args.excluded_out.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(text, end="")
    print(f"# n={len(codes)} excluded={len(excluded)}", flush=True)
    for row in excluded:
        print(f"# excluded {row['code']}: {row.get('reason')}", flush=True)


if __name__ == "__main__":
    main()
