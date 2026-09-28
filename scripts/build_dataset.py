#!/usr/bin/env python3
"""Build SPaDe V9 train/test CSVs from crystal-info_CSD_filtered.csv (170,278 rows)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellnet.spade import extract_v9_csv, split_crystal_info_csv  # noqa: E402

DEFAULT_SPADE_DIR = ROOT / "datasets/spade-csp"
DEFAULT_CRYSTAL_INFO = DEFAULT_SPADE_DIR / "crystal-info_CSD_filtered.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split crystal-info (170k) and extract V9 Hall/cell tables"
    )
    parser.add_argument(
        "--crystal-info",
        type=Path,
        default=DEFAULT_CRYSTAL_INFO,
        help="Full SPaDe filtered CSD table (170,278 rows)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SPADE_DIR)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Only extract; read split_train/test.csv already in --out-dir",
    )
    parser.add_argument(
        "--no-rhombo-hex",
        action="store_true",
        help="Disable rhombohedral→hexagonal cell conversion",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    crystal_info = args.crystal_info.resolve()
    train_raw = out_dir / "split_train.csv"
    test_raw = out_dir / "split_test.csv"
    train_out = out_dir / "spade_train.csv"
    test_out = out_dir / "spade_test.csv"

    if not args.skip_split:
        n_train, n_test = split_crystal_info_csv(
            crystal_info,
            train_raw,
            test_raw,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        print(
            f"Split {crystal_info.name}: {n_train:,} train + {n_test:,} test "
            f"({args.test_fraction:.0%} test, seed={args.seed})"
        )

    convert_rhombo = not args.no_rhombo_hex
    total_written = 0
    total_alias = total_fallback = total_rhombo = 0
    for src, dst in ((train_raw, train_out), (test_raw, test_out)):
        if not src.is_file():
            raise FileNotFoundError(f"Missing split file: {src}")
        stats = extract_v9_csv(src, dst, convert_rhombo=convert_rhombo)
        total_written += stats.written
        total_alias += stats.alias_mapped
        total_fallback += stats.fallback_mapped
        total_rhombo += stats.rhombo_converted
        print(
            f"Wrote {stats.written:,} rows → {dst} "
            f"(alias={stats.alias_mapped:,}, fallback={stats.fallback_mapped:,}, "
            f"rhombo→hex={stats.rhombo_converted:,}, "
            f"skipped hall={stats.skipped_unmapped_hall:,}, "
            f"skipped SMILES={stats.skipped_missing_smiles:,})"
        )

    print(
        f"V9 total: {total_written:,} structures "
        f"(alias={total_alias:,}, fallback={total_fallback:,}, rhombo→hex={total_rhombo:,})"
    )


if __name__ == "__main__":
    main()
