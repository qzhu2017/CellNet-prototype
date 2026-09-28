"""Tests for SPaDe CSV extraction and Hall lookup."""

import csv
from pathlib import Path

import numpy as np

from cellnet.precomputed import (
    PRECOMPUTED_COLUMNS,
    is_precomputed_csv,
    load_precomputed_csv,
    precompute_sample,
    precomputed_graphs_path,
    write_precomputed_csv,
)
from cellnet.spade import (
    SPADE_EXPORT_COLUMNS,
    extract_spade_csv,
    hall_number_from_sg_symbol,
    hall_number_from_spg,
    load_spade_csv,
)


def test_hall_from_spg():
    assert hall_number_from_spg(2) == 2
    assert hall_number_from_spg(19) == 115
    assert hall_number_from_spg(14) == 81


def test_hall_from_sg_symbol():
    assert hall_number_from_sg_symbol(14, "P21/c") == (81, "P21/c", "P 1 21/c 1")
    assert hall_number_from_sg_symbol(14, "P21/n") == (82, "P21/n", "P 1 21/n 1")
    assert hall_number_from_sg_symbol(14, "P21/a") == (83, "P21/a", "P 1 21/a 1")
    assert hall_number_from_sg_symbol(15, "I2/a") == (92, "I2/a", "I 1 2/a 1")
    assert hall_number_from_sg_symbol(14, "P1121/b") == (81, "P21/c", "P 1 21/c 1")
    assert hall_number_from_sg_symbol(152, "P3121") == (441, "P312", "P 31 2 1")


def test_extract_and_load(tmp_path: Path):
    src = tmp_path / "mini.csv"
    dst = tmp_path / "out.csv"
    src.write_text(
        "refcode,SMILES,MW,sg_symbol,sg_number,Z-prime,Z-value,a,b,c,alpha,beta,gamma,density\n"
        "ABC123,CCO,46.0,P-1,2,1.0,2.0,5.0,6.0,7.0,90.0,90.0,90.0,1.2\n"
    )
    stats = extract_spade_csv(src, dst)
    assert stats.written == 1
    with dst.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["hall_number"] == "2"
    assert set(row.keys()) == set(SPADE_EXPORT_COLUMNS)

    samples = load_spade_csv(dst, verbose=False)
    assert len(samples) == 1
    assert samples[0].hall_number == 2
    assert samples[0].spg_num == 2
    assert samples[0].zprime == 1.0
    assert samples[0].cellpar[0] == 5.0


def test_extract_skips_unmapped_hall(tmp_path: Path):
    src = tmp_path / "mini.csv"
    dst = tmp_path / "out.csv"
    src.write_text(
        "refcode,SMILES,MW,sg_symbol,sg_number,Z-prime,Z-value,a,b,c,alpha,beta,gamma,density\n"
        "GOOD01,CCO,46.0,P21/c,14,1.0,4.0,5.0,6.0,7.0,90.0,100.0,90.0,1.2\n"
        "BAD001,CCO,46.0,F-1,2,1.0,2.0,5.0,6.0,7.0,90.0,100.0,90.0,1.2\n"
    )
    stats = extract_spade_csv(src, dst)
    assert stats.written == 1
    assert stats.skipped_unmapped_hall == 1
    with dst.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["hall_number"] == "81"


def test_precomputed_round_trip(tmp_path: Path):
    src = tmp_path / "mini.csv"
    dst = tmp_path / "out.csv"
    src.write_text(
        "refcode,SMILES,MW,sg_symbol,sg_number,Z-prime,Z-value,a,b,c,alpha,beta,gamma,density\n"
        "ABC123,CCO,46.0,P-1,2,1.0,2.0,5.0,6.0,7.0,90.0,90.0,90.0,1.2\n"
    )
    extract_spade_csv(src, dst)
    raw = load_spade_csv(dst, verbose=False)
    precomputed = precompute_sample(raw[0])
    assert precomputed is not None

    out_csv = tmp_path / "precomputed.csv"
    write_precomputed_csv([precomputed], out_csv)
    graphs_path = precomputed_graphs_path(out_csv)
    assert graphs_path.is_file()
    assert is_precomputed_csv(out_csv)

    loaded = load_precomputed_csv(out_csv, verbose=False)
    assert len(loaded) == 1
    assert loaded[0].graph is not None
    assert loaded[0].selling_log1p.shape == (6,)
    assert loaded[0].log_successive_minima.shape == (3,)
    np.testing.assert_allclose(loaded[0].cellpar, precomputed.cellpar)

    with out_csv.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert set(row.keys()) == set(PRECOMPUTED_COLUMNS)
