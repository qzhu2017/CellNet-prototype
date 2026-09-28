#!/usr/bin/env python3
"""Tests for SPaDe V9 extraction (full 170k corpus, rhombo→hex)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cellnet.spade import (
    convert_cellpar_to_hexagonal_setting,
    extract_v9_csv,
    hall_number_from_sg_symbol,
    hall_number_to_hex_setting,
    is_rhombohedral_cellpar,
)


def test_r3c_maps_to_hex_hall_452():
    assert hall_number_to_hex_setting(453) == 452
    assert hall_number_from_sg_symbol(161, "R3c") == (452, "R3c", "R 3 c:H")


def test_rhombohedral_cell_detected():
    cp = np.array([9.38, 9.38, 9.38, 108.7, 108.7, 108.7])
    assert is_rhombohedral_cellpar(cp)
    assert not is_rhombohedral_cellpar(np.array([11.5, 11.5, 13.0, 90.0, 90.0, 120.0]))


def test_convert_rhombo_to_conventional_hex_cell():
    cp = np.array([9.38, 9.38, 9.38, 108.7, 108.7, 108.7])
    from pyxtal.lattice import Lattice

    vol_in = Lattice.from_para(*cp, ltype="trigonal").volume
    out, hall = convert_cellpar_to_hexagonal_setting(cp, 452)
    assert hall == 452
    assert abs(out[3] - 90.0) < 0.01
    assert abs(out[4] - 90.0) < 0.01
    assert abs(out[5] - 120.0) < 0.01
    assert abs(out[0] - out[1]) < 0.01
    vol_out = Lattice.from_para(*out, ltype="trigonal").volume
    assert abs(vol_out / vol_in - 3.0) < 0.02


def test_p3121_alias():
    assert hall_number_from_sg_symbol(152, "P3121") == (441, "P312", "P 31 2 1")


def test_extract_v9_full_row(tmp_path: Path):
    src = tmp_path / "mini.csv"
    dst = tmp_path / "out.csv"
    src.write_text(
        "refcode,SMILES,MW,sg_symbol,sg_number,Z-prime,Z-value,a,b,c,alpha,beta,gamma,density\n"
        "CAFVUA,CC(N)=O,59.0,R3c,161,1.0,6.0,9.38,9.38,9.38,108.7,108.7,108.7,1.696\n"
        "GOOD01,CCO,46.0,P21/c,14,1.0,4.0,5.0,6.0,7.0,90.0,100.0,90.0,1.2\n"
    )
    stats = extract_v9_csv(src, dst)
    assert stats.written == 2
    assert stats.skipped_unmapped_hall == 0
    rows = list(csv.DictReader(dst.open()))
    by_ref = {row["refcode"]: row for row in rows}
    assert by_ref["GOOD01"]["hall_number"] == "81"
    assert by_ref["CAFVUA"]["hall_number"] == "452"
    assert float(by_ref["CAFVUA"]["gamma"]) == 120.0
