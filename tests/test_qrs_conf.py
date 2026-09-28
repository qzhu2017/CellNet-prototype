"""Helpers for the conformational (packing) QRS in cellnet/qrs_conf.py."""

from pathlib import Path

import pyxtal
import pytest

from cellnet import qrs_conf

BUNDLED_TEST_DB = Path(pyxtal.__file__).resolve().parent / "database" / "test.db"


def test_clamp_delta_and_soft_clash_argument():
    assert qrs_conf._clamp_delta(5.0) == 10.0
    assert qrs_conf._clamp_delta(45.0) == 45.0
    assert qrs_conf._clamp_delta(120.0) == 90.0
    assert qrs_conf.resolve_soft_clash_buffer_arg("auto") is None
    assert qrs_conf.resolve_soft_clash_buffer_arg("0.3") == pytest.approx(0.3)


@pytest.mark.skipif(not BUNDLED_TEST_DB.is_file(), reason="PyXtal test.db not installed")
def test_sites_from_reference_crystal():
    from pyxtal.db import database

    xtal = database(str(BUNDLED_TEST_DB)).get_pyxtal("UREAXX02")
    sites = qrs_conf.build_sites_from_reference(xtal)
    assert len(sites) == len(xtal.numMols)  # one list of Wyckoff labels per molecule type
    assert all(isinstance(label, str) for group in sites for label in group)
