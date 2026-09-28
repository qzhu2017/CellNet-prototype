"""Argument handling and metadata lookup in scripts/run_pipeline.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pyxtal
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pipeline.py"
BUNDLED_TEST_DB = Path(pyxtal.__file__).resolve().parent / "database" / "test.db"


@pytest.fixture(scope="module")
def run_pipeline():
    spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_test_db_falls_back_to_the_copy_shipped_with_pyxtal(run_pipeline, tmp_path):
    assert run_pipeline._default_test_db(tmp_path) == BUNDLED_TEST_DB
    local = tmp_path / "datasets" / "test.db"
    local.parent.mkdir()
    local.touch()
    assert run_pipeline._default_test_db(tmp_path) == local


@pytest.mark.skipif(not BUNDLED_TEST_DB.is_file(), reason="PyXtal test.db not installed")
def test_benchmark_code_metadata_comes_from_test_db(run_pipeline):
    smiles, hall, zprime, cellpar, db_path, from_test_db = run_pipeline._resolve_crystal_metadata(
        "UREAXX02",
        csv_path=str(ROOT / "datasets/spade-csp/spade_test.csv"),
        db_path=None,
        smiles_override=None,
        hall_override=None,
        zprime_override=None,
    )
    assert smiles == "C(=O)(N)N"
    assert hall == 6
    assert zprime == 1.0
    assert cellpar is not None and len(cellpar) == 6
    assert from_test_db  # enables reference matching automatically


def test_smiles_run_needs_hall_and_zprime_and_has_no_default_code():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--smiles", "NC(N)=O", "--skip-conf-qrs"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert result.returncode == 2
    assert "Provide --csd-code or (--smiles --hall --zprime)" in result.stderr
