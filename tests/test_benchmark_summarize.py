"""benchmark/summarize.py on a small synthetic results tree."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmark" / "summarize.py"


@pytest.fixture(scope="module")
def summarize():
    spec = importlib.util.spec_from_file_location("summarize", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(root: Path, code: str, primary_sr: list, fallback_sr: list = ()):
    recs = [{"workdir": f"{root}/{code}/conf_qrs_{i:03d}", "relax_lattice": True, "success_rate": sr}
            for i, sr in enumerate(primary_sr)]
    recs += [{"workdir": f"{root}/{code}/adaptive_fixed_lattice/conf_qrs_{i:03d}", "relax_lattice": False,
              "success_rate": sr} for i, sr in enumerate(fallback_sr)]
    summary = {"hall_number": 81, "zprime": 1.0, "smiles": "CCO", "conf_qrs": recs}
    (root / code).mkdir(parents=True)
    (root / code / f"{code}_pipeline_summary.json").write_text(json.dumps(summary))


def test_outcomes_and_counts(summarize, tmp_path):
    _write_summary(tmp_path, "HIT", [0.0, 12.5, None, 3.0])
    _write_summary(tmp_path, "FROZEN", [0.0, 0.0], [0.0, 5.0])
    _write_summary(tmp_path, "MISS", [0.0, 0.0], [0.0, 0.0])

    hit = summarize.summarize_code(tmp_path, "HIT")
    assert hit["outcome"] == "covered"
    assert hit["n_cells"] == 4 and hit["n_hit_cells"] == 2
    assert hit["first_hit_cell"] == 2
    assert hit["max_sr_percent"] == "12.5"

    frozen = summarize.summarize_code(tmp_path, "FROZEN")
    assert frozen["outcome"] == "covered_frozen_lattice"
    assert frozen["n_fallback_cells"] == 2 and frozen["n_fallback_hit_cells"] == 1

    assert summarize.summarize_code(tmp_path, "MISS")["outcome"] == "miss"
    assert summarize.summarize_code(tmp_path, "ABSENT") is None


def test_shipped_results_table_is_self_consistent():
    rows = list(csv.DictReader((ROOT / "benchmark" / "results.csv").open()))
    codes = [l.strip() for l in (ROOT / "benchmark" / "codes.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert sorted(r["code"] for r in rows) == sorted(codes)
    outcomes = [r["outcome"] for r in rows]
    assert outcomes.count("covered") == 80
    assert outcomes.count("covered_frozen_lattice") == 2
    assert outcomes.count("miss") == 2
    for r in rows:
        covered = int(r["n_hit_cells"]) > 0
        assert (r["outcome"] == "covered") == covered, r["code"]
