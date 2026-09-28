"""Cell-blind cell selection: target-free quotas, candidate filters and diversity."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cellnet.lattice_conf_pipeline import (
    LatticeQRSRecord,
    UniqueCellRecord,
    _lattice_qrs_candidate_indices,
    run_conf_qrs_one,
    select_diverse_unique_cells,
    select_target_free_quota_unique_cells,
    unique_cell_is_qrs_origin,
    unique_cell_is_sell_only,
)
from cellnet.packing import cellpar_in_predicted_volume_range


HALL_TRICLINIC = 2


def _cubic_cell_for_volume(volume):
    edge = float(volume) ** (1.0 / 3.0)
    return np.array([edge, edge, edge, 90.0, 90.0, 90.0])


def test_selling_density_ratio_factor_includes_endpoints_and_rejects_beyond():
    target_volume = 1000.0
    factor = 1.25

    assert cellpar_in_predicted_volume_range(
        _cubic_cell_for_volume(target_volume / factor),
        target_volume,
        density_ratio_factor=factor,
    )
    assert cellpar_in_predicted_volume_range(
        _cubic_cell_for_volume(target_volume * factor),
        target_volume,
        density_ratio_factor=factor,
    )
    assert not cellpar_in_predicted_volume_range(
        _cubic_cell_for_volume(target_volume / factor - 1.0),
        target_volume,
        density_ratio_factor=factor,
    )
    assert not cellpar_in_predicted_volume_range(
        _cubic_cell_for_volume(target_volume * factor + 1.0),
        target_volume,
        density_ratio_factor=factor,
    )
    assert not cellpar_in_predicted_volume_range(
        _cubic_cell_for_volume(target_volume),
        target_volume,
        density_ratio_factor=float("nan"),
    )
    assert not cellpar_in_predicted_volume_range(
        _cubic_cell_for_volume(target_volume),
        target_volume,
        density_ratio_factor=4.01,
    )


def _quota_fixture():
    cells = []
    qrs = []
    selling = []
    for idx in range(48):
        cp = np.array(
            [7.0 + idx * 0.03, 8.0 + idx * 0.02, 9.0 + idx * 0.01, 90, 90, 90],
            dtype=float,
        )
        cells.append(UniqueCellRecord(idx, [idx], cp.copy(), sources=["qrs"]))
        qrs.append(
            LatticeQRSRecord(
                idx,
                cp.copy(),
                0.01 + idx / 1000.0,
                1.2 + idx / 1000.0,
                0.001 + idx / 1000.0,
                0.002 + idx / 1000.0,
                "qrs",
            )
        )
    for idx in range(24):
        cp = np.array(
            [
                7.2 + idx * 0.04,
                8.3 + idx * 0.025,
                9.4 + idx * 0.015,
                65.0 + idx * 0.7,
                82.0 + idx * 0.4,
                101.0 + idx * 0.3,
            ],
            dtype=float,
        )
        cells.append(
            UniqueCellRecord(100 + idx, [idx], cp.copy(), sources=["selling"])
        )
        selling.append(
            LatticeQRSRecord(
                idx, cp.copy(), float("inf"), 1.3, 0.0, 0.0, "selling"
            )
        )
    return cells, qrs, selling


def test_target_free_quota_has_exact_source_split_and_deterministic_provenance():
    assert "true_cellpar" not in inspect.signature(
        select_target_free_quota_unique_cells
    ).parameters
    assert "reference_cellpar" not in inspect.signature(
        select_target_free_quota_unique_cells
    ).parameters
    cells, qrs, selling = _quota_fixture()

    first, first_provenance = select_target_free_quota_unique_cells(
        cells, qrs, selling, 72, HALL_TRICLINIC, selling_quota=24
    )
    second, second_provenance = select_target_free_quota_unique_cells(
        cells, qrs, selling, 72, HALL_TRICLINIC, selling_quota=24
    )

    assert len(first) == 72
    assert sum(unique_cell_is_qrs_origin(cell) for cell in first) == 48
    assert sum(unique_cell_is_sell_only(cell) for cell in first) == 24
    assert [cell.dedup_idx for cell in first] == [
        cell.dedup_idx for cell in second
    ]
    assert first_provenance == second_provenance
    selling_channels = [
        first_provenance[cell.dedup_idx]
        for cell in first
        if unique_cell_is_sell_only(cell)
    ]
    assert selling_channels.count("selling_angle_disagreement") == 8
    assert selling_channels.count("selling_shape_diversity") == 8
    assert selling_channels.count("selling_parent_lambda_consistency") == 4
    assert selling_channels.count("selling_deterministic_fallback") == 4


def test_target_free_quota_backfills_missing_selling_cells_with_qrs():
    cells, qrs, selling = _quota_fixture()
    for idx in range(48, 50):
        cp = np.array(
            [7.0 + idx * 0.03, 8.0 + idx * 0.02, 9.0 + idx * 0.01, 90, 90, 90],
            dtype=float,
        )
        cells.append(UniqueCellRecord(idx, [idx], cp.copy(), sources=["qrs"]))
        qrs.append(
            LatticeQRSRecord(
                idx,
                cp.copy(),
                0.01 + idx / 1000.0,
                1.2 + idx / 1000.0,
                0.001 + idx / 1000.0,
                0.002 + idx / 1000.0,
                "qrs",
            )
        )
    kept_selling_ids = {100 + idx for idx in range(22)}
    cells = [
        cell
        for cell in cells
        if not unique_cell_is_sell_only(cell) or cell.dedup_idx in kept_selling_ids
    ]
    selling = selling[:22]

    selected, provenance = select_target_free_quota_unique_cells(
        cells, qrs, selling, 72, HALL_TRICLINIC, selling_quota=24
    )

    assert len(selected) == 72
    assert sum(unique_cell_is_qrs_origin(cell) for cell in selected) == 50
    assert sum(unique_cell_is_sell_only(cell) for cell in selected) == 22
    assert len(provenance) == 72


def test_reference_matching_can_be_disabled_without_losing_csd_template(monkeypatch, tmp_path):
    calls = {}
    wp = SimpleNamespace(letter="a")
    template = SimpleNamespace(
        numMols=[1],
        mol_sites=[SimpleNamespace(type=0, wp=wp)],
        lattice=object(),
        group=SimpleNamespace(hall_number=HALL_TRICLINIC),
        get_zprime=lambda: [1],
    )

    def fake_template(smiles, hall_number, zprime, cellpar, **kwargs):
        calls["template"] = (smiles, hall_number, zprime, kwargs)
        return template

    helpers = SimpleNamespace(
        build_sites_from_reference=lambda value: calls.setdefault("sites", value) or [],
        select_delta_angle=lambda molecules, composition: 10.0,
        select_soft_clash_buffer=lambda molecules, lattice, composition: (0.5, None),
    )

    class FakeQRS:
        def __init__(self, smiles, lattice, composition, molecules, sites, **kwargs):
            self.atom_info = {}

        def run(self, **kwargs):
            calls["run"] = kwargs
            return 0.0

    monkeypatch.setattr(
        "cellnet.lattice_conf_pipeline.build_pyxtal_template", fake_template
    )
    monkeypatch.setattr(
        "cellnet.lattice_conf_pipeline._import_qrs_conf_helpers", lambda: helpers
    )
    monkeypatch.setattr(
        "cellnet.lattice_conf_pipeline._get_component_pool",
        lambda *args, **kwargs: ([object()], "mock"),
    )
    monkeypatch.setattr("pyxtal.optimize.QRS", FakeQRS)

    run_conf_qrs_one(
        0,
        "CC",
        HALL_TRICLINIC,
        1.0,
        np.array([7, 8, 9, 90, 90, 90], dtype=float),
        Path(tmp_path) / "conf",
        csd_code="QAXMEH53",
        db_path=Path(tmp_path) / "template.db",
        ref_pmg=object(),
        enable_reference_matching=False,
        ngen=1,
        npop=1,
    )

    assert calls["template"][3]["csd_code"] == "QAXMEH53"
    assert calls["template"][3]["db_path"] == Path(tmp_path) / "template.db"
    assert calls["sites"] is template
    assert "ref_pmg" not in calls["run"]


def test_extreme_lattice_qrs_draws_are_skipped():
    lam = np.array(
        [
            [5.0, 10.0, 25.0],
            [3.98, 15.95, 36.67],
            [6.0, 8.0, 24.0],
        ]
    )
    selected, skipped = _lattice_qrs_candidate_indices(np.log(lam), 8.0)
    assert selected == [0, 2]
    assert skipped[0][0] == 1
    assert skipped[0][1] > 9.0

    selected, skipped = _lattice_qrs_candidate_indices(np.log(lam), 0.0)
    assert selected == [0, 1, 2]
    assert skipped == []


def _record(idx: int, cellpar: list[float], loss: float, support: int = 1):
    qrs = [
        LatticeQRSRecord(
            flow_idx=idx + j,
            cellpar=np.asarray(cellpar, dtype=np.float64),
            qrs_loss=loss + 0.01 * j,
            density=1.2,
            lambda_mse=0.0,
            lambda_recip_mse=0.0,
        )
        for j in range(support)
    ]
    unique = UniqueCellRecord(
        dedup_idx=idx,
        source_flow_indices=[record.flow_idx for record in qrs],
        cellpar=np.asarray(cellpar, dtype=np.float64),
    )
    return unique, qrs


def test_diverse_selection_keeps_quality_and_rare_shape():
    specs = [
        ([8.0, 9.0, 10.0, 90.0, 90.0, 90.0], 0.10),
        ([8.1, 9.1, 10.1, 90.0, 90.0, 90.0], 0.11),
        ([7.9, 8.9, 10.2, 90.0, 90.0, 90.0], 0.12),
        ([5.8, 7.4, 24.7, 90.0, 92.0, 90.0], 0.20),
    ]
    unique_cells = []
    records = []
    for idx, (cellpar, loss) in enumerate(specs):
        unique, recs = _record(idx, cellpar, loss)
        unique_cells.append(unique)
        records.extend(recs)

    selected = select_diverse_unique_cells(unique_cells, records, max_cells=2)
    selected_ids = {cell.dedup_idx for cell in selected}
    assert 0 in selected_ids  # best-QRS seed
    assert 3 in selected_ids  # distinct elongated shape
