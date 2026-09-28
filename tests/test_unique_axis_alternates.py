"""Monoclinic unique-axis alternates (the MERRAF failure case).

The lattice-QRS seed puts the monoclinic unique axis b on λ₁ and the (λ, λ*, ρ)
objective cannot distinguish which edge carries the 2₁ axis, so cells with b on
λ₃ (30 % of CSD monoclinic structures) were almost never swept. These tests
cover the target-free relabeling and its selection channel.
"""

from types import SimpleNamespace

import numpy as np

from cellnet.lattice_conf_pipeline import (
    AXIS_ALTERNATE_SOURCE,
    LatticeQRSRecord,
    deduplicate_cellpars,
    lattice_records_from_unique_axis_alternates,
    monoclinic_unique_axis_alternates,
    monoclinic_unique_axis_rank,
    posterior_effective_mode_count,
    select_multichannel_unique_cells,
    unique_cell_is_axis_alternate,
    unique_cell_is_qrs_origin,
)
from cellnet.lattice_invariants import successive_minima

HALL_P21 = 6  # MERRAF is run as P2₁ (unique axis b), Z′ = 1
HALL_PBCA = 290
HALL_P1BAR = 2

# Lattice-QRS cell for MERRAF flow draw 42 of the benchmark run and the true cell
# in the P2₁ setting; only the unique-axis assignment differs.
QRS_CELL_42 = np.array([20.996, 3.868, 7.494, 90.0, 89.65, 90.0])
TRUE_MERRAF = np.array([4.032, 21.157, 7.312, 90.0, 84.2, 90.0])


def test_unique_axis_rank_of_edges():
    assert monoclinic_unique_axis_rank(QRS_CELL_42) == 0
    assert monoclinic_unique_axis_rank(TRUE_MERRAF) == 2
    assert monoclinic_unique_axis_rank([5.0, 7.0, 9.0, 90.0, 95.0, 90.0]) == 1


def test_alternates_move_b_onto_each_other_edge_and_keep_beta():
    alternates = monoclinic_unique_axis_alternates(QRS_CELL_42, HALL_P21)
    ranks = [rank for rank, _ in alternates]
    assert sorted(ranks) == [1, 2]
    # Most common CSD setting first: b on the longest edge (30 %) beats middle (24 %).
    assert ranks[0] == 2
    for rank, cell in alternates:
        assert monoclinic_unique_axis_rank(cell) == rank
        assert np.allclose(sorted(cell[:3]), sorted(QRS_CELL_42[:3]))
        assert cell[0] <= cell[2]
        assert np.allclose(cell[[3, 5]], 90.0)
        assert np.isclose(cell[4], QRS_CELL_42[4])
        # Same successive minima and volume: the lattice QRS cannot rank them.
        assert np.allclose(successive_minima(cell), successive_minima(QRS_CELL_42), rtol=0.01)
    b_longest = dict(alternates)[2]
    assert np.allclose(b_longest[:3], TRUE_MERRAF[:3], rtol=0.05)


def test_alternates_skip_degenerate_edges_and_other_systems():
    assert monoclinic_unique_axis_alternates([7.0, 7.0, 12.0, 90.0, 90.0, 90.0], HALL_PBCA) == []
    assert monoclinic_unique_axis_alternates([7.0, 8.0, 12.0, 80.0, 85.0, 95.0], HALL_P1BAR) == []
    # a == b: moving b onto a reproduces the same cell, so only one alternate remains.
    alternates = monoclinic_unique_axis_alternates([7.0, 7.0, 12.0, 90.0, 95.0, 90.0], HALL_P21)
    assert len(alternates) == 1
    assert monoclinic_unique_axis_rank(alternates[0][1]) == 2


def _flow(hall=HALL_P21, k=4):
    lam = np.log([[3.87, 7.49, 21.0]] * k)
    return SimpleNamespace(
        hall_number=hall,
        all_log_lambda=lam,
        all_log_lambda_recip=np.log(2.0 * np.pi / np.exp(lam)[:, ::-1]),
        k=k,
    )


def _record(idx, cellpar, lam_mse, source="qrs"):
    return LatticeQRSRecord(
        flow_idx=idx,
        cellpar=np.asarray(cellpar, dtype=np.float64),
        qrs_loss=0.5 + idx,
        density=1.85,
        lambda_mse=lam_mse,
        lambda_recip_mse=0.0,
        source=source,
    )


def test_alternate_records_follow_top_lambda_parents_only():
    records = [
        _record(0, [20.996, 3.868, 7.494, 90.0, 89.65, 90.0], 1e-5),
        _record(1, [12.0, 5.0, 10.0, 90.0, 95.0, 90.0], 1e-3),
        _record(2, [9.0, 6.0, 11.0, 90.0, 100.0, 90.0], 1e-1),
        _record(3, [3.9, 7.5, 21.0, 90.0, 90.0, 90.0], np.inf, source="selling"),
    ]
    alts = lattice_records_from_unique_axis_alternates(_flow(), records, n_top=2)
    assert len(alts) == 4
    assert {rec.flow_idx for rec in alts} == {0, 1}
    assert all(rec.source == AXIS_ALTERNATE_SOURCE for rec in alts)
    assert all(np.isfinite(rec.lambda_mse) for rec in alts)
    assert lattice_records_from_unique_axis_alternates(_flow(hall=HALL_PBCA), records) == []
    # Alternates are never re-expanded.
    again = lattice_records_from_unique_axis_alternates(_flow(), records + alts, n_top=2)
    assert len(again) == 4


def test_dedup_keeps_alternates_distinct_but_merges_into_existing_settings():
    parent = _record(0, QRS_CELL_42, 1e-5)
    already_b_longest = _record(1, [3.87, 21.0, 7.49, 90.0, 89.6, 90.0], 2e-5)
    alts = lattice_records_from_unique_axis_alternates(_flow(), [parent, already_b_longest], n_top=1)
    unique = deduplicate_cellpars([parent, already_b_longest] + alts, HALL_P21)
    # parent, b-longest (merged with the alternate), b-middle (new)
    assert len(unique) == 3
    pure = [u for u in unique if unique_cell_is_axis_alternate(u)]
    assert len(pure) == 1
    assert monoclinic_unique_axis_rank(pure[0].cellpar) == 1
    merged = [u for u in unique if set(u.sources) == {"qrs", AXIS_ALTERNATE_SOURCE}]
    assert len(merged) == 1
    assert all(unique_cell_is_qrs_origin(u) for u in unique)
    # Derived cells do not widen the posterior used by --conf-budget auto.
    assert posterior_effective_mode_count(unique, [parent, already_b_longest] + alts) == (
        posterior_effective_mode_count(unique[:2], [parent, already_b_longest])
    )


def _pool():
    rng = np.random.default_rng(0)
    records = [_record(0, QRS_CELL_42, 1e-5)]
    for idx in range(1, 12):
        a, b, c = sorted(rng.uniform(4.0, 22.0, size=3))
        # b on the shortest edge, like the lattice-QRS seed
        records.append(_record(idx, [b, a, c, 90.0, rng.uniform(80.0, 100.0), 90.0], 1e-4 * idx))
    alts = lattice_records_from_unique_axis_alternates(_flow(k=12), records, n_top=3)
    unique = deduplicate_cellpars(records + alts, HALL_P21)
    return records, alts, unique


def test_multichannel_axis_alternate_channel_and_provenance():
    records, alts, unique = _pool()
    selected, provenance = select_multichannel_unique_cells(
        unique,
        records + alts,
        [],
        10,
        HALL_P21,
        lambda_quota=2,
        qrs_loss_quota=1,
        density_quota=0,
        disagreement_quota=0,
        qrs_origin_only=True,
        axis_alternate_quota=3,
    )
    channels = [provenance[u.dedup_idx] for u in selected]
    assert channels.count("unique_axis_alternate") == 3
    assert len(selected) == 10
    alt_cells = [u for u in selected if provenance[u.dedup_idx] == "unique_axis_alternate"]
    assert all(unique_cell_is_axis_alternate(u) for u in alt_cells)
    # Ranked by parent λ-consistency (draw 0 first) then CSD prior (b longest first).
    assert alt_cells[0].source_flow_indices == [0]
    assert monoclinic_unique_axis_rank(alt_cells[0].cellpar) == 2
    assert np.allclose(alt_cells[0].cellpar[:3], TRUE_MERRAF[:3], rtol=0.05)
    # Alternates never take fixed λ / QRS-loss slots.
    for u in selected:
        if provenance[u.dedup_idx] in ("lambda_consistency", "qrs_loss"):
            assert not unique_cell_is_axis_alternate(u)


def test_alternates_do_not_take_disagreement_slots():
    records, alts, unique = _pool()
    # Every draw has a Selling twin with a different β, so every draw "disagrees".
    selling = [
        _record(rec.flow_idx, np.r_[rec.cellpar[:4], rec.cellpar[4] + 10.0, 90.0], np.inf, source="selling")
        for rec in records
    ]
    unique = deduplicate_cellpars(records + alts + selling, HALL_P21)
    _, provenance = select_multichannel_unique_cells(
        unique, records + alts, selling, 12, HALL_P21, lambda_quota=0, qrs_loss_quota=0,
        density_quota=0, disagreement_quota=6, qrs_origin_only=True, axis_alternate_quota=3,
    )
    by_uid = {u.dedup_idx: u for u in unique}
    for uid, channel in provenance.items():
        if channel == "qrs_selling_disagreement":
            assert not unique_cell_is_axis_alternate(by_uid[uid])
    assert list(provenance.values()).count("unique_axis_alternate") == 3


def test_multichannel_zero_quota_matches_previous_behaviour():
    records, alts, unique = _pool()
    baseline_unique = deduplicate_cellpars(records, HALL_P21)
    base, base_prov = select_multichannel_unique_cells(
        baseline_unique, records, [], 8, HALL_P21, lambda_quota=2, qrs_loss_quota=1,
        density_quota=0, disagreement_quota=0, qrs_origin_only=True,
    )
    with_alts, prov = select_multichannel_unique_cells(
        unique, records + alts, [], 8, HALL_P21, lambda_quota=2, qrs_loss_quota=1,
        density_quota=0, disagreement_quota=0, qrs_origin_only=True, axis_alternate_quota=0,
    )
    assert "unique_axis_alternate" not in prov.values()
    assert [tuple(np.round(u.cellpar, 3)) for u in with_alts] == [
        tuple(np.round(u.cellpar, 3)) for u in base
    ]
    assert [prov[u.dedup_idx] for u in with_alts] == [base_prov[u.dedup_idx] for u in base]


def test_axis_alternate_channel_is_monoclinic_only():
    records = [_record(i, [5.0 + i, 7.0, 9.0, 90.0, 90.0, 90.0], 1e-4 * (i + 1)) for i in range(4)]
    unique = deduplicate_cellpars(records, HALL_PBCA)
    _, provenance = select_multichannel_unique_cells(
        unique, records, [], 4, HALL_PBCA, lambda_quota=1, qrs_loss_quota=1,
        density_quota=0, disagreement_quota=0, axis_alternate_quota=4,
    )
    assert "unique_axis_alternate" not in provenance.values()
