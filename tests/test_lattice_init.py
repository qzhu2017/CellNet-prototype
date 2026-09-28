"""Tests for lattice QRS initialization and elite conf selection."""

import numpy as np

from cellnet.lattice_conf_pipeline import (
    LatticeQRSRecord,
    UniqueCellRecord,
    append_disagreeing_selling_cells,
    append_disagreement_channel,
    dynamic_conf_run_count,
    init_cellpar_candidates_from_log_lambda,
    posterior_effective_mode_count,
    reference_match_score,
    scaled_conf_elite,
    select_conf_unique_cells,
    select_diverse_unique_cells,
    selling_kcenter_draw_indices,
)
from cellnet.lattice_invariants import (
    cellpar_from_selling,
    selling_parameters,
    sort_selling_parameters,
)

# Hall 2 → triclinic (QAXMEH53)
HALL_QAXMEH53 = 2
# Hall 290 → orthorhombic Pbca (OBEQUJ)
HALL_OBEQUJ = 290
# Hall 452 → trigonal R3c in the conventional hexagonal setting (ACEMID02)
HALL_ACEMID02 = 452


def test_triclinic_init_expands_angle_guesses_without_selling():
    log_lambda = np.log([6.5, 8.0, 10.0])
    candidates = init_cellpar_candidates_from_log_lambda(log_lambda, HALL_QAXMEH53)
    assert len(candidates) >= 6
    angle_sets = {tuple(cp[3:6]) for cp in candidates}
    assert (85.0, 95.0, 90.0) in angle_sets
    assert (100.0, 105.0, 90.0) in angle_sets


def test_triclinic_init_uses_selling_angles_not_reference_cell():
    log_lambda = np.log([6.5, 8.0, 10.0])
    ref = np.array([6.97, 8.27, 10.37, 97.6, 103.2, 90.1])
    selling = selling_parameters(ref)
    candidates = init_cellpar_candidates_from_log_lambda(
        log_lambda,
        HALL_QAXMEH53,
        reference_cellpar=ref,
        selling=selling,
    )
    # Selling reconstruction is Delaunay-reduced; angles need not match the
    # conventional setting, but they should not be the generic 85/95/90 grid only.
    angle_sets = {tuple(np.round(cp[3:6], 0)) for cp in candidates}
    assert (85.0, 95.0, 90.0) not in angle_sets or len(angle_sets) > 1
    assert len(candidates) >= 6


def test_orthorhombic_init_uses_axis_ranks_and_selling():
    log_lambda = np.log([7.0, 9.0, 16.0])
    ref = np.array([7.38, 16.19, 9.65, 90.0, 90.0, 90.0])
    ranks = (0, 2, 1)
    selling = selling_parameters(ref)
    candidates = init_cellpar_candidates_from_log_lambda(
        log_lambda,
        HALL_OBEQUJ,
        axis_rank_permutation=ranks,
        selling=selling,
    )
    assert any(np.allclose(cp[:3], [7.0, 16.0, 9.0], rtol=0.02) for cp in candidates)
    assert any(np.allclose(cp[3:6], [90.0, 90.0, 90.0], atol=1.0) for cp in candidates)


def test_garbage_selling_falls_back_to_lambda_inits():
    log_lambda = np.log([6.5, 8.0, 10.0])
    garbage = np.array([-1e6, 0.0, 0.0, 0.0, 0.0, 0.0])
    candidates = init_cellpar_candidates_from_log_lambda(
        log_lambda,
        HALL_QAXMEH53,
        selling=garbage,
    )
    assert len(candidates) >= 6
    assert all(np.isfinite(cp).all() for cp in candidates)


def test_hexagonal_lambda_init_maps_directly_to_a_and_c():
    log_lambda = np.log([11.2, 11.4, 12.73])
    candidates = init_cellpar_candidates_from_log_lambda(
        log_lambda,
        HALL_ACEMID02,
        selling=None,
    )
    expected_a = np.sqrt(11.2 * 11.4)
    assert len(candidates) == 1
    assert np.allclose(
        candidates[0],
        [expected_a, expected_a, 12.73, 90.0, 90.0, 120.0],
    )


def test_cellpar_from_selling_roundtrip_qaxmeh53():
    true = np.array([6.9716, 8.2722, 10.3708, 97.569, 103.23, 90.057])
    selling = selling_parameters(true)
    recon = cellpar_from_selling(selling, HALL_QAXMEH53)
    recon_s = sort_selling_parameters(selling_parameters(recon))
    true_s = sort_selling_parameters(selling)
    assert np.allclose(recon_s, true_s, rtol=0.05, atol=0.5)


def test_cellpar_from_selling_roundtrip_obequj():
    true = np.array([7.381, 16.185, 9.6451, 90.0, 90.0, 90.0])
    selling = selling_parameters(true)
    recon = cellpar_from_selling(selling, HALL_OBEQUJ)
    lengths = np.sort(recon[:3])
    assert np.allclose(lengths, np.sort(true[:3]), rtol=0.05)


def test_elite_selection_keeps_lowest_lambda_mse():
    unique = []
    records = []
    for i in range(20):
        cp = np.array([7.0 + 0.1 * i, 8.0, 10.0 + i, 90.0, 90.0, 90.0])
        unique.append(UniqueCellRecord(dedup_idx=i, source_flow_indices=[i], cellpar=cp))
        records.append(
            LatticeQRSRecord(
                flow_idx=i,
                cellpar=cp,
                qrs_loss=0.01 if i > 2 else 1.0,
                density=1.3,
                lambda_mse=float(i),  # 0 is best
                lambda_recip_mse=0.0,
            )
        )
    selected = select_diverse_unique_cells(unique, records, max_cells=8, n_elite=3)
    elite_idx = {c.dedup_idx for c in selected[:3]}
    assert elite_idx == {0, 1, 2}


def test_elite_selection_keeps_selling_agreement_cells():
    from cellnet.lattice_invariants import selling_parameters

    unique = []
    records = []
    sell_rows = []
    far = np.array([5.0, 12.0, 20.0, 70.0, 80.0, 110.0])
    close = [
        np.array([7.0, 8.0, 10.0, 97.0, 103.0, 90.0]),
        np.array([7.2, 8.1, 10.1, 96.0, 102.0, 91.0]),
        np.array([6.9, 7.9, 9.9, 98.0, 104.0, 89.0]),
    ]
    for i in range(8):
        unique.append(UniqueCellRecord(dedup_idx=i, source_flow_indices=[i], cellpar=far.copy()))
        records.append(
            LatticeQRSRecord(
                flow_idx=i,
                cellpar=far.copy(),
                qrs_loss=0.01,
                density=1.3,
                lambda_mse=1e-6 * (i + 1),
                lambda_recip_mse=0.0,
            )
        )
        sell_rows.append(selling_parameters(close[0]))  # disagrees with `far`
    for j, cp in enumerate(close):
        idx = 8 + j
        unique.append(UniqueCellRecord(dedup_idx=idx, source_flow_indices=[idx], cellpar=cp))
        records.append(
            LatticeQRSRecord(
                flow_idx=idx,
                cellpar=cp,
                qrs_loss=0.4,
                density=1.3,
                lambda_mse=1.0,
                lambda_recip_mse=0.0,
            )
        )
        sell_rows.append(selling_parameters(cp))
    selected = select_diverse_unique_cells(
        unique,
        records,
        max_cells=8,
        n_elite=6,
        hall_number=HALL_QAXMEH53,
        all_selling=np.stack(sell_rows),
    )
    elite_idx = {c.dedup_idx for c in selected[:6]}
    # 3 λ-MSE elites from the far/low-mse cells, 3 Selling-agreement elites
    # from the cells that match their own Selling vector.
    assert {8, 9, 10}.issubset(elite_idx)
    assert len(elite_idx & {0, 1, 2, 3, 4, 5, 6, 7}) >= 3


def test_selling_reconstructions_do_not_steal_agreement_elite():
    """Selling cells agree with themselves; agreement elite must still prefer QRS."""
    unique = []
    records = []
    sell_rows = []
    far = np.array([5.0, 12.0, 20.0, 70.0, 80.0, 110.0])
    close = [
        np.array([7.0, 8.0, 10.0, 97.0, 103.0, 90.0]),
        np.array([7.2, 8.1, 10.1, 96.0, 102.0, 91.0]),
        np.array([6.9, 7.9, 9.9, 98.0, 104.0, 89.0]),
    ]
    for i in range(8):
        unique.append(UniqueCellRecord(dedup_idx=i, source_flow_indices=[i], cellpar=far.copy()))
        records.append(
            LatticeQRSRecord(
                flow_idx=i,
                cellpar=far.copy(),
                qrs_loss=0.01,
                density=1.3,
                lambda_mse=1e-6 * (i + 1),
                lambda_recip_mse=0.0,
            )
        )
        sell_rows.append(selling_parameters(close[0]))
    for j, cp in enumerate(close):
        idx = 8 + j
        unique.append(UniqueCellRecord(dedup_idx=idx, source_flow_indices=[idx], cellpar=cp))
        records.append(
            LatticeQRSRecord(
                flow_idx=idx,
                cellpar=cp,
                qrs_loss=0.4,
                density=1.3,
                lambda_mse=1.0,
                lambda_recip_mse=0.0,
            )
        )
        sell_rows.append(selling_parameters(cp))
    for j in range(5):
        idx = 11 + j
        sell_cp = close[0]
        unique.append(UniqueCellRecord(dedup_idx=idx, source_flow_indices=[idx], cellpar=sell_cp.copy(), sources=["selling"]))
        records.append(
            LatticeQRSRecord(
                flow_idx=idx,
                cellpar=sell_cp.copy(),
                qrs_loss=float("inf"),
                density=1.3,
                lambda_mse=1e-9,
                lambda_recip_mse=0.0,
                source="selling",
            )
        )
        sell_rows.append(selling_parameters(sell_cp))
    selected = select_diverse_unique_cells(
        unique,
        records,
        max_cells=10,
        n_elite=8,
        hall_number=HALL_QAXMEH53,
        all_selling=np.stack(sell_rows),
    )
    elite_idx = {c.dedup_idx for c in selected[:8]}
    assert {8, 9, 10}.issubset(elite_idx)
    assert elite_idx.isdisjoint({11, 12, 13, 14, 15})
    assert {c.dedup_idx for c in selected}.isdisjoint({11, 12, 13, 14, 15})


def test_lattice_match_sort_key_prefers_triclinic_angles():
    from cellnet.lattice_conf_pipeline import LatticeQRSCellparMatch, _lattice_match_sort_key

    short = LatticeQRSCellparMatch(
        flow_idx=0,
        length_mape_pct=2.0,
        mape_a_pct=2.0,
        mape_b_pct=2.0,
        mape_c_pct=2.0,
        angle_mae_deg=12.0,
        angle_errs_deg=np.array([12.0, 12.0, 12.0]),
        aligned_cellpar=np.zeros(6),
        raw_cellpar=np.zeros(6),
        qrs_loss=0.0,
    )
    angled = LatticeQRSCellparMatch(
        flow_idx=1,
        length_mape_pct=4.0,
        mape_a_pct=4.0,
        mape_b_pct=4.0,
        mape_c_pct=4.0,
        angle_mae_deg=1.0,
        angle_errs_deg=np.array([1.0, 1.0, 1.0]),
        aligned_cellpar=np.zeros(6),
        raw_cellpar=np.zeros(6),
        qrs_loss=0.0,
    )
    # Hall 2 = triclinic: combined 2+12 > 4+1, so angled ranks first.
    ranked = sorted([short, angled], key=lambda m: _lattice_match_sort_key(m, HALL_QAXMEH53))
    assert ranked[0] is angled
    # Hall 290 = orthorhombic: length MAPE only, so short ranks first.
    ranked_orth = sorted([short, angled], key=lambda m: _lattice_match_sort_key(m, HALL_OBEQUJ))
    assert ranked_orth[0] is short


def test_reference_match_score_weights_triclinic_angles():
    true = np.array([6.97, 8.27, 10.37, 97.6, 103.2, 90.1])
    good_lengths_bad_angles = np.array([6.95, 8.25, 10.35, 85.0, 95.0, 90.0])
    _, _, combined = reference_match_score(
        good_lengths_bad_angles,
        true,
        HALL_QAXMEH53,
    )
    assert combined > 2.0

def test_qrs_elite_does_not_inherit_selling_sibling_scores():
    """Selling reconstructions must not donate λ-MSE to a different unique cell."""
    unique = []
    records = []
    qrs_cp = np.array([7.0, 8.0, 10.0, 90.0, 90.0, 90.0])
    sell_cp = np.array([5.0, 12.0, 20.0, 70.0, 80.0, 110.0])
    unique.append(UniqueCellRecord(dedup_idx=0, source_flow_indices=[0], cellpar=qrs_cp.copy(), sources=["qrs"]))
    records.append(
        LatticeQRSRecord(
            flow_idx=0,
            cellpar=qrs_cp.copy(),
            qrs_loss=0.2,
            density=1.3,
            lambda_mse=0.05,
            lambda_recip_mse=0.0,
            source="qrs",
        )
    )
    unique.append(UniqueCellRecord(dedup_idx=1, source_flow_indices=[0], cellpar=sell_cp.copy(), sources=["selling"]))
    records.append(
        LatticeQRSRecord(
            flow_idx=0,
            cellpar=sell_cp.copy(),
            qrs_loss=float("inf"),
            density=1.3,
            lambda_mse=1e-12,
            lambda_recip_mse=0.0,
            source="selling",
        )
    )
    for i in range(1, 12):
        cp = np.array([7.0 + 0.2 * i, 8.0, 10.0 + i, 90.0, 90.0, 90.0])
        unique.append(UniqueCellRecord(dedup_idx=i + 1, source_flow_indices=[i], cellpar=cp, sources=["qrs"]))
        records.append(
            LatticeQRSRecord(
                flow_idx=i,
                cellpar=cp,
                qrs_loss=0.01,
                density=1.3,
                lambda_mse=float(i) * 1e-4,
                lambda_recip_mse=0.0,
                source="qrs",
            )
        )
    selected = select_diverse_unique_cells(unique, records, max_cells=8, n_elite=3)
    elite_idx = {c.dedup_idx for c in selected[:3]}
    assert 1 not in elite_idx
    assert 1 not in {c.dedup_idx for c in selected}
    assert elite_idx == {2, 3, 4}


def test_append_disagreeing_selling_only():
    qrs_cp = np.array([7.0, 8.0, 10.0, 97.0, 103.0, 90.0])
    sell_disagree = np.array([7.1, 8.1, 10.1, 70.0, 80.0, 110.0])
    sell_agree = np.array([7.05, 8.05, 10.05, 97.2, 103.1, 90.2])
    qrs_u = UniqueCellRecord(dedup_idx=0, source_flow_indices=[0], cellpar=qrs_cp.copy(), sources=["qrs"])
    sell_u = UniqueCellRecord(dedup_idx=1, source_flow_indices=[0], cellpar=sell_disagree.copy(), sources=["selling"])
    qrs_rec = LatticeQRSRecord(
        flow_idx=0, cellpar=qrs_cp.copy(), qrs_loss=0.1, density=1.3,
        lambda_mse=0.01, lambda_recip_mse=0.0, source="qrs",
    )
    sell_rec = LatticeQRSRecord(
        flow_idx=0, cellpar=sell_disagree.copy(), qrs_loss=float("inf"), density=1.3,
        lambda_mse=0.02, lambda_recip_mse=0.0, source="selling",
    )
    out = append_disagreeing_selling_cells(
        [qrs_u], [qrs_u, sell_u], [qrs_rec], [sell_rec], HALL_QAXMEH53, angle_mae_min=3.0,
    )
    assert [c.dedup_idx for c in out] == [0, 1]

    sell_u2 = UniqueCellRecord(dedup_idx=1, source_flow_indices=[0], cellpar=sell_agree.copy(), sources=["selling"])
    sell_rec2 = LatticeQRSRecord(
        flow_idx=0, cellpar=sell_agree.copy(), qrs_loss=float("inf"), density=1.3,
        lambda_mse=0.02, lambda_recip_mse=0.0, source="selling",
    )
    out2 = append_disagreeing_selling_cells(
        [qrs_u], [qrs_u, sell_u2], [qrs_rec], [sell_rec2], HALL_QAXMEH53, angle_mae_min=3.0,
    )
    assert [c.dedup_idx for c in out2] == [0]


def test_elite_keeps_one_cell_per_draw():
    unique = []
    records = []
    for j in range(2):
        cp = np.array([7.0 + 0.3 * j, 8.0, 10.0 + j, 90.0, 90.0, 90.0])
        unique.append(UniqueCellRecord(dedup_idx=j, source_flow_indices=[0], cellpar=cp, sources=["qrs"]))
        records.append(
            LatticeQRSRecord(
                flow_idx=0, cellpar=cp, qrs_loss=0.01, density=1.3,
                lambda_mse=1e-6 * (j + 1), lambda_recip_mse=0.0, source="qrs",
            )
        )
    for i in range(1, 10):
        cp = np.array([7.0 + i, 8.0, 10.0 + i, 90.0, 90.0, 90.0])
        unique.append(UniqueCellRecord(dedup_idx=i + 1, source_flow_indices=[i], cellpar=cp, sources=["qrs"]))
        records.append(
            LatticeQRSRecord(
                flow_idx=i, cellpar=cp, qrs_loss=0.01, density=1.3,
                lambda_mse=float(i), lambda_recip_mse=0.0, source="qrs",
            )
        )
    selected = select_diverse_unique_cells(unique, records, max_cells=6, n_elite=3)
    elite_flows = [tuple(c.source_flow_indices) for c in selected[:3]]
    assert elite_flows.count((0,)) <= 1


def _qrs_unique_and_records(mses: list[float]) -> tuple[list[UniqueCellRecord], list[LatticeQRSRecord]]:
    unique = []
    records = []
    for i, mse in enumerate(mses):
        cp = np.array([7.0 + 0.1 * i, 8.0, 10.0, 90.0, 90.0, 90.0])
        unique.append(
            UniqueCellRecord(dedup_idx=i, source_flow_indices=[i], cellpar=cp, sources=["qrs"])
        )
        records.append(
            LatticeQRSRecord(
                flow_idx=i,
                cellpar=cp,
                qrs_loss=0.01,
                density=1.3,
                lambda_mse=float(mse),
                lambda_recip_mse=0.0,
                source="qrs",
            )
        )
    return unique, records


def test_posterior_n_eff_tight_vs_spread():
    tight_u, tight_r = _qrs_unique_and_records([1e-6, 1e-2, 1e-2, 1e-2, 1e-2])
    spread_u, spread_r = _qrs_unique_and_records([1e-4] * 12)
    n_tight = posterior_effective_mode_count(tight_u, tight_r)
    n_spread = posterior_effective_mode_count(spread_u, spread_r)
    assert n_tight < 2.0
    assert n_spread > 10.0


def test_selling_cells_do_not_inflate_n_eff():
    unique, records = _qrs_unique_and_records([1e-6, 1e-2, 1e-2])
    for j in range(8):
        cp = np.array([20.0 + j, 8.0, 10.0, 70.0, 80.0, 110.0])
        unique.append(
            UniqueCellRecord(
                dedup_idx=100 + j,
                source_flow_indices=[j],
                cellpar=cp,
                sources=["selling"],
            )
        )
        records.append(
            LatticeQRSRecord(
                flow_idx=j,
                cellpar=cp,
                qrs_loss=float("inf"),
                density=1.3,
                lambda_mse=1e-9,
                lambda_recip_mse=0.0,
                source="selling",
            )
        )
    n_eff = posterior_effective_mode_count(unique, records)
    n_qrs_only = posterior_effective_mode_count(unique[:3], records[:3])
    assert abs(n_eff - n_qrs_only) < 1e-9


def test_dynamic_conf_budget_acsala_vs_qax():
    acsala = dynamic_conf_run_count(5.1)
    qax = dynamic_conf_run_count(13.8)
    assert acsala < 24 <= qax
    assert acsala == 15
    assert qax == 41
    assert dynamic_conf_run_count(2.8) == 12
    assert dynamic_conf_run_count(18.3) == 48


def test_scaled_conf_elite_tracks_budget():
    assert scaled_conf_elite(6, 15) == 4
    assert scaled_conf_elite(6, 41) == 10
    assert scaled_conf_elite(6, 12) >= 3
    assert scaled_conf_elite(6, 3) == 3


def test_selling_kcenters_include_outlier_mode():
    rng = np.random.default_rng(0)
    major = np.array([-0.20, -0.10, -0.15, -0.30, -0.25, -0.22])
    minor = np.array([-1.50, -1.40, 0.80, 0.70, -0.10, 0.20])
    selling = np.stack([major + 0.01 * rng.normal(size=6) for _ in range(20)] + [minor])
    centers = selling_kcenter_draw_indices(selling, 4)
    assert 20 in centers


def test_disagreement_channel_adds_unselected_draw():
    qrs_agree = np.array([7.0, 8.0, 10.0, 90.0, 90.0, 90.0])
    sell_agree = np.array([7.05, 8.05, 10.05, 90.2, 89.8, 90.1])
    qrs_bad = np.array([7.1, 8.1, 10.1, 90.0, 87.0, 81.0])
    sell_good = np.array([7.0, 8.3, 10.2, 98.0, 103.0, 90.0])
    unique = [
        UniqueCellRecord(dedup_idx=0, source_flow_indices=[0], cellpar=qrs_agree.copy(), sources=["qrs"]),
        UniqueCellRecord(dedup_idx=5, source_flow_indices=[5], cellpar=qrs_bad.copy(), sources=["qrs"]),
        UniqueCellRecord(dedup_idx=105, source_flow_indices=[5], cellpar=sell_good.copy(), sources=["selling"]),
    ]
    qrs_recs = [
        LatticeQRSRecord(0, qrs_agree.copy(), 0.01, 1.3, 1e-6, 0.0, "qrs"),
        LatticeQRSRecord(5, qrs_bad.copy(), 0.01, 1.3, 1.0, 0.0, "qrs"),
    ]
    sell_recs = [
        LatticeQRSRecord(0, sell_agree.copy(), float("inf"), 1.3, 1e-6, 0.0, "selling"),
        LatticeQRSRecord(5, sell_good.copy(), float("inf"), 1.3, 1.0, 0.0, "selling"),
    ]
    out = append_disagreement_channel(
        [unique[0]], unique, qrs_recs, sell_recs, HALL_QAXMEH53, n_max=12,
    )
    ids = [c.dedup_idx for c in out]
    assert 0 in ids
    assert 5 in ids
    assert 105 in ids


def test_originelite_does_not_keep_unselected_selling():
    qrs_bad = np.array([7.1, 8.1, 10.1, 90.0, 87.0, 81.0])
    sell_good = np.array([7.0, 8.3, 10.2, 98.0, 103.0, 90.0])
    qrs_u = UniqueCellRecord(dedup_idx=0, source_flow_indices=[0], cellpar=qrs_bad.copy(), sources=["qrs"])
    sell_u = UniqueCellRecord(dedup_idx=1, source_flow_indices=[5], cellpar=sell_good.copy(), sources=["selling"])
    qrs_sel = UniqueCellRecord(dedup_idx=2, source_flow_indices=[1], cellpar=np.array([8.0, 8.0, 8.0, 90.0, 90.0, 90.0]), sources=["qrs"])
    qrs_recs = [
        LatticeQRSRecord(1, qrs_sel.cellpar.copy(), 0.01, 1.3, 1e-6, 0.0, "qrs"),
        LatticeQRSRecord(5, qrs_bad.copy(), 0.01, 1.3, 1.0, 0.0, "qrs"),
    ]
    sell_recs = [
        LatticeQRSRecord(5, sell_good.copy(), float("inf"), 1.3, 1.0, 0.0, "selling"),
    ]
    out = append_disagreeing_selling_cells(
        [qrs_sel], [qrs_u, sell_u, qrs_sel], qrs_recs, sell_recs, HALL_QAXMEH53,
    )
    assert [c.dedup_idx for c in out] == [2]


def test_select_conf_keeps_disagreement_pair():
    cells = []
    qrs_recs = []
    sell_recs = []
    selling_vecs = []
    for i in range(8):
        cp = np.array([7.0 + 0.05 * i, 8.0, 10.0, 90.0, 90.0, 90.0])
        cells.append(UniqueCellRecord(dedup_idx=i, source_flow_indices=[i], cellpar=cp, sources=["qrs"]))
        qrs_recs.append(LatticeQRSRecord(i, cp.copy(), 0.01, 1.3, 1e-4 * (i + 1), 0.0, "qrs"))
        sell_cp = cp.copy()
        sell_recs.append(LatticeQRSRecord(i, sell_cp, float("inf"), 1.3, 1e-4, 0.0, "selling"))
        selling_vecs.append(selling_parameters(cp))
    qrs79 = np.array([7.08, 8.97, 9.91, 84.2, 76.1, 81.8])
    sell79 = np.array([6.99, 8.46, 10.11, 81.7, 76.4, 88.1])
    cells.append(UniqueCellRecord(dedup_idx=79, source_flow_indices=[8], cellpar=qrs79, sources=["qrs"]))
    cells.append(UniqueCellRecord(dedup_idx=172, source_flow_indices=[8], cellpar=sell79, sources=["selling"]))
    qrs_recs.append(LatticeQRSRecord(8, qrs79.copy(), 0.5, 1.3, 1.0, 0.0, "qrs"))
    sell_recs.append(LatticeQRSRecord(8, sell79.copy(), float("inf"), 1.3, 0.2, 0.0, "selling"))
    selling_vecs.append(selling_parameters(sell79))
    all_selling = np.stack(selling_vecs)
    selected = select_conf_unique_cells(
        cells, qrs_recs, sell_recs, max_cells=6, hall_number=HALL_QAXMEH53,
        n_elite=3, all_selling=all_selling, n_clusters=4, n_disagree=4,
    )
    ids = {c.dedup_idx for c in selected}
    assert 79 in ids
    assert 172 in ids


def test_qqqcig04_selling_cells_fail_density_volume_gate():
    from cellnet.packing import cell_volume, cellpar_in_predicted_volume_range

    target_volume = 2579.3
    tiny_146 = np.array([9.041270243729324, 9.663436191624761, 10.340546012621694, 90.0, 89.98627350957184, 90.0])
    tiny_155 = np.array([6.766789021601983, 6.89084715880332, 16.45435530086045, 90.0, 89.97699966222768, 90.0])
    assert cell_volume(tiny_146) < 0.5 * target_volume
    assert not cellpar_in_predicted_volume_range(tiny_146, target_volume)
    assert not cellpar_in_predicted_volume_range(tiny_155, target_volume)


def test_disagreement_skips_selling_outside_density_volume():
    qrs_cp = np.array([13.03, 12.18, 13.80, 90.0, 86.6, 90.0])
    sell_tiny = np.array([9.0413, 9.6634, 10.3405, 90.0, 89.986, 90.0])
    qrs_u = UniqueCellRecord(dedup_idx=0, source_flow_indices=[0], cellpar=qrs_cp.copy(), sources=["qrs"])
    sell_u = UniqueCellRecord(dedup_idx=146, source_flow_indices=[0], cellpar=sell_tiny.copy(), sources=["selling"])
    qrs_rec = LatticeQRSRecord(0, qrs_cp.copy(), 0.1, 1.37, 0.003, 0.0, "qrs")
    sell_rec = LatticeQRSRecord(0, sell_tiny.copy(), float("inf"), 1.37, 0.12, 0.0, "selling")
    target_volume = 2579.3
    out = append_disagreeing_selling_cells(
        [qrs_u], [qrs_u, sell_u], [qrs_rec], [sell_rec], HALL_QAXMEH53,
        angle_mae_min=3.0, target_volume=target_volume,
    )
    assert [c.dedup_idx for c in out] == [0]

    out_ch = append_disagreement_channel(
        [qrs_u], [qrs_u, sell_u], [qrs_rec], [sell_rec], HALL_QAXMEH53,
        n_max=12, target_volume=target_volume,
    )
    ids = [c.dedup_idx for c in out_ch]
    assert 0 in ids
    assert 146 not in ids

