"""Tests for crystal-system-aware cell-parameter alignment."""

import numpy as np

from cellnet.symmetry import (
    align_orthorhombic_cellpar_to_axis_ranks,
    align_cellpar_to_reference,
    allowed_axis_permutations,
    angle_mae_deg,
    axis_permutation_equivalence_mask,
    cellpar_to_free,
    free_to_cellpar,
    hall_preserving_axis_permutations,
    select_axis_rank_permutation,
)


# Hall 6 → monoclinic (XELYUJ), unique axis b.
HALL_MONO = 6
# Hall 1 → triclinic P1
HALL_TRI = 1
# Orthorhombic settings with special c and fully equivalent axes, respectively.
HALL_P21212 = 112
HALL_PMM2 = 125
HALL_FDDD = 335
HALL_TETRAGONAL = 400
HALL_ACEMID02 = 452


def test_tetragonal_and_hexagonal_free_cellpar_roundtrip():
    tetragonal = np.array([8.2, 8.2, 15.4, 90.0, 90.0, 90.0])
    hexagonal = np.array([11.3, 11.3, 12.73, 90.0, 90.0, 120.0])

    for hall, cellpar in (
        (HALL_TETRAGONAL, tetragonal),
        (HALL_ACEMID02, hexagonal),
    ):
        free = cellpar_to_free(cellpar, hall)
        assert np.allclose(free, np.log(cellpar[[0, 2]]))
        assert np.allclose(free_to_cellpar(free, hall), cellpar)


def test_monoclinic_allowed_permutations_fix_unique_axis():
    perms = allowed_axis_permutations(HALL_MONO)
    assert (0, 1, 2) in perms
    assert (2, 1, 0) in perms
    assert len(perms) == 2
    for perm in perms:
        assert perm[1] == 1


def test_monoclinic_align_does_not_swap_unique_axis():
    true = np.array([5.816, 7.253, 25.286, 90.0, 96.3, 90.0])
    pred = np.array([25.872, 6.198, 7.400, 90.0, 86.8, 90.0])

    aligned = align_cellpar_to_reference(pred, true, HALL_MONO)

    assert aligned[1] == pred[1]
    rel = np.abs(aligned[:3] - true[:3]) / true[:3]
    assert float(np.mean(rel) * 100) > 10.0
    assert float(rel[1] * 100) > 10.0


def test_monoclinic_align_can_swap_a_and_c_only():
    true = np.array([5.816, 7.253, 25.286, 90.0, 96.3, 90.0])
    pred = np.array([7.400, 6.198, 25.872, 90.0, 86.8, 90.0])

    aligned = align_cellpar_to_reference(pred, true, HALL_MONO)

    assert aligned[0] == pred[0]
    assert aligned[1] == pred[1]
    assert aligned[2] == pred[2]
    rel = np.abs(aligned[:3] - true[:3]) / true[:3]
    assert float(np.mean(rel) * 100) < 15.0


def test_monoclinic_supplementary_beta_xelyuj_flow48():
    """β=86.5° vs true β=96.3° → ~2.8° after supplementary-angle equivalence."""
    true = np.array([5.816, 7.253, 25.286, 90.0, 96.3, 90.0])
    pred = np.array([5.787, 7.794, 24.842, 90.0, 86.5, 90.0])

    mae, errs = angle_mae_deg(pred, true, HALL_MONO)
    assert mae < 4.0
    assert abs(mae - 2.8) < 0.5
    assert abs(errs[1] - mae) < 1e-6


def test_triclinic_axis_flip_maps_supplementary_alpha():
    """Reversing **b** maps (α,γ) → (180°−α, 180°−γ) simultaneously."""
    true = np.array([5.0, 6.0, 7.0, 82.0, 85.0, 88.0])
    pred = np.array([5.0, 6.0, 7.0, 98.0, 85.0, 92.0])

    mae, errs = angle_mae_deg(pred, true, HALL_TRI)
    assert mae < 0.5
    assert np.all(errs[:3] < 0.5)


def test_triclinic_naive_supplement_can_mismatch_other_angles():
    """α=100° alone is not equivalent to true α=82° if γ must stay matched."""
    true = np.array([5.0, 6.0, 7.0, 82.0, 85.0, 88.0])
    pred = np.array([5.0, 6.0, 7.0, 100.0, 85.0, 88.0])

    mae, errs = angle_mae_deg(pred, true, HALL_TRI)
    assert mae < 3.0
    assert errs[0] < 3.0


def test_hall_preserving_permutations_detect_special_c_axis():
    expected = ((0, 1, 2), (1, 0, 2))
    assert hall_preserving_axis_permutations(HALL_P21212) == expected
    assert hall_preserving_axis_permutations(HALL_PMM2) == expected


def test_fddd_merges_all_axis_permutations():
    assert len(hall_preserving_axis_permutations(HALL_FDDD)) == 6
    mask = axis_permutation_equivalence_mask(
        np.array([7.0, 11.0, 9.0, 90.0, 90.0, 90.0]),
        HALL_FDDD,
    )
    assert mask.all()


def test_axis_probabilities_merge_ab_swap_for_special_c():
    probabilities = np.zeros(6)
    probabilities[0] = 0.2  # (0,1,2)
    probabilities[2] = 0.3  # (1,0,2), equivalent a/b swap
    probabilities[1] = 0.5  # (0,2,1), different c rank

    selected, collapsed = select_axis_rank_permutation(
        probabilities,
        HALL_P21212,
    )

    assert selected == (0, 1, 2)
    assert np.isclose(collapsed.sum(), 1.0)
    assert np.isclose(collapsed[0], 0.5)


def test_align_orthorhombic_cellpar_uses_predicted_axis_ranks():
    cellpar = np.array([12.0, 5.0, 8.0, 90.0, 90.0, 90.0])
    aligned = align_orthorhombic_cellpar_to_axis_ranks(
        cellpar,
        (1, 2, 0),
        HALL_PMM2,
    )
    assert np.allclose(aligned[:3], [8.0, 12.0, 5.0])
