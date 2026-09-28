"""
Regression tests for the Delaunay-reduced successive-minima path and the
invariant Selling canonicalization.
"""

import numpy as np
import torch

from cellnet.lattice_invariants import (
    _enumerate_candidates,
    _greedy_successive_minima,
    canonical_selling_parameters,
    delaunay_reduce_matrix,
    lattice_invariants_from_cellpar,
    log_reciprocal_successive_minima,
    log_successive_minima,
    selling_parameters,
    selling_parameters_from_basis,
    sort_selling_parameters,
    successive_minima,
)
from cellnet.reciprocal import direct_matrix_from_cellpar, metric_tensor


def _brute_force_minima(cellpar, radius=8):
    """Large-box enumeration on the reduced basis: the ground truth."""
    reduced = delaunay_reduce_matrix(direct_matrix_from_cellpar(np.asarray(cellpar, float)))
    sq, hkl = _enumerate_candidates(metric_tensor(reduced), radius)
    return _greedy_successive_minima(sq, hkl)


# Strongly oblique unreduced cells on which the old edge-length search radius
# missed the true third minimum (over-estimating λ₃ by up to ~20%).
OBLIQUE_CELLS = [
    [3.58, 40.9, 25.67, 76.22, 61.6, 137.37],
    [4.39, 35.57, 42.28, 132.98, 44.59, 89.13],
    [22.85, 5.01, 18.66, 52.87, 71.73, 123.39],
    [10.95, 4.36, 38.56, 59.08, 96.13, 43.6],
    [29.43, 14.58, 19.01, 45.6, 89.48, 134.25],
]


def test_successive_minima_match_brute_force_on_oblique_cells():
    for cellpar in OBLIQUE_CELLS:
        truth = _brute_force_minima(cellpar)
        got = successive_minima(np.array(cellpar, dtype=np.float64))
        assert np.allclose(got, truth, rtol=1e-9, atol=1e-9), (cellpar, got, truth)


def test_successive_minima_match_brute_force_on_random_cells():
    rng = np.random.default_rng(1234)
    n_checked = 0
    while n_checked < 200:
        cellpar = np.array(
            [
                *np.exp(rng.uniform(np.log(3.0), np.log(40.0), 3)),
                *rng.uniform(45.0, 135.0, 3),
            ]
        )
        a, b, g = np.radians(cellpar[3:])
        ca, cb, cg = np.cos([a, b, g])
        if 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg < 1e-2:
            continue
        truth = _brute_force_minima(cellpar)
        got = successive_minima(cellpar)
        assert np.allclose(got, truth, rtol=1e-9, atol=1e-9), (cellpar, got, truth)
        n_checked += 1


def test_shared_reduction_matches_separate_kernels():
    for cellpar in OBLIQUE_CELLS + [[5.0, 7.0, 9.0, 90.0, 90.0, 90.0]]:
        cp = np.array(cellpar, dtype=np.float64)
        sell, lam, lam_r = lattice_invariants_from_cellpar(cp)
        assert np.allclose(sell, selling_parameters(cp), rtol=1e-9, atol=1e-9)
        assert np.allclose(lam, log_successive_minima(cp), rtol=1e-9, atol=1e-9)
        assert np.allclose(lam_r, log_reciprocal_successive_minima(cp), rtol=1e-9, atol=1e-9)


def test_shared_reduction_respects_flags():
    cp = np.array([5.0, 7.0, 9.0, 90.0, 90.0, 90.0])
    sell, lam, lam_r = lattice_invariants_from_cellpar(
        cp, with_selling=False, with_lambda=True, with_lambda_recip=False
    )
    assert sell is None and lam_r is None and lam is not None


def test_canonical_selling_is_invariant_under_vertex_relabeling():
    rng = np.random.default_rng(7)
    from itertools import permutations

    for _ in range(50):
        basis = rng.normal(size=(3, 3)) * 5.0
        if abs(np.linalg.det(basis)) < 1.0:
            continue
        b1, b2, b3 = basis
        b4 = -(b1 + b2 + b3)
        vecs = [b1, b2, b3, b4]
        ref = None
        for perm in permutations(range(4)):
            # Any three of the relabeled superbase vectors form a basis of the same lattice.
            new_basis = np.stack([vecs[perm[0]], vecs[perm[1]], vecs[perm[2]]])
            canon = canonical_selling_parameters(selling_parameters_from_basis(new_basis))
            if ref is None:
                ref = canon
            assert np.allclose(canon, ref, atol=1e-9)


def test_legacy_sort_is_documented_as_non_invariant():
    """The legacy function is kept for checkpoint compatibility; show it is not invariant."""
    s = np.array([-0.9, -0.6, 10.5, -0.46, 17.45, 26.11])
    # Swap b1<->b2: (S12,S13,S14,S23,S24,S34) -> (S12,S23,S24,S13,S14,S34)
    swapped = s[[0, 3, 4, 1, 2, 5]]
    assert not np.allclose(sort_selling_parameters(s), sort_selling_parameters(swapped))
    assert np.allclose(canonical_selling_parameters(s), canonical_selling_parameters(swapped))


def test_polymorph_bank_pads_missing_rows_with_fallback():
    from types import SimpleNamespace

    from cellnet.polymorph import SellingPolymorphBank

    stats = SimpleNamespace(
        selling_mean=np.zeros(6, dtype=np.float32),
        selling_std=np.ones(6, dtype=np.float32),
        log_density_mean=0.0,
        log_density_std=1.0,
        log_lambda_mean=np.zeros(3, dtype=np.float32),
        log_lambda_std=np.ones(3, dtype=np.float32),
        log_lambda_recip_mean=np.zeros(3, dtype=np.float32),
        log_lambda_recip_std=np.ones(3, dtype=np.float32),
        idx_to_hall={0: 2},
        zprime_values=[1.0],
    )

    def sample(smiles, k):
        return SimpleNamespace(
            smiles=smiles,
            hall_number=2,
            zprime=1.0,
            selling_log1p=np.full(6, float(k), dtype=np.float32),
            log_density=0.1 * k,
            log_successive_minima=np.full(3, float(k), dtype=np.float32),
            log_reciprocal_successive_minima=np.full(3, -float(k), dtype=np.float32),
        )

    bank = SellingPolymorphBank([sample("A", 1), sample("A", 2), sample("B", 3)], stats)
    fallback_lat = torch.full((3, 12), 9.0)
    fallback_ld = torch.tensor([0.5, 0.6, 0.7])
    lat, ld = bank.batch_lattice_tensors(
        ["A", "B", "MISSING"],
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 0, 0]),
        stats.idx_to_hall,
        stats.zprime_values,
        torch.device("cpu"),
        fallback_lattice=fallback_lat,
        fallback_log_density=fallback_ld,
    )
    assert lat.shape == (3, 2, 12) and ld.shape == (3, 2)
    # Group A has two entries.
    assert torch.allclose(lat[0, 0, :6], torch.full((6,), 1.0))
    assert torch.allclose(lat[0, 1, :6], torch.full((6,), 2.0))
    # Group B pads its single entry into slot 1.
    assert torch.allclose(lat[1, 1], lat[1, 0])
    # Missing key: fallback fills every K slot (a zero row would be a spurious min-over-K target).
    assert torch.allclose(lat[2, 0], fallback_lat[2]) and torch.allclose(lat[2, 1], fallback_lat[2])
    assert torch.allclose(ld[2], torch.tensor([0.7, 0.7]))
