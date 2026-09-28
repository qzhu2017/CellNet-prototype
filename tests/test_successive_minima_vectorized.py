"""
Property tests for the vectorized Minkowski successive-minima path.

These check mathematical invariants rather than agreement with any particular
implementation, so they stay meaningful if the enumeration is rewritten again.
"""

import numpy as np

from cellnet.lattice_invariants import _enumerate_candidates, successive_minima
from cellnet.reciprocal import (
    cellpar_from_direct_matrix,
    direct_matrix_from_cellpar,
    metric_tensor,
)


def _gram(cellpar):
    return metric_tensor(direct_matrix_from_cellpar(np.asarray(cellpar, dtype=np.float64)))


def test_enumerate_candidates_contract():
    """Ascending squared norms, no zero vector, norms consistent with the Gram matrix."""
    gram = _gram([5.0, 7.0, 9.0, 80.0, 95.0, 105.0])
    radius = 3
    sq_norms, hkl = _enumerate_candidates(gram, radius)

    assert sq_norms.shape[0] == hkl.shape[0]
    assert hkl.shape[1] == 3
    # No zero vector, and nothing outside the box.
    assert np.all(np.any(hkl != 0, axis=1))
    assert np.all(np.abs(hkl) <= radius)
    # (2r+1)^3 - 1 lattice points for a positive-definite Gram matrix.
    assert sq_norms.shape[0] == (2 * radius + 1) ** 3 - 1
    # Sorted ascending.
    assert np.all(np.diff(sq_norms) >= 0.0)
    # Norms agree with an explicit per-vector evaluation.
    explicit = np.array([float(v @ gram @ v) for v in hkl.astype(np.float64)])
    assert np.allclose(sq_norms, explicit, rtol=0.0, atol=1e-9)


def test_successive_minima_are_ordered_and_positive():
    for cellpar in (
        [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
        [4.5, 8.1, 11.3, 78.0, 91.0, 103.0],
        [6.0, 6.0, 14.0, 90.0, 90.0, 120.0],
    ):
        lam = successive_minima(np.array(cellpar, dtype=np.float64))
        assert lam.shape == (3,)
        assert np.all(lam > 0.0)
        assert lam[0] <= lam[1] + 1e-12 <= lam[2] + 1e-12


def test_cubic_successive_minima_are_the_edge():
    a = 6.25
    lam = successive_minima(np.array([a, a, a, 90.0, 90.0, 90.0]))
    assert np.allclose(lam, [a, a, a], atol=1e-9)


def test_orthorhombic_successive_minima_are_the_edges():
    """For a mildly anisotropic orthorhombic cell the minima are the three edges."""
    a, b, c = 5.0, 7.0, 9.0
    lam = successive_minima(np.array([a, b, c, 90.0, 90.0, 90.0]))
    assert np.allclose(lam, [a, b, c], atol=1e-9)


def test_successive_minima_invariant_under_unimodular_basis_change():
    """
    λ depends on the lattice, not the basis: any SL(3, Z) change of cell must
    leave it fixed. This is the invariant the flow targets rely on.
    """
    unimodular = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64),
    ]
    base = [
        [5.0, 7.0, 9.0, 90.0, 90.0, 90.0],
        [4.8, 6.1, 7.7, 82.0, 96.0, 101.0],
        [6.4, 6.4, 10.2, 90.0, 90.0, 120.0],
    ]
    for cellpar in base:
        reference = successive_minima(np.array(cellpar, dtype=np.float64))
        matrix = direct_matrix_from_cellpar(np.array(cellpar, dtype=np.float64))
        for u in unimodular:
            assert abs(abs(np.linalg.det(u)) - 1.0) < 1e-12
            transformed = cellpar_from_direct_matrix(u @ matrix)
            assert np.allclose(
                successive_minima(transformed), reference, rtol=1e-9, atol=1e-9
            ), f"lambda changed under a unimodular map for {cellpar}"
