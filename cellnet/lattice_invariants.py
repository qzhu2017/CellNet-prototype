"""
Minkowski successive minima λ₁, λ₂, λ₃ for 3D direct lattices.

The k-th successive minimum λ_k(L) is the smallest radius r such that the closed
ball of radius r about the origin contains k linearly independent lattice vectors.
Equivalently,

    λ_k(L) = min{ max_{1≤i≤k} ‖v_i‖ : v_1,…,v_k ∈ L are ℝ-linearly independent }.

These depend only on the lattice point set and are invariant under SL(3, ℤ) changes
of basis (unimodular transformations).
"""

from __future__ import annotations

import numpy as np

from cellnet.reciprocal import direct_matrix_from_cellpar, metric_tensor

# Minkowski 2nd theorem (Euclidean ball): V ≤ λ₁λ₂λ₃ ≤ (6/π)V for 3D lattices.
LAMBDA_PRODUCT_LOWER_FRAC = 1.0
LAMBDA_PRODUCT_UPPER_FRAC = 6.0 / np.pi  # ≈ 1.9099


def lambda_product_from_log(log_lambdas: np.ndarray) -> float:
    """Return λ₁λ₂λ₃ from natural-log successive minima."""
    log_l = np.asarray(log_lambdas, dtype=np.float64).reshape(-1)
    return float(np.exp(np.sum(log_l)))


def constrain_log_lambda_product(
    log_lambdas: np.ndarray,
    target_volume: float,
    lower_frac: float = LAMBDA_PRODUCT_LOWER_FRAC,
    upper_frac: float = LAMBDA_PRODUCT_UPPER_FRAC,
) -> tuple[np.ndarray, bool]:
    """
    Uniformly scale log λ so λ₁λ₂λ₃ lies in [lower_frac·V, upper_frac·V].

    Preserves λ ratios (shape); only adjusts the overall scale.
    """
    log_l = np.asarray(log_lambdas, dtype=np.float64).reshape(-1)
    if log_l.size != 3 or target_volume <= 0:
        return log_l.copy(), False

    log_prod = float(np.sum(log_l))
    log_lo = float(np.log(max(lower_frac * target_volume, 1e-30)))
    log_hi = float(np.log(max(upper_frac * target_volume, 1e-30)))

    if log_prod < log_lo:
        return log_l + (log_lo - log_prod) / 3.0, True
    if log_prod > log_hi:
        return log_l + (log_hi - log_prod) / 3.0, True
    return log_l.copy(), False


def constrain_log_lambda_batch(
    log_lambdas: np.ndarray,
    target_volume: float,
    lower_frac: float = LAMBDA_PRODUCT_LOWER_FRAC,
    upper_frac: float = LAMBDA_PRODUCT_UPPER_FRAC,
) -> tuple[np.ndarray, list[bool]]:
    """Apply ``constrain_log_lambda_product`` row-wise to shape (K, 3)."""
    rows = np.asarray(log_lambdas, dtype=np.float64)
    if rows.ndim == 1:
        adj, clipped = constrain_log_lambda_product(rows, target_volume, lower_frac, upper_frac)
        return adj, [clipped]
    out = np.empty_like(rows)
    clipped: list[bool] = []
    for j in range(rows.shape[0]):
        out[j], was_clipped = constrain_log_lambda_product(
            rows[j], target_volume, lower_frac, upper_frac
        )
        clipped.append(was_clipped)
    return out, clipped


def _gram_matrix(cellpar: np.ndarray) -> np.ndarray:
    """Gram matrix G = M Mᵀ for direct lattice rows M."""
    m = direct_matrix_from_cellpar(cellpar)
    return metric_tensor(m)


def _enumerate_candidates(
    gram: np.ndarray,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    All non-zero lattice vectors with |h|,|k|,|l| ≤ search_radius, sorted by norm.

    Vectorized over the whole (2r+1)³ integer box: every squared norm hᵀ G h is
    formed in a single ``einsum`` instead of one Python iteration, one temporary
    array and one scalar matrix product per lattice point.

    Returns
    -------
    sq_norms : (N,) float64
        Squared norms, ascending.
    hkl : (N, 3) int64
        Integer indices in the matching order.

    Ties are broken in odometer order (h slowest, l fastest) via a stable sort,
    which is the generation order of the original triple loop, so selection
    among equal-norm vectors downstream is unchanged.
    """
    r = int(search_radius)
    axes = np.arange(-r, r + 1, dtype=np.int64)
    hkl = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1).reshape(-1, 3)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    hkl_f = hkl.astype(np.float64)
    sq_norms = np.einsum("ij,jk,ik->i", hkl_f, gram, hkl_f)
    keep = sq_norms > 0
    hkl = hkl[keep]
    sq_norms = sq_norms[keep]
    order = np.argsort(sq_norms, kind="stable")
    return sq_norms[order], hkl[order]


def _greedy_successive_minima(
    sq_norms: np.ndarray,
    hkl: np.ndarray,
) -> np.ndarray | None:
    """
    (λ₁, λ₂, λ₃) by ascending-norm greedy rank growth.

    Returns None when the candidate set does not span rank 3, which signals the
    caller to widen the search box.

    Walking candidates shortest-first and keeping each vector that raises the
    rank of the span yields exactly the successive minima: at the moment the
    k-th independent vector is accepted, no shorter set of k independent
    vectors exists, which is the definition of λ_k.

    The previous implementation re-ranked every prefix of the candidate list
    once per k -- an O(N²) sweep of SVDs over the whole box. This stops at the
    third rank increase, which for a reduced cell happens within the first
    handful of candidates.
    """
    basis: list[np.ndarray] = []
    lambdas: list[float] = []
    for i in range(sq_norms.shape[0]):
        trial = np.stack(basis + [hkl[i]], axis=0).astype(np.float64)
        if int(np.linalg.matrix_rank(trial, tol=1e-9)) > len(basis):
            basis.append(hkl[i])
            lambdas.append(float(np.sqrt(sq_norms[i])))
            if len(lambdas) == 3:
                return np.array(lambdas, dtype=np.float64)
    return None


def _selling_candidate_vectors(reduced_basis: np.ndarray) -> np.ndarray:
    """
    The seven lattice vectors (up to sign) that can attain the successive minima
    of a Delaunay-reduced 3D lattice.

    For an obtuse superbase ``b1, b2, b3, b4 = -(b1 + b2 + b3)`` (all pairwise dot
    products ≤ 0, which is exactly what ``spglib.delaunay_reduce`` returns), the
    Voronoi-relevant vectors are ``±b_i`` and ``±(b_i + b_j)`` (Conway & Sloane,
    "Low-dimensional lattices VI"). In three dimensions a shortest basis is made
    of Voronoi-relevant vectors, so λ₁, λ₂, λ₃ can be read off this set; no
    integer-box enumeration is needed.
    """
    b1, b2, b3 = np.asarray(reduced_basis, dtype=np.float64)
    return np.array(
        [b1, b2, b3, -(b1 + b2 + b3), b1 + b2, b1 + b3, b2 + b3],
        dtype=np.float64,
    )


def _successive_minima_from_vectors(vecs: np.ndarray) -> np.ndarray | None:
    """Greedy rank growth over an explicit candidate vector set (shortest first)."""
    sq_norms = np.einsum("ij,ij->i", vecs, vecs)
    order = np.argsort(sq_norms, kind="stable")
    basis: list[np.ndarray] = []
    lambdas: list[float] = []
    for i in order:
        trial = np.stack(basis + [vecs[i]], axis=0)
        if int(np.linalg.matrix_rank(trial, tol=1e-9)) > len(basis):
            basis.append(vecs[i])
            lambdas.append(float(np.sqrt(sq_norms[i])))
            if len(lambdas) == 3:
                return np.array(lambdas, dtype=np.float64)
    return None


def _successive_minima_by_enumeration(
    matrix: np.ndarray,
    search_radius: int | None = None,
) -> np.ndarray:
    """
    Fallback: integer-box enumeration on a direct basis (rows).

    Only used when Delaunay reduction fails. The auto radius is derived from the
    basis lengths, which is sufficient for a reduced basis but can miss λ₃ for
    strongly oblique unreduced cells -- another reason the reduced path is
    preferred.
    """
    gram = metric_tensor(matrix)
    lengths = np.linalg.norm(matrix, axis=1)
    lmin = max(float(np.min(lengths)), 1e-6)
    lmax = float(np.max(lengths))
    if search_radius is None:
        search_radius = max(2, int(np.ceil(lmax / lmin)) + 2)

    for radius in range(search_radius, search_radius + 12):
        sq_norms, hkl = _enumerate_candidates(gram, radius)
        if sq_norms.size == 0:
            continue
        lambdas = _greedy_successive_minima(sq_norms, hkl)
        if lambdas is not None:
            return lambdas

    raise RuntimeError(
        f"Could not determine three successive minima within search radius {radius}"
    )


def successive_minima_from_matrix(
    matrix: np.ndarray,
    search_radius: int | None = None,
) -> np.ndarray:
    """
    (λ₁, λ₂, λ₃) for a lattice given by a 3×3 basis matrix (rows = vectors).

    Delaunay-reduces the basis with spglib and reads the minima off the seven
    Selling vectors. Falls back to integer enumeration if the reduction fails.
    """
    m = np.asarray(matrix, dtype=np.float64)
    try:
        reduced = delaunay_reduce_matrix(m)
    except Exception:
        return _successive_minima_by_enumeration(m, search_radius=search_radius)
    lambdas = _successive_minima_from_vectors(_selling_candidate_vectors(reduced))
    if lambdas is None:
        return _successive_minima_by_enumeration(reduced, search_radius=search_radius)
    return lambdas


def successive_minima(
    cellpar: np.ndarray,
    search_radius: int | None = None,
) -> np.ndarray:
    """
    Compute (λ₁, λ₂, λ₃) in Å for a direct unit cell.

    ``search_radius`` is only consulted by the enumeration fallback (see
    ``successive_minima_from_matrix``).
    """
    return successive_minima_from_matrix(
        direct_matrix_from_cellpar(cellpar), search_radius=search_radius
    )


def log_successive_minima(cellpar: np.ndarray, search_radius: int | None = None) -> np.ndarray:
    """Natural log of direct-space (λ₁, λ₂, λ₃) in Å."""
    lambdas = successive_minima(cellpar, search_radius=search_radius)
    return np.log(np.clip(lambdas, 1e-12, None))


def reciprocal_successive_minima(cellpar: np.ndarray, search_radius: int | None = None) -> np.ndarray:
    """Minkowski successive minima (λ*₁, λ*₂, λ*₃) on the reciprocal lattice (Å⁻¹ scale)."""
    from cellnet.reciprocal import reciprocal_matrix

    return successive_minima_from_matrix(
        reciprocal_matrix(direct_matrix_from_cellpar(cellpar)), search_radius=search_radius
    )


def log_reciprocal_successive_minima(cellpar: np.ndarray, search_radius: int | None = None) -> np.ndarray:
    """Natural log of reciprocal-space (λ*₁, λ*₂, λ*₃)."""
    lambdas = reciprocal_successive_minima(cellpar, search_radius=search_radius)
    return np.log(np.clip(lambdas, 1e-12, None))


def successive_minima_ratios(cellpar: np.ndarray, search_radius: int | None = None) -> np.ndarray:
    """Scale-free shape invariants (λ₂/λ₁, λ₃/λ₁, λ₃/λ₂)."""
    lambdas = successive_minima(cellpar, search_radius=search_radius)
    return np.array(
        [lambdas[1] / lambdas[0], lambdas[2] / lambdas[0], lambdas[2] / lambdas[1]],
        dtype=np.float64,
    )


SELLING_DIM = 6
SELLING_PAIR_LABELS = ("12", "13", "14", "23", "24", "34")


def gram_from_selling(selling: np.ndarray) -> np.ndarray:
    """
    Recover the 3×3 Gram matrix of (b₁, b₂, b₃) from Selling scalars.

    With S_ij = −b_i · b_j and b₄ = −(b₁+b₂+b₃),

        ‖b₁‖² = S₁₂ + S₁₃ + S₁₄,   b₁·b₂ = −S₁₂
    """
    s = np.asarray(selling, dtype=np.float64).reshape(6)
    s12, s13, s14, s23, s24, s34 = s.tolist()
    return np.array(
        [
            [s12 + s13 + s14, -s12, -s13],
            [-s12, s12 + s23 + s24, -s23],
            [-s13, -s23, s13 + s23 + s34],
        ],
        dtype=np.float64,
    )


def cellpar_from_selling(
    selling: np.ndarray,
    hall_number: int | None = None,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Reconstruct a Delaunay-reduced cell from a 6-vector of Selling parameters.

    Used as a target-free QRS initialization (angles come from the flow, not
    from a known reference cell).
    """
    from cellnet.reciprocal import cellpar_from_direct_matrix

    gram = 0.5 * (gram_from_selling(selling) + gram_from_selling(selling).T)
    evals, evecs = np.linalg.eigh(gram)
    evals = np.clip(evals, eps, None)
    basis = evecs @ np.diag(np.sqrt(evals))
    cellpar = cellpar_from_direct_matrix(basis)
    return delaunay_standardize_cellpar(cellpar, hall_number=hall_number)


def selling_parameters_from_basis(basis: np.ndarray) -> np.ndarray:
    """
    Six Selling parameters for row-basis vectors ``b₁, b₂, b₃``.

    With ``b₄ = −(b₁ + b₂ + b₃)``,

        S_ij = −b_i · b_j   for 1 ≤ i < j ≤ 4

    Returned in lexicographic pair order (12, 13, 14, 23, 24, 34).
    Units are Å² (negated dot products of direct-lattice vectors).
    """
    m = np.asarray(basis, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) basis matrix, got {m.shape}")
    b1, b2, b3 = m
    b4 = -(b1 + b2 + b3)
    vecs = (b1, b2, b3, b4)
    return np.array(
        [-float(np.dot(vecs[i], vecs[j])) for i in range(4) for j in range(i + 1, 4)],
        dtype=np.float64,
    )


def delaunay_reduce_matrix(basis: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Delaunay-reduce a direct-lattice basis via spglib.

    Returns a unimodular-equivalent basis whose Selling scalars are canonical
    (all literature dot products ≤ 0, equivalently ``S_ij ≥ 0`` in our sign).
    """
    import spglib

    m = np.array(basis, dtype=np.float64, copy=True)
    reduced = spglib.delaunay_reduce(m, eps=eps)
    if reduced is None:
        raise RuntimeError("spglib.delaunay_reduce failed")
    return np.asarray(reduced, dtype=np.float64)


def delaunay_standardize_cellpar(
    cellpar: np.ndarray,
    hall_number: int | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Return an equivalent unit cell in its Delaunay-reduced setting.

    Applies ``spglib.delaunay_reduce`` to the direct basis, then optionally
    enforces Hall-number symmetry constraints on angles/lengths.
    """
    from cellnet.reciprocal import cellpar_from_direct_matrix, direct_matrix_from_cellpar
    from cellnet.symmetry import apply_cellpar_constraints

    m = delaunay_reduce_matrix(direct_matrix_from_cellpar(cellpar), eps=eps)
    cp = cellpar_from_direct_matrix(m)
    if hall_number is not None:
        cp = apply_cellpar_constraints(cp, hall_number)
    return cp


def selling_parameters(cellpar: np.ndarray, *, delaunay: bool = True, eps: float = 1e-5) -> np.ndarray:
    """
    Selling parameters for a unit cell.

    By default applies Delaunay reduction first so the result is an SL(3, ℤ)
    invariant of the lattice (up to the Selling-tetrahedron index permutation).
    """
    m = direct_matrix_from_cellpar(cellpar)
    if delaunay:
        m = delaunay_reduce_matrix(m, eps=eps)
    return selling_parameters_from_basis(m)


def lattice_invariants_from_cellpar(
    cellpar: np.ndarray,
    *,
    with_selling: bool = True,
    with_lambda: bool = True,
    with_lambda_recip: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    (Selling Å², log λ, log λ*) for one cell, sharing a single Delaunay reduction
    between the Selling parameters and the direct-space minima.

    Equivalent to calling ``selling_parameters``, ``log_successive_minima`` and
    ``log_reciprocal_successive_minima`` separately; used on the QRS hot path
    where every candidate needs all three.
    """
    from cellnet.reciprocal import reciprocal_matrix

    m = direct_matrix_from_cellpar(cellpar)
    selling = None
    log_lam = None
    log_lam_r = None
    if with_selling or with_lambda:
        reduced = delaunay_reduce_matrix(m)
        if with_selling:
            selling = selling_parameters_from_basis(reduced)
        if with_lambda:
            lam = _successive_minima_from_vectors(_selling_candidate_vectors(reduced))
            if lam is None:
                lam = _successive_minima_by_enumeration(reduced)
            log_lam = np.log(np.clip(lam, 1e-12, None))
    if with_lambda_recip:
        lam_r = successive_minima_from_matrix(reciprocal_matrix(m))
        log_lam_r = np.log(np.clip(lam_r, 1e-12, None))
    return selling, log_lam, log_lam_r


def selling_mse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative MSE between two 6-vectors of Selling parameters."""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(ref), np.abs(pred)), 1e-8)
    return float(np.mean(((pred - ref) / scale) ** 2))


# Selling pair index (i, j) for each of the 6 slots, matching SELLING_PAIR_LABELS.
_SELLING_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_SELLING_SLOT = {pair: k for k, pair in enumerate(_SELLING_PAIRS)}


def _selling_vertex_permutation_table() -> np.ndarray:
    """(24, 6) slot permutations induced by relabeling the four superbase vectors."""
    from itertools import permutations

    rows = []
    for perm in permutations(range(4)):
        rows.append(
            [_SELLING_SLOT[tuple(sorted((perm[i], perm[j])))] for i, j in _SELLING_PAIRS]
        )
    return np.array(rows, dtype=np.int64)


_SELLING_PERMS = _selling_vertex_permutation_table()


def canonical_selling_parameters(selling: np.ndarray) -> np.ndarray:
    """
    Relabeling-invariant canonical form of a Selling 6-vector.

    Returns the lexicographically smallest of the 24 vectors obtained by
    permuting the superbase vertices (b₁, b₂, b₃, b₄). Unlike
    ``sort_selling_parameters`` this is exactly invariant under any vertex
    relabeling, so equivalent Delaunay-reduced cells map to one vector.

    Not used for the shipped checkpoints' training targets: switching the
    target canonicalization requires retraining. Intended for the next model.
    """
    s = np.asarray(selling, dtype=np.float64).reshape(6)
    candidates = s[_SELLING_PERMS]  # (24, 6)
    order = np.lexsort(candidates.T[::-1])  # lexicographic on slot 0, then 1, ...
    return candidates[order[0]].copy()


def sort_selling_parameters(selling: np.ndarray) -> np.ndarray:
    """
    Legacy canonicalization used for the shipped checkpoints' Selling targets.

    Sorts slots 3:6 = (S₂₃, S₂₄, S₃₄) ascending and applies the same permutation
    to slots 0:3 = (S₁₂, S₁₃, S₁₄). This is *not* invariant under relabeling of
    the superbase vertices (the body terms are S₁₄, S₂₄, S₃₄ at slots 2, 4, 5,
    and a vertex permutation does not act slot-wise like this), so equivalent
    cells can receive different targets. Kept unchanged because the existing
    models were trained with it; see ``canonical_selling_parameters`` for the
    invariant form to use when retraining.
    """
    s = np.asarray(selling, dtype=np.float64).copy()
    body = s[3:6]
    order = np.argsort(body)
    cross = s[:3][order]
    body_sorted = body[order]
    return np.array([cross[0], cross[1], cross[2], body_sorted[0], body_sorted[1], body_sorted[2]])

