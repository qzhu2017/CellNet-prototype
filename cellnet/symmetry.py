"""Crystal-system cell parameter constraints from Hall number."""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations

import numpy as np
import spglib


AXIS_PERMUTATIONS: tuple[tuple[int, int, int], ...] = tuple(permutations((0, 1, 2)))


def crystal_system_from_spg(spg_number: int) -> str:
    """Map international space-group number to crystal system."""
    if spg_number <= 2:
        return "triclinic"
    if spg_number <= 15:
        return "monoclinic"
    if spg_number <= 74:
        return "orthorhombic"
    if spg_number <= 142:
        return "tetragonal"
    if spg_number <= 167:
        return "trigonal"
    if spg_number <= 194:
        return "hexagonal"
    return "cubic"


@lru_cache(maxsize=None)
def crystal_system_from_hall(hall_number: int) -> str:
    """Derive crystal system from Hall number via spglib (pure int → str; cached)."""
    sg = spglib.get_spacegroup_type(int(hall_number))
    return crystal_system_from_spg(sg.number)


def apply_cellpar_constraints(cellpar: np.ndarray, hall_number: int) -> np.ndarray:
    """
    Enforce standard-setting cell constraints for the predicted Hall number.

    ASE convention for monoclinic: unique axis b, alpha = gamma = 90 deg.
    """
    cp = np.array(cellpar, dtype=np.float64, copy=True)
    system = crystal_system_from_hall(hall_number)

    if system == "monoclinic":
        cp[3] = 90.0
        cp[5] = 90.0
    elif system == "orthorhombic":
        cp[3:6] = 90.0
    elif system == "tetragonal":
        cp[0] = cp[1] = 0.5 * (cp[0] + cp[1])
        cp[3:6] = 90.0
    elif system in ("trigonal", "hexagonal"):
        cp[0] = cp[1] = 0.5 * (cp[0] + cp[1])
        cp[3] = 90.0
        cp[4] = 90.0
        cp[5] = 120.0
    elif system == "cubic":
        cp[0] = cp[1] = cp[2] = cp[:3].mean()
        cp[3:6] = 90.0

    return cp


def cellpar_to_free(cellpar: np.ndarray, hall_number: int) -> np.ndarray:
    """Encode cell parameters to symmetry-reduced optimization variables (log lengths)."""
    cp = apply_cellpar_constraints(cellpar, hall_number)
    system = crystal_system_from_hall(hall_number)

    if system == "triclinic":
        return np.array([np.log(cp[0]), np.log(cp[1]), np.log(cp[2]), cp[3], cp[4], cp[5]])
    if system == "monoclinic":
        return np.array([np.log(cp[0]), np.log(cp[1]), np.log(cp[2]), cp[4]])
    if system == "orthorhombic":
        return np.array([np.log(cp[0]), np.log(cp[1]), np.log(cp[2])])
    if system == "tetragonal":
        return np.array([np.log(cp[0]), np.log(cp[2])])
    if system in ("trigonal", "hexagonal"):
        return np.array([np.log(cp[0]), np.log(cp[2])])
    if system == "cubic":
        return np.array([np.log(cp[0])])
    raise ValueError(f"Unknown crystal system for hall {hall_number}: {system}")


def free_to_cellpar(free: np.ndarray, hall_number: int) -> np.ndarray:
    """Decode symmetry-reduced variables to full cell parameters."""
    system = crystal_system_from_hall(hall_number)

    if system == "triclinic":
        return np.array([np.exp(free[0]), np.exp(free[1]), np.exp(free[2]), free[3], free[4], free[5]])
    if system == "monoclinic":
        return np.array([np.exp(free[0]), np.exp(free[1]), np.exp(free[2]), 90.0, free[3], 90.0])
    if system == "orthorhombic":
        return np.array([np.exp(free[0]), np.exp(free[1]), np.exp(free[2]), 90.0, 90.0, 90.0])
    if system == "tetragonal":
        a = np.exp(free[0])
        return np.array([a, a, np.exp(free[1]), 90.0, 90.0, 90.0])
    if system in ("trigonal", "hexagonal"):
        a = np.exp(free[0])
        return np.array([a, a, np.exp(free[1]), 90.0, 90.0, 120.0])
    if system == "cubic":
        a = np.exp(free[0])
        return np.array([a, a, a, 90.0, 90.0, 90.0])
    raise ValueError(f"Unknown crystal system for hall {hall_number}: {system}")


def free_param_bounds(
    hall_number: int,
    min_length: float = 2.0,
    max_length: float = 60.0,
    min_angle: float = 30.0,
    max_angle: float = 150.0,
) -> list[tuple[float, float]]:
    """
    Box bounds for symmetry-reduced optimization variables.

    Lengths are optimized in log-space; angles in degrees. Matches PyXtal
    ``Lattice.get_bounds`` DOF layout for each crystal system.
    """
    system = crystal_system_from_hall(hall_number)
    log_bounds = (float(np.log(min_length)), float(np.log(max_length)))
    ang_bounds = (min_angle, max_angle)
    length_bounds = [log_bounds, log_bounds, log_bounds]

    if system == "triclinic":
        return length_bounds + [ang_bounds, ang_bounds, ang_bounds]
    if system == "monoclinic":
        return length_bounds + [ang_bounds]
    if system == "orthorhombic":
        return length_bounds
    if system == "tetragonal":
        return [log_bounds, log_bounds]
    if system in ("trigonal", "hexagonal"):
        return [log_bounds, log_bounds]
    if system == "cubic":
        return [log_bounds]
    raise ValueError(f"Unknown crystal system for hall {hall_number}: {system}")


def optimize_lattice_cellpar(
    cellpar: np.ndarray,
    hall_number: int,
    iterations: int = 5,
) -> np.ndarray:
    """
    Standardize and optimize inclination angles via PyXtal ``Lattice``.

    Equivalent to ``pyxtal.optimize_lattice`` on the lattice object: picks an
    equivalent unit-cell setting with well-conditioned angles (monoclinic β,
    triclinic α/β/γ) without changing the underlying Bravais lattice.
    """
    from pyxtal.lattice import Lattice

    cp = apply_cellpar_constraints(cellpar, hall_number)
    system = crystal_system_from_hall(hall_number)
    a, b, c, alpha, beta, gamma = cp
    try:
        lat = Lattice.from_para(a, b, c, alpha, beta, gamma, ltype=system, force_symmetry=True)
    except ValueError:
        return cp
    lat.standardize()
    lat, _ = lat.optimize_multi(iterations=iterations)
    return apply_cellpar_constraints(np.array(lat.get_para(degree=True), dtype=np.float64), hall_number)


def permute_cellpar(cellpar: np.ndarray, axis_perm: tuple[int, int, int]) -> np.ndarray:
    """Reorder ``(a,b,c)`` and ``(α,β,γ)`` with the same axis permutation."""
    cp = np.asarray(cellpar, dtype=np.float64)
    lengths = cp[:3][list(axis_perm)]
    angles = cp[3:6][list(axis_perm)]
    return np.array([lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2]])


def _sort_lengths_and_angles(cellpar: np.ndarray) -> np.ndarray:
    """Sort axis lengths ascending and permute angles with the same order."""
    cp = np.asarray(cellpar, dtype=np.float64)
    order = np.argsort(cp[:3])
    lengths = cp[:3][order]
    angles = cp[3:6][order]
    return np.array([lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2]])


def canonicalize_cellpar_for_comparison(cellpar: np.ndarray, hall_number: int) -> np.ndarray:
    """
    Put cell parameters in a canonical form before comparing pred vs true.

    Orthorhombic / triclinic: sort (a, b, c) low → high and permute (α, β, γ)
    with the same axis order.

    Monoclinic (unique axis b): sort only a and c (keep b fixed); map β to
    ``180° − β`` when β > 90°.
    """
    cp = apply_cellpar_constraints(cellpar, hall_number)
    system = crystal_system_from_hall(hall_number)

    if system in ("orthorhombic", "triclinic"):
        return _sort_lengths_and_angles(cp)

    if system == "monoclinic":
        a, b, c, alpha, beta, gamma = cp
        if a > c:
            a, c = c, a
        if beta > 90.0:
            beta = 180.0 - beta
        return np.array([a, b, c, 90.0, beta, 90.0], dtype=np.float64)

    return cp.copy()


def flip_cellpar_by_axis_signs(
    cellpar: np.ndarray,
    *,
    flip_a: bool = False,
    flip_b: bool = False,
    flip_c: bool = False,
) -> np.ndarray:
    """
    Reverse crystal-axis directions (same Bravais lattice, supplementary angles).

    α is the angle between **b** and **c**; β between **a** and **c**; γ between **a** and **b**.
    Reversing **a** flips β and γ; **b** flips α and γ; **c** flips α and β.
    """
    cp = np.asarray(cellpar, dtype=np.float64).copy()
    a, b, c, alpha, beta, gamma = cp
    if flip_a:
        beta = 180.0 - beta
        gamma = 180.0 - gamma
    if flip_b:
        alpha = 180.0 - alpha
        gamma = 180.0 - gamma
    if flip_c:
        alpha = 180.0 - alpha
        beta = 180.0 - beta
    return np.array([a, b, c, alpha, beta, gamma], dtype=np.float64)


def _axis_sign_flip_patterns() -> list[tuple[bool, bool, bool]]:
    return [
        (flip_a, flip_b, flip_c)
        for flip_a in (False, True)
        for flip_b in (False, True)
        for flip_c in (False, True)
    ]


def free_angle_indices(hall_number: int) -> list[int]:
    """Indices into cellpar (3=α, 4=β, 5=γ) that are free for this Hall number."""
    system = crystal_system_from_hall(hall_number)
    if system == "triclinic":
        return [3, 4, 5]
    if system == "monoclinic":
        return [4]
    return []


def equivalent_cellpar_settings(cellpar: np.ndarray, hall_number: int) -> list[np.ndarray]:
    """
    Equivalent ``(a,b,c,α,β,γ)`` descriptions in the standard crystal setting.

    Includes symmetry-respecting axis permutations, triclinic axis-direction flips
    (supplementary angles), and monoclinic ``β ↔ 180° − β``.
    """
    system = crystal_system_from_hall(hall_number)
    cp = np.asarray(cellpar, dtype=np.float64)
    out: list[np.ndarray] = []
    flip_patterns = _axis_sign_flip_patterns() if system == "triclinic" else [(False, False, False)]

    for perm in allowed_axis_permutations(hall_number):
        permuted = permute_cellpar(cp, perm)
        for flip_a, flip_b, flip_c in flip_patterns:
            cand = flip_cellpar_by_axis_signs(
                permuted, flip_a=flip_a, flip_b=flip_b, flip_c=flip_c
            )
            cand = apply_cellpar_constraints(cand, hall_number)
            out.append(cand)
            if system == "monoclinic" and abs(cand[4] - 90.0) > 1e-3:
                alt = cand.copy()
                alt[4] = 180.0 - alt[4]
                out.append(alt)
    return out


def angle_mae_deg(
    pred_cellpar: np.ndarray,
    true_cellpar: np.ndarray,
    hall_number: int,
    *,
    length_aligned_pred: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """
    Minimum mean absolute error on free angles over equivalent cell settings.

    Monoclinic: compares β with supplementary-angle equivalence (e.g. 86.5° vs 96.3°
    via canonical β ≤ 90° or ``180° − β``). Triclinic: also tries axis-direction
    flips that map each angle to its supplement.
    """
    ang_idx = free_angle_indices(hall_number)
    angle_errs = np.full(3, np.nan, dtype=np.float64)
    if not ang_idx:
        return 0.0, angle_errs

    true_can = canonicalize_cellpar_for_comparison(true_cellpar, hall_number)
    base = np.asarray(
        length_aligned_pred if length_aligned_pred is not None else pred_cellpar,
        dtype=np.float64,
    )

    best_mae = float("inf")
    for cand in equivalent_cellpar_settings(base, hall_number):
        cand_can = canonicalize_cellpar_for_comparison(cand, hall_number)
        errs = np.abs(cand_can[ang_idx] - true_can[ang_idx])
        mae = float(np.mean(errs))
        if mae < best_mae:
            best_mae = mae
            angle_errs[:] = np.nan
            for idx, err in zip(ang_idx, errs):
                angle_errs[idx - 3] = float(err)
    return best_mae, angle_errs


def _length_alignment_score(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Sum of relative length errors (lower is better)."""
    rel = np.abs(candidate[:3] - reference[:3]) / np.clip(reference[:3], 1e-6, None)
    return float(rel.sum())


def allowed_axis_permutations(hall_number: int) -> list[tuple[int, int, int]]:
    """
    Axis relabelings that preserve the standard-setting symmetry convention.

    Monoclinic (unique axis b): only swap a ↔ c — never move the unique axis.
    Tetragonal / trigonal / hexagonal: swap equivalent a ↔ b.
    Orthorhombic / triclinic: all permutations.
    """
    system = crystal_system_from_hall(hall_number)
    if system == "monoclinic":
        return [(0, 1, 2), (2, 1, 0)]
    if system in ("tetragonal", "trigonal", "hexagonal"):
        return [(0, 1, 2), (1, 0, 2)]
    if system == "cubic":
        return [(0, 1, 2)]
    if system == "orthorhombic":
        return list(permutations((0, 1, 2)))
    return list(permutations((0, 1, 2)))


def _symmetry_operation_key(
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    denominator: int = 48,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Hash a database symmetry operation using exact fractional translations."""
    rot_key = tuple(np.asarray(rotation, dtype=np.int64).reshape(-1).tolist())
    trans = np.rint((np.asarray(translation) % 1.0) * denominator).astype(np.int64)
    trans_key = tuple((trans % denominator).tolist())
    return rot_key, trans_key


@lru_cache(maxsize=512)
def hall_preserving_axis_permutations(hall_number: int) -> tuple[tuple[int, int, int], ...]:
    """
    Axis permutations that leave a Hall setting's full operation set unchanged.

    Unlike :func:`allowed_axis_permutations`, this distinguishes orthorhombic
    settings with a special axis.  For example, P2₁2₁2 and Pmm2 permit only
    ``a ↔ b``, while Fddd permits all six permutations.
    """
    symmetry = spglib.get_symmetry_from_database(int(hall_number))
    rotations = np.asarray(symmetry["rotations"], dtype=np.int64)
    translations = np.asarray(symmetry["translations"], dtype=np.float64)
    reference = {
        _symmetry_operation_key(rotation, translation)
        for rotation, translation in zip(rotations, translations)
    }

    allowed: list[tuple[int, int, int]] = []
    for perm in AXIS_PERMUTATIONS:
        transform = np.eye(3, dtype=np.int64)[list(perm)]
        inverse = transform.T
        transformed = {
            _symmetry_operation_key(
                transform @ rotation @ inverse,
                transform @ translation,
            )
            for rotation, translation in zip(rotations, translations)
        }
        if transformed == reference:
            allowed.append(perm)
    return tuple(allowed or [(0, 1, 2)])


def axis_rank_permutation(cellpar: np.ndarray) -> tuple[int, int, int]:
    """
    Return the ascending length rank assigned to conventional ``(a,b,c)``.

    For ``a < b < c`` this is ``(0,1,2)``; for ``c < a < b`` it is
    ``(1,2,0)``. Stable sorting makes equal lengths deterministic.
    """
    lengths = np.asarray(cellpar, dtype=np.float64)[:3]
    order = np.argsort(lengths, kind="stable")
    ranks = np.empty(3, dtype=np.int64)
    ranks[order] = np.arange(3, dtype=np.int64)
    return tuple(int(rank) for rank in ranks)


def equivalent_axis_rank_permutations(
    axis_ranks: tuple[int, int, int],
    hall_number: int,
) -> tuple[tuple[int, int, int], ...]:
    """Symmetry-equivalent conventional-axis rank assignments."""
    ranks = np.asarray(axis_ranks, dtype=np.int64)
    return tuple(
        sorted(
            {
                tuple(int(value) for value in ranks[list(perm)])
                for perm in hall_preserving_axis_permutations(hall_number)
            }
        )
    )


def canonical_axis_rank_permutation(
    axis_ranks: tuple[int, int, int],
    hall_number: int,
) -> tuple[int, int, int]:
    """Canonical representative of a Hall-equivalent axis-assignment class."""
    return equivalent_axis_rank_permutations(axis_ranks, hall_number)[0]


def axis_permutation_index(axis_ranks: tuple[int, int, int]) -> int:
    """Index an axis-rank permutation in the fixed six-class vocabulary."""
    return AXIS_PERMUTATIONS.index(tuple(int(value) for value in axis_ranks))


def axis_permutation_equivalence_mask(
    cellpar: np.ndarray,
    hall_number: int,
) -> np.ndarray:
    """Boolean six-class target mask for symmetry-aware classification loss."""
    mask = np.zeros(len(AXIS_PERMUTATIONS), dtype=bool)
    ranks = axis_rank_permutation(cellpar)
    for equivalent in equivalent_axis_rank_permutations(ranks, hall_number):
        mask[axis_permutation_index(equivalent)] = True
    return mask


def collapse_axis_permutation_probabilities(
    probabilities: np.ndarray,
    hall_number: int,
) -> np.ndarray:
    """Merge probabilities belonging to the same Hall-equivalent axis class."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(len(AXIS_PERMUTATIONS))
    probs = np.clip(probs, 0.0, None)
    collapsed = np.zeros_like(probs)
    for index, permutation in enumerate(AXIS_PERMUTATIONS):
        canonical = canonical_axis_rank_permutation(permutation, hall_number)
        collapsed[axis_permutation_index(canonical)] += probs[index]
    total = float(collapsed.sum())
    if total <= 0.0:
        collapsed[axis_permutation_index((0, 1, 2))] = 1.0
        return collapsed
    return collapsed / total


def select_axis_rank_permutation(
    probabilities: np.ndarray,
    hall_number: int,
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Return the most likely Hall-distinct axis assignment and class probabilities."""
    collapsed = collapse_axis_permutation_probabilities(probabilities, hall_number)
    selected = AXIS_PERMUTATIONS[int(np.argmax(collapsed))]
    return selected, collapsed


def align_orthorhombic_cellpar_to_axis_ranks(
    cellpar: np.ndarray,
    axis_ranks: tuple[int, int, int] | np.ndarray,
    hall_number: int,
) -> np.ndarray:
    """
    Assign sorted orthorhombic lengths to predicted conventional axes.

    This changes only the axis labels; invariant λ, λ*, Selling, volume, and
    density objectives are unchanged.
    """
    cp = apply_cellpar_constraints(cellpar, hall_number)
    if crystal_system_from_hall(hall_number) != "orthorhombic":
        return cp
    ranks = tuple(int(value) for value in np.asarray(axis_ranks).reshape(3))
    if ranks not in AXIS_PERMUTATIONS:
        raise ValueError(f"Invalid axis-rank permutation: {ranks}")
    lengths = np.sort(cp[:3], kind="stable")
    cp[:3] = lengths[list(ranks)]
    return cp


def align_cellpar_to_reference(
    pred_cellpar: np.ndarray,
    true_cellpar: np.ndarray,
    hall_number: int,
) -> np.ndarray:
    """
    Pick an equivalent axis labeling of ``pred`` that best matches ``true`` lengths.

    Uses ``equivalent_cellpar_settings`` (symmetry-respecting permutations,
    triclinic axis flips, monoclinic ``β ↔ 180° − β``).
    """
    true = np.asarray(true_cellpar, dtype=np.float64)
    best = np.asarray(pred_cellpar, dtype=np.float64)
    best_score = _length_alignment_score(
        apply_cellpar_constraints(best, hall_number), true
    )

    for cand in equivalent_cellpar_settings(pred_cellpar, hall_number):
        aligned = apply_cellpar_constraints(cand, hall_number)
        score = _length_alignment_score(aligned, true)
        if score < best_score:
            best_score = score
            best = aligned
    return best


@lru_cache(maxsize=512)
def general_position_multiplicity(hall_number: int) -> int:
    """
    Multiplicity of the general Wyckoff position for a spglib Hall number.

    Matches PyXtal's ``get_zprime`` convention: Z′ = Z / multiplicity.
    """
    from pyxtal.symmetry import Group

    group = Group(hall_number, use_hall=True, style="spglib")
    return len(group[0])


def zprime_to_Z(zprime: float, hall_number: int) -> float:
    """Convert Z′ (molecules per asymmetric unit) to Z (molecules per unit cell)."""
    return float(zprime) * general_position_multiplicity(hall_number)
