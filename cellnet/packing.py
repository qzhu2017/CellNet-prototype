"""Packing decomposition: volume + shape ratios + angles."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from cellnet.lattice_invariants import (
    LAMBDA_PRODUCT_LOWER_FRAC,
    LAMBDA_PRODUCT_UPPER_FRAC,
)

MAX_PREDICTED_DENSITY_RATIO_FACTOR = 4.0


def cell_volume(cellpar: np.ndarray) -> float:
    """Unit cell volume in Å³."""
    a, b, c, alpha, beta, gamma = cellpar
    ar, br, gr = np.radians([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos([ar, br, gr])
    term = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    return float(a * b * c * np.sqrt(max(0.0, term)))


from cellnet.symmetry import zprime_to_Z


def packing_density(
    cellpar: np.ndarray,
    mol_weight: float,
    zprime: float,
    hall_number: int,
) -> float:
    """Crystal density in g/cm³ using Z = Z′ × general-position multiplicity."""
    Z = zprime_to_Z(zprime, hall_number)
    volume_ang3 = cell_volume(cellpar)
    mass_g = mol_weight * Z / 6.02214076e23
    volume_cm3 = volume_ang3 * 1e-24
    if volume_cm3 <= 0:
        return 0.0
    return float(mass_g / volume_cm3)


def volume_from_density(
    rho: float,
    mol_weight: float,
    zprime: float,
    hall_number: int,
) -> float:
    """Unit cell volume (Å³) implied by crystal density."""
    if rho <= 0:
        return 0.0
    Z = zprime_to_Z(zprime, hall_number)
    mass_g = mol_weight * Z / 6.02214076e23
    volume_cm3 = mass_g / rho
    return float(volume_cm3 / 1e-24)


def cellpar_in_predicted_volume_range(
    cellpar: np.ndarray,
    target_volume: float | None,
    *,
    lower_frac: float = LAMBDA_PRODUCT_LOWER_FRAC,
    upper_frac: float = LAMBDA_PRODUCT_UPPER_FRAC,
    density_ratio_factor: float | None = None,
) -> bool:
    """True if cell volume lies in ``[lower_frac, upper_frac] ×`` density-implied V.

    Same window used to clip flow λ (Minkowski: V ≤ λ₁λ₂λ₃ ≤ (6/π)V). Tiny
    Selling reconstructions (high density) fail the lower bound.
    """
    if density_ratio_factor is not None:
        try:
            factor = float(density_ratio_factor)
        except (TypeError, ValueError):
            return False
        if (
            not np.isfinite(factor)
            or factor < 1.0
            or factor > MAX_PREDICTED_DENSITY_RATIO_FACTOR
        ):
            return False
        lower_frac = 1.0 / factor
        upper_frac = factor
    if target_volume is None:
        return density_ratio_factor is None
    try:
        target_volume = float(target_volume)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(target_volume) or target_volume <= 0:
        return False
    try:
        vol = cell_volume(np.asarray(cellpar, dtype=np.float64))
    except (TypeError, ValueError):
        return False
    if not np.isfinite(vol) or vol <= 0:
        return False
    lo = float(lower_frac) * float(target_volume)
    hi = float(upper_frac) * float(target_volume)
    return bool(
        (vol > lo or np.isclose(vol, lo, rtol=1e-12, atol=1e-12))
        and (vol < hi or np.isclose(vol, hi, rtol=1e-12, atol=1e-12))
    )


@lru_cache(maxsize=4096)
def molecular_weight(smiles: str) -> float | None:
    """RDKit molecular weight (g/mol); cached because QRS asks per stage per draw."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))
