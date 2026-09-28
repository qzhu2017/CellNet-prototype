"""Density-implied volume constraint on the sampled successive minima."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cellnet.lattice_invariants import (
    LAMBDA_PRODUCT_LOWER_FRAC,
    LAMBDA_PRODUCT_UPPER_FRAC,
    constrain_log_lambda_batch,
)
from cellnet.packing import molecular_weight, volume_from_density
from cellnet.sequential import norm_log_lambda


@dataclass
class ConflictResolutionConfig:
    """Bounds for rescaling λ₁λ₂λ₃ to the density-implied cell volume."""

    lambda_volume_constraint: bool = True
    lambda_product_lower_frac: float = LAMBDA_PRODUCT_LOWER_FRAC
    lambda_product_upper_frac: float = LAMBDA_PRODUCT_UPPER_FRAC


def target_volume_from_density(
    target_rho: float,
    smiles: str,
    zprime: float,
    hall_number: int,
) -> float | None:
    """Volume (Å³) implied by predicted density and composition."""
    mw = molecular_weight(smiles)
    if mw is None or target_rho <= 0:
        return None
    return volume_from_density(target_rho, mw, zprime, hall_number)


def apply_lambda_volume_constraint(
    log_lambdas: np.ndarray,
    target_volume: float | None,
    config: ConflictResolutionConfig,
    stats=None,
) -> tuple[np.ndarray, np.ndarray | None, list[bool]]:
    """
    Project log(λ) targets onto [V, upper_frac·V] using density-implied volume.

    Returns (log_lambda_phys, log_lambda_norm, clipped_flags).
    """
    rows = np.asarray(log_lambdas, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    if not config.lambda_volume_constraint or target_volume is None or target_volume <= 0:
        norms = (
            np.stack([norm_log_lambda(rows[j], stats) for j in range(rows.shape[0])], axis=0)
            if stats is not None
            else None
        )
        return rows.copy(), norms, [False] * rows.shape[0]

    adjusted, clipped = constrain_log_lambda_batch(
        rows,
        target_volume,
        lower_frac=config.lambda_product_lower_frac,
        upper_frac=config.lambda_product_upper_frac,
    )
    norms = (
        np.stack([norm_log_lambda(adjusted[j], stats) for j in range(adjusted.shape[0])], axis=0)
        if stats is not None
        else None
    )
    return adjusted, norms, clipped
