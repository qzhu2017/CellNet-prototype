"""Joint (Selling, λ) pairing and adaptive QRS weights for hybrid cell recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cellnet.lattice_invariants import (
    LAMBDA_PRODUCT_LOWER_FRAC,
    LAMBDA_PRODUCT_UPPER_FRAC,
    constrain_log_lambda_batch,
    lambda_product_from_log,
)
from cellnet.packing import molecular_weight, volume_from_density
from cellnet.sequential import denorm_log_lambda, norm_log_lambda, LOG_LAMBDA_DIM


@dataclass
class ConflictResolutionConfig:
    """Controls joint target pairing and uncertainty-aware QRS weights."""

    enabled: bool = True
    joint_pairs: bool = True
    lambda_perturb_std: float = 0.2
    w_lambda_min_frac: float = 0.15
    uncertainty_gain: float = 2.0
    sample_gain: float = 1.5
    conflict_gain: float = 1.0
    boost_selling: bool = True
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


def selling_flow_uncertainty(selling_norm: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Estimate flow uncertainty from K Selling samples in normalized space.

    Returns (global_spread, per_sample_outlier_score) with shape (K,).
    """
    s = np.asarray(selling_norm, dtype=np.float64)
    if s.ndim == 1:
        return 0.0, np.zeros(1, dtype=np.float64)
    centroid = s.mean(axis=0)
    global_spread = float(np.mean(s.std(axis=0)))
    per_sample = np.linalg.norm(s - centroid, axis=1) / np.sqrt(max(s.shape[1], 1))
    return global_spread, per_sample


def sample_joint_log_lambda_pairs(
    log_lambda_norm: np.ndarray,
    stats,
    k: int,
    perturb_std_frac: float = 0.2,
    seed: int | None = None,
    target_volume: float | None = None,
    volume_constraint: ConflictResolutionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build K joint log(λ) targets by perturbing the model prediction in norm space.

    Returns (log_lambda_phys (K,3), log_lambda_norm (K,3)).
    """
    center = np.asarray(log_lambda_norm, dtype=np.float64).reshape(-1)
    if center.size != LOG_LAMBDA_DIM:
        raise ValueError(f"Expected {LOG_LAMBDA_DIM} log λ dims, got {center.size}")
    std = np.asarray(stats.log_lambda_std, dtype=np.float64)
    scale = float(perturb_std_frac) * std
    rng = np.random.default_rng(seed)
    norms = center + rng.normal(0.0, 1.0, size=(k, LOG_LAMBDA_DIM)) * scale
    phys = np.stack([denorm_log_lambda(norms[j], stats) for j in range(k)], axis=0)

    cfg = volume_constraint or ConflictResolutionConfig()
    phys, norms, _ = apply_lambda_volume_constraint(phys, target_volume, cfg, stats=stats)
    if norms is None:
        norms = np.stack([norm_log_lambda(phys[j], stats) for j in range(k)], axis=0)
    return phys, norms


def pair_conflict_scores(
    selling_norm: np.ndarray,
    log_lambda_norm: np.ndarray,
) -> np.ndarray:
    """
    Per-pair Selling vs λ scale mismatch in normalized space.

    Body Selling dims (3:6) and log λ are both length-scale proxies; large
    disagreement suggests conflicting QRS targets for that pair.
    """
    s = np.asarray(selling_norm, dtype=np.float64)
    lam = np.asarray(log_lambda_norm, dtype=np.float64)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    if lam.ndim == 1:
        lam = lam.reshape(1, -1)
    sell_scale = np.mean(s[:, 3:6], axis=1)
    lam_scale = np.mean(lam, axis=1)
    return np.abs(sell_scale - lam_scale)


def adaptive_hybrid_weights(
    base_w_selling: float,
    base_w_lambda: float,
    global_spread: float,
    sample_outlier: float,
    pair_conflict: float,
    config: ConflictResolutionConfig,
) -> tuple[float, float]:
    """Down-weight λ (and optionally up-weight Selling) when flow is uncertain."""
    if not config.enabled:
        return base_w_selling, base_w_lambda

    penalty = (
        1.0
        + config.uncertainty_gain * global_spread
        + config.sample_gain * float(sample_outlier)
        + config.conflict_gain * float(pair_conflict)
    )
    lam_frac = max(config.w_lambda_min_frac, 1.0 / penalty)
    w_lambda = base_w_lambda * lam_frac

    w_selling = base_w_selling
    if config.boost_selling:
        confidence = 1.0 / (1.0 + float(sample_outlier))
        w_selling = base_w_selling * (0.75 + 0.25 * confidence)

    return w_selling, w_lambda
