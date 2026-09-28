"""Quasi-random search (QRS) over symmetry-reduced cell parameters."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from cellnet.lattice_invariants import (
    delaunay_standardize_cellpar,
    lattice_invariants_from_cellpar,
    selling_mse as _selling_mse,
)
from cellnet.packing import molecular_weight, packing_density
from cellnet.sequential import compute_selling_parameters
from cellnet.symmetry import (
    apply_cellpar_constraints,
    cellpar_to_free,
    free_param_bounds,
    free_to_cellpar,
)


@dataclass
class QRSConfig:
    """Progressive quasi-random search settings."""

    n_stages: int = 12
    samples_per_stage: int = 1024
    top_k: int = 32
    shrink: float = 0.4
    min_span_frac: float = 0.02
    w_density: float = 2.0
    w_lambda: float = 1.0
    w_lambda_recip: float = 1.0
    w_selling: float = 1.0
    n_workers: int = 4
    max_log_density_dev: float = 0.25
    seed: int = 42
    evaluate_initial_candidate: bool = False


@dataclass
class QRSResult:
    cellpar: np.ndarray
    free: np.ndarray
    loss: float
    log_density_err: float
    density: float
    stage: int
    n_evaluated: int
    lambda_mse: float = 0.0
    lambda_recip_mse: float = 0.0
    selling_mse: float = 0.0
    n_density_rejected: int = 0
    n_id_evaluated: int = 0


def _log_density_error(rho: float, target_rho: float) -> float:
    if rho <= 0 or target_rho <= 0:
        return float("inf")
    return float(np.log(rho) - np.log(target_rho)) ** 2


def _log_density_deviation(rho: float, target_rho: float) -> float:
    if rho <= 0 or target_rho <= 0:
        return float("inf")
    return abs(float(np.log(rho) - np.log(target_rho)))


def _lambda_mse(lam_pred: np.ndarray, lam_ref: np.ndarray) -> float:
    return float(np.mean((lam_pred - lam_ref) ** 2))


def _decode_cellpar(free: np.ndarray, hall_number: int) -> np.ndarray | None:
    try:
        return apply_cellpar_constraints(free_to_cellpar(free, hall_number), hall_number)
    except ValueError:
        return None


def _evaluate_candidate(
    cellpar: np.ndarray,
    rho: float,
    target_rho: float,
    config: QRSConfig,
    target_log_lambdas: np.ndarray | None = None,
    target_log_lambdas_recip: np.ndarray | None = None,
    target_selling: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float, np.ndarray]:
    lam_loss = 0.0
    lam_recip_loss = 0.0
    sell_loss = 0.0

    pred_sell, pred_lam, pred_lam_r = lattice_invariants_from_cellpar(
        cellpar,
        with_selling=target_selling is not None,
        with_lambda=target_log_lambdas is not None,
        with_lambda_recip=target_log_lambdas_recip is not None,
    )
    if target_log_lambdas is not None:
        lam_loss = _lambda_mse(pred_lam, target_log_lambdas)
    if target_log_lambdas_recip is not None:
        lam_recip_loss = _lambda_mse(pred_lam_r, target_log_lambdas_recip)
    if target_selling is not None:
        sell_loss = _selling_mse(pred_sell, target_selling)

    d_loss = _log_density_error(rho, target_rho)
    total = config.w_density * d_loss
    if target_log_lambdas is not None:
        total += config.w_lambda * lam_loss
    if target_log_lambdas_recip is not None:
        total += config.w_lambda_recip * lam_recip_loss
    if target_selling is not None:
        total += config.w_selling * sell_loss
    return total, lam_loss, lam_recip_loss, sell_loss, d_loss, rho, cellpar


def _evaluate_batch(
    free_samples: np.ndarray,
    hall_number: int,
    smiles: str,
    zprime: float,
    target_rho: float,
    config: QRSConfig,
    target_log_lambdas: np.ndarray | None = None,
    target_log_lambdas_recip: np.ndarray | None = None,
    target_selling: np.ndarray | None = None,
) -> tuple[list[tuple[float, np.ndarray, float, float, float, float, float, np.ndarray]], int]:
    mw = molecular_weight(smiles)
    if mw is None:
        return [], len(free_samples)

    rows: list[tuple[float, np.ndarray, float, float, float, float, float, np.ndarray]] = []
    n_density_rejected = 0
    for free in free_samples:
        cellpar = _decode_cellpar(free, hall_number)
        if cellpar is None:
            n_density_rejected += 1
            continue
        rho = packing_density(cellpar, mw, zprime, hall_number)
        if rho <= 0 or _log_density_deviation(rho, target_rho) > config.max_log_density_dev:
            n_density_rejected += 1
            continue

        total, lam_loss, lam_recip_loss, sell_loss, d_loss, rho, cellpar = _evaluate_candidate(
            cellpar,
            rho,
            target_rho,
            config,
            target_log_lambdas=target_log_lambdas,
            target_log_lambdas_recip=target_log_lambdas_recip,
            target_selling=target_selling,
        )
        rows.append((total, free.copy(), lam_loss, lam_recip_loss, sell_loss, d_loss, rho, cellpar.copy()))
    return rows, n_density_rejected


def _evaluate_batch_worker(payload: tuple) -> tuple[list, int]:
    """ProcessPool worker entry point."""
    (
        free_samples,
        hall_number,
        smiles,
        zprime,
        target_rho,
        cfg_dict,
        target_log_lambdas,
        target_log_lambdas_recip,
        target_selling,
    ) = payload
    config = QRSConfig(**cfg_dict)
    return _evaluate_batch(
        free_samples,
        hall_number,
        smiles,
        zprime,
        target_rho,
        config,
        target_log_lambdas=target_log_lambdas,
        target_log_lambdas_recip=target_log_lambdas_recip,
        target_selling=target_selling,
    )


def _config_to_dict(cfg: QRSConfig) -> dict:
    return {
        "n_stages": cfg.n_stages,
        "samples_per_stage": cfg.samples_per_stage,
        "top_k": cfg.top_k,
        "shrink": cfg.shrink,
        "min_span_frac": cfg.min_span_frac,
        "w_density": cfg.w_density,
        "w_lambda": cfg.w_lambda,
        "w_lambda_recip": cfg.w_lambda_recip,
        "w_selling": cfg.w_selling,
        "n_workers": 1,
        "max_log_density_dev": cfg.max_log_density_dev,
        "seed": cfg.seed,
        "evaluate_initial_candidate": cfg.evaluate_initial_candidate,
    }


def _sample_unit(n: int, dim: int, seed: int, stage: int = 0) -> np.ndarray:
    """
    Scrambled Sobol points for one stage.

    The scramble is seeded from the pair (seed, stage) via ``SeedSequence`` so
    two stages never share a point set. The previous ``seed + stage`` scheme
    made draw j / stage s and draw j' / stage s' identical whenever
    j + s == j' + s' (callers seed draws as ``base + j``), so with K=96 draws
    and 12 stages nearly every point set was reused ~12 times.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stage)]))
    engine = qmc.Sobol(d=dim, scramble=True, seed=rng)
    m = int(np.ceil(np.log2(max(n, 1))))
    return engine.random_base2(m=m)[:n]


def _map_to_bounds(unit: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + unit * (hi - lo)


def _shrink_bounds(
    lo: np.ndarray,
    hi: np.ndarray,
    center: np.ndarray,
    shrink: float,
    global_lo: np.ndarray,
    global_hi: np.ndarray,
    min_span_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    span = hi - lo
    half = 0.5 * shrink * span
    new_lo = np.maximum(center - half, global_lo)
    new_hi = np.minimum(center + half, global_hi)
    min_span = min_span_frac * (global_hi - global_lo)
    for j in range(len(lo)):
        if new_hi[j] - new_lo[j] < min_span[j]:
            mid = 0.5 * (new_lo[j] + new_hi[j])
            new_lo[j] = max(global_lo[j], mid - 0.5 * min_span[j])
            new_hi[j] = min(global_hi[j], mid + 0.5 * min_span[j])
    return new_lo, new_hi


def qrs_search_cellpar(
    smiles: str,
    hall_number: int,
    zprime: float,
    target_rho: float,
    config: QRSConfig | None = None,
    target_log_lambdas: np.ndarray | None = None,
    target_log_lambdas_recip: np.ndarray | None = None,
    target_selling: np.ndarray | None = None,
    initial_cellpar: np.ndarray | None = None,
    verbose: bool = False,
) -> QRSResult:
    """
    Progressive Sobol QRS in symmetry-reduced coordinates.

    Match log(λ), log(λ*), Selling parameters, and/or density.
    Density is always enforced via pre-filter and loss term.
    """
    if (
        target_log_lambdas is None
        and target_log_lambdas_recip is None
        and target_selling is None
    ):
        raise ValueError("Provide at least one lattice target (λ, λ*, or Selling)")

    cfg = config or QRSConfig()
    global_bounds = np.array(free_param_bounds(hall_number), dtype=np.float64)
    global_lo, global_hi = global_bounds[:, 0], global_bounds[:, 1]
    lo, hi = global_lo.copy(), global_hi.copy()
    dim = len(global_lo)

    if initial_cellpar is not None:
        center = cellpar_to_free(initial_cellpar, hall_number)
        lo, hi = _shrink_bounds(lo, hi, center, 0.6, global_lo, global_hi, cfg.min_span_frac)

    best: QRSResult | None = None
    n_eval = 0
    n_density_rejected = 0
    n_id_evaluated = 0

    # A supplied seed is normally only used to center the first Sobol box.
    # For tightly constrained crystal systems, retaining the exact seed avoids
    # losing a physically good λ-derived cell to a broad first-stage search.
    if initial_cellpar is not None and cfg.evaluate_initial_candidate:
        seed_cellpar = _decode_cellpar(center, hall_number)
        if seed_cellpar is not None:
            mw = molecular_weight(smiles)
            rho = 0.0 if mw is None else packing_density(
                seed_cellpar, mw, zprime, hall_number
            )
            if (
                rho > 0
                and _log_density_deviation(rho, target_rho)
                <= cfg.max_log_density_dev
            ):
                total, lam_loss, lam_recip_loss, sell_loss, d_loss, rho, cellpar = (
                    _evaluate_candidate(
                        seed_cellpar,
                        rho,
                        target_rho,
                        cfg,
                        target_log_lambdas=target_log_lambdas,
                        target_log_lambdas_recip=target_log_lambdas_recip,
                        target_selling=target_selling,
                    )
                )
                best = QRSResult(
                    cellpar=cellpar,
                    free=center.copy(),
                    loss=total,
                    lambda_mse=lam_loss,
                    lambda_recip_mse=lam_recip_loss,
                    selling_mse=sell_loss,
                    log_density_err=d_loss,
                    density=rho,
                    stage=-1,
                    n_evaluated=1,
                )
                n_eval = 1
                n_id_evaluated = 1

    for stage in range(cfg.n_stages):
        unit = _sample_unit(cfg.samples_per_stage, dim, cfg.seed, stage)
        samples = _map_to_bounds(unit, lo, hi)

        stage_rows: list[tuple[float, np.ndarray, float, float, float, float, float, np.ndarray]] = []
        stage_rejected = 0
        if cfg.n_workers > 1:
            n_workers = min(cfg.n_workers, cfg.samples_per_stage)
            chunks = np.array_split(samples, n_workers)
            payloads = [
                (
                    chunk,
                    hall_number,
                    smiles,
                    zprime,
                    target_rho,
                    _config_to_dict(cfg),
                    target_log_lambdas,
                    target_log_lambdas_recip,
                    target_selling,
                )
                for chunk in chunks
                if len(chunk) > 0
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for part, rejected in pool.map(_evaluate_batch_worker, payloads):
                    stage_rows.extend(part)
                    stage_rejected += rejected
        else:
            stage_rows, stage_rejected = _evaluate_batch(
                samples,
                hall_number,
                smiles,
                zprime,
                target_rho,
                cfg,
                target_log_lambdas=target_log_lambdas,
                target_log_lambdas_recip=target_log_lambdas_recip,
                target_selling=target_selling,
            )

        n_density_rejected += stage_rejected
        n_id_evaluated += len(stage_rows)
        n_eval += len(stage_rows)

        if not stage_rows:
            continue

        stage_rows.sort(key=lambda row: row[0])
        total, free, lam_loss, lam_recip_loss, sell_loss, d_loss, rho, cellpar = stage_rows[0]
        candidate = QRSResult(
            cellpar=cellpar,
            free=free,
            loss=total,
            lambda_mse=lam_loss,
            lambda_recip_mse=lam_recip_loss,
            selling_mse=sell_loss,
            log_density_err=d_loss,
            density=rho,
            stage=stage,
            n_evaluated=n_eval,
        )
        if best is None or candidate.loss < best.loss:
            best = candidate

        if verbose:
            spec_msg = ""
            if target_log_lambdas is not None:
                spec_msg += f"λ={lam_loss:.6f} "
            if target_log_lambdas_recip is not None:
                spec_msg += f"λ*={lam_recip_loss:.6f} "
            if target_selling is not None:
                spec_msg += f"S={sell_loss:.6f} "
            print(
                f"  stage {stage + 1}/{cfg.n_stages}: best loss={total:.6f} "
                f"{spec_msg}ρ={rho:.3f} "
                f"(evals={len(stage_rows)}, density rejected={stage_rejected})",
                flush=True,
            )

        top = stage_rows[: min(cfg.top_k, len(stage_rows))]
        center = np.mean([row[1] for row in top], axis=0)
        lo, hi = _shrink_bounds(
            lo, hi, center, cfg.shrink, global_lo, global_hi, cfg.min_span_frac
        )

    if best is None:
        raise RuntimeError("QRS found no valid candidates; check bounds and inputs")
    best.n_density_rejected = n_density_rejected
    best.n_id_evaluated = n_id_evaluated
    if target_selling is not None:
        std_cp = delaunay_standardize_cellpar(best.cellpar, hall_number)
        if np.isfinite(std_cp).all():
            s_old = compute_selling_parameters(best.cellpar)
            s_new = compute_selling_parameters(std_cp)
            if _selling_mse(s_new, s_old) < 1e-4:
                best.cellpar = std_cp
                best.free = cellpar_to_free(std_cp, hall_number)
    return best
