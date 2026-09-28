"""Lattice-flow → lattice QRS → dedup → PyXtal conformational QRS pipeline."""

from __future__ import annotations

import warnings

# Noisy third-party import warnings (repeat per spawn worker)
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch_geometric\.typing")

import copy
import csv
import json
import multiprocessing as mp
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cellnet.hybrid_qrs import (
    ConflictResolutionConfig,
    apply_lambda_volume_constraint,
    target_volume_from_density,
)
from cellnet.packing import cellpar_in_predicted_volume_range
from cellnet.inference import (
    _prepare_batch,
    denorm_log_density,
    load_flow_qrs_model,
)
from cellnet.models import GivenHZConditionalLatticeFlowGNN
from cellnet.qrs import QRSConfig, _config_to_dict, qrs_search_cellpar
from cellnet.sequential import (
    compute_log_reciprocal_successive_minima,
    compute_log_successive_minima,
    denorm_log_lambda,
    denorm_log_reciprocal_lambda,
    denorm_selling_log1p,
    sample_k_given_hz_conditional_lattice_flow,
    split_lattice_flow_norm,
)
from cellnet.symmetry import (
    AXIS_PERMUTATIONS,
    align_orthorhombic_cellpar_to_axis_ranks,
    align_cellpar_to_reference,
    angle_mae_deg,
    apply_cellpar_constraints,
    canonicalize_cellpar_for_comparison,
    crystal_system_from_hall,
    optimize_lattice_cellpar,
    select_axis_rank_permutation,
    zprime_to_Z,
)


ROOT = Path(__file__).resolve().parents[1]

# Triclinic QRS seeds when no reference cell is available (α, β, γ in degrees).
_TRICLINIC_ANGLE_GUESSES: tuple[tuple[float, float, float], ...] = (
    (90.0, 90.0, 90.0),
    (85.0, 95.0, 90.0),
    (95.0, 85.0, 90.0),
    (100.0, 100.0, 90.0),
    (90.0, 100.0, 90.0),
    (100.0, 105.0, 90.0),
)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible flow sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flow_npz_path(out_dir: Path, tag: str, k: int) -> Path:
    return out_dir / f"{tag}_flow_k{k}.npz"


def save_flow_batch(
    flow: FlowSampleBatch,
    path: str | Path,
    *,
    seed: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        all_log_lambda=flow.all_log_lambda,
        all_log_lambda_recip=flow.all_log_lambda_recip,
        all_selling=flow.all_selling,
        target_rho=flow.target_rho,
        pred_log_density_norm=flow.pred_log_density_norm,
        hall_number=flow.hall_number,
        zprime=flow.zprime,
        smiles=flow.smiles,
        seed=np.int64(-1 if seed is None else seed),
        axis_rank_permutation=np.asarray(flow.axis_rank_permutation, dtype=np.int64),
        all_axis_rank_permutations=(
            flow.all_axis_rank_permutations
            if flow.all_axis_rank_permutations is not None
            else np.empty((0, 3), dtype=np.int64)
        ),
        axis_permutation_probabilities=(
            flow.axis_permutation_probabilities
            if flow.axis_permutation_probabilities is not None
            else np.empty((0,), dtype=np.float64)
        ),
        axis_permutation_source=flow.axis_permutation_source,
    )


def load_flow_batch(path: str | Path) -> FlowSampleBatch:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    smiles = str(data["smiles"])
    if hasattr(smiles, "item"):
        smiles = smiles.item()
    k = int(len(data["all_log_lambda"]))
    return FlowSampleBatch(
        smiles=smiles,
        hall_number=int(data["hall_number"]),
        zprime=float(data["zprime"]),
        target_rho=float(data["target_rho"]),
        pred_log_density_norm=float(data["pred_log_density_norm"])
        if "pred_log_density_norm" in data
        else float("nan"),
        all_selling=np.asarray(data["all_selling"]),
        all_log_lambda=np.asarray(data["all_log_lambda"]),
        all_log_lambda_recip=np.asarray(data["all_log_lambda_recip"]),
        k=k,
        axis_rank_permutation=(
            tuple(int(value) for value in data["axis_rank_permutation"])
            if "axis_rank_permutation" in data
            else (0, 1, 2)
        ),
        all_axis_rank_permutations=(
            np.asarray(data["all_axis_rank_permutations"], dtype=np.int64)
            if "all_axis_rank_permutations" in data
            and data["all_axis_rank_permutations"].size
            else None
        ),
        axis_permutation_probabilities=(
            np.asarray(data["axis_permutation_probabilities"], dtype=np.float64)
            if "axis_permutation_probabilities" in data
            and data["axis_permutation_probabilities"].size
            else None
        ),
        axis_permutation_source=(
            str(data["axis_permutation_source"].item())
            if "axis_permutation_source" in data
            else "legacy"
        ),
    )


@dataclass
class FlowSampleBatch:
    smiles: str
    hall_number: int
    zprime: float
    target_rho: float
    pred_log_density_norm: float
    all_selling: np.ndarray
    all_log_lambda: np.ndarray
    all_log_lambda_recip: np.ndarray
    k: int
    axis_rank_permutation: tuple[int, int, int] = (0, 1, 2)
    all_axis_rank_permutations: np.ndarray | None = None
    axis_permutation_probabilities: np.ndarray | None = None
    axis_permutation_source: str = "default"

    def axis_ranks_for_draw(self, flow_idx: int) -> tuple[int, int, int]:
        if self.all_axis_rank_permutations is not None:
            return tuple(
                int(value) for value in self.all_axis_rank_permutations[flow_idx]
            )
        return self.axis_rank_permutation


@dataclass
class LatticeQRSRecord:
    flow_idx: int
    cellpar: np.ndarray
    qrs_loss: float
    density: float
    lambda_mse: float
    lambda_recip_mse: float
    source: str = "qrs"


@dataclass
class LatticeQRSCellparMatch:
    flow_idx: int
    length_mape_pct: float
    mape_a_pct: float
    mape_b_pct: float
    mape_c_pct: float
    angle_mae_deg: float
    angle_errs_deg: np.ndarray
    aligned_cellpar: np.ndarray
    raw_cellpar: np.ndarray
    qrs_loss: float

    @property
    def beta_mae_deg(self) -> float:
        """Backward-compatible alias (monoclinic β error)."""
        return self.angle_mae_deg


def _format_angle_err_summary(match: LatticeQRSCellparMatch, hall_number: int) -> str:
    system = crystal_system_from_hall(hall_number)
    if system == "monoclinic":
        return f"β err={match.angle_mae_deg:.1f}°"
    if system == "triclinic":
        al, be, ga = match.angle_errs_deg
        return (
            f"angle MAE={match.angle_mae_deg:.1f}° "
            f"(α {al:.1f}°, β {be:.1f}°, γ {ga:.1f}°)"
        )
    return f"angle MAE={match.angle_mae_deg:.1f}°"


def _lattice_match_sort_key(
    match: LatticeQRSCellparMatch,
    hall_number: int,
) -> tuple[float, float, float]:
    """Rank lattice matches; triclinic weights angles equally with lengths."""
    if crystal_system_from_hall(hall_number) == "triclinic":
        return (
            match.length_mape_pct + match.angle_mae_deg,
            match.length_mape_pct,
            match.angle_mae_deg,
        )
    return (match.length_mape_pct, match.angle_mae_deg, 0.0)


@dataclass
class UniqueCellRecord:
    dedup_idx: int
    source_flow_indices: list[int]
    cellpar: np.ndarray
    sources: list[str] = field(default_factory=list)


@dataclass
class ConfQRSRecord:
    dedup_idx: int
    cellpar: list[float]
    success_rate: float | None
    time_min: float
    workdir: str
    n_conformers: int
    error: str | None = None
    restart: int = 1
    seed: int | None = None
    selection_channel: str | None = None
    reference_matching_enabled: bool = True
    relax_lattice: bool = False


@dataclass
class PipelineResult:
    tag: str
    smiles: str
    hall_number: int
    zprime: float
    true_cellpar: np.ndarray | None
    flow: FlowSampleBatch
    lattice_qrs: list[LatticeQRSRecord] = field(default_factory=list)
    unique_cells: list[UniqueCellRecord] = field(default_factory=list)
    conf_qrs: list[ConfQRSRecord] = field(default_factory=list)
    top_lattice_matches: list[LatticeQRSCellparMatch] = field(default_factory=list)
    # Populated only by --adaptive-relax runs; see summarize_adaptive_sweeps().
    adaptive_relax: dict | None = None


def _dedupe_cellpar_candidates(candidates: list[np.ndarray]) -> list[np.ndarray]:
    """Drop near-duplicate initial cells (same lengths/angles within tolerance)."""
    unique: list[np.ndarray] = []
    for candidate in candidates:
        cp = np.asarray(candidate, dtype=np.float64)
        if any(
            np.allclose(cp[:3], seen[:3], rtol=1e-4, atol=1e-3)
            and np.allclose(cp[3:], seen[3:], atol=0.25)
            for seen in unique
        ):
            continue
        unique.append(cp)
    return unique


def reference_lattice_match_score(
    cellpar: np.ndarray,
    true_cellpar: np.ndarray,
    hall_number: int,
) -> tuple[float, float, float]:
    """
    Return (length MAPE %, angle MAE °, combined score) vs a reference cell.

    Triclinic combined score weights angles equally with lengths because conf QRS
    is sensitive to both; other systems use length MAPE only.
    """
    true = np.asarray(true_cellpar, dtype=np.float64)
    raw = np.asarray(cellpar, dtype=np.float64)
    aligned = align_cellpar_to_reference(raw, true, hall_number)
    rel = np.abs(aligned[:3] - true[:3]) / np.clip(true[:3], 1e-6, None)
    length_mape = float(np.mean(rel) * 100.0)
    ang_mae, _ = angle_mae_deg(raw, true, hall_number, length_aligned_pred=aligned)
    system = crystal_system_from_hall(hall_number)
    if system == "triclinic":
        combined = length_mape + ang_mae
    else:
        combined = length_mape
    return length_mape, ang_mae, combined


# Public alias used by the pipeline runner and tests.
reference_match_score = reference_lattice_match_score


def init_cellpar_candidates_from_log_lambda(
    log_lambda: np.ndarray,
    hall_number: int,
    *,
    beta_guess: float = 95.0,
    axis_rank_permutation: tuple[int, int, int] | np.ndarray | None = None,
    reference_cellpar: np.ndarray | None = None,
    selling: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Build QRS initial cells from sorted successive minima λ₁ ≤ λ₂ ≤ λ₃.

    λ_k are not crystallographic axis lengths; assignments differ by crystal system.
    Monoclinic (unique axis b): fix b ← λ₁, try both orderings of λ₂/λ₃ on a/c.
    Orthorhombic: use the flow sample's predicted conventional-axis ranks.
    Triclinic: all axis permutations.

    When a flow Selling vector is provided, reconstruct a Delaunay cell and use
    its angles (target-free). ``reference_cellpar`` is optional benchmark-only
    and is not inserted as a QRS seed.
    """
    from itertools import permutations

    from cellnet.lattice_invariants import cellpar_from_selling

    lam = np.exp(np.asarray(log_lambda, dtype=np.float64).reshape(3))
    system = crystal_system_from_hall(hall_number)

    selling_cellpar: np.ndarray | None = None
    selling_angles: tuple[float, float, float] | None = None
    if selling is not None:
        try:
            selling_cellpar = cellpar_from_selling(selling, hall_number)
            selling_cellpar = _finalize_lattice_qrs_cellpar(
                selling_cellpar,
                hall_number,
                axis_rank_permutation,
            )
            if _cellpar_init_is_valid(selling_cellpar):
                selling_angles = (
                    float(selling_cellpar[3]),
                    float(selling_cellpar[4]),
                    float(selling_cellpar[5]),
                )
            else:
                selling_cellpar = None
        except Exception:
            selling_cellpar = None
            selling_angles = None

    ref_angles: tuple[float, float, float] | None = None
    if reference_cellpar is not None:
        ref = apply_cellpar_constraints(reference_cellpar, hall_number)
        try:
            ref = optimize_lattice_cellpar(ref, hall_number)
        except Exception:
            pass
        ref = apply_cellpar_constraints(ref, hall_number)
        ref_angles = (float(ref[3]), float(ref[4]), float(ref[5]))

    candidates: list[np.ndarray] = []
    if selling_cellpar is not None:
        candidates.append(selling_cellpar)

    if system == "monoclinic":
        beta = (
            selling_angles[1]
            if selling_angles is not None
            else (ref_angles[1] if ref_angles is not None else beta_guess)
        )
        candidates.extend(
            [
                np.array([lam[1], lam[0], lam[2], 90.0, beta, 90.0], dtype=np.float64),
                np.array([lam[2], lam[0], lam[1], 90.0, beta, 90.0], dtype=np.float64),
            ]
        )
        return _dedupe_cellpar_candidates(candidates)
    if system == "orthorhombic":
        ranks = (
            tuple(int(value) for value in np.asarray(axis_rank_permutation).reshape(3))
            if axis_rank_permutation is not None
            else (0, 1, 2)
        )
        if ranks not in AXIS_PERMUTATIONS:
            raise ValueError(f"Invalid axis-rank permutation: {ranks}")
        lengths = lam[list(ranks)]
        candidates.append(np.array([*lengths, 90.0, 90.0, 90.0], dtype=np.float64))
        return _dedupe_cellpar_candidates(candidates)
    if system == "triclinic":
        if selling_angles is not None:
            angle_guesses: tuple[tuple[float, float, float], ...] = (selling_angles,)
        elif ref_angles is not None:
            angle_guesses = (ref_angles,) + _TRICLINIC_ANGLE_GUESSES
        else:
            angle_guesses = _TRICLINIC_ANGLE_GUESSES
        candidates.extend(
            np.array([lam[i], lam[j], lam[k], alpha, beta, gamma], dtype=np.float64)
            for i, j, k in permutations((0, 1, 2))
            for alpha, beta, gamma in angle_guesses
        )
        return _dedupe_cellpar_candidates(candidates)
    if system == "tetragonal":
        a = float(np.sqrt(lam[0] * lam[1]))
        candidates.append(np.array([a, a, lam[2], 90.0, 90.0, 90.0], dtype=np.float64))
        return _dedupe_cellpar_candidates(candidates)
    if system in ("trigonal", "hexagonal"):
        a = float(np.sqrt(lam[0] * lam[1]))
        candidates.append(np.array([a, a, lam[2], 90.0, 90.0, 120.0], dtype=np.float64))
        return _dedupe_cellpar_candidates(candidates)
    if system == "cubic":
        a = float(lam[0] ** (1.0 / 3.0) * (6.0 / np.pi) ** (1.0 / 3.0))
        candidates.append(np.array([a, a, a, 90.0, 90.0, 90.0], dtype=np.float64))
        return _dedupe_cellpar_candidates(candidates)
    candidates.append(np.array([lam[0], lam[1], lam[2], 90.0, 90.0, 90.0], dtype=np.float64))
    return _dedupe_cellpar_candidates(candidates)


def init_cellpar_from_log_lambda(log_lambda: np.ndarray, hall_number: int) -> np.ndarray:
    """Return the first (primary) λ → cellpar initialization candidate."""
    return init_cellpar_candidates_from_log_lambda(log_lambda, hall_number)[0]


def _finalize_lattice_qrs_cellpar(
    cellpar: np.ndarray,
    hall_number: int,
    axis_rank_permutation: tuple[int, int, int] | np.ndarray | None = None,
) -> np.ndarray:
    cp = apply_cellpar_constraints(cellpar, hall_number)
    try:
        cp = optimize_lattice_cellpar(cp, hall_number)
    except Exception:
        pass
    cp = apply_cellpar_constraints(cp, hall_number)
    if axis_rank_permutation is not None:
        cp = align_orthorhombic_cellpar_to_axis_ranks(
            cp,
            axis_rank_permutation,
            hall_number,
        )
    return cp


def _cellpar_init_is_valid(cellpar: np.ndarray) -> bool:
    """Reject non-finite or geometrically illegal QRS seeds."""
    cp = np.asarray(cellpar, dtype=np.float64).reshape(-1)
    if cp.size != 6 or not np.all(np.isfinite(cp)):
        return False
    if np.any(cp[:3] < 1.5) or np.any(cp[:3] > 80.0):
        return False
    if np.any(cp[3:] < 35.0) or np.any(cp[3:] > 145.0):
        return False
    return True


def _search_lattice_qrs_one(
    flow_idx: int,
    *,
    smiles: str,
    hall_number: int,
    zprime: float,
    target_rho: float,
    log_lambda: np.ndarray,
    log_lambda_recip: np.ndarray,
    target_selling: np.ndarray | None,
    cfg: QRSConfig,
    use_selling: bool,
    axis_rank_permutation: tuple[int, int, int] | np.ndarray | None = None,
    reference_cellpar: np.ndarray | None = None,
    verbose: bool = False,
) -> LatticeQRSRecord:
    effective_selling = target_selling if use_selling else None
    inits = [
        init
        for init in init_cellpar_candidates_from_log_lambda(
            log_lambda,
            hall_number,
            axis_rank_permutation=axis_rank_permutation,
            reference_cellpar=reference_cellpar,
            selling=effective_selling,
        )
        if _cellpar_init_is_valid(init)
    ]
    if not inits:
        inits = [
            init
            for init in init_cellpar_candidates_from_log_lambda(
                log_lambda,
                hall_number,
                axis_rank_permutation=axis_rank_permutation,
            )
            if _cellpar_init_is_valid(init)
        ]
    best_qrs = None
    last_error: Exception | None = None
    for init_idx, init in enumerate(inits):
        try:
            qrs = qrs_search_cellpar(
                smiles=smiles,
                hall_number=hall_number,
                zprime=zprime,
                target_rho=target_rho,
                config=cfg,
                target_selling=effective_selling,
                target_log_lambdas=log_lambda,
                target_log_lambdas_recip=log_lambda_recip,
                initial_cellpar=init,
                verbose=verbose and init_idx == 0,
            )
        except Exception as exc:
            last_error = exc
            continue
        if best_qrs is None or qrs.loss < best_qrs.loss:
            best_qrs = qrs

    if best_qrs is None:
        raise RuntimeError(
            f"QRS found no valid candidates for flow {flow_idx}"
            + (f": {last_error}" if last_error is not None else "")
        )
    cellpar = _finalize_lattice_qrs_cellpar(
        best_qrs.cellpar,
        hall_number,
        axis_rank_permutation,
    )
    pred_lam = compute_log_successive_minima(cellpar)
    pred_lam_r = compute_log_reciprocal_successive_minima(cellpar)
    lam_mse = float(np.mean((pred_lam - log_lambda) ** 2))
    lam_r_mse = float(np.mean((pred_lam_r - log_lambda_recip) ** 2))
    return LatticeQRSRecord(
        flow_idx=flow_idx,
        cellpar=cellpar,
        qrs_loss=float(best_qrs.loss),
        density=float(best_qrs.density),
        lambda_mse=lam_mse,
        lambda_recip_mse=lam_r_mse,
        source="qrs",
    )


@torch.no_grad()
def sample_flow_batch(
    checkpoint: str | Path,
    smiles: str,
    hall_number: int,
    zprime: float,
    k: int = 100,
    device: torch.device | None = None,
    seed: int | None = None,
) -> FlowSampleBatch:
    if seed is not None:
        set_random_seed(seed)

    model, stats, model_args, kind = load_flow_qrs_model(checkpoint, device)
    if kind != "lattice_flow":
        raise ValueError(f"Expected lattice_flow checkpoint, got {kind}")

    dev = device or next(model.parameters()).device
    batch, hall_t, zprime_t = _prepare_batch(stats, smiles, hall_number, zprime, dev)
    samples_norm, pred_ld = sample_k_given_hz_conditional_lattice_flow(
        model, batch, hall_t, zprime_t, k=k
    )

    pred_ld_norm = float(pred_ld.squeeze(0).cpu().numpy())
    target_rho = float(denorm_log_density(pred_ld_norm, stats))
    flow_norm = samples_norm.squeeze(0).cpu().numpy()
    pred_s_norm, pred_l_norm, pred_lr_norm = split_lattice_flow_norm(flow_norm, stats)
    all_selling = np.stack(
        [denorm_selling_log1p(pred_s_norm[j], stats) for j in range(k)], axis=0
    )
    all_log_lambda = denorm_log_lambda(pred_l_norm, stats)
    all_log_lambda_recip = denorm_log_reciprocal_lambda(pred_lr_norm, stats)

    cr_cfg = ConflictResolutionConfig(lambda_volume_constraint=True)
    target_volume = target_volume_from_density(target_rho, smiles, zprime, hall_number)
    all_log_lambda, _, _ = apply_lambda_volume_constraint(
        all_log_lambda, target_volume, cr_cfg, stats=stats
    )

    axis_logits = getattr(model, "_last_axis_permutation_logits", None)
    if axis_logits is not None:
        raw_axis_probs = torch.softmax(axis_logits, dim=-1).squeeze(0).cpu().numpy()
        axis_source = "neural"
    else:
        prior = np.asarray(
            stats.axis_permutation_priors.get(hall_number, []),
            dtype=np.float64,
        )
        if prior.size == len(AXIS_PERMUTATIONS):
            axis_source = "hall_prior"
        else:
            prior = np.zeros(len(AXIS_PERMUTATIONS), dtype=np.float64)
            prior[0] = 1.0
            axis_source = "default"
        raw_axis_probs = np.repeat(prior[np.newaxis, :], k, axis=0)
    selected_axis = [
        select_axis_rank_permutation(probabilities, hall_number)
        for probabilities in raw_axis_probs
    ]
    all_axis_ranks = np.asarray(
        [ranks for ranks, _ in selected_axis],
        dtype=np.int64,
    )
    axis_probs = np.stack(
        [probabilities for _, probabilities in selected_axis],
        axis=0,
    )
    axis_ranks = tuple(int(value) for value in all_axis_ranks[0])
    return FlowSampleBatch(
        smiles=smiles,
        hall_number=hall_number,
        zprime=zprime,
        target_rho=target_rho,
        pred_log_density_norm=pred_ld_norm,
        all_selling=all_selling,
        all_log_lambda=all_log_lambda,
        all_log_lambda_recip=all_log_lambda_recip,
        k=k,
        axis_rank_permutation=axis_ranks,
        all_axis_rank_permutations=all_axis_ranks,
        axis_permutation_probabilities=axis_probs,
        axis_permutation_source=axis_source,
    )


def _lattice_qrs_record(
    flow_idx: int,
    flow: FlowSampleBatch,
    cfg: QRSConfig,
    *,
    use_selling: bool,
    reference_cellpar: np.ndarray | None = None,
    verbose: bool = False,
) -> LatticeQRSRecord:
    return _search_lattice_qrs_one(
        flow_idx,
        smiles=flow.smiles,
        hall_number=flow.hall_number,
        zprime=flow.zprime,
        target_rho=flow.target_rho,
        log_lambda=flow.all_log_lambda[flow_idx],
        log_lambda_recip=flow.all_log_lambda_recip[flow_idx],
        target_selling=flow.all_selling[flow_idx],
        cfg=cfg,
        use_selling=use_selling,
        axis_rank_permutation=flow.axis_ranks_for_draw(flow_idx),
        reference_cellpar=reference_cellpar,
        verbose=verbose,
    )


def _lattice_qrs_worker(payload: tuple) -> LatticeQRSRecord:
    """ProcessPool entry: one lattice QRS run for a single flow draw."""
    (
        flow_idx,
        smiles,
        hall_number,
        zprime,
        target_rho,
        cfg_dict,
        use_selling,
        target_selling,
        target_log_lambda,
        target_log_lambda_recip,
        axis_rank_permutation,
        reference_cellpar,
    ) = payload
    cfg = QRSConfig(**cfg_dict)
    cfg.n_workers = 1
    ref = None if reference_cellpar is None else np.asarray(reference_cellpar, dtype=np.float64)
    return _search_lattice_qrs_one(
        flow_idx,
        smiles=smiles,
        hall_number=hall_number,
        zprime=zprime,
        target_rho=target_rho,
        log_lambda=target_log_lambda,
        log_lambda_recip=target_log_lambda_recip,
        target_selling=target_selling,
        cfg=cfg,
        use_selling=use_selling,
        axis_rank_permutation=axis_rank_permutation,
        reference_cellpar=ref,
        verbose=False,
    )


def rank_lattice_qrs_by_reference(
    records: list[LatticeQRSRecord],
    true_cellpar: np.ndarray,
    hall_number: int,
) -> list[LatticeQRSCellparMatch]:
    """Rank lattice QRS cells by axis-aligned length MAPE vs a reference cell."""
    true = np.asarray(true_cellpar, dtype=np.float64)
    matches: list[LatticeQRSCellparMatch] = []
    for rec in records:
        raw = np.asarray(rec.cellpar, dtype=np.float64)
        aligned = align_cellpar_to_reference(raw, true, hall_number)
        rel = np.abs(aligned[:3] - true[:3]) / np.clip(true[:3], 1e-6, None)
        mapes = rel * 100.0
        ang_mae, ang_errs = angle_mae_deg(raw, true, hall_number, length_aligned_pred=aligned)
        matches.append(
            LatticeQRSCellparMatch(
                flow_idx=rec.flow_idx,
                length_mape_pct=float(np.mean(mapes)),
                mape_a_pct=float(mapes[0]),
                mape_b_pct=float(mapes[1]),
                mape_c_pct=float(mapes[2]),
                angle_mae_deg=ang_mae,
                angle_errs_deg=ang_errs,
                aligned_cellpar=aligned,
                raw_cellpar=raw,
                qrs_loss=rec.qrs_loss,
            )
        )
    matches.sort(key=lambda m: _lattice_match_sort_key(m, hall_number))
    return matches


def print_top_lattice_qrs_cellpar_matches(
    records: list[LatticeQRSRecord],
    true_cellpar: np.ndarray,
    hall_number: int,
    *,
    top_n: int = 5,
) -> list[LatticeQRSCellparMatch]:
    """Print and return the best lattice QRS matches vs a known reference cell."""
    true = np.asarray(true_cellpar, dtype=np.float64)
    ranked = rank_lattice_qrs_by_reference(records, true, hall_number)
    n = min(top_n, len(ranked))
    if n == 0:
        return ranked

    print(
        f"\n=== Top {n} lattice QRS vs reference "
        f"(a={true[0]:.3f} b={true[1]:.3f} c={true[2]:.3f} "
        f"β={true[4]:.1f}°, symmetry-respecting length MAPE) ===",
        flush=True,
    )
    for rank, match in enumerate(ranked[:n], start=1):
        cp = match.aligned_cellpar
        print(
            f"  #{rank} flow {match.flow_idx + 1}: length MAPE={match.length_mape_pct:.2f}% "
            f"(a {match.mape_a_pct:.1f}%, b {match.mape_b_pct:.1f}%, c {match.mape_c_pct:.1f}%) "
            f"{_format_angle_err_summary(match, hall_number)}  QRS loss={match.qrs_loss:.5f}",
            flush=True,
        )
        print(
            f"      aligned a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f} β={cp[4]:.1f}°",
            flush=True,
        )
    return ranked[:n]


def _print_lattice_qrs_record(rec: LatticeQRSRecord, k: int) -> None:
    cp = rec.cellpar
    lam = np.exp(compute_log_successive_minima(cp))
    print(
        f"  lattice QRS {rec.flow_idx + 1}/{k}: loss={rec.qrs_loss:.5f} "
        f"λ=({lam[0]:.2f},{lam[1]:.2f},{lam[2]:.2f}) "
        f"a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f} "
        f"β={cp[4]:.1f} λMSE={rec.lambda_mse:.4f}",
        flush=True,
    )


def _lattice_qrs_candidate_indices(
    all_log_lambda: np.ndarray,
    max_lambda_ratio: float | None,
) -> tuple[list[int], list[tuple[int, float]]]:
    """Select QRS draws, excluding pathological successive-minima ratios."""
    n_draws = len(all_log_lambda)
    if max_lambda_ratio is None or max_lambda_ratio <= 0:
        return list(range(n_draws)), []

    lam = np.exp(np.asarray(all_log_lambda, dtype=np.float64))
    ratios = lam.max(axis=1) / np.clip(lam.min(axis=1), 1e-12, None)
    selected = [i for i, ratio in enumerate(ratios) if ratio <= max_lambda_ratio]
    skipped = [
        (i, float(ratio))
        for i, ratio in enumerate(ratios)
        if ratio > max_lambda_ratio
    ]
    return selected, skipped


def run_lattice_qrs_batch(
    flow: FlowSampleBatch,
    *,
    qrs_config: QRSConfig | None = None,
    use_selling: bool = False,
    qrs_seed: int = 42,
    n_proc: int = 1,
    max_lambda_ratio: float | None = None,
    reference_cellpar: np.ndarray | None = None,
    verbose: bool = False,
) -> list[LatticeQRSRecord]:
    cfg = copy.deepcopy(qrs_config) if qrs_config is not None else QRSConfig(
        w_selling=0.0,
        w_lambda=2.0,
        w_lambda_recip=2.0,
        w_density=2.0,
        seed=qrs_seed,
        n_workers=1,
    )
    if use_selling:
        cfg.w_selling = 2.0
    system = crystal_system_from_hall(flow.hall_number)
    if system in ("trigonal", "hexagonal"):
        # In two-DOF hexagonal settings, λ already maps directly to
        # a=sqrt(λ1 λ2), c=λ3. Selling and a full-strength reciprocal-λ
        # objective can drag that good seed into a very elongated cell.
        use_selling = False
        cfg.w_selling = 0.0
        cfg.w_lambda_recip = min(cfg.w_lambda_recip, 0.5)
        cfg.evaluate_initial_candidate = True
        print(
            "  trigonal/hexagonal QRS policy: direct λ seed retained, "
            "Selling disabled, w_λ*=0.5",
            flush=True,
        )
    if n_proc > 1:
        cfg.n_workers = 1

    candidate_indices, skipped = _lattice_qrs_candidate_indices(
        flow.all_log_lambda,
        max_lambda_ratio,
    )
    if skipped:
        skipped_summary = ", ".join(
            f"{idx + 1} ({ratio:.2f}x)" for idx, ratio in skipped
        )
        print(
            f"  skipped {len(skipped)}/{flow.k} extreme lattice QRS draws "
            f"(λmax/λmin > {max_lambda_ratio:g}): {skipped_summary}",
            flush=True,
        )
    if not candidate_indices:
        raise ValueError(
            "All lattice QRS draws exceeded the configured λmax/λmin cutoff"
        )

    if n_proc <= 1:
        records: list[LatticeQRSRecord] = []
        for j in candidate_indices:
            draw_cfg = replace(cfg, seed=qrs_seed + j)
            try:
                rec = _lattice_qrs_record(
                    j,
                    flow,
                    draw_cfg,
                    use_selling=use_selling,
                    reference_cellpar=reference_cellpar,
                    verbose=verbose and j == 0,
                )
            except Exception as exc:
                print(
                    f"  skipped lattice QRS draw {j + 1}/{flow.k}: {exc}",
                    flush=True,
                )
                continue
            records.append(rec)
            if verbose:
                _print_lattice_qrs_record(rec, flow.k)
        if not records:
            raise RuntimeError("All lattice QRS draws failed")
        return records

    cfg_dict = _config_to_dict(cfg)
    ref_payload = (
        None
        if reference_cellpar is None
        else np.asarray(reference_cellpar, dtype=np.float64).tolist()
    )
    payloads = [
        (
            j,
            flow.smiles,
            flow.hall_number,
            flow.zprime,
            flow.target_rho,
            {**cfg_dict, "seed": qrs_seed + j},
            use_selling,
            flow.all_selling[j],
            flow.all_log_lambda[j],
            flow.all_log_lambda_recip[j],
            flow.axis_ranks_for_draw(j),
            ref_payload,
        )
        for j in candidate_indices
    ]
    workers = min(n_proc, len(candidate_indices))
    records_by_idx: list[LatticeQRSRecord | None] = [None] * flow.k
    # spawn avoids fork-after-CUDA hangs when step 1 used the GPU in the parent process
    mp_ctx = mp.get_context("spawn")
    print(f"  parallel lattice QRS: {workers} workers (spawn)", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
        futures = {pool.submit(_lattice_qrs_worker, payload): payload[0] for payload in payloads}
        failed: list[tuple[int, str]] = []
        for fut in as_completed(futures):
            flow_idx = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                failed.append((flow_idx, str(exc)))
                print(
                    f"  skipped lattice QRS draw {flow_idx + 1}/{flow.k}: {exc}",
                    flush=True,
                )
                continue
            records_by_idx[rec.flow_idx] = rec
            if verbose:
                _print_lattice_qrs_record(rec, flow.k)
    n_ok = sum(1 for i in candidate_indices if records_by_idx[i] is not None)
    print(
        f"  parallel lattice QRS: {n_ok}/{len(candidate_indices)} draws finished"
        + (f" ({len(skipped)} ratio-skipped)" if skipped else "")
        + (f" ({len(failed)} QRS-failed)" if failed else ""),
        flush=True,
    )
    if n_ok == 0:
        raise RuntimeError("All lattice QRS draws failed")
    return [
        records_by_idx[i]
        for i in candidate_indices
        if records_by_idx[i] is not None
    ]


def deduplicate_cellpars(
    records: list[LatticeQRSRecord],
    hall_number: int,
    *,
    length_rtol: float = 0.02,
    length_atol: float = 0.15,
    angle_atol: float = 2.0,
    volume_rtol: float = 0.02,
) -> list[UniqueCellRecord]:
    from cellnet.packing import cell_volume

    unique: list[UniqueCellRecord] = []

    for rec in records:
        cp_a = canonicalize_cellpar_for_comparison(rec.cellpar, hall_number)
        vol = float(cell_volume(cp_a))
        matched: UniqueCellRecord | None = None
        for u in unique:
            u_a = canonicalize_cellpar_for_comparison(u.cellpar, hall_number)
            u_vol = float(cell_volume(u_a))
            if abs(vol - u_vol) / max(vol, u_vol, 1e-6) > volume_rtol:
                continue
            if not np.allclose(cp_a[:3], u_a[:3], rtol=length_rtol, atol=length_atol):
                continue
            if not np.allclose(cp_a[3:], u_a[3:], atol=angle_atol):
                continue
            matched = u
            break
        if matched is None:
            unique.append(
                UniqueCellRecord(
                    dedup_idx=len(unique),
                    source_flow_indices=[rec.flow_idx],
                    cellpar=rec.cellpar.copy(),
                    sources=[getattr(rec, "source", "qrs")],
                )
            )
        else:
            matched.source_flow_indices.append(rec.flow_idx)
            matched.sources.append(getattr(rec, "source", "qrs"))

    return unique


def lattice_records_from_selling(
    flow: FlowSampleBatch,
    *,
    density_ratio_factor: float | None = None,
) -> list[LatticeQRSRecord]:
    """Delaunay reconstruction of each draw's Selling vector.

    Volume must lie in the same density-implied window used to clip flow λ.
    """
    from cellnet.lattice_invariants import cellpar_from_selling, successive_minima

    target_volume = target_volume_from_density(
        flow.target_rho, flow.smiles, flow.zprime, flow.hall_number
    )
    records: list[LatticeQRSRecord] = []
    skipped = 0
    for i in range(flow.k):
        try:
            cellpar = cellpar_from_selling(flow.all_selling[i], flow.hall_number)
            cellpar = _finalize_lattice_qrs_cellpar(
                cellpar,
                flow.hall_number,
                flow.axis_ranks_for_draw(i),
            )
            if not _cellpar_init_is_valid(cellpar):
                continue
            rec_lam = successive_minima(cellpar)
            log_rec = np.log(np.clip(rec_lam, 1e-12, None))
            lam_mse = float(np.mean((log_rec - flow.all_log_lambda[i]) ** 2))
        except Exception:
            continue
        cellpar = np.asarray(cellpar, dtype=np.float64)
        if not cellpar_in_predicted_volume_range(
            cellpar,
            target_volume,
            density_ratio_factor=density_ratio_factor,
        ):
            skipped += 1
            continue
        records.append(
            LatticeQRSRecord(
                flow_idx=i,
                cellpar=cellpar,
                qrs_loss=float("inf"),
                density=float(flow.target_rho),
                lambda_mse=lam_mse,
                lambda_recip_mse=0.0,
                source="selling",
            )
        )
    if skipped:
        print(
            f"  skipped {skipped} Selling reconstructions outside density volume range "
            f"(predicted ρ={flow.target_rho:.4f} g/cm³, V={target_volume:.0f} Å³)",
            flush=True,
        )
    return records


# Source tag for cells derived by moving the monoclinic unique axis b onto
# another edge of a lattice-QRS cell (see ``monoclinic_unique_axis_alternates``).
AXIS_ALTERNATE_SOURCE = "qrs_axis_alt"

# Which of the three cell edges carries the monoclinic unique axis b in the
# SPADE training split (73,852 monoclinic cells, unique axis b, measured
# 2026-09-12): shortest 45.6 %, middle 24.2 %, longest 30.2 %. The lattice-QRS
# seed puts b on λ₁ and the (λ, λ*, ρ) objective is invariant to the choice, so
# without alternates the sweep tries the b = λ₃ setting in only ~1 % of cells.
MONOCLINIC_UNIQUE_AXIS_RANK_PRIOR: tuple[float, float, float] = (0.456, 0.242, 0.302)


def monoclinic_unique_axis_rank(cellpar: np.ndarray) -> int:
    """Length rank (0 shortest … 2 longest) of the ``b`` edge among ``(a, b, c)``."""
    cp = np.asarray(cellpar, dtype=np.float64).reshape(-1)
    order = np.argsort(cp[:3], kind="stable")
    return int(np.flatnonzero(order == 1)[0])


def monoclinic_unique_axis_alternates(
    cellpar: np.ndarray,
    hall_number: int,
    *,
    length_rtol: float = 0.02,
) -> list[tuple[int, np.ndarray]]:
    """Relabel a monoclinic cell so the unique axis sits on each other edge.

    A monoclinic lattice-QRS cell ``(a, b, c, β)`` with ``b`` on λ₁ has two
    target-free siblings with ``b`` on λ₂ or λ₃: the same three edge lengths and
    the same β between the two remaining edges. They are different lattices
    (a different pair of edges is oblique) with the same λ, λ*, and volume, so
    the lattice QRS cannot rank them and the conformational QRS cannot relax
    from one to another: the 2₁ axis is fixed to ``b`` by the Hall setting.

    Returns ``(b_rank, cellpar)`` pairs ordered by the CSD prior of the new
    ``b`` rank (most common first); ``[]`` for non-monoclinic Hall numbers.
    Relabelings whose new ``b`` equals the current one within ``length_rtol``
    are skipped.
    """
    if crystal_system_from_hall(hall_number) != "monoclinic":
        return []
    cp = apply_cellpar_constraints(cellpar, hall_number)
    lengths = cp[:3]
    beta = float(cp[4])
    own_rank = monoclinic_unique_axis_rank(cp)
    out: list[tuple[int, np.ndarray]] = []
    for axis in (0, 2):
        b_new = float(lengths[axis])
        if np.isclose(b_new, float(lengths[1]), rtol=length_rtol, atol=1e-3):
            continue
        rest = sorted(float(lengths[j]) for j in range(3) if j != axis)
        alt = np.array([rest[0], b_new, rest[1], 90.0, beta, 90.0], dtype=np.float64)
        rank = monoclinic_unique_axis_rank(alt)
        if rank == own_rank or any(existing == rank for existing, _ in out):
            continue
        out.append((rank, alt))
    out.sort(key=lambda item: (-MONOCLINIC_UNIQUE_AXIS_RANK_PRIOR[item[0]], item[0]))
    return out


def _lambda_consistency_score(record: LatticeQRSRecord) -> float:
    score = float(record.lambda_mse) + float(record.lambda_recip_mse)
    return score if np.isfinite(score) else float("inf")


def lattice_records_from_unique_axis_alternates(
    flow: FlowSampleBatch,
    lattice_records: list[LatticeQRSRecord],
    *,
    n_top: int = 6,
) -> list[LatticeQRSRecord]:
    """Unique-axis alternates of the ``n_top`` most λ-consistent QRS cells.

    Monoclinic only (returns ``[]`` otherwise). Parents are ranked by
    ``lambda_mse + lambda_recip_mse`` exactly like the λ-consistency selection
    channel, so the alternates cover the settings of the cells the pipeline
    already trusts most. Each alternate keeps its parent's ``flow_idx`` and QRS
    loss, recomputes its own λ / λ* errors, and is tagged
    ``AXIS_ALTERNATE_SOURCE`` so the selection channels can tell it apart.
    """
    if crystal_system_from_hall(flow.hall_number) != "monoclinic":
        return []
    n_top = max(int(n_top), 0)
    parents = [
        rec
        for rec in lattice_records
        if _record_source(rec) not in ("selling", AXIS_ALTERNATE_SOURCE)
        and np.isfinite(_lambda_consistency_score(rec))
    ]
    parents.sort(key=lambda rec: (_lambda_consistency_score(rec), int(rec.flow_idx)))
    records: list[LatticeQRSRecord] = []
    for rec in parents[:n_top]:
        idx = int(rec.flow_idx)
        for _rank, alt in monoclinic_unique_axis_alternates(rec.cellpar, flow.hall_number):
            if not _cellpar_init_is_valid(alt):
                continue
            try:
                pred_lam = compute_log_successive_minima(alt)
                pred_lam_r = compute_log_reciprocal_successive_minima(alt)
            except Exception:
                continue
            lam_mse = float(np.mean((pred_lam - flow.all_log_lambda[idx]) ** 2))
            lam_r_mse = float(np.mean((pred_lam_r - flow.all_log_lambda_recip[idx]) ** 2))
            records.append(
                LatticeQRSRecord(
                    flow_idx=idx,
                    cellpar=alt,
                    qrs_loss=float(rec.qrs_loss),
                    density=float(rec.density),
                    lambda_mse=lam_mse,
                    lambda_recip_mse=lam_r_mse,
                    source=AXIS_ALTERNATE_SOURCE,
                )
            )
    return records


SELLING_DISAGREE_ANGLE_MAE_DEG = 3.0


def _record_source(record: LatticeQRSRecord) -> str:
    return getattr(record, "source", "qrs") or "qrs"


def _unique_cell_sources(cell: UniqueCellRecord) -> list[str]:
    sources = list(getattr(cell, "sources", []) or [])
    n = len(cell.source_flow_indices)
    if len(sources) == n:
        return sources
    return ["qrs"] * n


def unique_cell_is_qrs_origin(cell: UniqueCellRecord) -> bool:
    sources = _unique_cell_sources(cell)
    return (not sources) or any(src != "selling" for src in sources)


def unique_cell_is_sell_only(cell: UniqueCellRecord) -> bool:
    sources = _unique_cell_sources(cell)
    return bool(sources) and all(src == "selling" for src in sources)


def unique_cell_is_axis_alternate(cell: UniqueCellRecord) -> bool:
    """True when every origin of the cell is a relabeled unique-axis alternate."""
    sources = _unique_cell_sources(cell)
    return bool(sources) and all(src == AXIS_ALTERNATE_SOURCE for src in sources)


def _origin_records_for_unique_cell(
    cell: UniqueCellRecord,
    lattice_records: list[LatticeQRSRecord],
) -> list[LatticeQRSRecord]:
    wanted = set(zip(cell.source_flow_indices, _unique_cell_sources(cell)))
    matched = [
        rec
        for rec in lattice_records
        if (rec.flow_idx, _record_source(rec)) in wanted
    ]
    if matched:
        return matched
    return [rec for rec in lattice_records if rec.flow_idx in set(cell.source_flow_indices)]


def _selling_angle_mae(
    cellpar: np.ndarray,
    source_flow_indices: list[int],
    all_selling: np.ndarray,
    hall_number: int,
) -> float:
    from cellnet.lattice_invariants import cellpar_from_selling
    from cellnet.symmetry import angle_mae_deg

    best = float("inf")
    for idx in source_flow_indices:
        if idx < 0 or idx >= len(all_selling):
            continue
        try:
            selling_cell = cellpar_from_selling(all_selling[idx], hall_number)
            mae, _ = angle_mae_deg(cellpar, selling_cell, hall_number)
        except Exception:
            continue
        if mae < best:
            best = float(mae)
    return best


def _cellpar_angle_mae(cell_a: np.ndarray, cell_b: np.ndarray, hall_number: int) -> float:
    from cellnet.symmetry import angle_mae_deg

    try:
        mae, _ = angle_mae_deg(cell_a, cell_b, hall_number)
    except Exception:
        return float("inf")
    return float(mae)


def qrs_origin_unique_cells(unique_cells: list[UniqueCellRecord]) -> list[UniqueCellRecord]:
    """QRS-origin cells, excluding pure unique-axis alternates (derived, not searched)."""
    qrs_cells = [
        cell
        for cell in unique_cells
        if unique_cell_is_qrs_origin(cell) and not unique_cell_is_axis_alternate(cell)
    ]
    return qrs_cells or list(unique_cells)


def posterior_effective_mode_count(
    unique_cells: list[UniqueCellRecord],
    lattice_records: list[LatticeQRSRecord],
) -> float:
    """Target-free count of how many QRS cells share the posterior mass.

    Weight unique QRS-origin cells by ``1 / λ-MSE``. If a few cells reconstruct
    the predicted λ much better than the rest, ``n_eff`` is small (tight
    posterior → fewer conf runs). If many cells are similarly good, ``n_eff``
    is large (spread posterior → more conf runs).
    """
    # Unique-axis alternates are relabelings of searched cells, not new draws.
    lattice_records = [
        rec for rec in lattice_records if _record_source(rec) != AXIS_ALTERNATE_SOURCE
    ]
    mses: list[float] = []
    for cell in qrs_origin_unique_cells(unique_cells):
        origin = _origin_records_for_unique_cell(cell, lattice_records)
        if not origin:
            continue
        mses.append(min(float(r.lambda_mse) for r in origin))
    if len(mses) < 2:
        return 1.0
    weights = 1.0 / np.clip(np.asarray(mses, dtype=np.float64), 1e-12, None)
    weights /= float(weights.sum())
    return float(1.0 / np.sum(weights * weights))


def dynamic_conf_run_count(
    n_eff: float,
    *,
    n_min: int = 12,
    n_max: int = 48,
    n_ref: float = 8.0,
    n_base: int = 24,
) -> int:
    """Map ``n_eff`` to a conf budget. ``n_eff == n_ref`` → ``n_base`` cells."""
    n_min = max(int(n_min), 1)
    n_max = max(int(n_max), n_min)
    n_ref = max(float(n_ref), 1e-6)
    n = int(round(float(n_base) * float(n_eff) / n_ref))
    return int(min(max(n, n_min), n_max))


def scaled_conf_elite(n_elite: int, n_conf: int, *, n_ref: int = 24) -> int:
    """Keep elite at ~1/4 of the conf budget, at least 3."""
    n_conf = max(int(n_conf), 1)
    n_ref = max(int(n_ref), 1)
    scaled = int(round(int(n_elite) * n_conf / n_ref))
    return int(min(max(scaled, 3), n_conf))


def append_disagreeing_selling_cells(
    selected: list[UniqueCellRecord],
    unique_cells: list[UniqueCellRecord],
    qrs_records: list[LatticeQRSRecord],
    selling_records: list[LatticeQRSRecord],
    hall_number: int,
    *,
    angle_mae_min: float = SELLING_DISAGREE_ANGLE_MAE_DEG,
    target_volume: float | None = None,
    density_ratio_factor: float | None = None,
) -> list[UniqueCellRecord]:
    """After QRS draw selection, add Selling cells that disagree with those draws."""
    selected_ids = {getattr(cell, "dedup_idx") for cell in selected}
    selected_flows = {idx for cell in selected for idx in cell.source_flow_indices}
    qrs_by_flow = {rec.flow_idx: rec for rec in qrs_records}
    sell_by_flow = {rec.flow_idx: rec for rec in selling_records}
    extra: list[UniqueCellRecord] = []
    skipped_density = 0
    for flow_idx in sorted(selected_flows):
        qrs = qrs_by_flow.get(flow_idx)
        sell = sell_by_flow.get(flow_idx)
        if qrs is None or sell is None:
            continue
        if _cellpar_angle_mae(qrs.cellpar, sell.cellpar, hall_number) < angle_mae_min:
            continue
        cell = next(
            (
                unique
                for unique in unique_cells
                if unique_cell_is_sell_only(unique) and flow_idx in unique.source_flow_indices
            ),
            None,
        )
        if cell is None or getattr(cell, "dedup_idx") in selected_ids:
            continue
        if not cellpar_in_predicted_volume_range(
            cell.cellpar,
            target_volume,
            density_ratio_factor=density_ratio_factor,
        ):
            skipped_density += 1
            continue
        extra.append(cell)
        selected_ids.add(getattr(cell, "dedup_idx"))
    if extra:
        print(
            f"  +{len(extra)} Selling cells disagree with selected QRS "
            f"(angle MAE ≥ {angle_mae_min:g}°)",
            flush=True,
        )
    if skipped_density:
        print(
            f"  skipped {skipped_density} Selling extras outside density volume range",
            flush=True,
        )
    return list(selected) + extra


def _cell_uid(cell: UniqueCellRecord) -> int:
    return int(getattr(cell, "dedup_idx"))


def lowest_lambda_mse_draw_indices(
    qrs_records: list[LatticeQRSRecord],
    n_pool: int,
) -> list[int]:
    recs = [rec for rec in qrs_records if _record_source(rec) != "selling"]
    recs.sort(key=lambda rec: float(rec.lambda_mse))
    n_pool = max(int(n_pool), 0)
    return [int(rec.flow_idx) for rec in recs[:n_pool]]


def selling_kcenter_draw_indices(
    all_selling: np.ndarray,
    n_clusters: int,
    *,
    subset: list[int] | None = None,
) -> list[int]:
    """Farthest-point cluster centers on sorted Selling vectors (target-free).

    If ``subset`` is given, cluster only those draw indices (used to ignore
    high-λ-MSE reconstructions, which otherwise dominate as geometric outliers).
    """
    from cellnet.lattice_invariants import sort_selling_parameters

    selling = np.asarray(all_selling, dtype=np.float64)
    if selling.ndim != 2 or len(selling) == 0:
        return []
    if subset is not None:
        subset = [int(i) for i in subset if 0 <= int(i) < len(selling)]
        if not subset:
            return []
        local = selling_kcenter_draw_indices(
            selling[np.asarray(subset)],
            n_clusters,
            subset=None,
        )
        return [subset[i] for i in local]
    n_clusters = int(min(max(int(n_clusters), 1), len(selling)))
    features = np.stack([sort_selling_parameters(row) for row in selling])
    scale = np.std(features, axis=0)
    scale[scale < 1e-6] = 1.0
    z = (features - np.mean(features, axis=0)) / scale
    centers = [int(np.argmin(np.linalg.norm(z - z.mean(axis=0), axis=1)))]
    while len(centers) < n_clusters:
        delta = z[:, None, :] - z[np.asarray(centers)][None, :, :]
        dmin = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
        dmin[np.asarray(centers)] = -np.inf
        centers.append(int(np.argmax(dmin)))
    return centers


def merge_selling_cluster_representatives(
    selected: list[UniqueCellRecord],
    unique_cells: list[UniqueCellRecord],
    all_selling: np.ndarray,
    *,
    n_clusters: int,
    max_cells: int,
    n_elite: int,
    qrs_records: list[LatticeQRSRecord] | None = None,
    n_pool: int = 32,
) -> list[UniqueCellRecord]:
    """Replace diversity-tail QRS cells with Selling-space cluster representatives.

    Elite prefix is kept. K-centers run only on the ``n_pool`` lowest-λ-MSE
    QRS draws so pathological reconstructions cannot occupy the outlier slots.
    """
    if n_clusters <= 0 or max_cells <= 0 or all_selling is None:
        return list(selected)
    qrs_cells = qrs_origin_unique_cells(unique_cells)
    subset = None
    if qrs_records:
        n_pool = max(int(n_pool), int(n_clusters))
        subset = lowest_lambda_mse_draw_indices(qrs_records, n_pool)
    centers = selling_kcenter_draw_indices(all_selling, n_clusters, subset=subset)
    reps: list[UniqueCellRecord] = []
    seen: set[int] = set()
    for draw in centers:
        cell = next(
            (
                unique
                for unique in qrs_cells
                if int(draw) in unique.source_flow_indices
            ),
            None,
        )
        if cell is None:
            continue
        uid = _cell_uid(cell)
        if uid in seen:
            continue
        reps.append(cell)
        seen.add(uid)
    n_elite_keep = min(max(int(n_elite), 0), len(selected), max_cells)
    out: list[UniqueCellRecord] = list(selected[:n_elite_keep])
    have = {_cell_uid(cell) for cell in out}
    for cell in reps:
        uid = _cell_uid(cell)
        if uid in have:
            continue
        out.append(cell)
        have.add(uid)
        if len(out) >= max_cells:
            break
    if len(out) < max_cells:
        for cell in selected[n_elite_keep:]:
            uid = _cell_uid(cell)
            if uid in have:
                continue
            out.append(cell)
            have.add(uid)
            if len(out) >= max_cells:
                break
    print(
        f"  Selling k-centers: {len(centers)} draws → {len(reps)} QRS reps"
        f"{'' if subset is None else f' (pool={len(subset)} low-λ-MSE)'}; "
        f"{len(out)} QRS-origin cells after merge (elite={n_elite_keep})",
        flush=True,
    )
    return out


def append_disagreement_channel(
    selected: list[UniqueCellRecord],
    unique_cells: list[UniqueCellRecord],
    qrs_records: list[LatticeQRSRecord],
    selling_records: list[LatticeQRSRecord],
    hall_number: int,
    *,
    n_max: int = 12,
    angle_mae_min: float = SELLING_DISAGREE_ANGLE_MAE_DEG,
    allowed_flows: set[int] | None = None,
    target_volume: float | None = None,
    density_ratio_factor: float | None = None,
) -> list[UniqueCellRecord]:
    """Add QRS+Selling unique cells for draws with the largest QRS–Selling angle MAE.

    Unlike ``append_disagreeing_selling_cells``, this does **not** require the
    QRS draw to already be selected. Cap ``n_max`` pairs so ACSALA cannot explode.
    Selling reconstructions outside the density-implied volume window are skipped.
    """
    if n_max <= 0:
        return list(selected)
    qrs_by_flow = {rec.flow_idx: rec for rec in qrs_records}
    sell_by_flow = {rec.flow_idx: rec for rec in selling_records}

    def _match(flow_idx: int, source: str) -> UniqueCellRecord | None:
        for unique in unique_cells:
            sources = _unique_cell_sources(unique)
            for src_flow, src_name in zip(unique.source_flow_indices, sources):
                if int(src_flow) == int(flow_idx) and src_name == source:
                    return unique
        return None

    ranked: list[tuple[float, int]] = []
    for flow_idx, qrs in qrs_by_flow.items():
        sell = sell_by_flow.get(flow_idx)
        if sell is None:
            continue
        if allowed_flows is not None and int(flow_idx) not in allowed_flows:
            continue
        mae = _cellpar_angle_mae(qrs.cellpar, sell.cellpar, hall_number)
        qrs_cell = _match(flow_idx, "qrs")
        sell_cell = _match(flow_idx, "selling")
        distinct = (
            qrs_cell is not None
            and sell_cell is not None
            and _cell_uid(qrs_cell) != _cell_uid(sell_cell)
        )
        if not distinct and (not np.isfinite(mae) or mae < angle_mae_min):
            continue
        score = float(mae) if np.isfinite(mae) else 0.0
        if distinct:
            score = max(score, float(angle_mae_min))
        ranked.append((score, int(flow_idx)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    ranked = ranked[: int(n_max)]

    out = list(selected)
    have = {_cell_uid(cell) for cell in out}
    added_flows: list[int] = []
    skipped_density = 0
    for mae, flow_idx in ranked:
        added_this = False
        for source in ("qrs", "selling"):
            cell = _match(flow_idx, source)
            if cell is None:
                continue
            uid = _cell_uid(cell)
            if uid in have:
                continue
            if source == "selling" and not cellpar_in_predicted_volume_range(
                cell.cellpar,
                target_volume,
                density_ratio_factor=density_ratio_factor,
            ):
                skipped_density += 1
                continue
            out.append(cell)
            have.add(uid)
            added_this = True
        if added_this:
            added_flows.append(flow_idx)
    if added_flows:
        print(
            f"  disagreement channel: +{len(added_flows)} draws "
            f"(angle MAE ≥ {angle_mae_min:g}°, cap={n_max}); flows={added_flows}",
            flush=True,
        )
    if skipped_density:
        print(
            f"  skipped {skipped_density} disagreement Selling cells "
            "outside density volume range",
            flush=True,
        )
    return out


def select_conf_unique_cells(
    unique_cells: list[UniqueCellRecord],
    qrs_records: list[LatticeQRSRecord],
    selling_records: list[LatticeQRSRecord],
    max_cells: int,
    hall_number: int,
    *,
    n_elite: int = 6,
    all_selling: np.ndarray | None = None,
    n_clusters: int = 12,
    n_disagree: int = 12,
    target_volume: float | None = None,
    density_ratio_factor: float | None = None,
) -> list[UniqueCellRecord]:
    """Originelite QRS selection, then Selling k-centers, then disagreement extras."""
    qrs_unique = qrs_origin_unique_cells(unique_cells)
    selected = select_diverse_unique_cells(
        qrs_unique,
        qrs_records,
        max_cells,
        hall_number=hall_number,
        n_elite=n_elite,
        all_selling=all_selling,
    )
    pool = set(lowest_lambda_mse_draw_indices(qrs_records, max(32, n_clusters)))
    if n_clusters > 0 and all_selling is not None:
        selected = merge_selling_cluster_representatives(
            selected,
            unique_cells,
            all_selling,
            n_clusters=n_clusters,
            max_cells=max_cells,
            n_elite=n_elite,
            qrs_records=qrs_records,
            n_pool=max(32, n_clusters),
        )
    selected = append_disagreeing_selling_cells(
        selected,
        unique_cells,
        qrs_records,
        selling_records,
        hall_number,
        target_volume=target_volume,
        density_ratio_factor=density_ratio_factor,
    )
    selected = append_disagreement_channel(
        selected,
        unique_cells,
        qrs_records,
        selling_records,
        hall_number,
        n_max=n_disagree,
        allowed_flows=pool if n_disagree > 0 else None,
        target_volume=target_volume,
        density_ratio_factor=density_ratio_factor,
    )
    return selected


def select_multichannel_unique_cells(
    unique_cells: list[UniqueCellRecord],
    qrs_records: list[LatticeQRSRecord],
    selling_records: list[LatticeQRSRecord],
    max_cells: int,
    hall_number: int,
    *,
    target_rho: float | None = None,
    target_volume: float | None = None,
    lambda_quota: int = 6,
    qrs_loss_quota: int = 6,
    density_quota: int = 4,
    disagreement_quota: int = 4,
    qrs_origin_only: bool = False,
    density_ratio_factor: float | None = None,
    axis_alternate_quota: int = 0,
) -> tuple[list[UniqueCellRecord], dict[int, str]]:
    """Select fixed, target-free channels followed by shape-space k-centers.

    The API intentionally has no reference-cell or match-result input. A cell is
    assigned to the first channel that selects it; unfilled channel capacity is
    recovered by the final diversity channel.

    ``axis_alternate_quota`` > 0 adds a ``unique_axis_alternate`` channel that
    takes relabeled monoclinic cells (source ``AXIS_ALTERNATE_SOURCE``; see
    ``lattice_records_from_unique_axis_alternates``) ranked by their parent's
    λ-consistency and the CSD prior of the new ``b`` rank. Alternates never
    enter the λ, QRS-loss, density, or disagreement channels, and their shape
    features coincide with their parent's, so the k-centers channel does not
    pick them while the parent is selected.
    """
    from cellnet.lattice_invariants import successive_minima
    from cellnet.packing import cell_volume

    max_cells = max(int(max_cells), 0)
    if max_cells == 0:
        return [], {}

    qrs_origin_pool = [cell for cell in unique_cells if unique_cell_is_qrs_origin(cell)]
    qrs_pool = (
        [cell for cell in qrs_origin_pool if not unique_cell_is_axis_alternate(cell)]
        if qrs_origin_only
        else qrs_origin_unique_cells(unique_cells)
    )
    qrs_cells = sorted(qrs_pool, key=_cell_uid)
    # Pure unique-axis alternates stay eligible for their own channel and for
    # the shape-diversity fill even when the selection is QRS-origin only.
    eligible_cells = (
        sorted(qrs_origin_pool, key=_cell_uid) if qrs_origin_only else list(unique_cells)
    )
    qrs_by_flow = {
        int(record.flow_idx): record
        for record in qrs_records
        if _record_source(record) not in ("selling", AXIS_ALTERNATE_SOURCE)
    }
    sell_by_flow = {int(record.flow_idx): record for record in selling_records}
    selected: list[UniqueCellRecord] = []
    provenance: dict[int, str] = {}
    selected_ids: set[int] = set()

    def _take_ranked(
        ranked: list[UniqueCellRecord],
        quota: int,
        channel: str,
    ) -> None:
        n_added = 0
        for cell in ranked:
            if len(selected) >= max_cells or n_added >= max(int(quota), 0):
                break
            uid = _cell_uid(cell)
            if uid in selected_ids:
                continue
            selected.append(cell)
            selected_ids.add(uid)
            provenance[uid] = channel
            n_added += 1

    def _origins(cell: UniqueCellRecord) -> list[LatticeQRSRecord]:
        return [
            qrs_by_flow[flow_idx]
            for flow_idx in sorted(set(int(i) for i in cell.source_flow_indices))
            if flow_idx in qrs_by_flow
        ]

    def _lambda_key(cell: UniqueCellRecord) -> tuple[float, float, int]:
        scores = [
            float(record.lambda_mse) + float(record.lambda_recip_mse)
            for record in _origins(cell)
            if np.isfinite(record.lambda_mse)
            and np.isfinite(record.lambda_recip_mse)
        ]
        score = min(scores, default=float("inf"))
        loss = min(
            (float(record.qrs_loss) for record in _origins(cell)),
            default=float("inf"),
        )
        return score, loss, _cell_uid(cell)

    lambda_ranked = [cell for cell in qrs_cells if np.isfinite(_lambda_key(cell)[0])]
    lambda_ranked.sort(key=_lambda_key)
    _take_ranked(lambda_ranked, lambda_quota, "lambda_consistency")

    def _qrs_loss_key(cell: UniqueCellRecord) -> tuple[float, float, int]:
        losses = [
            float(record.qrs_loss)
            for record in _origins(cell)
            if np.isfinite(record.qrs_loss)
        ]
        return min(losses, default=float("inf")), _lambda_key(cell)[0], _cell_uid(cell)

    loss_ranked = [cell for cell in qrs_cells if np.isfinite(_qrs_loss_key(cell)[0])]
    loss_ranked.sort(key=_qrs_loss_key)
    _take_ranked(loss_ranked, qrs_loss_quota, "qrs_loss")

    def _density_key(cell: UniqueCellRecord) -> tuple[float, float, int]:
        terms: list[float] = []
        if target_rho is not None and np.isfinite(target_rho) and target_rho > 0:
            density_errors = [
                abs(np.log(float(record.density) / float(target_rho)))
                for record in _origins(cell)
                if np.isfinite(record.density) and record.density > 0
            ]
            if density_errors:
                terms.append(min(density_errors))
        if target_volume is not None and np.isfinite(target_volume) and target_volume > 0:
            try:
                volume = float(cell_volume(cell.cellpar))
            except Exception:
                volume = float("nan")
            if np.isfinite(volume) and volume > 0:
                terms.append(abs(np.log(volume / float(target_volume))))
        score = float(sum(terms)) if terms else float("inf")
        return score, _lambda_key(cell)[0], _cell_uid(cell)

    density_ranked = [cell for cell in qrs_cells if np.isfinite(_density_key(cell)[0])]
    density_ranked.sort(key=_density_key)
    _take_ranked(density_ranked, density_quota, "density_volume")

    disagreement_scores: dict[int, float] = {}
    for flow_idx, qrs_record in qrs_by_flow.items():
        selling_record = sell_by_flow.get(flow_idx)
        if selling_record is None:
            continue
        score = _cellpar_angle_mae(qrs_record.cellpar, selling_record.cellpar, hall_number)
        if not np.isfinite(score):
            continue
        for cell in eligible_cells:
            if flow_idx not in cell.source_flow_indices:
                continue
            if unique_cell_is_axis_alternate(cell):
                # Relabelings inherit the parent's draw; the parent carries the
                # disagreement, the alternates have their own channel.
                continue
            if unique_cell_is_sell_only(cell) and not cellpar_in_predicted_volume_range(
                cell.cellpar,
                target_volume,
                density_ratio_factor=density_ratio_factor,
            ):
                continue
            uid = _cell_uid(cell)
            disagreement_scores[uid] = max(disagreement_scores.get(uid, -float("inf")), score)
    disagreement_ranked = [
        cell for cell in sorted(eligible_cells, key=_cell_uid)
        if _cell_uid(cell) in disagreement_scores
    ]
    disagreement_ranked.sort(
        key=lambda cell: (-disagreement_scores[_cell_uid(cell)], _cell_uid(cell))
    )
    _take_ranked(disagreement_ranked, disagreement_quota, "qrs_selling_disagreement")

    if int(axis_alternate_quota) > 0 and crystal_system_from_hall(hall_number) == "monoclinic":

        def _axis_alternate_key(cell: UniqueCellRecord) -> tuple[float, float, int]:
            parent_score = min(
                (_lambda_consistency_score(record) for record in _origins(cell)),
                default=float("inf"),
            )
            prior = MONOCLINIC_UNIQUE_AXIS_RANK_PRIOR[monoclinic_unique_axis_rank(cell.cellpar)]
            return parent_score, -prior, _cell_uid(cell)

        alternate_ranked = [
            cell
            for cell in eligible_cells
            if unique_cell_is_axis_alternate(cell)
            and np.isfinite(_axis_alternate_key(cell)[0])
        ]
        alternate_ranked.sort(key=_axis_alternate_key)
        _take_ranked(alternate_ranked, axis_alternate_quota, "unique_axis_alternate")

    diversity_pool = sorted(eligible_cells, key=_cell_uid)
    features: dict[int, np.ndarray] = {}
    for cell in diversity_pool:
        if unique_cell_is_sell_only(cell) and not cellpar_in_predicted_volume_range(
            cell.cellpar,
            target_volume,
            density_ratio_factor=density_ratio_factor,
        ):
            continue
        try:
            lam = successive_minima(cell.cellpar)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        log_lam = np.log(np.clip(lam, 1e-12, None))
        angles = np.asarray(cell.cellpar[3:6], dtype=np.float64) / 180.0
        feature = np.concatenate(
            ([log_lam[1] - log_lam[0], log_lam[2] - log_lam[0]], angles)
        )
        if np.all(np.isfinite(feature)):
            features[_cell_uid(cell)] = feature

    if features and len(selected) < max_cells:
        feature_ids = sorted(features)
        matrix = np.stack([features[uid] for uid in feature_ids])
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-6] = 1.0
        mean = np.mean(matrix, axis=0)
        normalized = {uid: (features[uid] - mean) / scale for uid in feature_ids}
        candidates = [
            cell for cell in diversity_pool
            if _cell_uid(cell) in normalized and _cell_uid(cell) not in selected_ids
        ]
        while candidates and len(selected) < max_cells:
            if selected_ids & normalized.keys():
                def _distance_key(cell: UniqueCellRecord) -> tuple[float, int]:
                    uid = _cell_uid(cell)
                    distance = min(
                        float(np.linalg.norm(normalized[uid] - normalized[other]))
                        for other in selected_ids
                        if other in normalized
                    )
                    return -distance, uid
            else:
                def _distance_key(cell: UniqueCellRecord) -> tuple[float, int]:
                    uid = _cell_uid(cell)
                    return float(np.linalg.norm(normalized[uid])), uid

            best = min(candidates, key=_distance_key)
            uid = _cell_uid(best)
            selected.append(best)
            selected_ids.add(uid)
            provenance[uid] = "shape_diversity"
            candidates.remove(best)

    # Invalid shape features should not make the requested fixed budget shrink.
    _take_ranked(
        [cell for cell in diversity_pool if _cell_uid(cell) not in selected_ids],
        max_cells - len(selected),
        "deterministic_fallback",
    )
    return selected, provenance


def _scaled_selling_channel_quotas(quota: int) -> tuple[int, int, int, int]:
    """Scale the v7 8/8/4/4 allocation with deterministic largest remainders."""
    quota = max(int(quota), 0)
    weights = np.asarray([8, 8, 4, 4], dtype=np.float64) / 24.0
    raw = weights * quota
    counts = np.floor(raw).astype(int)
    remainder = quota - int(counts.sum())
    order = sorted(range(4), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    return tuple(int(value) for value in counts)


def select_target_free_quota_unique_cells(
    unique_cells: list[UniqueCellRecord],
    qrs_records: list[LatticeQRSRecord],
    selling_records: list[LatticeQRSRecord],
    n_select: int,
    hall_number: int,
    *,
    selling_quota: int | None = None,
    target_rho: float | None = None,
    target_volume: float | None = None,
    density_ratio_factor: float | None = None,
    lambda_quota: int = 6,
    qrs_loss_quota: int = 6,
    density_quota: int = 4,
    disagreement_quota: int = 4,
    axis_alternate_quota: int = 0,
) -> tuple[list[UniqueCellRecord], dict[int, str]]:
    """Select separate QRS and Selling budgets without reference-cell information."""
    from cellnet.lattice_invariants import successive_minima

    n_select = max(int(n_select), 0)
    if selling_quota is None:
        selling_quota = int(round(n_select / 3.0))
    selling_quota = min(max(int(selling_quota), 0), n_select)
    qrs_quota = n_select - selling_quota

    qrs_selected, provenance = select_multichannel_unique_cells(
        unique_cells,
        qrs_records,
        selling_records,
        qrs_quota,
        hall_number,
        target_rho=target_rho,
        target_volume=target_volume,
        lambda_quota=lambda_quota,
        qrs_loss_quota=qrs_loss_quota,
        density_quota=density_quota,
        disagreement_quota=disagreement_quota,
        qrs_origin_only=True,
        density_ratio_factor=density_ratio_factor,
        axis_alternate_quota=axis_alternate_quota,
    )

    selling_cells = [
        cell
        for cell in sorted(unique_cells, key=_cell_uid)
        if unique_cell_is_sell_only(cell)
        and cellpar_in_predicted_volume_range(
            cell.cellpar,
            target_volume,
            density_ratio_factor=density_ratio_factor,
        )
    ]
    qrs_by_flow = {
        int(record.flow_idx): record
        for record in qrs_records
        if _record_source(record) != "selling"
    }
    sell_by_flow = {int(record.flow_idx): record for record in selling_records}
    selected_selling: list[UniqueCellRecord] = []
    selected_ids: set[int] = set()

    def _take(
        ranked: list[UniqueCellRecord],
        quota: int,
        channel: str,
    ) -> None:
        added = 0
        for cell in ranked:
            if len(selected_selling) >= selling_quota or added >= quota:
                break
            uid = _cell_uid(cell)
            if uid in selected_ids:
                continue
            selected_selling.append(cell)
            selected_ids.add(uid)
            provenance[uid] = channel
            added += 1

    disagreement_scores: dict[int, float] = {}
    for cell in selling_cells:
        scores = []
        for flow_idx in sorted(set(int(i) for i in cell.source_flow_indices)):
            qrs = qrs_by_flow.get(flow_idx)
            selling = sell_by_flow.get(flow_idx)
            if qrs is not None and selling is not None:
                score = _cellpar_angle_mae(qrs.cellpar, selling.cellpar, hall_number)
                if np.isfinite(score):
                    scores.append(float(score))
        disagreement_scores[_cell_uid(cell)] = max(scores, default=-float("inf"))
    disagreement_ranked = sorted(
        selling_cells,
        key=lambda cell: (-disagreement_scores[_cell_uid(cell)], _cell_uid(cell)),
    )

    disagree_n, shape_n, lambda_n, fallback_n = _scaled_selling_channel_quotas(
        selling_quota
    )
    _take(
        [
            cell
            for cell in disagreement_ranked
            if np.isfinite(disagreement_scores[_cell_uid(cell)])
        ],
        disagree_n,
        "selling_angle_disagreement",
    )

    features: dict[int, np.ndarray] = {}
    for cell in selling_cells:
        if _cell_uid(cell) in selected_ids:
            continue
        try:
            lam = successive_minima(cell.cellpar)
            log_lam = np.log(np.clip(lam, 1e-12, None))
            feature = np.concatenate(
                (
                    [log_lam[1] - log_lam[0], log_lam[2] - log_lam[0]],
                    np.asarray(cell.cellpar[3:6], dtype=np.float64) / 180.0,
                )
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(feature)):
            features[_cell_uid(cell)] = feature
    if features and shape_n > 0:
        ids = sorted(features)
        matrix = np.stack([features[uid] for uid in ids])
        scale = np.std(matrix, axis=0)
        scale[scale < 1e-6] = 1.0
        normalized = {
            uid: (features[uid] - np.mean(matrix, axis=0)) / scale for uid in ids
        }
        candidates = [
            cell
            for cell in selling_cells
            if _cell_uid(cell) in normalized and _cell_uid(cell) not in selected_ids
        ]
        ranked_shape: list[UniqueCellRecord] = []
        anchors = set(selected_ids) & set(normalized)
        while candidates and len(ranked_shape) < shape_n:
            if anchors:
                best = min(
                    candidates,
                    key=lambda cell: (
                        -min(
                            float(
                                np.linalg.norm(
                                    normalized[_cell_uid(cell)] - normalized[anchor]
                                )
                            )
                            for anchor in anchors
                        ),
                        _cell_uid(cell),
                    ),
                )
            else:
                best = min(
                    candidates,
                    key=lambda cell: (
                        -float(np.linalg.norm(normalized[_cell_uid(cell)])),
                        _cell_uid(cell),
                    ),
                )
            ranked_shape.append(best)
            anchors.add(_cell_uid(best))
            candidates.remove(best)
        _take(ranked_shape, shape_n, "selling_shape_diversity")

    def _parent_lambda_key(cell: UniqueCellRecord) -> tuple[float, int]:
        scores = [
            float(qrs_by_flow[idx].lambda_mse)
            + float(qrs_by_flow[idx].lambda_recip_mse)
            for idx in sorted(set(int(i) for i in cell.source_flow_indices))
            if idx in qrs_by_flow
            and np.isfinite(qrs_by_flow[idx].lambda_mse)
            and np.isfinite(qrs_by_flow[idx].lambda_recip_mse)
        ]
        return min(scores, default=float("inf")), _cell_uid(cell)

    lambda_ranked = sorted(
        [
            cell
            for cell in selling_cells
            if _cell_uid(cell) not in selected_ids
            and np.isfinite(_parent_lambda_key(cell)[0])
        ],
        key=_parent_lambda_key,
    )
    _take(lambda_ranked, lambda_n, "selling_parent_lambda_consistency")
    _take(
        [cell for cell in selling_cells if _cell_uid(cell) not in selected_ids],
        fallback_n + max(selling_quota - disagree_n - shape_n - lambda_n - fallback_n, 0),
        "selling_deterministic_fallback",
    )
    # Recover unused channel capacity deterministically while retaining its own provenance.
    _take(
        [cell for cell in selling_cells if _cell_uid(cell) not in selected_ids],
        selling_quota - len(selected_selling),
        "selling_deterministic_fallback",
    )
    deficit = n_select - len(qrs_selected) - len(selected_selling)
    if deficit > 0:
        selling_provenance = {
            _cell_uid(cell): provenance[_cell_uid(cell)]
            for cell in selected_selling
        }
        qrs_selected, qrs_provenance = select_multichannel_unique_cells(
            unique_cells,
            qrs_records,
            selling_records,
            qrs_quota + deficit,
            hall_number,
            target_rho=target_rho,
            target_volume=target_volume,
            lambda_quota=lambda_quota,
            qrs_loss_quota=qrs_loss_quota,
            density_quota=density_quota,
            disagreement_quota=disagreement_quota,
            qrs_origin_only=True,
            density_ratio_factor=density_ratio_factor,
        )
        provenance = dict(qrs_provenance)
        provenance.update(selling_provenance)
    return qrs_selected + selected_selling, provenance


def select_diverse_unique_cells(
    unique_cells: list[UniqueCellRecord],
    lattice_records: list[LatticeQRSRecord],
    max_cells: int,
    *,
    true_cellpar: np.ndarray | None = None,
    hall_number: int | None = None,
    reference_weight: float = 0.0,
    n_elite: int = 6,
    all_selling: np.ndarray | None = None,
) -> list[UniqueCellRecord]:
    """
    Select a diverse QRS-origin subset for conf QRS without the ground-truth cell.

    Elite slots (first ``n_elite``) use only QRS-origin cells, at most one per
    flow draw, ranked by:
      * lowest flow-to-cell λ MSE
      * lowest angle MAE vs the same draw's Selling reconstruction

    Remaining slots maximize diversity in successive-minima ratio space among
    QRS-origin cells. Selling reconstructions are not part of this pool;
    add them afterward with ``append_disagreeing_selling_cells``.
    ``true_cellpar`` is ignored here.
    """
    del true_cellpar, reference_weight
    unique_cells = qrs_origin_unique_cells(unique_cells)
    if max_cells <= 0:
        return []
    if len(unique_cells) <= max_cells:
        return list(unique_cells)

    from cellnet.lattice_invariants import successive_minima

    features: list[np.ndarray] = []
    losses: list[float] = []
    lam_mses: list[float] = []
    sell_angs: list[float] = []
    supports: list[float] = []
    valid_cells: list[UniqueCellRecord] = []
    use_selling_agree = all_selling is not None and hall_number is not None
    for cell in unique_cells:
        try:
            lam = successive_minima(cell.cellpar)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        log_lam = np.log(np.clip(lam, 1e-12, None))
        features.append(
            np.array(
                [
                    log_lam.mean(),
                    log_lam[1] - log_lam[0],
                    log_lam[2] - log_lam[0],
                ],
                dtype=np.float64,
            )
        )
        origin = _origin_records_for_unique_cell(cell, lattice_records)
        losses.append(min(r.qrs_loss for r in origin) if origin else float("inf"))
        lam_mses.append(min(r.lambda_mse for r in origin) if origin else float("inf"))
        if use_selling_agree:
            sell_angs.append(
                _selling_angle_mae(
                    cell.cellpar,
                    cell.source_flow_indices,
                    all_selling,
                    hall_number,
                )
            )
        supports.append(float(len(cell.source_flow_indices)))
        valid_cells.append(cell)

    if not valid_cells:
        return list(unique_cells[:max_cells])

    x = np.stack(features)
    scale = np.std(x, axis=0)
    scale[scale < 1e-6] = 1.0
    x = (x - np.mean(x, axis=0)) / scale

    loss_arr = np.asarray(losses, dtype=np.float64)
    lam_arr = np.asarray(lam_mses, dtype=np.float64)
    loss_order = np.argsort(np.argsort(loss_arr))
    quality = 1.0 - loss_order / max(len(valid_cells) - 1, 1)
    support = np.log1p(np.asarray(supports, dtype=np.float64))
    support /= max(float(support.max()), 1e-12)

    n_keep = min(max(int(n_elite), 0), max_cells, len(valid_cells))
    # Leave at least one diversity slot when elite would consume the whole budget.
    if max_cells > 1 and n_keep >= max_cells:
        n_keep = max_cells - 1
    n_lam = n_keep if not use_selling_agree else (n_keep + 1) // 2
    n_agree = n_keep - n_lam
    used_flows: set[int] = set()

    def _take_ranked(order: np.ndarray, n_take: int, start: list[int]) -> list[int]:
        elite = list(start)
        for idx in order:
            idx = int(idx)
            if idx in elite:
                continue
            flows = set(valid_cells[idx].source_flow_indices)
            if flows & used_flows:
                continue
            elite.append(idx)
            used_flows.update(flows)
            if len(elite) >= n_take:
                break
        return elite

    elite = _take_ranked(np.lexsort((loss_arr, lam_arr)), n_lam, [])
    elite_mode = "lambda-mse"
    if n_agree:
        ang_arr = np.asarray(sell_angs, dtype=np.float64)
        elite = _take_ranked(np.lexsort((lam_arr, ang_arr)), n_lam + n_agree, elite)
        elite_mode = "lambda-mse+selling-agree"
    selected = list(dict.fromkeys(int(i) for i in elite))
    elite_ids = [getattr(valid_cells[i], "dedup_idx") for i in selected]
    elite_flows = [valid_cells[i].source_flow_indices for i in selected]
    print(
        f"  elite ({elite_mode}, n={len(selected)}): unique={elite_ids} flows={elite_flows}",
        flush=True,
    )
    remaining = set(range(len(valid_cells))) - set(selected)
    while remaining and len(selected) < max_cells:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            min_distance = min(float(np.linalg.norm(x[idx] - x[j])) for j in selected)
            score = min_distance + 0.20 * float(quality[idx]) + 0.10 * float(support[idx])
            if score > best_score:
                best_idx = idx
                best_score = score
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [valid_cells[idx] for idx in selected]


def _import_qrs_conf_helpers():
    from cellnet import qrs_conf

    return qrs_conf


@dataclass
class DbCrystalInfo:
    csd_code: str
    smiles: str
    hall_number: int
    zprime: float
    cellpar: np.ndarray
    space_group_number: int
    db_path: str


def _pyxtal_db_paths(extra: str | Path | None = None) -> list[Path]:
    import pyxtal

    paths: list[Path] = []
    if extra is not None:
        paths.append(Path(extra))
    paths.extend(
        [
            ROOT / "datasets" / "test.db",
            Path(pyxtal.__file__).resolve().parent / "database" / "test.db",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_crystal_from_db(db_path: str | Path, csd_code: str) -> DbCrystalInfo:
    """Load SMILES, symmetry, Z′, and cell parameters from a PyXtal ASE database."""
    from pyxtal.db import database

    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"PyXtal database not found: {db_path}")

    db = database(str(db_path))
    if csd_code not in db.get_all_codes():
        raise ValueError(f"CSD code not found in {db_path}: {csd_code}")

    row = db.get_row(code=csd_code)
    xtal = db.get_pyxtal(code=csd_code)
    if xtal.has_special_site():
        xtal = xtal.to_subgroup()

    zprime_vals = xtal.get_zprime()
    if isinstance(zprime_vals, (list, tuple, np.ndarray)):
        zprime = float(zprime_vals[0]) if len(zprime_vals) == 1 else float(row.Zprime)
    else:
        zprime = float(row.Zprime)

    return DbCrystalInfo(
        csd_code=csd_code,
        smiles=str(row.mol_smi).strip(),
        hall_number=int(xtal.group.hall_number),
        zprime=zprime,
        cellpar=np.array(xtal.lattice.get_para(degree=True), dtype=np.float64),
        space_group_number=int(xtal.group.number),
        db_path=str(db_path.resolve()),
    )


def _load_pyxtal_from_db(csd_code: str, db_path: str | Path | None = None):
    from pyxtal.db import database

    for path in _pyxtal_db_paths(db_path):
        try:
            db = database(str(path))
            if csd_code not in db.get_all_codes():
                continue
            xtal = db.get_pyxtal(code=csd_code)
            if xtal.has_special_site():
                xtal = xtal.to_subgroup()
            return xtal
        except Exception:
            continue
    return None


def _pregen_component_pool(smi: str, wps, *, n_iter: int = 20, n_conf: int = 200):
    """Mirror qrs_conf.pregen_component_pool without version-specific kwargs."""
    from pyxtal.constants import single_smiles
    from pyxtal.molecule import generate_molecules, pyxtal_molecule

    if smi in single_smiles:
        m0 = pyxtal_molecule(smi + ".smi", fix=True)
        _, valid = m0.get_orientations_in_wps(wps)
        if not valid:
            return None, "single-species"
        return [m0], "single-species"

    m0 = pyxtal_molecule(smi + ".smi", fix=True)
    if not m0.torsionlist:
        _, valid = m0.get_orientations_in_wps(wps)
        if not valid:
            return None, "rigid"
        return [m0], "rigid"

    pool = generate_molecules(
        smi,
        wps=wps,
        N_iter=n_iter,
        N_conf=n_conf,
        tol=0.5,
    )
    return pool, "flexible"


def _component_pool_cache_key(smi: str, wps) -> tuple:
    """Build a stable key for a molecule and its required Wyckoff positions."""
    return (
        smi,
        tuple(
            (
                getattr(wp, "hall_number", None),
                getattr(wp, "index", None),
                getattr(wp, "letter", None),
                getattr(wp, "multiplicity", None),
            )
            for wp in wps
        ),
    )


def _get_component_pool(smi: str, wps, qrs_conf, cache: dict | None = None):
    """Generate/filter a conformer pool once and return isolated copies on reuse."""
    key = _component_pool_cache_key(smi, wps)
    if cache is not None and key in cache:
        print(f"Reusing cached conformer pool ({len(cache[key])} conformers)", flush=True)
        return copy.deepcopy(cache[key]), "cached"

    started = perf_counter()
    pool, pool_kind = _pregen_component_pool(smi, wps)
    if pool is None or len(pool) == 0:
        return pool, pool_kind
    pool = qrs_conf.filter_similar_molecules(pool, rmsd_tol=0.5)
    if cache is not None:
        cache[key] = copy.deepcopy(pool)
    print(
        f"Generated conformer pool ({len(pool)} conformers) in "
        f"{(perf_counter() - started) / 60.0:.1f} min",
        flush=True,
    )
    return pool, pool_kind


def _forcefield_cache_key(smiles: str, ff_style: str = "openff") -> tuple[str, str]:
    """Force-field atom typing depends on chemistry/style, not the lattice cell."""
    return smiles.strip(), ff_style


def _get_cached_forcefield_info(
    cache: dict | None,
    smiles: str,
    ff_style: str = "openff",
):
    if cache is None:
        return None
    entry = cache.get(_forcefield_cache_key(smiles, ff_style))
    if isinstance(entry, dict) and entry.get("_cellnet_forcefield_bundle"):
        info = entry["atom_info"]
    else:
        info = entry
    return copy.deepcopy(info) if info is not None else None


def _materialize_cached_forcefield_files(
    cache: dict | None,
    smiles: str,
    calc_dir: str | Path,
    ff_style: str = "openff",
) -> None:
    """Restore CHARMM topology/parameter files skipped by cached QRS setup."""
    if cache is None:
        return
    entry = cache.get(_forcefield_cache_key(smiles, ff_style))
    if not (
        isinstance(entry, dict)
        and entry.get("_cellnet_forcefield_bundle")
        and entry.get("calc_files")
    ):
        return
    calc_dir = Path(calc_dir)
    calc_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in entry["calc_files"].items():
        (calc_dir / name).write_bytes(contents)


def _store_forcefield_info(
    cache: dict | None,
    smiles: str,
    info,
    calc_dir: str | Path | None = None,
    ff_style: str = "openff",
) -> None:
    if cache is not None and info is not None:
        calc_files: dict[str, bytes] = {}
        if calc_dir is not None:
            calc_path = Path(calc_dir)
            for name in ("pyxtal.rtf", "pyxtal.prm"):
                path = calc_path / name
                if path.is_file():
                    calc_files[name] = path.read_bytes()
        cache[_forcefield_cache_key(smiles, ff_style)] = {
            "_cellnet_forcefield_bundle": True,
            "atom_info": copy.deepcopy(info),
            "calc_files": calc_files,
        }


def build_pyxtal_template(
    smiles: str,
    hall_number: int,
    zprime: float,
    cellpar: np.ndarray,
    seed: int = 0,
    *,
    csd_code: str | None = None,
    db_path: str | Path | None = None,
):
    from pyxtal import pyxtal
    from pyxtal.lattice import Lattice
    from pyxtal.molecule import pyxtal_molecule

    a, b, c, alpha, beta, gamma = cellpar
    system = crystal_system_from_hall(hall_number)
    lattice = Lattice.from_para(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        ltype=system,
        force_symmetry=True,
    )

    if csd_code:
        ref = _load_pyxtal_from_db(csd_code, db_path=db_path)
        if ref is not None:
            ref.lattice = lattice
            if ref.has_special_site():
                ref = ref.to_subgroup()
            return ref
        # A random packing is a different experiment from a CSD template
        # (different conformer, different Wyckoff assignment); never do it quietly.
        print(
            f"  Warning: CSD template {csd_code!r} not found in any PyXtal database; "
            "falling back to a random packing for the conf-QRS template",
            flush=True,
        )

    z_mols = int(round(zprime_to_Z(zprime, hall_number)))
    mol = pyxtal_molecule(smiles.strip() + ".smi", fix=True)
    xtal = pyxtal(molecular=True)
    xtal.from_random(
        dim=3,
        group=hall_number,
        species=[mol],
        numIons=[z_mols],
        lattice=lattice,
        use_hall=True,
        seed=seed,
        max_count=30,
    )
    if xtal.has_special_site():
        xtal = xtal.to_subgroup()
    return xtal


def run_conf_qrs_one(
    dedup_idx: int,
    smiles: str,
    hall_number: int,
    zprime: float,
    cellpar: np.ndarray,
    workdir: Path,
    *,
    csd_code: str | None = None,
    db_path: str | Path | None = None,
    ref_pmg=None,
    ngen: int = 100,
    npop: int = 48,
    nproc: int = 1,
    seed: int = 0,
    restart: int = 1,
    selection_channel: str | None = None,
    component_pool_cache: dict | None = None,
    forcefield_info_cache: dict | None = None,
    min_matches: int = 3,
    relax_lattice: bool = False,
    check_stable: bool = True,
    enable_reference_matching: bool = True,
) -> ConfQRSRecord:
    qrs_conf = _import_qrs_conf_helpers()
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pyxtal.optimize import QRS

    set_random_seed(seed)
    workdir.mkdir(parents=True, exist_ok=True)
    template = build_pyxtal_template(
        smiles,
        hall_number,
        zprime,
        cellpar,
        seed=seed,
        csd_code=csd_code,
        db_path=db_path,
    )

    type_wps: list[list] = [[] for _ in range(len(template.numMols))]
    for site in template.mol_sites:
        type_wps[site.type].append(site.wp)

    molecules = []
    smi = smiles.strip()
    for type_idx, wps in enumerate(type_wps):
        pool, _pool_kind = _get_component_pool(
            smi, wps, qrs_conf, component_pool_cache
        )
        if pool is None or len(pool) == 0:
            raise RuntimeError(f"No conformer pool for component {type_idx} ({smi})")
        if not pool:
            raise RuntimeError(f"All conformers filtered for component {type_idx}")
        molecules.append(pool)

    sites = qrs_conf.build_sites_from_reference(template)

    n_conformers = sum(len(p) for p in molecules)
    composition = [int(a) for a in template.get_zprime()]
    selected_deltas = qrs_conf.select_delta_angle(molecules, composition)
    soft_clash_buffer, _ = qrs_conf.select_soft_clash_buffer(
        molecules, template.lattice, composition
    )

    matcher = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=5.0)
    param_xml = workdir / "parameters.xml"
    if param_xml.exists():
        param_xml.unlink()

    qrs_kwargs = {
        "smiles": smiles,
        "workdir": str(workdir),
        "sg": template.group.hall_number,
        "tag": workdir.name,
        "use_hall": True,
        "lattice": template.lattice,
        "composition": composition,
        "molecules": molecules,
        "sites": sites,
        "N_gen": ngen,
        "N_pop": npop,
        "N_cpu": nproc,
        "random_state": seed,
        "cif": "all.cif",
        "skip_mlp": True,
        "verbose": False,
        "delta_length": 1.1,
        "delta_angle": selected_deltas,
        "matcher": matcher,
        "check_stable": check_stable,
    }
    cached_forcefield_info = _get_cached_forcefield_info(
        forcefield_info_cache, smiles
    )
    if cached_forcefield_info is not None:
        qrs_kwargs["info"] = cached_forcefield_info
        _materialize_cached_forcefield_files(
            forcefield_info_cache, smiles, workdir / "calc"
        )
    import inspect

    supported = set(inspect.signature(QRS.__init__).parameters) - {"self"}
    optional = {
        "soft_clash_check": True,
        "soft_clash_buffer": soft_clash_buffer,
        "close_grid_cutoff": "0",
        "N_min_matches": min_matches,
        "opt_lat": relax_lattice,
    }
    qrs_kwargs.update({k: v for k, v in optional.items() if k in supported})
    setup_started = perf_counter()
    qrs = QRS(**{k: v for k, v in qrs_kwargs.items() if k in supported})
    setup_time_min = (perf_counter() - setup_started) / 60.0
    if cached_forcefield_info is None:
        _store_forcefield_info(
            forcefield_info_cache,
            smiles,
            qrs.atom_info,
            workdir / "calc",
        )
        setup_kind = "generated"
    else:
        setup_kind = "reused"
    print(
        f"Force-field atom info {setup_kind} in {setup_time_min:.2f} min",
        flush=True,
    )
    if relax_lattice and "opt_lat" not in supported:
        # A supplied lattice makes QRS default to fixed-cell CHARMM runs.
        # Retain compatibility with older PyXtal versions lacking the public
        # opt_lat constructor option.
        qrs.opt_lat = True

    t0 = perf_counter()
    if enable_reference_matching and ref_pmg is not None:
        success_rate = qrs.run(ref_pmg=ref_pmg, max_rmsd=0.3)
    else:
        success_rate = qrs.run(max_rmsd=0.3)
    time_min = (perf_counter() - t0) / 60.0

    return ConfQRSRecord(
        dedup_idx=dedup_idx,
        cellpar=cellpar.tolist(),
        success_rate=float(success_rate) if success_rate is not None else None,
        time_min=time_min,
        workdir=str(workdir),
        n_conformers=n_conformers,
        restart=restart,
        seed=seed,
        selection_channel=selection_channel,
        reference_matching_enabled=enable_reference_matching,
        relax_lattice=relax_lattice,
    )


def conf_sweep_had_hit(records: list[ConfQRSRecord]) -> bool:
    """
    True when any conformational-QRS record in a sweep matched the reference.

    A sweep that finishes every selected cell with no match at all is the v9
    outright-failure signature: QAXMEH53, XAFPAY, XULDUD01 and OBEQUJ each
    completed all 72 cells with zero hits. ``success_rate is None`` marks a run
    that raised, and counts as no hit.
    """
    return any(
        record.success_rate is not None and float(record.success_rate) > 0.0
        for record in records
    )


def conf_sweep_all_errored(records: list[ConfQRSRecord]) -> bool:
    """
    True when every record in a non-empty sweep raised instead of finishing.

    A sweep like this says nothing about the lattice: a broken force-field
    install, a missing CHARMM binary or a bad conformer pool fails identically
    with the lattice free or frozen. The adaptive fallback skips it rather than
    spending a second sweep reproducing the same error.
    """
    return bool(records) and all(record.error for record in records)


def summarize_adaptive_sweeps(
    *,
    enabled: bool,
    primary_relax_lattice: bool,
    primary: list[ConfQRSRecord],
    fallback: list[ConfQRSRecord] | None,
    fallback_root: str | None,
    skipped_reason: str | None = None,
) -> dict:
    """
    Machine-readable account of an adaptive-relax run for the pipeline summary.

    ``conf_qrs`` in the summary JSON holds both passes back to back, so a
    consumer that counts any positive success rate (the v8/v9 suite
    summarizers) would silently credit fallback hits as v9-protocol coverage.
    This block keeps the two passes countable on their own.
    """

    def _hits(records: list[ConfQRSRecord]) -> int:
        return sum(
            1
            for r in records
            if r.success_rate is not None and float(r.success_rate) > 0.0
        )

    primary_hit = conf_sweep_had_hit(primary)
    block = {
        "enabled": bool(enabled),
        "primary_relax_lattice": bool(primary_relax_lattice),
        "primary_n_runs": len(primary),
        "primary_n_hit": _hits(primary),
        "primary_all_errored": conf_sweep_all_errored(primary),
        "fallback_ran": fallback is not None,
        "fallback_relax_lattice": (not primary_relax_lattice) if fallback is not None else None,
        "fallback_root": fallback_root,
        "fallback_n_runs": len(fallback) if fallback is not None else 0,
        "fallback_n_hit": _hits(fallback) if fallback is not None else 0,
        "fallback_skipped_reason": skipped_reason,
    }
    block["recovered_by_fallback"] = (
        not primary_hit and fallback is not None and conf_sweep_had_hit(fallback)
    )
    return block


def conf_restart_seed(
    base_seed: int,
    dedup_idx: int,
    restart: int,
    seed_stride: int = 100_000,
) -> int:
    """Return a deterministic one-based restart seed; restart 1 is v5-compatible."""
    if restart < 1:
        raise ValueError("restart must be >= 1")
    if seed_stride < 1:
        raise ValueError("seed_stride must be >= 1")
    return int(base_seed) + int(dedup_idx) + (int(restart) - 1) * int(seed_stride)


def conf_restart_workdir(
    out_dir: str | Path,
    dedup_idx: int,
    restart: int,
    n_restarts: int,
) -> Path:
    """Return collision-free restart paths while preserving the v5 single path."""
    if restart < 1 or n_restarts < 1 or restart > n_restarts:
        raise ValueError("restart must be in [1, n_restarts]")
    root = Path(out_dir)
    legacy = root / f"conf_qrs_{int(dedup_idx):03d}"
    if n_restarts == 1:
        return legacy
    return root / f"{legacy.name}_restart_{int(restart):02d}"


def conf_restart_schedule(
    out_dir: str | Path,
    dedup_idx: int,
    n_restarts: int,
    base_seed: int,
    seed_stride: int = 100_000,
) -> list[tuple[int, int, Path]]:
    """Materialize every fixed restart before execution."""
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    return [
        (
            restart,
            conf_restart_seed(base_seed, dedup_idx, restart, seed_stride),
            conf_restart_workdir(out_dir, dedup_idx, restart, n_restarts),
        )
        for restart in range(1, int(n_restarts) + 1)
    ]


def save_pipeline_artifacts(
    result: PipelineResult,
    out_dir: Path,
    *,
    flow_seed: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = result.tag

    save_flow_batch(
        result.flow,
        flow_npz_path(out_dir, tag, result.flow.k),
        seed=flow_seed,
    )

    lat_path = out_dir / f"{tag}_lattice_qrs.csv"
    with lat_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "flow_idx",
                "lambda1",
                "lambda2",
                "lambda3",
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
                "qrs_loss",
                "density",
                "lambda_mse",
                "lambda_recip_mse",
                "source",
            ]
        )
        for r in result.lattice_qrs:
            cp = r.cellpar
            lam = np.exp(compute_log_successive_minima(cp))
            w.writerow(
                [
                    r.flow_idx,
                    *lam.tolist(),
                    *cp.tolist(),
                    r.qrs_loss,
                    r.density,
                    r.lambda_mse,
                    r.lambda_recip_mse,
                    getattr(r, "source", "qrs"),
                ]
            )

    dedup_path = out_dir / f"{tag}_unique_cells.csv"
    with dedup_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dedup_idx",
                "n_sources",
                "source_flow_indices",
                "sources",
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
            ]
        )
        for u in result.unique_cells:
            cp = u.cellpar
            w.writerow(
                [
                    u.dedup_idx,
                    len(u.source_flow_indices),
                    ";".join(str(i) for i in u.source_flow_indices),
                    ";".join(_unique_cell_sources(u)),
                    *cp.tolist(),
                ]
            )

    conf_path = out_dir / f"{tag}_conf_qrs.csv"
    with conf_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dedup_idx",
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
                "success_rate",
                "time_min",
                "n_conformers",
                "workdir",
                "error",
                "restart",
                "seed",
                "selection_channel",
            ]
        )
        for c in result.conf_qrs:
            cp = c.cellpar
            w.writerow(
                [
                    c.dedup_idx,
                    *cp,
                    c.success_rate if c.success_rate is not None else "",
                    c.time_min,
                    c.n_conformers,
                    c.workdir,
                    c.error or "",
                    getattr(c, "restart", 1),
                    getattr(c, "seed", None) if getattr(c, "seed", None) is not None else "",
                    getattr(c, "selection_channel", None) or "",
                ]
            )

    summary = {
        "tag": result.tag,
        "smiles": result.smiles,
        "hall_number": result.hall_number,
        "zprime": result.zprime,
        "true_cellpar": result.true_cellpar.tolist() if result.true_cellpar is not None else None,
        "k_flow": result.flow.k,
        "n_lattice_qrs": len(result.lattice_qrs),
        "n_unique_cells": len(result.unique_cells),
        "top_lattice_matches": [
            {
                "rank": i + 1,
                "flow_idx": m.flow_idx,
                "length_mape_pct": m.length_mape_pct,
                "mape_a_pct": m.mape_a_pct,
                "mape_b_pct": m.mape_b_pct,
                "mape_c_pct": m.mape_c_pct,
                "angle_mae_deg": m.angle_mae_deg,
                "angle_errs_deg": [
                    None if np.isnan(x) else float(x) for x in m.angle_errs_deg
                ],
                "beta_mae_deg": m.angle_mae_deg,
                "aligned_cellpar": m.aligned_cellpar.tolist(),
                "raw_cellpar": m.raw_cellpar.tolist(),
                "qrs_loss": m.qrs_loss,
            }
            for i, m in enumerate(result.top_lattice_matches)
        ],
        "conf_qrs": [
            {
                "dedup_idx": c.dedup_idx,
                "cellpar": c.cellpar,
                "success_rate": c.success_rate,
                "time_min": c.time_min,
                "workdir": c.workdir,
                "n_conformers": c.n_conformers,
                "error": c.error,
                "restart": getattr(c, "restart", 1),
                "seed": getattr(c, "seed", None),
                "selection_channel": getattr(c, "selection_channel", None),
                "relax_lattice": getattr(c, "relax_lattice", False),
                "reference_matching_enabled": getattr(
                    c, "reference_matching_enabled", True
                ),
            }
            for c in result.conf_qrs
        ],
        "adaptive_relax": result.adaptive_relax,
    }
    (out_dir / f"{tag}_pipeline_summary.json").write_text(json.dumps(summary, indent=2))
