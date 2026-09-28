"""End-to-end inference: lattice-flow model → QRS cell recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from cellnet.data import DatasetStats
from cellnet.graph import smiles_to_graph
from cellnet.models import GivenHZConditionalLatticeFlowGNN, GivenHZLatticeFlowGNN
from cellnet.hybrid_qrs import (
    ConflictResolutionConfig,
    adaptive_hybrid_weights,
    apply_lambda_volume_constraint,
    selling_flow_uncertainty,
    target_volume_from_density,
)
from cellnet.qrs import QRSConfig, QRSResult, qrs_search_cellpar
from cellnet.sequential import (
    denorm_log_lambda,
    denorm_log_reciprocal_lambda,
    denorm_selling_log1p,
    lattice_flow_slices,
    sample_k_given_hz_conditional_lattice_flow,
    sample_k_given_hz_lattice_flow,
    split_lattice_flow_norm,
)
from cellnet.symmetry import (
    AXIS_PERMUTATIONS,
    align_orthorhombic_cellpar_to_axis_ranks,
    cellpar_to_free,
    select_axis_rank_permutation,
)


@dataclass
class ModelSellingQRSResult:
    """Combined lattice-flow prediction + QRS reconstruction."""

    cellpar: np.ndarray
    target_rho: float
    pred_log_density_norm: float
    pred_selling: np.ndarray
    sample_idx: int
    qrs: QRSResult
    all_qrs: list[QRSResult]
    all_pred_selling: np.ndarray
    pred_log_lambda: np.ndarray | None = None
    all_pred_log_lambda: np.ndarray | None = None
    pred_log_lambda_recip: np.ndarray | None = None
    all_pred_log_lambda_recip: np.ndarray | None = None
    w_lambda_per_sample: list[float] | None = None
    w_selling_per_sample: list[float] | None = None
    flow_uncertainty: float = 0.0
    target_volume: float | None = None
    lambda_volume_clipped: list[bool] | None = None


def _resolve_device(model: torch.nn.Module, device: torch.device | None) -> torch.device:
    if device is None:
        return next(model.parameters()).device
    return device


def _prepare_batch(
    stats: DatasetStats,
    smiles: str,
    hall_number: int,
    zprime: float,
    device: torch.device,
) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    graph = smiles_to_graph(smiles)
    if graph is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if hall_number not in stats.hall_to_idx:
        raise ValueError(f"Hall {hall_number} not in training vocabulary")
    hall_idx = stats.hall_to_idx[hall_number]
    z_idx = min(
        range(len(stats.zprime_values)),
        key=lambda i: abs(stats.zprime_values[i] - zprime),
    )
    batch = Batch.from_data_list([graph]).to(device)
    hall_t = torch.tensor([hall_idx], dtype=torch.long, device=device)
    zprime_t = torch.tensor([z_idx], dtype=torch.long, device=device)
    return batch, hall_t, zprime_t


def load_given_hz_lattice_flow(
    checkpoint: str | Path,
    device: torch.device | None = None,
) -> tuple[GivenHZLatticeFlowGNN, DatasetStats, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_path = Path(checkpoint)
    stats = DatasetStats.load(ckpt_path.parent / "stats.json")
    model_args = ckpt.get("args", ckpt)
    model = GivenHZLatticeFlowGNN(
        node_dim=stats.n_node_features,
        edge_dim=stats.n_edge_features,
        lattice_flow_dim=stats.lattice_flow_dim,
        n_halls=stats.n_halls,
        n_zprimes=stats.n_zprimes,
        hidden_dim=model_args.get("hidden_dim", 512),
        gnn_layers=model_args.get("gnn_layers", 4),
        dropout=model_args.get("dropout", 0.3),
        flow_steps=model_args.get("flow_steps", 20),
        gnn_pooling=model_args.get("gnn_pooling", "mean"),
        n_shape_bins=model_args.get("shape_bins", 0),
        shape_emb_dim=model_args.get("shape_emb_dim", 16),
        shape_exploration=model_args.get("shape_exploration", 0.10),
        predict_axis_permutation=model_args.get("w_axis_permutation", 0.0) > 0.0,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    model = model.to(device)
    model.eval()
    return model, stats, model_args


def load_given_hz_conditional_lattice_flow(
    checkpoint: str | Path,
    device: torch.device | None = None,
) -> tuple[GivenHZConditionalLatticeFlowGNN, DatasetStats, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_path = Path(checkpoint)
    stats = DatasetStats.load(ckpt_path.parent / "stats.json")
    model_args = ckpt.get("args", ckpt)
    lambda_dim = int(stats.log_lambda_dim + stats.log_lambda_recip_dim)
    model = GivenHZConditionalLatticeFlowGNN(
        node_dim=stats.n_node_features,
        edge_dim=stats.n_edge_features,
        selling_dim=stats.selling_dim,
        lambda_dim=lambda_dim,
        n_halls=stats.n_halls,
        n_zprimes=stats.n_zprimes,
        hidden_dim=model_args.get("hidden_dim", 512),
        gnn_layers=model_args.get("gnn_layers", 4),
        dropout=model_args.get("dropout", 0.3),
        flow_steps=model_args.get("flow_steps", 20),
        gnn_pooling=model_args.get("gnn_pooling", "mean"),
        predict_axis_permutation=model_args.get("w_axis_permutation", 0.0) > 0.0,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    model = model.to(device)
    model.eval()
    return model, stats, model_args


def describe_flow_checkpoint(checkpoint: str | Path) -> dict[str, str | int | None]:
    """Lightweight checkpoint metadata (path, model_name, epoch) without building the model."""
    path = Path(checkpoint).resolve()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model_name = ckpt.get("model_name", args.get("model", "given_hz_lattice_flow_gnn"))
    return {
        "path": str(path),
        "model_name": str(model_name),
        "epoch": ckpt.get("epoch"),
    }


def load_flow_qrs_model(
    checkpoint: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, DatasetStats, dict, str]:
    """Load a joint or conditional lattice-flow checkpoint for QRS inference."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", ckpt.get("args", {}).get("model", "given_hz_lattice_flow_gnn"))
    if model_name == "given_hz_lattice_flow_gnn":
        model, stats, args = load_given_hz_lattice_flow(checkpoint, device)
        return model, stats, args, "lattice_flow"
    if model_name == "given_hz_conditional_lattice_flow_gnn":
        model, stats, args = load_given_hz_conditional_lattice_flow(checkpoint, device)
        return model, stats, args, "lattice_flow"
    raise ValueError(
        f"Unsupported checkpoint model {model_name!r}; "
        "expected given_hz_lattice_flow_gnn or given_hz_conditional_lattice_flow_gnn"
    )


def denorm_log_density(log_density_norm: float, stats: DatasetStats) -> float:
    """Map normalized log ρ back to physical density (g/cm³)."""
    log_rho = float(log_density_norm * stats.log_density_std + stats.log_density_mean)
    return float(np.exp(log_rho))


def reconstruct_cellpar_from_lattice_model(
    model: GivenHZLatticeFlowGNN | GivenHZConditionalLatticeFlowGNN,
    stats: DatasetStats,
    smiles: str,
    hall_number: int,
    zprime: float,
    qrs_config: QRSConfig | None = None,
    k: int = 10,
    target_rho_override: float | None = None,
    use_lambda: bool = True,
    use_lambda_recip: bool = True,
    use_selling: bool = True,
    conflict_resolution: ConflictResolutionConfig | None = None,
    initial_cellpar: np.ndarray | None = None,
    verbose: bool = False,
    device: torch.device | None = None,
    qrs_seed: int | None = None,
) -> ModelSellingQRSResult:
    """Joint lattice flow → K paired (Selling, log λ, log λ*) samples → QRS → pick best."""
    device = _resolve_device(model, device)
    if hall_number is None or zprime is None:
        raise ValueError("hall_number and zprime are required")
    batch, hall_t, zprime_t = _prepare_batch(stats, smiles, hall_number, zprime, device)
    with torch.no_grad():
        if isinstance(model, GivenHZConditionalLatticeFlowGNN):
            samples_norm, pred_ld = sample_k_given_hz_conditional_lattice_flow(
                model, batch, hall_t, zprime_t, k=k
            )
        else:
            samples_norm, pred_ld = sample_k_given_hz_lattice_flow(
                model, batch, hall_t, zprime_t, k=k
            )

    pred_ld_norm = float(pred_ld.squeeze(0).cpu().numpy())
    target_rho = denorm_log_density(pred_ld_norm, stats)
    if target_rho_override is not None:
        target_rho = float(target_rho_override)

    flow_norm = samples_norm.squeeze(0).cpu().numpy()
    pred_selling_norm, pred_lam_norm, pred_lam_r_norm = split_lattice_flow_norm(flow_norm, stats)
    all_selling = np.stack(
        [denorm_selling_log1p(pred_selling_norm[j], stats) for j in range(k)],
        axis=0,
    )

    cr_cfg = conflict_resolution if conflict_resolution is not None else ConflictResolutionConfig()
    target_volume = target_volume_from_density(target_rho, smiles, zprime, hall_number)

    all_log_lambda = denorm_log_lambda(pred_lam_norm, stats)
    all_log_lambda_recip = denorm_log_reciprocal_lambda(pred_lam_r_norm, stats)
    if use_lambda and cr_cfg.lambda_volume_constraint:
        all_log_lambda, _, lambda_clipped = apply_lambda_volume_constraint(
            all_log_lambda,
            target_volume,
            cr_cfg,
            stats=stats,
        )
    else:
        lambda_clipped = [False] * k

    axis_logits = getattr(model, "_last_axis_permutation_logits", None)
    if axis_logits is not None:
        raw_axis_probs = torch.softmax(axis_logits, dim=-1).squeeze(0).cpu().numpy()
    else:
        prior = np.asarray(
            stats.axis_permutation_priors.get(hall_number, []),
            dtype=np.float64,
        )
        if prior.size != len(AXIS_PERMUTATIONS):
            prior = np.zeros(len(AXIS_PERMUTATIONS), dtype=np.float64)
            prior[0] = 1.0
        raw_axis_probs = np.repeat(prior[np.newaxis, :], k, axis=0)
    all_axis_ranks = [
        select_axis_rank_permutation(probabilities, hall_number)[0]
        for probabilities in raw_axis_probs
    ]

    global_spread, sample_outlier = selling_flow_uncertainty(pred_selling_norm)
    base_cfg = qrs_config or QRSConfig(
        w_selling=2.0 if use_selling else 0.0,
        w_density=2.0,
        w_lambda=2.0 if use_lambda else 0.0,
        w_lambda_recip=2.0 if use_lambda_recip else 0.0,
    )

    w_lambda_per: list[float] = []
    w_selling_per: list[float] = []
    all_qrs: list[QRSResult] = []
    for j in range(k):
        w_s, w_l = adaptive_hybrid_weights(
            base_cfg.w_selling if use_selling else 0.0,
            base_cfg.w_lambda if use_lambda else 0.0,
            global_spread,
            float(sample_outlier[j]),
            0.0,
            cr_cfg,
        )
        w_selling_per.append(w_s)
        w_lambda_per.append(w_l)
        cfg_j = replace(
            base_cfg,
            w_selling=w_s if use_selling else 0.0,
            w_lambda=w_l if use_lambda else 0.0,
            w_lambda_recip=base_cfg.w_lambda_recip if use_lambda_recip else 0.0,
        )
        if qrs_seed is not None:
            cfg_j = replace(cfg_j, seed=qrs_seed + j)
        qrs = qrs_search_cellpar(
            smiles=smiles,
            hall_number=hall_number,
            zprime=zprime,
            target_rho=target_rho,
            config=cfg_j,
            target_selling=all_selling[j] if use_selling else None,
            target_log_lambdas=all_log_lambda[j] if use_lambda else None,
            target_log_lambdas_recip=all_log_lambda_recip[j] if use_lambda_recip else None,
            initial_cellpar=initial_cellpar,
            verbose=verbose and j == 0,
        )
        qrs.cellpar = align_orthorhombic_cellpar_to_axis_ranks(
            qrs.cellpar,
            all_axis_ranks[j],
            hall_number,
        )
        qrs.free = cellpar_to_free(qrs.cellpar, hall_number)
        all_qrs.append(qrs)

    best_idx = int(np.argmin([q.loss for q in all_qrs]))
    best_qrs = all_qrs[best_idx]
    return ModelSellingQRSResult(
        cellpar=best_qrs.cellpar,
        target_rho=target_rho,
        pred_log_density_norm=pred_ld_norm,
        pred_selling=all_selling[best_idx],
        sample_idx=best_idx,
        qrs=best_qrs,
        all_qrs=all_qrs,
        all_pred_selling=all_selling,
        pred_log_lambda=all_log_lambda[best_idx] if use_lambda else None,
        all_pred_log_lambda=all_log_lambda if use_lambda else None,
        pred_log_lambda_recip=all_log_lambda_recip[best_idx] if use_lambda_recip else None,
        all_pred_log_lambda_recip=all_log_lambda_recip if use_lambda_recip else None,
        w_lambda_per_sample=w_lambda_per,
        w_selling_per_sample=w_selling_per,
        flow_uncertainty=global_spread,
        target_volume=target_volume,
        lambda_volume_clipped=lambda_clipped if lambda_clipped else None,
    )
