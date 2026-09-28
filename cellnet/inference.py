"""Load the conditional lattice-flow checkpoint and prepare inference inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from cellnet.data import DatasetStats
from cellnet.graph import smiles_to_graph
from cellnet.models import GivenHZConditionalLatticeFlowGNN


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
    model_name = ckpt.get("model_name", args.get("model", "given_hz_conditional_lattice_flow_gnn"))
    return {
        "path": str(path),
        "model_name": str(model_name),
        "epoch": ckpt.get("epoch"),
    }


def load_flow_qrs_model(
    checkpoint: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, DatasetStats, dict, str]:
    """Load a conditional lattice-flow checkpoint for QRS inference."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_name = ckpt.get(
        "model_name", ckpt.get("args", {}).get("model", "given_hz_conditional_lattice_flow_gnn")
    )
    if model_name == "given_hz_conditional_lattice_flow_gnn":
        model, stats, args = load_given_hz_conditional_lattice_flow(checkpoint, device)
        return model, stats, args, "lattice_flow"
    raise ValueError(
        f"Unsupported checkpoint model {model_name!r}; "
        "expected given_hz_conditional_lattice_flow_gnn"
    )


def denorm_log_density(log_density_norm: float, stats: DatasetStats) -> float:
    """Map normalized log ρ back to physical density (g/cm³)."""
    log_rho = float(log_density_norm * stats.log_density_std + stats.log_density_mean)
    return float(np.exp(log_rho))
