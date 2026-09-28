#!/usr/bin/env python3
"""
Train lattice-flow GNN models on SPaDe-CSP data.

Models:
  - given_hz_lattice_flow_gnn: joint 12-d flow (Selling + log λ + log λ*)
  - given_hz_conditional_lattice_flow_gnn: Selling flow, then conditional λ flow

Example:
  python scripts/train.py --model given_hz_conditional_lattice_flow_gnn --csv datasets/spade-csp/spade_train_precomputed.csv
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellnet.data import (
    CrystalGraphDataset,
    attach_lattice_invariants,
    attach_sequential_targets,
    attach_selling_targets,
    compute_stats,
    graphify_samples,
    load_hem_database,
)
from cellnet.metrics import (
    given_hz_conditional_lattice_flow_loss,
    given_hz_lattice_flow_loss,
)
from cellnet.models import GivenHZConditionalLatticeFlowGNN, GivenHZLatticeFlowGNN
from cellnet.polymorph import SellingPolymorphBank
from cellnet.rare_shapes import RareShapeWeighter
from cellnet.precomputed import (
    HybridGraphStore,
    ShardGroupedBatchSampler,
    has_sharded_graphs,
    is_precomputed_csv,
    load_precomputed_csv,
    precomputed_graphs_path,
    sharded_graphs_meta_path,
)
from cellnet.retrieval import ShapeBinRetrievalIndex
from cellnet.splits import split_by_smiles_group
from cellnet.spade import load_structure_csv


def load_compatible_state_dict(model: torch.nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    """Load checkpoint weights with matching names and shapes; skip incompatible tensors."""
    current = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in current and current[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    skipped = [k for k in state_dict if k not in filtered]
    return missing, unexpected, skipped


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_gnn_model(model_name: str) -> bool:
    return model_name.endswith("_gnn")


def uses_given_hz_lattice_flow(model_name: str) -> bool:
    return model_name == "given_hz_lattice_flow_gnn"


def uses_given_hz_conditional_lattice_flow(model_name: str) -> bool:
    return model_name == "given_hz_conditional_lattice_flow_gnn"


def uses_given_hz_lattice_flow_any(model_name: str) -> bool:
    return uses_given_hz_lattice_flow(model_name) or uses_given_hz_conditional_lattice_flow(model_name)


def get_batch_tensors(batch, device, gnn: bool):
    batch = batch.to(device)
    return (
        batch,
        batch.y.view(batch.num_graphs, -1),
        batch.hall.view(-1),
        batch.zprime.view(-1),
        batch.log_density.view(-1),
    )


def build_model(name: str, stats, args):
    gnn_common = dict(
        node_dim=stats.n_node_features,
        edge_dim=stats.n_edge_features,
        n_halls=stats.n_halls,
        n_zprimes=stats.n_zprimes,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
    )
    if name == "given_hz_lattice_flow_gnn":
        gnn_ps = {**gnn_common, "dropout": args.dropout}
        return (
            GivenHZLatticeFlowGNN(
                **gnn_ps,
                lattice_flow_dim=stats.lattice_flow_dim,
                flow_steps=args.flow_steps,
                gnn_pooling=args.gnn_pooling,
                n_shape_bins=args.shape_bins,
                shape_emb_dim=args.shape_emb_dim,
                shape_exploration=args.shape_exploration,
                predict_axis_permutation=args.w_axis_permutation > 0.0,
            ),
            "lattice_flow",
            "lattice_flow",
        )
    if name == "given_hz_conditional_lattice_flow_gnn":
        gnn_ps = {**gnn_common, "dropout": args.dropout}
        lambda_dim = int(stats.log_lambda_dim + stats.log_lambda_recip_dim)
        return (
            GivenHZConditionalLatticeFlowGNN(
                **gnn_ps,
                selling_dim=stats.selling_dim,
                lambda_dim=lambda_dim,
                flow_steps=args.flow_steps,
                gnn_pooling=args.gnn_pooling,
                predict_axis_permutation=args.w_axis_permutation > 0.0,
            ),
            "lattice_flow",
            "lattice_flow",
        )
    raise ValueError(
        f"Unknown model: {name!r}; "
        "expected given_hz_lattice_flow_gnn or given_hz_conditional_lattice_flow_gnn"
    )


def _compute_lattice_loss(
    model_name: str,
    model,
    x,
    y,
    hall,
    zprime,
    log_density,
    stats,
    args,
    polymorph_bank,
    batch,
    gnn: bool,
    device: torch.device,
):
    s_mean = torch.as_tensor(stats.selling_mean, device=device, dtype=torch.float32)
    s_std = torch.as_tensor(stats.selling_std, device=device, dtype=torch.float32)
    lam_mean = torch.as_tensor(stats.log_lambda_mean, device=device, dtype=torch.float32)
    lam_std = torch.as_tensor(stats.log_lambda_std, device=device, dtype=torch.float32)
    lam_r_mean = torch.as_tensor(stats.log_lambda_recip_mean, device=device, dtype=torch.float32)
    lam_r_std = torch.as_tensor(stats.log_lambda_recip_std, device=device, dtype=torch.float32)
    poly_lat, poly_ld = _lattice_polymorph_tensors(
        polymorph_bank, batch, gnn, hall, zprime, stats, device,
        fallback_lattice=y, fallback_log_density=log_density,
    )
    sample_weight = _batch_lattice_weight(batch, gnn, device)
    axis_permutation_mask = _batch_axis_permutation_mask(batch, gnn, device)
    if uses_given_hz_lattice_flow(model_name):
        shape_bin = _batch_lattice_shape_bin(batch, gnn, device)
        out = model(x, hall, zprime, x1=y, shape_bin=shape_bin)
        loss, _ = given_hz_lattice_flow_loss(
            out, y, log_density, s_mean, s_std, lam_mean, lam_std, lam_r_mean, lam_r_std,
            selling_dim=stats.selling_dim,
            log_lambda_dim=stats.log_lambda_dim,
            w_flow=args.w_flow,
            w_s=args.w_volume,
            w_density=args.w_density,
            w_lambda=args.w_lambda,
            w_lambda_recip=args.w_lambda_recip,
            lambda_loss_mode=args.lambda_loss_mode,
            w_lambda_relative=args.w_lambda_relative,
            w_lambda_product=args.w_lambda_product,
            polymorph_lattice=poly_lat,
            polymorph_log_density=poly_ld,
            sample_weight=sample_weight,
            target_shape_bin=shape_bin,
            w_shape_bin=args.w_shape_bin,
            target_axis_permutation_mask=axis_permutation_mask,
            w_axis_permutation=args.w_axis_permutation,
        )
        return loss
    out = model(x, hall, zprime, x1=y)
    loss, _ = given_hz_conditional_lattice_flow_loss(
        out, y, log_density, s_mean, s_std, lam_mean, lam_std, lam_r_mean, lam_r_std,
        selling_dim=stats.selling_dim,
        log_lambda_dim=stats.log_lambda_dim,
        w_flow=args.w_flow,
        w_flow_lambda=args.w_flow_lambda,
        w_s=args.w_volume,
        w_density=args.w_density,
        w_lambda=args.w_lambda,
        w_lambda_recip=args.w_lambda_recip,
        lambda_loss_mode=args.lambda_loss_mode,
        w_lambda_relative=args.w_lambda_relative,
        w_lambda_product=args.w_lambda_product,
        polymorph_lattice=poly_lat,
        polymorph_log_density=poly_ld,
        sample_weight=sample_weight,
        target_axis_permutation_mask=axis_permutation_mask,
        w_axis_permutation=args.w_axis_permutation,
    )
    return loss


def _batch_smiles(batch, gnn: bool) -> list[str]:
    if gnn:
        return list(batch.smiles)
    return list(batch["smiles"])


def _batch_lattice_weight(batch, gnn: bool, device: torch.device) -> torch.Tensor | None:
    if gnn and hasattr(batch, "lattice_weight"):
        return batch.lattice_weight.view(-1).to(device)
    if not gnn and "lattice_weight" in batch:
        return batch["lattice_weight"].view(-1).to(device)
    return None


def _batch_lattice_shape_bin(batch, gnn: bool, device: torch.device) -> torch.Tensor | None:
    if gnn and hasattr(batch, "lattice_shape_bin"):
        return batch.lattice_shape_bin.view(-1).to(device)
    if not gnn and "lattice_shape_bin" in batch:
        return batch["lattice_shape_bin"].view(-1).to(device)
    return None


def _batch_axis_permutation_mask(
    batch,
    gnn: bool,
    device: torch.device,
) -> torch.Tensor | None:
    if gnn and hasattr(batch, "axis_permutation_mask"):
        return batch.axis_permutation_mask.view(-1, 6).to(device)
    if not gnn and "axis_permutation_mask" in batch:
        return batch["axis_permutation_mask"].view(-1, 6).to(device)
    return None


def _lattice_polymorph_tensors(
    polymorph_bank: SellingPolymorphBank | None,
    batch,
    gnn: bool,
    hall: torch.Tensor,
    zprime: torch.Tensor,
    stats,
    device: torch.device,
    fallback_lattice: torch.Tensor | None = None,
    fallback_log_density: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if polymorph_bank is None:
        return None, None
    return polymorph_bank.batch_lattice_tensors(
        _batch_smiles(batch, gnn),
        hall,
        zprime,
        stats.idx_to_hall,
        stats.zprime_values,
        device,
        fallback_lattice=fallback_lattice,
        fallback_log_density=fallback_log_density,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    model_name,
    stats=None,
    args=None,
    polymorph_bank: SellingPolymorphBank | None = None,
    epoch: int | None = None,
    show_progress: bool = True,
):
    model.train()
    gnn = is_gnn_model(model_name)
    totals = []
    desc = f"Train epoch {epoch}" if epoch is not None else "Train"
    batch_iter = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        disable=not show_progress,
        leave=False,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.2,
    )
    for batch in batch_iter:
        x, y, hall, zprime, log_density = get_batch_tensors(batch, device, gnn)
        optimizer.zero_grad()
        loss = _compute_lattice_loss(
            model_name, model, x, y, hall, zprime, log_density,
            stats, args, polymorph_bank, batch, gnn, device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals.append(loss.item())
        if show_progress:
            batch_iter.set_postfix(loss=f"{float(np.mean(totals)):.4f}")

    return float(np.mean(totals))


def main():
    parser = argparse.ArgumentParser(description="Train CellNet models")
    parser.add_argument("--db", type=str, default=str(ROOT / "HEM.db"))
    parser.add_argument("--csv", type=str, default=None, help="Training CSV (HEM or SPaDe format)")
    parser.add_argument("--test-csv", type=str, default=None, help="Held-out test CSV (SPaDe split)")
    parser.add_argument(
        "--precomputed-graphs",
        type=str,
        default=None,
        help="PyG graph sidecar .pt aligned with --csv (skips graphify and target prep)",
    )
    parser.add_argument(
        "--test-precomputed-graphs",
        type=str,
        default=None,
        help="PyG graph sidecar .pt aligned with --test-csv",
    )
    parser.add_argument(
        "--eager-graphs",
        action="store_true",
        help="Keep more graph shards hot in RAM (8 vs 4); warm OS disk cache for shards",
    )
    parser.add_argument(
        "--preload-all-graphs",
        action="store_true",
        help="Load every train/val graph shard at startup (needs lots of RAM; can OOM on GPU nodes)",
    )
    parser.add_argument(
        "--warm-graph-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-graphify unique SMILES when shards are unavailable (default: on)",
    )
    parser.add_argument("--warm-graph-workers", type=int, default=8)
    parser.add_argument(
        "--graph-cache-size",
        type=int,
        default=0,
        help="SMILES LRU cache size (0=unlimited when warming unique SMILES)",
    )
    parser.add_argument(
        "--graph-hot-shards",
        type=int,
        default=4,
        help="Number of graph shards (~10k graphs each) to keep in RAM",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="given_hz_lattice_flow_gnn",
        choices=[
            "given_hz_lattice_flow_gnn",
            "given_hz_conditional_lattice_flow_gnn",
        ],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--gnn-layers", type=int, default=4, help="GNN message-passing layers")
    parser.add_argument(
        "--gnn-pooling",
        choices=["mean", "multiscale"],
        default="mean",
        help="Graph pooling; multiscale keeps molecular size/extent via mean+sum+max",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs"))
    parser.add_argument("--w-flow", type=float, default=1.0, help="Flow velocity loss weight")
    parser.add_argument(
        "--w-flow-lambda",
        type=float,
        default=None,
        help="Conditional λ-flow velocity weight (defaults to --w-flow)",
    )
    parser.add_argument("--w-volume", type=float, default=0.5, help="Selling endpoint loss weight")
    parser.add_argument("--w-density", type=float, default=0.25, help="Density regression loss weight")
    parser.add_argument("--w-lambda", type=float, default=1.0, help="log(λ) flow endpoint loss weight")
    parser.add_argument("--w-lambda-recip", type=float, default=1.0, help="log(λ*) flow endpoint loss weight")
    parser.add_argument(
        "--lambda-loss-mode",
        type=str,
        default="phys_log",
        choices=["phys_log", "norm", "relative"],
        help="λ endpoint loss: physical log MSE (default), normalized MSE, or relative linear λ",
    )
    parser.add_argument(
        "--w-lambda-relative",
        type=float,
        default=0.0,
        help="Extra relative linear-λ loss weight (added to phys_log mode)",
    )
    parser.add_argument(
        "--w-lambda-product",
        type=float,
        default=0.0,
        help="Weight for log(λ₁λ₂λ₃) product match (reduces compensating errors)",
    )
    parser.add_argument(
        "--freeze-except-lambda",
        action="store_true",
        help="Fine-tune only lambda_flow_head (keep encoder/Selling/axis frozen)",
    )
    parser.add_argument(
        "--rare-shape-max-weight",
        type=float,
        default=1.0,
        help="Maximum conditional-quantile weight on lattice flow loss (1 disables)",
    )
    parser.add_argument(
        "--rare-shape-weight-power",
        type=float,
        default=2.0,
        help="Tail emphasis exponent for rare lattice-shape weighting",
    )
    parser.add_argument(
        "--shape-bins",
        type=int,
        default=0,
        help="Conditional anisotropy-quantile bins for joint lattice flow (0 disables)",
    )
    parser.add_argument("--shape-emb-dim", type=int, default=16)
    parser.add_argument(
        "--shape-exploration",
        type=float,
        default=0.10,
        help="Uniform probability mixed into predicted shape-bin sampling",
    )
    parser.add_argument(
        "--w-shape-bin",
        type=float,
        default=0.3,
        help="Auxiliary anisotropy-bin classification loss weight",
    )
    parser.add_argument(
        "--w-axis-permutation",
        type=float,
        default=0.3,
        help=(
            "Hall-aware orthorhombic axis-assignment classification weight "
            "(0 disables the prediction head)"
        ),
    )
    parser.add_argument("--shape-retrieval-k", type=int, default=16)
    parser.add_argument(
        "--shape-retrieval-blend",
        type=float,
        default=0.85,
        help="Maximum retrieval-prior blend at inference (0 disables index)",
    )
    parser.add_argument(
        "--split-by-smiles",
        dest="split_by_smiles",
        action="store_true",
        help="Split train/val/test by SMILES (keeps polymorphs in one partition)",
    )
    parser.add_argument(
        "--no-split-by-smiles",
        dest="split_by_smiles",
        action="store_false",
        help="Random structure split (legacy)",
    )
    parser.add_argument(
        "--polymorph-loss",
        dest="polymorph_loss",
        action="store_true",
        help="Best-of-K polymorph loss for Selling flow (train only)",
    )
    parser.add_argument(
        "--no-polymorph-loss",
        dest="polymorph_loss",
        action="store_false",
        help="Disable polymorph best-of-K loss",
    )
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Load weights before training (strict=False)")
    parser.add_argument("--flow-steps", type=int, default=20, help="ODE steps for flow sampling at inference")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable batch-level tqdm progress bars during training",
    )
    parser.add_argument(
        "--no-shard-grouped-batches",
        action="store_true",
        help="Disable shard-grouped train batches (slower IO, more random shuffle)",
    )
    parser.add_argument(
        "--no-cuda-warmup",
        action="store_true",
        help="Skip the small pre-epoch CUDA kernel warmup batch",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip full test-set evaluation after training (use validate_flow.py separately)",
    )
    parser.set_defaults(split_by_smiles=None, polymorph_loss=None)
    args = parser.parse_args()
    if args.split_by_smiles is None:
        args.split_by_smiles = True
    if args.polymorph_loss is None:
        args.polymorph_loss = True
    if args.w_flow_lambda is None:
        args.w_flow_lambda = args.w_flow

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    print(f"Device: {device}")

    use_gnn = True
    if not is_gnn_model(args.model):
        raise ValueError(f"Only GNN lattice-flow models are supported: {args.model}")

    def _load_csv_samples(
        csv_path: str,
        graphs_path: str | None,
        label: str,
    ) -> tuple[list, bool]:
        use_precomputed = graphs_path is not None or is_precomputed_csv(csv_path)
        if use_precomputed:
            resolved_graphs = graphs_path or str(precomputed_graphs_path(csv_path))
            loaded = load_precomputed_csv(
                csv_path,
                resolved_graphs,
                max_samples=args.max_samples,
                verbose=False,
                lazy_graphs=True,
            )
            print(f"Loaded {len(loaded)} precomputed {label} from {csv_path}", flush=True)
            return loaded, True
        print(f"Loading {csv_path} ...")
        loaded = load_structure_csv(csv_path, max_samples=args.max_samples)
        loaded = graphify_samples(loaded)
        print(f"Loaded {len(loaded)} {label}")
        return loaded, False

    # Load data
    samples_precomputed = False
    if args.csv:
        samples, samples_precomputed = _load_csv_samples(
            args.csv, args.precomputed_graphs, "training samples"
        )
    else:
        print("Loading HEM.db ...")
        samples = graphify_samples(load_hem_database(args.db, max_samples=args.max_samples))
        print(f"Loaded {len(samples)} training samples")

    test_list: list = []
    test_precomputed = False
    if args.test_csv:
        test_list, test_precomputed = _load_csv_samples(
            args.test_csv, args.test_precomputed_graphs, "test samples"
        )

    if not samples_precomputed:
        attach_selling_targets(samples)
        attach_sequential_targets(samples)
        attach_lattice_invariants(samples)
        print("Attached Delaunay Selling + log-density + log(λ) + log(λ*) targets")
        if test_list and not test_precomputed:
            attach_selling_targets(test_list)
            attach_sequential_targets(test_list)
            attach_lattice_invariants(test_list)
    else:
        print("Using precomputed Selling + log-density + log(λ) + log(λ*) targets")

    # Split — ensure label vocabularies are built from all data first
    n = len(samples)
    if args.test_csv:
        if args.split_by_smiles:
            train_idx, val_idx, _ = split_by_smiles_group(
                samples, val_frac=args.val_frac, test_frac=0.0, seed=args.seed
            )
            train_list = [samples[i] for i in train_idx]
            val_list = [samples[i] for i in val_idx]
        else:
            n_val = int(n * args.val_frac)
            n_train = n - n_val
            train_samples, val_samples = random_split(
                samples,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(args.seed),
            )
            train_list = [samples[i] for i in train_samples.indices]
            val_list = [samples[i] for i in val_samples.indices]
        print(
            f"Fixed test CSV: train={len(train_list)} val={len(val_list)} test={len(test_list)}"
        )
    elif args.split_by_smiles:
        train_idx, val_idx, test_idx = split_by_smiles_group(
            samples, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
        train_list = [samples[i] for i in train_idx]
        val_list = [samples[i] for i in val_idx]
        test_list = [samples[i] for i in test_idx]
        n_train_smiles = len({s.smiles for s in train_list})
        from collections import Counter

        smile_counts = Counter(s.smiles for s in samples)
        n_poly_smiles = sum(1 for c in smile_counts.values() if c > 1)
        print(
            f"SMILES-group split: train={len(train_list)} val={len(val_list)} test={len(test_list)} "
            f"({n_train_smiles} train SMILES, {n_poly_smiles} polymorphic SMILES in corpus)"
        )
    else:
        n_test = int(n * args.test_frac)
        n_val = int(n * args.val_frac)
        n_train = n - n_test - n_val
        train_samples, val_samples, test_samples = random_split(
            samples, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_list = [samples[i] for i in train_samples.indices]
        val_list = [samples[i] for i in val_samples.indices]
        test_list = [samples[i] for i in test_samples.indices]
        print(f"Random split: train={len(train_list)} val={len(val_list)} test={len(test_list)}")

    if uses_given_hz_lattice_flow_any(args.model) and (
        args.rare_shape_max_weight > 1.0 or args.shape_bins > 1
    ):
        shape_weighter = RareShapeWeighter.fit(
            train_list,
            max_weight=args.rare_shape_max_weight,
            power=args.rare_shape_weight_power,
        )
        train_weights = shape_weighter.assign(train_list, n_bins=args.shape_bins)
        val_weights = shape_weighter.assign(val_list, n_bins=args.shape_bins)
        shape_weighter.assign(test_list, n_bins=args.shape_bins)
        print(
            "Rare-shape flow weighting: "
            f"train mean={train_weights.mean():.3f} max={train_weights.max():.3f}; "
            f"val mean={val_weights.mean():.3f} max={val_weights.max():.3f}"
        )
        if args.shape_bins > 1:
            counts = np.bincount(
                [sample.lattice_shape_bin for sample in train_list],
                minlength=args.shape_bins,
            )
            print(f"Conditional anisotropy bins ({args.shape_bins}): {counts.tolist()}")

    # Build label vocab from all splits; normalize continuous targets from train only
    vocab_samples = samples + test_list if test_list else samples
    all_halls = sorted({s.hall_number for s in vocab_samples})
    all_zprimes = sorted({s.zprime for s in vocab_samples})
    hall_to_idx = {h: i for i, h in enumerate(all_halls)}
    idx_to_hall = {i: h for h, i in hall_to_idx.items()}
    zprime_to_idx = {z: i for i, z in enumerate(all_zprimes)}

    stats = compute_stats(
        train_list,
        use_graph=True,
        use_selling=True,
        use_log_lambda=True,
        use_log_density=True,
    )
    stats.hall_to_idx = hall_to_idx
    stats.idx_to_hall = idx_to_hall
    stats.zprime_to_idx = zprime_to_idx
    stats.zprime_values = all_zprimes
    stats.n_halls = len(all_halls)
    stats.n_zprimes = len(all_zprimes)

    polymorph_bank = None
    if args.polymorph_loss:
        polymorph_bank = SellingPolymorphBank(train_list, stats)
        print(
            f"Polymorph best-of-K loss: {polymorph_bank.n_groups} (S,H,Z′) groups with >1 packing, "
            f"{polymorph_bank.n_polymorph_structures} train structures"
        )

    shape_retrieval_index = None
    if (
        uses_given_hz_lattice_flow(args.model)
        and args.shape_bins > 1
        and args.shape_retrieval_blend > 0.0
    ):
        shape_index_path = (
            Path(args.output_dir) / args.model / "shape_retrieval_index.pkl"
        )
        if shape_index_path.is_file():
            try:
                cached = ShapeBinRetrievalIndex.load(shape_index_path)
                if (
                    cached.n_bins == args.shape_bins
                    and len(cached.smiles) == len(train_list)
                    and getattr(cached, "descriptors", None) is not None
                ):
                    shape_retrieval_index = cached
                    print(
                        f"Loaded shape retrieval index: {len(cached.smiles)} references"
                    )
            except Exception as exc:
                print(f"Rebuilding incompatible shape retrieval index: {exc}")
        if shape_retrieval_index is None:
            print(
                f"Building Hall/Z′ shape retrieval index "
                f"(k={args.shape_retrieval_k}) ..."
            )
            shape_retrieval_index = ShapeBinRetrievalIndex.build(
                train_list,
                n_bins=args.shape_bins,
                k=args.shape_retrieval_k,
            )

    target_mode = "lattice_flow"

    train_batch_sampler = None
    from torch_geometric.loader import DataLoader as PyGDataLoader

    graph_store_by_path: dict[str, HybridGraphStore] = {}

    def _get_graph_store(
        csv_path: str | None,
        graphs_path: str | None,
        sample_list: list,
        is_precomputed: bool,
        *,
        allow_preload_all: bool = False,
    ) -> HybridGraphStore | None:
        if not is_precomputed or not csv_path or not sample_list:
            return None
        resolved = str(Path(graphs_path or precomputed_graphs_path(csv_path)).resolve())
        if resolved in graph_store_by_path:
            return graph_store_by_path[resolved]

        hot_shards = args.graph_hot_shards
        sharded = has_sharded_graphs(resolved)
        if sharded:
            if allow_preload_all and args.preload_all_graphs:
                meta = json.loads(sharded_graphs_meta_path(resolved).read_text())
                hot_shards = int(meta["n_shards"])
            elif args.eager_graphs:
                hot_shards = max(hot_shards, 8)
        store = HybridGraphStore(
            resolved,
            hot_shards=hot_shards,
            cache_size=args.graph_cache_size,
        )
        if store.uses_shards:
            if allow_preload_all and args.preload_all_graphs:
                print(
                    f"  preloading {hot_shards} graph shards for {Path(resolved).name} ...",
                    flush=True,
                )
                n = store.preload_shards(verbose=True)
                print(f"  preloaded {n} graphs into RAM", flush=True)
            else:
                print(
                    f"  graph loader: sharded sidecar ({hot_shards} hot shards) "
                    f"for {Path(resolved).name}",
                    flush=True,
                )
                if allow_preload_all:
                    store.warm_disk_cache(verbose=True)
                    print(
                        "  batch graph fetch uses transient shard reads (low RAM)",
                        flush=True,
                    )
        elif (args.warm_graph_cache or args.eager_graphs) and not store.uses_shards:
            unique = list({s.smiles for s in sample_list})
            workers = args.warm_graph_workers
            label = "preloading" if args.eager_graphs else "warming"
            print(
                f"  {label} {len(unique)} unique SMILES graphs "
                f"(workers={workers}) ...",
                flush=True,
            )
            n = store.warm_smiles(unique, workers=workers)
            print(f"  {label} done: {n} graphs in cache", flush=True)
        graph_store_by_path[resolved] = store
        return store

    train_graph_store = _get_graph_store(
        args.csv, args.precomputed_graphs, train_list, samples_precomputed,
        allow_preload_all=True,
    )
    val_graph_store = _get_graph_store(
        args.csv, args.precomputed_graphs, val_list, samples_precomputed,
        allow_preload_all=True,
    )
    test_graph_store = _get_graph_store(
        args.test_csv, args.test_precomputed_graphs, test_list, test_precomputed,
        allow_preload_all=False,
    )

    train_ds = CrystalGraphDataset(
        train_list, stats, target=target_mode,
        with_log_density=True,
        graph_store=train_graph_store,
        graph_cache_size=args.graph_cache_size,
    )
    val_ds = CrystalGraphDataset(
        val_list, stats, target=target_mode,
        with_log_density=True,
        graph_store=val_graph_store,
        graph_cache_size=args.graph_cache_size,
    )
    test_ds = CrystalGraphDataset(
        test_list, stats, target=target_mode,
        with_log_density=True,
        graph_store=test_graph_store,
        graph_cache_size=args.graph_cache_size,
    )
    graph_shard_size: int | None = None
    if (
        not args.no_shard_grouped_batches
        and train_graph_store is not None
        and train_graph_store.uses_shards
    ):
        graphs_path = str(
            Path(args.precomputed_graphs or precomputed_graphs_path(args.csv)).resolve()
        )
        meta = json.loads(sharded_graphs_meta_path(graphs_path).read_text())
        graph_shard_size = int(meta["shard_size"])
        train_batch_sampler = ShardGroupedBatchSampler(
            train_list,
            batch_size=args.batch_size,
            shard_size=graph_shard_size,
            seed=args.seed,
        )
        print(
            f"  train loader: shard-grouped batches "
            f"({train_batch_sampler.__len__()} batches, 1 shard load each)",
            flush=True,
        )
    if train_batch_sampler is not None:
        train_loader = PyGDataLoader(
            train_ds, batch_sampler=train_batch_sampler, num_workers=0
        )
        val_batch_sampler = ShardGroupedBatchSampler(
            val_list,
            batch_size=args.batch_size,
            shard_size=graph_shard_size,
            seed=args.seed,
            shuffle_shards=False,
            shuffle_within_shard=False,
        )
        val_loader = PyGDataLoader(
            val_ds, batch_sampler=val_batch_sampler, num_workers=0
        )
        print(
            f"  val loader: shard-grouped batches ({val_batch_sampler.__len__()} batches)",
            flush=True,
        )
    else:
        train_loader = PyGDataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model, target_mode, target_key = build_model(args.model, stats, args)
    model = model.to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected, skipped = load_compatible_state_dict(model, ckpt["model"])
        print(
            f"Loaded init checkpoint {args.init_checkpoint} "
            f"(loaded={len(ckpt['model']) - len(skipped)}, missing={len(missing)}, "
            f"skipped_shape={len(skipped)}, unexpected={len(unexpected)})"
        )
    if getattr(args, "freeze_except_lambda", False):
        n_frozen = 0
        n_train = 0
        trainable_keys = ("flow_head", "lambda_flow_head")
        for name, param in model.named_parameters():
            if any(key in name for key in trainable_keys):
                param.requires_grad = True
                n_train += param.numel()
            else:
                param.requires_grad = False
                n_frozen += param.numel()
        if n_train == 0:
            raise RuntimeError(
                "freeze-except-lambda matched no parameters "
                "(expected flow_head or lambda_flow_head)"
            )
        print(
            f"  freeze-except-lambda: trainable={n_train:,} frozen={n_frozen:,}",
            flush=True,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters for optimizer")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.save(out_dir / "stats.json")
    if shape_retrieval_index is not None:
        shape_retrieval_index.save(out_dir / "shape_retrieval_index.pkl")
        (out_dir / "shape_retrieval_metadata.json").write_text(
            json.dumps(shape_retrieval_index.metadata(), indent=2)
        )

    best_val = float("inf")
    patience_counter = 0
    history = []

    if (
        device.type == "cuda"
        and use_gnn
        and not args.no_cuda_warmup
        and train_batch_sampler is None
    ):
        from torch_geometric.loader import DataLoader as PyGWarmLoader

        warm_bs = min(8, len(train_ds))
        warm_loader = PyGWarmLoader(train_ds, batch_size=warm_bs, shuffle=False, num_workers=0)
        print(f"  CUDA warmup: loading {warm_bs}-sample batch ...", flush=True)
        model.eval()
        t_warm = time.time()
        with torch.no_grad():
            warm_batch = next(iter(warm_loader))
            t_load = time.time()
            print(f"  CUDA warmup: batch loaded in {t_load - t_warm:.1f}s, running forward ...", flush=True)
            x, y, hall, zprime, log_density = get_batch_tensors(warm_batch, device, True)
            model(x, hall, zprime, x1=y)
            torch.cuda.synchronize()
        model.train()
        print(f"  CUDA warmup done in {time.time() - t_warm:.1f}s", flush=True)

    for epoch in range(1, args.epochs + 1):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            args.model,
            stats=stats, args=args, polymorph_bank=polymorph_bank,
            epoch=epoch, show_progress=not args.no_progress,
        )
        scheduler.step()

        # Quick val loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y, hall, zprime, log_density = get_batch_tensors(batch, device, True)
                loss = _compute_lattice_loss(
                    args.model, model, x, y, hall, zprime, log_density,
                    stats, args, polymorph_bank, batch, True, device,
                )
                val_losses.append(float(loss.item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "model_name": args.model,
                "target_mode": target_mode,
                "target_key": target_key,
                "args": vars(args),
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break
    if args.skip_test_eval:
        results = {"test_metrics": None, "history": history, "args": vars(args)}
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"\nSkipped test eval (--skip-test-eval). Saved to {out_dir}")
        return

    # Final evaluation on test set
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    model.eval()
    all_pred, all_true_y = [], []
    all_pred_ld, all_true_ld = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x = batch
            y = batch.y.view(batch.num_graphs, -1)
            hall = batch.hall.view(-1)
            zprime = batch.zprime.view(-1)
            out = model(x, hall, zprime)
            all_pred.append(out["lattice_flow"].cpu().numpy())
            all_true_y.append(y.cpu().numpy())
            all_pred_ld.append(out["log_density"].cpu().numpy())
            all_true_ld.append(batch.log_density.view(-1).cpu().numpy())

    pred_norm = np.concatenate(all_pred, axis=0)
    true_target_norm = np.concatenate(all_true_y, axis=0)
    from cellnet.sequential import (
        denorm_log_lambda,
        denorm_log_reciprocal_lambda,
        split_lattice_flow_norm,
    )

    pred_s, pred_l, pred_r = split_lattice_flow_norm(pred_norm, stats)
    true_s, true_l, true_r = split_lattice_flow_norm(true_target_norm, stats)
    pred_l_phys = np.stack([denorm_log_lambda(pred_l[i], stats) for i in range(len(pred_l))])
    true_l_phys = np.stack([denorm_log_lambda(true_l[i], stats) for i in range(len(true_l))])
    pred_r_phys = np.stack([
        denorm_log_reciprocal_lambda(pred_r[i], stats) for i in range(len(pred_r))
    ])
    true_r_phys = np.stack([
        denorm_log_reciprocal_lambda(true_r[i], stats) for i in range(len(true_r))
    ])
    test_metrics = {
        "selling_mse_norm": float(np.mean(np.sum((pred_s - true_s) ** 2, axis=1))),
        "log_lambda_mse_norm": float(np.mean(np.sum((pred_l - true_l) ** 2, axis=1))),
        "log_lambda_recip_mse_norm": float(np.mean(np.sum((pred_r - true_r) ** 2, axis=1))),
        "log_lambda_mse_phys": float(np.mean(np.sum((pred_l_phys - true_l_phys) ** 2, axis=1))),
        "log_lambda_recip_mse_phys": float(np.mean(np.sum((pred_r_phys - true_r_phys) ** 2, axis=1))),
        "log_density_mae": float(np.mean(np.abs(np.concatenate(all_pred_ld) - np.concatenate(all_true_ld)))),
    }

    results = {"test_metrics": test_metrics, "history": history, "args": vars(args)}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nTest metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"Saved to {out_dir}")
if __name__ == "__main__":
    main()
