#!/usr/bin/env python3
"""K-sample lattice-flow validation using SPaDe CSV (no internal re-split)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cellnet.data import (
    CrystalGraphDataset,
    attach_lattice_invariants,
    attach_selling_targets,
    graphify_samples,
)
from cellnet.metrics import evaluate_lattice_flow_k_samples
from cellnet.sequential import (
    denorm_log_lambda,
    denorm_log_reciprocal_lambda,
    lattice_flow_slices,
    split_lattice_flow_norm,
)
from cellnet.spade import load_structure_csv
from eval_lattice_flow_k100 import load_model, sample_k


def _best_k_index(pred: np.ndarray, true: np.ndarray) -> int:
    per_k = np.mean((pred - true) ** 2, axis=-1)
    return int(np.argmin(per_k))


def _write_prediction_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_prediction_dump(
    dump_path: Path,
    stats,
    records: list[dict[str, object]],
) -> None:
    n = len(records)
    if n == 0:
        return

    k = int(records[0]["pred_samples_norm"].shape[0])
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        dump_path,
        csd_code=np.array([r["csd_code"] for r in records], dtype=object),
        smiles=np.array([r["smiles"] for r in records], dtype=object),
        hall_number=np.array([r["hall_number"] for r in records], dtype=np.int32),
        zprime=np.array([r["zprime"] for r in records], dtype=np.float64),
        true_selling_norm=np.stack([r["true_selling_norm"] for r in records], axis=0),
        true_lambda_norm=np.stack([r["true_lambda_norm"] for r in records], axis=0),
        true_lambda_recip_norm=np.stack([r["true_lambda_recip_norm"] for r in records], axis=0),
        true_log_density_norm=np.array([r["true_log_density_norm"] for r in records], dtype=np.float64),
        pred_samples_norm=np.stack([r["pred_samples_norm"] for r in records], axis=0),
        pred_log_density_norm=np.array([r["pred_log_density_norm"] for r in records], dtype=np.float64),
        true_log_lambda_phys=np.stack([r["true_log_lambda_phys"] for r in records], axis=0),
        true_log_lambda_recip_phys=np.stack([r["true_log_lambda_recip_phys"] for r in records], axis=0),
        pred_log_lambda_phys=np.stack([r["pred_log_lambda_phys"] for r in records], axis=0),
        pred_log_lambda_recip_phys=np.stack([r["pred_log_lambda_recip_phys"] for r in records], axis=0),
        best_k_selling=np.array([r["best_k_selling"] for r in records], dtype=np.int32),
        best_k_lambda=np.array([r["best_k_lambda"] for r in records], dtype=np.int32),
        best_k_lambda_recip=np.array([r["best_k_lambda_recip"] for r in records], dtype=np.int32),
        selling_mse_norm_mean=np.array([r["selling_mse_norm_mean"] for r in records], dtype=np.float64),
        selling_mse_norm_best=np.array([r["selling_mse_norm_best"] for r in records], dtype=np.float64),
        lambda_mse_norm_mean=np.array([r["lambda_mse_norm_mean"] for r in records], dtype=np.float64),
        lambda_mse_norm_best=np.array([r["lambda_mse_norm_best"] for r in records], dtype=np.float64),
        lambda_recip_mse_norm_mean=np.array([r["lambda_recip_mse_norm_mean"] for r in records], dtype=np.float64),
        lambda_recip_mse_norm_best=np.array([r["lambda_recip_mse_norm_best"] for r in records], dtype=np.float64),
        density_mae_norm=np.array([r["density_mae_norm"] for r in records], dtype=np.float64),
        k=np.array([k], dtype=np.int32),
    )

    csv_rows: list[dict[str, object]] = []
    for r in records:
        bk = int(r["best_k_lambda"])
        bkr = int(r["best_k_lambda_recip"])
        true_l = r["true_log_lambda_phys"]
        pred_l = r["pred_log_lambda_phys"][bk]
        true_lr = r["true_log_lambda_recip_phys"]
        pred_lr = r["pred_log_lambda_recip_phys"][bkr]
        row: dict[str, object] = {
            "csd_code": r["csd_code"],
            "hall_number": r["hall_number"],
            "zprime": r["zprime"],
            "lambda_mse_norm_best": r["lambda_mse_norm_best"],
            "lambda_recip_mse_norm_best": r["lambda_recip_mse_norm_best"],
            "lambda_mse_norm_mean": r["lambda_mse_norm_mean"],
            "lambda_recip_mse_norm_mean": r["lambda_recip_mse_norm_mean"],
            "best_k_lambda": bk,
            "best_k_lambda_recip": bkr,
            "density_mae_norm": r["density_mae_norm"],
        }
        for j in range(3):
            row[f"true_log_lambda_{j + 1}_phys"] = float(true_l[j])
            row[f"pred_log_lambda_{j + 1}_phys"] = float(pred_l[j])
            row[f"true_lambda_{j + 1}_ang"] = float(np.exp(true_l[j]))
            row[f"pred_lambda_{j + 1}_ang"] = float(np.exp(pred_l[j]))
            row[f"true_log_lambda_recip_{j + 1}_phys"] = float(true_lr[j])
            row[f"pred_log_lambda_recip_{j + 1}_phys"] = float(pred_lr[j])
            row[f"true_lambda_recip_{j + 1}_invA"] = float(np.exp(true_lr[j]))
            row[f"pred_lambda_recip_{j + 1}_invA"] = float(np.exp(pred_lr[j]))
        csv_rows.append(row)

    _write_prediction_csv(dump_path.with_suffix(".csv"), csv_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="SPaDe K-sample flow validation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--dump-predictions",
        type=str,
        default=None,
        help="Save per-structure true vs K predictions to .npz (+ readable .csv summary)",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    checkpoint = Path(args.checkpoint)
    model, stats, kind, model_name = load_model(checkpoint, device)
    if kind != "lattice_flow":
        raise SystemExit(f"Expected lattice_flow checkpoint, got {kind}")

    samples = load_structure_csv(args.csv)
    attach_selling_targets(samples, verbose=False)
    attach_lattice_invariants(samples, verbose=False)
    samples = graphify_samples(samples, verbose=False)
    if args.max_samples is not None and len(samples) > args.max_samples:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(samples), size=args.max_samples, replace=False)
        samples = [samples[i] for i in pick]

    ds = CrystalGraphDataset(samples, stats, target="lattice_flow", with_log_density=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    s_sl, l_sl, r_sl = lattice_flow_slices(stats)
    totals = {
        "selling_mse_norm_mean": [],
        "selling_mse_norm_best": [],
        "lambda_mse_norm_mean": [],
        "lambda_mse_norm_best": [],
        "lambda_recip_mse_norm_mean": [],
        "lambda_recip_mse_norm_best": [],
        "density_mae_norm": [],
    }
    dump_records: list[dict[str, object]] = []
    sample_idx = 0

    for batch in loader:
        batch = batch.to(device)
        hall = batch.hall.view(-1)
        zprime = batch.zprime.view(-1)
        true_y = batch.y.view(batch.num_graphs, -1).cpu().numpy()
        true_ld = batch.log_density.view(-1).cpu().numpy()
        samples_norm, pred_ld, _ = sample_k(model, batch, hall, zprime, args.k, kind)
        pred_ld = pred_ld.cpu().numpy()
        samples_norm = samples_norm.cpu().numpy()

        true_s = true_y[:, s_sl]
        true_l = true_y[:, l_sl]
        true_r = true_y[:, r_sl]
        pred_s, pred_l, pred_r = split_lattice_flow_norm(samples_norm, stats)
        for i in range(true_y.shape[0]):
            s = ds.samples[sample_idx]
            sample_idx += 1

            sell_mean = evaluate_lattice_flow_k_samples(pred_s[i], true_s[i], "selling")["selling_mse_norm_mean"]
            sell_best = evaluate_lattice_flow_k_samples(pred_s[i], true_s[i], "selling")["selling_mse_norm_best"]
            lam_mean = evaluate_lattice_flow_k_samples(pred_l[i], true_l[i], "lambda")["lambda_mse_norm_mean"]
            lam_best = evaluate_lattice_flow_k_samples(pred_l[i], true_l[i], "lambda")["lambda_mse_norm_best"]
            lam_r_mean = evaluate_lattice_flow_k_samples(pred_r[i], true_r[i], "lambda_recip")["lambda_recip_mse_norm_mean"]
            lam_r_best = evaluate_lattice_flow_k_samples(pred_r[i], true_r[i], "lambda_recip")["lambda_recip_mse_norm_best"]
            dens_mae = float(abs(pred_ld[i] - true_ld[i]))

            totals["selling_mse_norm_mean"].append(sell_mean)
            totals["selling_mse_norm_best"].append(sell_best)
            totals["lambda_mse_norm_mean"].append(lam_mean)
            totals["lambda_mse_norm_best"].append(lam_best)
            totals["lambda_recip_mse_norm_mean"].append(lam_r_mean)
            totals["lambda_recip_mse_norm_best"].append(lam_r_best)
            totals["density_mae_norm"].append(dens_mae)

            if args.dump_predictions:
                true_l_phys = denorm_log_lambda(true_l[i], stats)
                true_r_phys = denorm_log_reciprocal_lambda(true_r[i], stats)
                pred_l_phys = np.stack(
                    [denorm_log_lambda(pred_l[i, j], stats) for j in range(pred_l.shape[1])],
                    axis=0,
                )
                pred_r_phys = np.stack(
                    [denorm_log_reciprocal_lambda(pred_r[i, j], stats) for j in range(pred_r.shape[1])],
                    axis=0,
                )
                dump_records.append({
                    "csd_code": s.csd_code,
                    "smiles": s.smiles,
                    "hall_number": int(s.hall_number),
                    "zprime": float(s.zprime),
                    "true_selling_norm": true_s[i].astype(np.float64),
                    "true_lambda_norm": true_l[i].astype(np.float64),
                    "true_lambda_recip_norm": true_r[i].astype(np.float64),
                    "true_log_density_norm": float(true_ld[i]),
                    "pred_samples_norm": samples_norm[i].astype(np.float64),
                    "pred_log_density_norm": float(pred_ld[i]),
                    "true_log_lambda_phys": true_l_phys.astype(np.float64),
                    "true_log_lambda_recip_phys": true_r_phys.astype(np.float64),
                    "pred_log_lambda_phys": pred_l_phys.astype(np.float64),
                    "pred_log_lambda_recip_phys": pred_r_phys.astype(np.float64),
                    "best_k_selling": _best_k_index(pred_s[i], true_s[i]),
                    "best_k_lambda": _best_k_index(pred_l[i], true_l[i]),
                    "best_k_lambda_recip": _best_k_index(pred_r[i], true_r[i]),
                    "selling_mse_norm_mean": sell_mean,
                    "selling_mse_norm_best": sell_best,
                    "lambda_mse_norm_mean": lam_mean,
                    "lambda_mse_norm_best": lam_best,
                    "lambda_recip_mse_norm_mean": lam_r_mean,
                    "lambda_recip_mse_norm_best": lam_r_best,
                    "density_mae_norm": dens_mae,
                })

    summary = {
        "checkpoint": str(checkpoint),
        "model": model_name,
        "csv": args.csv,
        "k": args.k,
        "n_samples": len(samples),
    }
    for key, vals in totals.items():
        if vals:
            summary[key] = float(np.mean(vals))

    print(f"\n=== K={args.k} SPaDe validation (n={len(samples)}) ===")
    for key in sorted(summary):
        if key.endswith("_mean") or key.endswith("_best") or key.endswith("_norm"):
            print(f"  {key}: {summary[key]:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved: {out_path}")

    if args.dump_predictions:
        dump_path = Path(args.dump_predictions)
        _save_prediction_dump(dump_path, stats, dump_records)
        print(f"Saved predictions: {dump_path}")
        print(f"Saved summary CSV:   {dump_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
