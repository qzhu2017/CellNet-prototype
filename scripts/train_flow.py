#!/usr/bin/env python3
"""Train conditional lattice flow: Selling flow, then λ flow given Selling."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellnet.precomputed import precomputed_graphs_path  # noqa: E402

SPADE_DIR = ROOT / "datasets/spade-csp"
DEFAULT_TRAIN = SPADE_DIR / "spade_train.csv"
DEFAULT_TEST = SPADE_DIR / "spade_test.csv"
DEFAULT_TRAIN_PRE = SPADE_DIR / "spade_train_precomputed.csv"
DEFAULT_TEST_PRE = SPADE_DIR / "spade_test_precomputed.csv"
DEFAULT_OUT = ROOT / "outputs/conditional_lattice_flow_gnn"


def main() -> None:
    parser = argparse.ArgumentParser(description="SPaDe conditional lattice-flow training")
    parser.add_argument("--train-csv", type=str, default=str(DEFAULT_TRAIN))
    parser.add_argument("--test-csv", type=str, default=str(DEFAULT_TEST))
    parser.add_argument(
        "--train-graphs",
        type=str,
        default=None,
        help="Precomputed train graph sidecar .pt (auto-detected from precomputed CSV name)",
    )
    parser.add_argument(
        "--test-graphs",
        type=str,
        default=None,
        help="Precomputed test graph sidecar .pt (auto-detected from precomputed CSV name)",
    )
    parser.add_argument(
        "--precomputed",
        action="store_true",
        help="Use default precomputed train/test CSV + graph sidecars",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Rebuild spade_train/test.csv from crystal-info_CSD_filtered.csv before training",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Rebuild precomputed CSV + graph sidecars after --extract (slow; required for --precomputed)",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Load compatible weights (encoder, density; flows partially transfer from joint model)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--eager-graphs",
        action="store_true",
        help="Warm OS disk cache for graph shards at startup",
    )
    parser.add_argument(
        "--preload-all-graphs",
        action="store_true",
        help="Load all train/val graph shards at startup (high RAM; omit on GPU nodes)",
    )
    parser.add_argument(
        "--graph-hot-shards",
        type=int,
        default=None,
        help="Keep N graph shards (~10k each) hot in RAM",
    )
    parser.add_argument("--w-flow", type=float, default=0.5, help="Selling flow velocity weight")
    parser.add_argument(
        "--w-flow-lambda",
        type=float,
        default=None,
        help="Conditional λ flow velocity weight (defaults to --w-flow)",
    )
    parser.add_argument("--w-volume", type=float, default=0.5, help="Selling endpoint weight")
    parser.add_argument("--w-density", type=float, default=0.3)
    parser.add_argument("--w-lambda", type=float, default=3.0)
    parser.add_argument("--w-lambda-recip", type=float, default=3.0)
    parser.add_argument(
        "--lambda-loss-mode",
        type=str,
        default="phys_log",
        choices=["phys_log", "norm", "relative"],
    )
    parser.add_argument("--w-lambda-relative", type=float, default=1.0)
    parser.add_argument("--w-lambda-product", type=float, default=0.5)
    parser.add_argument("--w-axis-permutation", type=float, default=0.3)
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip full test eval after training (use validate_flow.py separately)",
    )
    args = parser.parse_args()

    train_csv = args.train_csv
    test_csv = args.test_csv
    output_dir = args.output_dir
    if args.extract or args.precompute:
        extract_script = ROOT / "scripts/build_dataset.py"
        extract_cmd = [sys.executable, str(extract_script)]
        print("Running:", " ".join(extract_cmd))
        subprocess.check_call(extract_cmd, cwd=str(ROOT))

    if args.precompute:
        train_name = Path(train_csv).name
        test_name = Path(test_csv).name
        for name in (train_name, test_name):
            pre_cmd = [
                sys.executable,
                str(ROOT / "scripts/precompute_lattice_cache.py"),
                "--input",
                str(SPADE_DIR / name),
            ]
            print("Running:", " ".join(pre_cmd))
            subprocess.check_call(pre_cmd, cwd=str(ROOT))

    train_graphs = args.train_graphs
    test_graphs = args.test_graphs
    if args.precomputed:
        if args.train_csv == str(DEFAULT_TRAIN):
            train_csv = str(DEFAULT_TRAIN_PRE)
            test_csv = str(DEFAULT_TEST_PRE)
        if train_graphs is None:
            train_graphs = str(precomputed_graphs_path(train_csv))
        if test_graphs is None:
            test_graphs = str(precomputed_graphs_path(test_csv))

    cmd = [
        sys.executable,
        str(ROOT / "scripts/train.py"),
        "--model",
        "given_hz_conditional_lattice_flow_gnn",
        "--csv",
        train_csv,
        "--test-csv",
        test_csv,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--patience",
        str(args.patience),
        "--w-flow",
        str(args.w_flow),
        "--w-volume",
        str(args.w_volume),
        "--w-density",
        str(args.w_density),
        "--w-lambda",
        str(args.w_lambda),
        "--w-lambda-recip",
        str(args.w_lambda_recip),
        "--lambda-loss-mode",
        args.lambda_loss_mode,
        "--w-lambda-relative",
        str(args.w_lambda_relative),
        "--w-lambda-product",
        str(args.w_lambda_product),
        "--w-axis-permutation",
        str(args.w_axis_permutation),
        "--flow-steps",
        "20",
        "--output-dir",
        output_dir,
        "--device",
        args.device,
        "--split-by-smiles",
        "--polymorph-loss",
    ]
    if args.w_flow_lambda is not None:
        cmd.extend(["--w-flow-lambda", str(args.w_flow_lambda)])
    if train_graphs:
        cmd.extend(["--precomputed-graphs", train_graphs])
    if test_graphs:
        cmd.extend(["--test-precomputed-graphs", test_graphs])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.init_checkpoint:
        cmd.extend(["--init-checkpoint", args.init_checkpoint])
    if args.eager_graphs:
        cmd.append("--eager-graphs")
    if args.preload_all_graphs:
        cmd.append("--preload-all-graphs")
    if args.graph_hot_shards is not None:
        cmd.extend(["--graph-hot-shards", str(args.graph_hot_shards)])
    if args.skip_test_eval:
        cmd.append("--skip-test-eval")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
