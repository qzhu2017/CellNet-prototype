#!/usr/bin/env python3
"""K-sample flow validation on SPaDe test split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPADE_DIR = ROOT / "datasets/spade-csp"
DEFAULT_TEST = SPADE_DIR / "spade_test.csv"
DEFAULT_CKPT = ROOT / "checkpoints/cellnet_flow/best.pt"
DEFAULT_OUT = ROOT / "outputs/flow_k100_test.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="SPaDe lattice-flow K-sample validation")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--csv", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    parser.add_argument(
        "--dump-predictions",
        type=str,
        default=None,
        help="Per-structure K-sample predictions (.npz + .csv); default: <output_stem>_predictions.npz",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    dump_predictions = args.dump_predictions
    if dump_predictions is None and args.output:
        dump_predictions = str(Path(args.output).with_name(Path(args.output).stem + "_predictions.npz"))

    cmd = [
        sys.executable,
        str(ROOT / "scripts/validate_flow_k100.py"),
        "--checkpoint",
        args.checkpoint,
        "--csv",
        args.csv,
        "--k",
        str(args.k),
        "--max-samples",
        str(args.max_samples),
        "--output",
        args.output,
        "--device",
        args.device,
    ]
    if dump_predictions:
        cmd.extend(["--dump-predictions", dump_predictions])
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
