#!/usr/bin/env python3
"""Split a monolithic precomputed graph .pt into RAM-friendly shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellnet.precomputed import shard_graph_sidecar, sharded_graphs_meta_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard precomputed graph sidecar")
    parser.add_argument("--input", type=str, required=True, help="Monolithic *_graphs.pt file")
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = shard_graph_sidecar(input_path, shard_size=args.shard_size, output_dir=args.output_dir)
    meta = sharded_graphs_meta_path(input_path)
    print(f"Wrote shards to {out_dir}")
    print(f"Metadata: {meta}")
    print(
        "\nTrain with lazy graphs (auto-detects shards):\n"
        "  python scripts/train_flow.py --precomputed"
    )


if __name__ == "__main__":
    main()
