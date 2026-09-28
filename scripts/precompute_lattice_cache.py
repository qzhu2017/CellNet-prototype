#!/usr/bin/env python3
"""Precompute PyG graphs, Selling parameters, and lattice invariants for SPaDe CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellnet.precomputed import (  # noqa: E402
    PRECOMPUTED_COLUMNS,
    graph_from_payload,
    graph_to_payload,
    precompute_sample,
    precomputed_graphs_path,
    sample_to_precomputed_row,
    _sample_from_row,
)


def _process_sample(sample_id: int, row: dict) -> tuple[int, dict | None, dict | None]:
    """Worker entry point: return picklable NumPy payloads, not torch tensors."""
    sample = _sample_from_row(row, sample_id)
    result = precompute_sample(sample)
    if result is None:
        return sample_id, None, None
    return sample_id, sample_to_precomputed_row(result), graph_to_payload(result.graph)


def _read_spade_rows(csv_path: Path, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            rows.append(row)
    return rows


def _checkpoint_paths(output_csv: Path, graphs_path: Path) -> tuple[Path, Path, Path]:
    meta = output_csv.with_suffix(".checkpoint.json")
    partial_csv = output_csv.with_suffix(".partial.csv")
    partial_graphs = graphs_path.with_suffix(".partial.pt")
    return meta, partial_csv, partial_graphs


def _load_checkpoint(
    meta_path: Path,
    partial_csv: Path,
    partial_graphs: Path,
) -> tuple[set[int], dict[int, tuple[dict, object]], int]:
    if not meta_path.is_file():
        return set(), {}, 0

    meta = json.loads(meta_path.read_text())
    skipped = int(meta.get("skipped", 0))
    ok_indices = [int(i) for i in meta.get("ok_indices", [])]
    finished = {int(i) for i in meta.get("finished_indices", ok_indices)}

    out_rows: list[dict] = []
    if partial_csv.is_file():
        with partial_csv.open(newline="") as handle:
            out_rows = list(csv.DictReader(handle))

    graphs: list[object] = []
    if partial_graphs.is_file():
        import torch

        payload = torch.load(partial_graphs, map_location="cpu", weights_only=False)
        graphs = payload["graphs"]

    if len(out_rows) != len(graphs) or len(out_rows) != len(ok_indices):
        raise ValueError(
            f"Checkpoint mismatch: {len(out_rows)} rows, {len(graphs)} graphs, "
            f"{len(ok_indices)} indices ({meta_path})"
        )

    results = {
        idx: (out_rows[j], graphs[j]) for j, idx in enumerate(ok_indices)
    }
    return finished, results, skipped


def _save_checkpoint(
    finished: set[int],
    skipped: int,
    results: dict[int, tuple[dict, object]],
    meta_path: Path,
    partial_csv: Path,
    partial_graphs: Path,
) -> None:
    import torch

    ok_indices = sorted(results)
    out_rows = [results[i][0] for i in ok_indices]
    graphs = [results[i][1] for i in ok_indices]

    meta_path.write_text(
        json.dumps(
            {
                "skipped": skipped,
                "ok_indices": ok_indices,
                "finished_indices": sorted(finished),
            },
            indent=2,
        )
    )

    with partial_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRECOMPUTED_COLUMNS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row[k] for k in PRECOMPUTED_COLUMNS})

    torch.save({"version": 1, "n_graphs": len(graphs), "graphs": graphs}, partial_graphs)


def precompute_rows(
    rows: list[dict],
    workers: int = 1,
    show_progress: bool = True,
    finished: set[int] | None = None,
    initial_results: dict[int, tuple[dict, object]] | None = None,
    initial_skipped: int = 0,
    on_checkpoint: Callable[[set[int], dict[int, tuple[dict, object]], int], None] | None = None,
    checkpoint_every: int = 0,
) -> tuple[list[dict], list[object], int]:
    """Return CSV rows, graphs, and skip count."""
    total = len(rows)
    results: dict[int, tuple[dict, object]] = dict(initial_results or {})
    finished = set(finished or [])
    skipped = initial_skipped
    since_checkpoint = 0

    pbar = tqdm(
        initial=len(finished),
        total=total,
        desc="Precomputing",
        disable=not show_progress,
        file=sys.stderr,
        mininterval=0.5,
        dynamic_ncols=True,
    )

    def _maybe_checkpoint() -> None:
        nonlocal since_checkpoint
        if on_checkpoint is None or checkpoint_every <= 0:
            return
        if since_checkpoint < checkpoint_every:
            return
        on_checkpoint(finished, results, skipped)
        since_checkpoint = 0

    def _store(idx: int, csv_row: dict | None, graph_payload: dict | None) -> None:
        nonlocal skipped, since_checkpoint
        finished.add(idx)
        since_checkpoint += 1
        if csv_row is None:
            skipped += 1
        else:
            results[idx] = (csv_row, graph_from_payload(graph_payload))
        pbar.update(1)
        pbar.set_postfix(ok=len(results), skip=skipped, refresh=False)
        _maybe_checkpoint()

    pending = [i for i in range(total) if i not in finished]

    if workers <= 1:
        for i in pending:
            _, csv_row, graph_payload = _process_sample(i, rows[i])
            _store(i, csv_row, graph_payload)
        pbar.close()
    else:
        chunk_size = max(32, workers * 4)
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for chunk_start in range(0, len(pending), chunk_size):
                    chunk = pending[chunk_start : chunk_start + chunk_size]
                    futures = {
                        pool.submit(_process_sample, i, rows[i]): i for i in chunk
                    }
                    for future in as_completed(futures):
                        idx, csv_row, graph_payload = future.result()
                        _store(idx, csv_row, graph_payload)
                    _maybe_checkpoint()
        except BrokenProcessPool as exc:
            pbar.close()
            if on_checkpoint is not None:
                on_checkpoint(finished, results, skipped)
            raise RuntimeError(
                "A worker process crashed (often OOM or torch IPC failure). "
                "Partial progress was checkpointed if --checkpoint-every > 0. "
                "Retry with --resume and/or fewer workers (e.g. --workers 4)."
            ) from exc
        pbar.close()

    out_rows = [results[i][0] for i in sorted(results)]
    graphs = [results[i][1] for i in sorted(results)]
    return out_rows, graphs, skipped


def _write_outputs(
    out_rows: list[dict],
    graphs: list[object],
    output_csv: Path,
    graphs_path: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRECOMPUTED_COLUMNS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row[k] for k in PRECOMPUTED_COLUMNS})

    import torch

    torch.save({"version": 1, "n_graphs": len(graphs), "graphs": graphs}, graphs_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute lattice-flow training cache")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input SPaDe CSV (e.g. datasets/spade-csp/spade_train.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output precomputed CSV (default: <input_stem>_precomputed.csv)",
    )
    parser.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="Output graph sidecar .pt (default: <output_stem>_graphs.pt)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel workers for precomputation (default: half of CPU count)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial checkpoint next to --output",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="Save partial checkpoint every N input rows (0 to disable)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_precomputed.csv"
    )
    graphs_path = Path(args.graphs) if args.graphs else precomputed_graphs_path(output_csv)
    meta_path, partial_csv, partial_graphs = _checkpoint_paths(output_csv, graphs_path)

    print(f"Loading {input_path} ...", flush=True)
    rows = _read_spade_rows(input_path, args.max_samples)
    print(f"  {len(rows)} rows", flush=True)

    finished: set[int] = set()
    initial_results: dict[int, tuple[dict, object]] = {}
    skipped = 0
    if args.resume:
        finished, initial_results, skipped = _load_checkpoint(
            meta_path, partial_csv, partial_graphs
        )
        if finished:
            print(
                f"Resuming with {len(finished)}/{len(rows)} rows done "
                f"({len(initial_results)} ok, {skipped} skipped so far)",
                flush=True,
            )

    def save_checkpoint(fin: set[int], results: dict, n_skip: int) -> None:
        _save_checkpoint(fin, n_skip, results, meta_path, partial_csv, partial_graphs)
        if not args.no_progress:
            print(f"  checkpoint saved ({len(fin)} done, {len(results)} ok)", flush=True)

    if args.workers > 1 and not args.no_progress and not finished:
        print(
            f"Starting {args.workers} workers "
            "(each imports RDKit/torch; first bar update may take ~30-60s) ...",
            flush=True,
        )

    out_rows, graphs, skipped = precompute_rows(
        rows,
        workers=args.workers,
        show_progress=not args.no_progress,
        finished=finished,
        initial_results=initial_results,
        initial_skipped=skipped,
        on_checkpoint=save_checkpoint if args.checkpoint_every > 0 else None,
        checkpoint_every=args.checkpoint_every,
    )
    _write_outputs(out_rows, graphs, output_csv, graphs_path)

    for path in (meta_path, partial_csv, partial_graphs):
        if path.is_file():
            path.unlink()

    print(
        f"Saved {len(out_rows)} samples to {output_csv}\n"
        f"Saved {len(graphs)} graphs to {graphs_path}\n"
        f"Skipped {skipped} invalid or failed structures"
    )
    print(
        "\nTrain with:\n"
        f"  python scripts/train_flow.py \\\n"
        f"    --train-csv {output_csv} \\\n"
        f"    --train-graphs {graphs_path}"
    )


if __name__ == "__main__":
    main()
