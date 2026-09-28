"""Precomputed lattice-flow cache: tabular targets + PyG graph sidecar."""

from __future__ import annotations

import csv
import json
import random
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from cellnet.data import (
    CrystalSample,
    attach_lattice_invariants,
    attach_sequential_targets,
    attach_selling_targets,
)
from cellnet.graph import smiles_to_graph
from cellnet.reciprocal import cellpar_to_reciprocal_metric
from cellnet.sequential import FREE_PARAM_DIM

PRECOMPUTED_MARKER = "selling_log1p_0"

SELLING_COLUMNS = [f"selling_log1p_{i}" for i in range(6)]
LOG_LAMBDA_COLUMNS = [f"log_lambda_{i}" for i in range(3)]
LOG_LAMBDA_RECIP_COLUMNS = [f"log_lambda_recip_{i}" for i in range(3)]
FREE_COLUMNS = [f"free_{i}" for i in range(FREE_PARAM_DIM)]
FREE_MASK_COLUMNS = [f"free_mask_{i}" for i in range(FREE_PARAM_DIM)]

PRECOMPUTED_SCALAR_COLUMNS = (
    SELLING_COLUMNS
    + LOG_LAMBDA_COLUMNS
    + LOG_LAMBDA_RECIP_COLUMNS
    + ["log_density"]
    + FREE_COLUMNS
    + FREE_MASK_COLUMNS
)

PRECOMPUTED_BASE_COLUMNS = [
    "refcode",
    "sg_symbol",
    "sg_number",
    "hall_number",
    "smiles",
    "zprime",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "density",
]

PRECOMPUTED_COLUMNS = PRECOMPUTED_BASE_COLUMNS + PRECOMPUTED_SCALAR_COLUMNS


def precomputed_graphs_path(csv_path: str | Path) -> Path:
    """Default sidecar path: ``spade_train_precomputed_graphs.pt``."""
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_graphs.pt")


def sharded_graphs_meta_path(graphs_path: str | Path) -> Path:
    graphs_path = Path(graphs_path)
    return graphs_path.with_name(f"{graphs_path.stem}_shards.json")


def sharded_graphs_dir(graphs_path: str | Path) -> Path:
    graphs_path = Path(graphs_path)
    return graphs_path.with_name(f"{graphs_path.stem}_shards")


def has_sharded_graphs(graphs_path: str | Path) -> bool:
    meta = sharded_graphs_meta_path(graphs_path)
    return meta.is_file()


def shard_graph_sidecar(
    graphs_path: str | Path,
    shard_size: int = 10_000,
    output_dir: str | Path | None = None,
) -> Path:
    """Split a monolithic graph .pt into shard files for RAM-bounded fast loading."""
    graphs_path = Path(graphs_path)
    if not graphs_path.is_file():
        raise FileNotFoundError(graphs_path)

    payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
    graphs = payload["graphs"]
    n_graphs = len(graphs)
    out_dir = Path(output_dir) if output_dir is not None else sharded_graphs_dir(graphs_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_shards = (n_graphs + shard_size - 1) // shard_size
    for shard_id in range(n_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, n_graphs)
        shard_path = out_dir / f"shard_{shard_id:04d}.pt"
        torch.save(
            {"version": 1, "start": start, "graphs": graphs[start:end]},
            shard_path,
        )

    meta = {
        "version": 1,
        "source": str(graphs_path),
        "shard_size": shard_size,
        "n_graphs": n_graphs,
        "n_shards": n_shards,
        "shard_dir": str(out_dir),
    }
    meta_path = sharded_graphs_meta_path(graphs_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return out_dir


class ShardedGraphStore:
    """Load precomputed graphs from disk shards; keep a few shards hot in RAM."""

    def __init__(self, graphs_path: str | Path, hot_shards: int = 3):
        graphs_path = Path(graphs_path)
        meta_path = sharded_graphs_meta_path(graphs_path)
        if not meta_path.is_file():
            raise FileNotFoundError(f"Shard metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text())
        self.graphs_path = graphs_path
        self.shard_size = int(meta["shard_size"])
        self.n_graphs = int(meta["n_graphs"])
        self.n_shards = int(meta["n_shards"])
        self.shard_dir = Path(meta["shard_dir"])
        self.hot_shards = max(1, hot_shards)
        self._loaded: OrderedDict[int, list] = OrderedDict()

    def _read_shard_graphs(self, shard_id: int) -> list:
        """Load one shard without updating the hot LRU cache."""
        shard_path = self.shard_dir / f"shard_{shard_id:04d}.pt"
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        return payload["graphs"]

    def _load_shard(self, shard_id: int) -> list:
        if shard_id in self._loaded:
            self._loaded.move_to_end(shard_id)
            return self._loaded[shard_id]
        graphs = self._read_shard_graphs(shard_id)
        self._loaded[shard_id] = graphs
        while len(self._loaded) > self.hot_shards:
            self._loaded.popitem(last=False)
        return graphs

    def preload_all(self, verbose: bool = True) -> int:
        """Load every shard into RAM (respecting hot_shards eviction cap)."""
        import sys

        from tqdm import tqdm

        shard_ids = range(self.n_shards)
        if verbose:
            shard_ids = tqdm(
                shard_ids,
                desc="Loading graph shards",
                file=sys.stderr,
                mininterval=1.0,
            )
        for shard_id in shard_ids:
            self._load_shard(shard_id)
        return self.n_graphs

    def warm_shards(self, n: int = 1) -> int:
        """Load the first n shards into the hot LRU cache."""
        n = max(0, min(int(n), self.n_shards))
        for shard_id in range(n):
            self._load_shard(shard_id)
        return n

    def warm_disk_cache(self, verbose: bool = True) -> None:
        """Read all shard files sequentially to warm the OS page cache."""
        import sys

        from tqdm import tqdm

        paths = [self.shard_dir / f"shard_{shard_id:04d}.pt" for shard_id in range(self.n_shards)]
        iterator = paths
        if verbose:
            iterator = tqdm(
                paths,
                desc="Warming graph shard disk cache",
                file=sys.stderr,
                mininterval=1.0,
            )
        for path in iterator:
            with path.open("rb") as handle:
                while handle.read(8 << 20):
                    pass

    def get_by_index(self, index: int):
        if index < 0 or index >= self.n_graphs:
            raise IndexError(f"Graph index {index} out of range [0, {self.n_graphs})")
        shard_id = index // self.shard_size
        offset = index % self.shard_size
        return self._load_shard(shard_id)[offset]

    def get_graphs_for_indices(self, index_pairs: list[tuple[int, int]]) -> dict[int, object]:
        """Fetch many graphs, loading each shard at most once (dataset_idx, sidecar_idx)."""
        from collections import defaultdict

        by_shard: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for dataset_idx, graph_index in index_pairs:
            if graph_index < 0 or graph_index >= self.n_graphs:
                raise IndexError(f"Graph index {graph_index} out of range [0, {self.n_graphs})")
            shard_id = graph_index // self.shard_size
            offset = graph_index % self.shard_size
            by_shard[shard_id].append((dataset_idx, offset))

        out: dict[int, object] = {}
        for shard_id, entries in by_shard.items():
            graphs = self._load_shard(shard_id)
            for dataset_idx, offset in entries:
                out[dataset_idx] = graphs[offset]
        return out


class ShardGroupedBatchSampler:
    """Batch indices grouped by graph shard so each batch triggers one shard load."""

    def __init__(
        self,
        samples: list[CrystalSample],
        batch_size: int,
        shard_size: int,
        drop_last: bool = False,
        seed: int = 0,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
    ):
        by_shard: dict[int, list[int]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            graph_index = sample.id if sample.id is not None else idx
            by_shard[graph_index // shard_size].append(idx)
        self.by_shard = dict(by_shard)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        n = 0
        for indices in self.by_shard.values():
            n_full, rem = divmod(len(indices), self.batch_size)
            n += n_full
            if rem and not self.drop_last:
                n += 1
        return n

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shard_ids = sorted(self.by_shard.keys())
        if self.shuffle_shards:
            rng.shuffle(shard_ids)
        for shard_id in shard_ids:
            indices = self.by_shard[shard_id][:]
            if self.shuffle_within_shard:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch


class HybridGraphStore:
    """Sharded precomputed graphs when available; otherwise SMILES cache."""

    def __init__(
        self,
        graphs_path: str | Path | None,
        hot_shards: int = 3,
        cache_size: int = 0,
    ):
        from cellnet.graph import SmilesGraphCache

        self._shard: ShardedGraphStore | None = None
        self._smiles = SmilesGraphCache(maxsize=cache_size)
        if graphs_path is not None and has_sharded_graphs(graphs_path):
            self._shard = ShardedGraphStore(graphs_path, hot_shards=hot_shards)

    @property
    def uses_shards(self) -> bool:
        return self._shard is not None

    def warm_smiles(self, smiles_list: list[str], workers: int = 8) -> int:
        if self._shard is not None:
            return 0
        return self._smiles.preload(smiles_list, workers=workers)

    def preload_shards(self, verbose: bool = True) -> int:
        if self._shard is None:
            return 0
        return self._shard.preload_all(verbose=verbose)

    def warm_shards(self, n: int = 1) -> int:
        if self._shard is None:
            return 0
        return self._shard.warm_shards(n)

    def warm_disk_cache(self, verbose: bool = True) -> None:
        if self._shard is not None:
            self._shard.warm_disk_cache(verbose=verbose)

    @property
    def n_graphs(self) -> int:
        if self._shard is not None:
            return self._shard.n_graphs
        return 0

    def get(self, sample: CrystalSample, index: int):
        if self._shard is not None:
            return self._shard.get_by_index(sample.id if sample.id is not None else index)
        return self._smiles.get(sample.smiles)

    def get_graphs_for_dataset_indices(
        self,
        items: list[tuple[int, CrystalSample]],
    ) -> dict[int, object]:
        if self._shard is None:
            return {idx: self._smiles.get(sample.smiles) for idx, sample in items}
        pairs = [
            (idx, sample.id if sample.id is not None else idx)
            for idx, sample in items
        ]
        return self._shard.get_graphs_for_indices(pairs)


def is_precomputed_csv(csv_path: str | Path) -> bool:
    with Path(csv_path).open(newline="") as handle:
        header = handle.readline()
    return PRECOMPUTED_MARKER in header


def _row_to_cellpar(row: dict) -> np.ndarray:
    return np.array(
        [
            float(row["a"]),
            float(row["b"]),
            float(row["c"]),
            float(row["alpha"]),
            float(row["beta"]),
            float(row["gamma"]),
        ],
        dtype=np.float64,
    )


def _sample_from_row(row: dict, sample_id: int) -> CrystalSample:
    cellpar = _row_to_cellpar(row)
    sg_number = int(row["sg_number"])
    return CrystalSample(
        id=sample_id,
        csd_code=str(row.get("refcode", f"SPADE_{sample_id}")),
        smiles=str(row["smiles"]),
        hall_number=int(row["hall_number"]),
        zprime=float(row["zprime"]),
        spg_num=sg_number,
        space_group=str(row.get("sg_symbol", "")),
        l_type="",
        cellpar=cellpar,
        reciprocal_metric=cellpar_to_reciprocal_metric(cellpar),
    )


def _attach_targets_from_row(sample: CrystalSample, row: dict) -> None:
    sample.selling_log1p = np.array(
        [float(row[c]) for c in SELLING_COLUMNS],
        dtype=np.float64,
    )
    sample.log_successive_minima = np.array(
        [float(row[c]) for c in LOG_LAMBDA_COLUMNS],
        dtype=np.float64,
    )
    sample.log_reciprocal_successive_minima = np.array(
        [float(row[c]) for c in LOG_LAMBDA_RECIP_COLUMNS],
        dtype=np.float64,
    )
    sample.log_density = float(row["log_density"])
    sample.free_padded = np.array(
        [float(row[c]) for c in FREE_COLUMNS],
        dtype=np.float64,
    )
    sample.free_mask = np.array(
        [float(row[c]) for c in FREE_MASK_COLUMNS],
        dtype=np.float64,
    )


def sample_to_precomputed_row(sample: CrystalSample) -> dict[str, str | float]:
    if (
        sample.selling_log1p is None
        or sample.log_successive_minima is None
        or sample.log_reciprocal_successive_minima is None
        or sample.free_padded is None
        or sample.free_mask is None
    ):
        raise ValueError(f"Sample {sample.csd_code} is missing precomputed targets")

    row: dict[str, str | float] = {
        "refcode": sample.csd_code,
        "sg_symbol": sample.space_group,
        "sg_number": sample.spg_num,
        "hall_number": sample.hall_number,
        "smiles": sample.smiles,
        "zprime": sample.zprime,
        "a": sample.cellpar[0],
        "b": sample.cellpar[1],
        "c": sample.cellpar[2],
        "alpha": sample.cellpar[3],
        "beta": sample.cellpar[4],
        "gamma": sample.cellpar[5],
        "density": 0.0,
        "log_density": sample.log_density,
    }
    for i, col in enumerate(SELLING_COLUMNS):
        row[col] = float(sample.selling_log1p[i])
    for i, col in enumerate(LOG_LAMBDA_COLUMNS):
        row[col] = float(sample.log_successive_minima[i])
    for i, col in enumerate(LOG_LAMBDA_RECIP_COLUMNS):
        row[col] = float(sample.log_reciprocal_successive_minima[i])
    for i, col in enumerate(FREE_COLUMNS):
        row[col] = float(sample.free_padded[i])
    for i, col in enumerate(FREE_MASK_COLUMNS):
        row[col] = float(sample.free_mask[i])
    return row


def graph_to_payload(graph: object) -> dict[str, np.ndarray]:
    """Convert PyG graph to plain NumPy arrays for multiprocessing IPC."""
    return {
        "x": graph.x.detach().cpu().numpy(),
        "edge_index": graph.edge_index.detach().cpu().numpy(),
        "edge_attr": graph.edge_attr.detach().cpu().numpy(),
    }


def graph_from_payload(payload: dict[str, np.ndarray]) -> object:
    """Rebuild a PyG graph from NumPy payload."""
    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(np.asarray(payload["x"], dtype=np.float32)),
        edge_index=torch.from_numpy(np.asarray(payload["edge_index"], dtype=np.int64)),
        edge_attr=torch.from_numpy(np.asarray(payload["edge_attr"], dtype=np.float32)),
    )


def precompute_sample(sample: CrystalSample) -> CrystalSample | None:
    """Graphify and attach lattice-flow targets for one structure."""
    graph = smiles_to_graph(sample.smiles)
    if graph is None:
        return None
    sample.graph = graph
    attach_selling_targets([sample], verbose=False)
    attach_sequential_targets([sample])
    attach_lattice_invariants([sample], verbose=False)
    if (
        sample.selling_log1p is None
        or sample.log_successive_minima is None
        or sample.log_reciprocal_successive_minima is None
    ):
        return None
    return sample


def write_precomputed_csv(
    samples: list[CrystalSample],
    csv_path: str | Path,
    graphs_path: str | Path | None = None,
) -> None:
    """Write precomputed CSV and aligned PyG graph sidecar."""
    csv_path = Path(csv_path)
    graphs_path = Path(graphs_path) if graphs_path is not None else precomputed_graphs_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRECOMPUTED_COLUMNS)
        writer.writeheader()
        for sample in samples:
            row = sample_to_precomputed_row(sample)
            writer.writerow({k: row[k] for k in PRECOMPUTED_COLUMNS})

    graphs = [sample.graph for sample in samples]
    torch.save({"version": 1, "n_graphs": len(graphs), "graphs": graphs}, graphs_path)


def load_precomputed_csv(
    csv_path: str | Path,
    graphs_path: str | Path | None = None,
    max_samples: int | None = None,
    verbose: bool = True,
    lazy_graphs: bool = False,
) -> list[CrystalSample]:
    """Load precomputed CSV + optional graph sidecar into CrystalSample records."""
    csv_path = Path(csv_path)
    graphs_path = Path(graphs_path) if graphs_path is not None else precomputed_graphs_path(csv_path)

    graphs: list | None = None
    if not lazy_graphs:
        if not graphs_path.is_file():
            raise FileNotFoundError(f"Graph sidecar not found: {graphs_path}")
        size_mb = graphs_path.stat().st_size / (1024 * 1024)
        if verbose:
            print(
                f"Loading graph sidecar {graphs_path.name} ({size_mb:.0f} MB) — "
                f"this can take several minutes ...",
                flush=True,
            )
        payload = torch.load(graphs_path, map_location="cpu", weights_only=False)
        graphs = payload["graphs"]
        if len(graphs) != payload.get("n_graphs", len(graphs)):
            raise ValueError(f"Graph count mismatch in {graphs_path}")
        if verbose:
            print(f"Loaded {len(graphs)} graphs from {graphs_path.name}", flush=True)

    samples: list[CrystalSample] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            if graphs is not None and i >= len(graphs):
                raise ValueError(
                    f"CSV has more rows than graphs ({i + 1} rows vs {len(graphs)} graphs)"
                )
            sample = _sample_from_row(row, i)
            _attach_targets_from_row(sample, row)
            if graphs is not None:
                sample.graph = graphs[i]
            samples.append(sample)

    if verbose:
        if lazy_graphs:
            print(
                f"Loaded {len(samples)} precomputed samples from {csv_path} "
                f"(lazy graphs: SMILES→PyG on batch load)"
            )
        else:
            print(
                f"Loaded {len(samples)} precomputed samples from {csv_path} "
                f"(graphs: {graphs_path.name})"
            )
    return samples
