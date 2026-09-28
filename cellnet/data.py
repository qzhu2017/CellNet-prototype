"""Crystal samples, normalization statistics and the PyG training dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cellnet.graph import BOND_TYPES, NODE_FEATURE_DIM, smiles_to_graph
from cellnet.sequential import (
    compute_log_density,
    compute_log_reciprocal_successive_minima,
    compute_log_successive_minima,
    compute_selling_parameters,
    signed_log1p,
)
from cellnet.symmetry import (
    AXIS_PERMUTATIONS,
    axis_permutation_equivalence_mask,
    axis_permutation_index,
    axis_rank_permutation,
    canonical_axis_rank_permutation,
    crystal_system_from_hall,
)


@dataclass
class CrystalSample:
    """Single crystal structure record and its lattice-flow targets."""

    id: int
    csd_code: str
    smiles: str
    hall_number: int
    zprime: float
    spg_num: int
    space_group: str
    l_type: str
    cellpar: np.ndarray  # [a, b, c, alpha, beta, gamma]
    log_density: float = 0.0
    log_successive_minima: np.ndarray | None = None  # (λ₁, λ₂, λ₃) in log-Å
    log_reciprocal_successive_minima: np.ndarray | None = None  # (λ*₁, λ*₂, λ*₃) in log(Å⁻¹)
    selling_log1p: np.ndarray | None = None  # signed-log1p(Delaunay Selling), shape (6,)
    graph: object | None = None  # torch_geometric.data.Data


def _array(values, default: list[float]) -> np.ndarray:
    return np.array(values if values is not None else default, dtype=np.float32)


@dataclass
class DatasetStats:
    """Normalization statistics and label encodings saved next to a checkpoint (stats.json)."""

    log_density_mean: float = 0.0
    log_density_std: float = 1.0
    selling_mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    selling_std: np.ndarray = field(default_factory=lambda: np.ones(6))
    selling_dim: int = 6
    log_lambda_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    log_lambda_std: np.ndarray = field(default_factory=lambda: np.ones(3))
    log_lambda_dim: int = 3
    log_lambda_recip_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    log_lambda_recip_std: np.ndarray = field(default_factory=lambda: np.ones(3))
    log_lambda_recip_dim: int = 3
    lattice_flow_dim: int = 12
    hall_to_idx: dict[int, int] = field(default_factory=dict)
    idx_to_hall: dict[int, int] = field(default_factory=dict)
    axis_permutation_priors: dict[int, list[float]] = field(default_factory=dict)
    zprime_values: list[float] = field(default_factory=list)
    zprime_to_idx: dict[float, int] = field(default_factory=dict)
    n_node_features: int = NODE_FEATURE_DIM
    n_edge_features: int = len(BOND_TYPES)
    n_halls: int = 0
    n_zprimes: int = 0

    def save(self, path: Path) -> None:
        data = {
            "log_density_mean": self.log_density_mean,
            "log_density_std": self.log_density_std,
            "selling_mean": self.selling_mean.tolist(),
            "selling_std": self.selling_std.tolist(),
            "selling_dim": self.selling_dim,
            "log_lambda_mean": self.log_lambda_mean.tolist(),
            "log_lambda_std": self.log_lambda_std.tolist(),
            "log_lambda_dim": self.log_lambda_dim,
            "log_lambda_recip_mean": self.log_lambda_recip_mean.tolist(),
            "log_lambda_recip_std": self.log_lambda_recip_std.tolist(),
            "log_lambda_recip_dim": self.log_lambda_recip_dim,
            "lattice_flow_dim": self.lattice_flow_dim,
            "hall_to_idx": {str(k): v for k, v in self.hall_to_idx.items()},
            "idx_to_hall": {str(k): v for k, v in self.idx_to_hall.items()},
            "axis_permutation_priors": {
                str(k): list(v) for k, v in self.axis_permutation_priors.items()
            },
            "zprime_values": self.zprime_values,
            "zprime_to_idx": {str(k): v for k, v in self.zprime_to_idx.items()},
            "n_node_features": self.n_node_features,
            "n_edge_features": self.n_edge_features,
            "n_halls": self.n_halls,
            "n_zprimes": self.n_zprimes,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "DatasetStats":
        """Read stats.json; keys written by older versions of CellNet are ignored."""
        data = json.loads(path.read_text())
        return cls(
            log_density_mean=float(data.get("log_density_mean", 0.0)),
            log_density_std=float(data.get("log_density_std", 1.0)),
            selling_mean=_array(data.get("selling_mean"), [0.0] * 6),
            selling_std=_array(data.get("selling_std"), [1.0] * 6),
            selling_dim=int(data.get("selling_dim", 6)),
            log_lambda_mean=_array(data.get("log_lambda_mean"), [0.0] * 3),
            log_lambda_std=_array(data.get("log_lambda_std"), [1.0] * 3),
            log_lambda_dim=int(data.get("log_lambda_dim", 3)),
            log_lambda_recip_mean=_array(data.get("log_lambda_recip_mean"), [0.0] * 3),
            log_lambda_recip_std=_array(data.get("log_lambda_recip_std"), [1.0] * 3),
            log_lambda_recip_dim=int(data.get("log_lambda_recip_dim", 3)),
            lattice_flow_dim=int(data.get("lattice_flow_dim", 12)),
            hall_to_idx={int(k): v for k, v in data["hall_to_idx"].items()},
            idx_to_hall={int(k): v for k, v in data["idx_to_hall"].items()},
            axis_permutation_priors={
                int(k): [float(value) for value in values]
                for k, values in data.get("axis_permutation_priors", {}).items()
            },
            zprime_values=data["zprime_values"],
            zprime_to_idx={float(k): v for k, v in data["zprime_to_idx"].items()},
            n_node_features=data.get("n_node_features", NODE_FEATURE_DIM),
            n_edge_features=data.get("n_edge_features", len(BOND_TYPES)),
            n_halls=data["n_halls"],
            n_zprimes=data["n_zprimes"],
        )


def graphify_samples(
    samples: list[CrystalSample],
    verbose: bool = True,
) -> list[CrystalSample]:
    """Attach molecular graphs to each sample; drop invalid SMILES."""
    valid: list[CrystalSample] = []
    skipped = 0
    for s in samples:
        graph = smiles_to_graph(s.smiles)
        if graph is None:
            skipped += 1
            continue
        s.graph = graph
        valid.append(s)
    if verbose:
        print(f"Graphified {len(valid)} samples (skipped invalid SMILES={skipped})")
    return valid


def attach_log_density(samples: list[CrystalSample]) -> None:
    """Compute the log-density target from the cell, molecule, Z′ and Hall setting."""
    for s in samples:
        s.log_density = compute_log_density(s.cellpar, s.smiles, s.zprime, s.hall_number)


def attach_selling_targets(samples: list[CrystalSample], verbose: bool = True) -> None:
    """Attach signed-log1p(Delaunay Selling) targets for each sample."""
    from cellnet.lattice_invariants import sort_selling_parameters

    skipped = 0
    for s in samples:
        try:
            selling = sort_selling_parameters(compute_selling_parameters(s.cellpar))
            s.selling_log1p = signed_log1p(selling).astype(np.float64)
        except Exception:
            skipped += 1
            s.selling_log1p = None
    if verbose:
        ok = sum(s.selling_log1p is not None for s in samples)
        print(f"Selling parameters: {ok} ok, {skipped} skipped")


def attach_lattice_invariants(samples: list[CrystalSample], verbose: bool = True) -> None:
    """Attach log(λ₁), log(λ₂), log(λ₃) and log(λ*₁), log(λ*₂), log(λ*₃)."""
    skipped = 0
    for s in samples:
        try:
            s.log_successive_minima = compute_log_successive_minima(s.cellpar)
            s.log_reciprocal_successive_minima = compute_log_reciprocal_successive_minima(s.cellpar)
        except Exception:
            skipped += 1
            s.log_successive_minima = None
            s.log_reciprocal_successive_minima = None
    if verbose:
        ok = sum(s.log_successive_minima is not None for s in samples)
        print(f"Successive minima: {ok} ok, {skipped} skipped")


def lattice_flow_target(sample: CrystalSample, stats: DatasetStats) -> np.ndarray:
    """Normalized [Selling(6), log λ(3), log λ*(3)] flow target."""
    sell = (sample.selling_log1p - stats.selling_mean) / stats.selling_std
    lam = (sample.log_successive_minima - stats.log_lambda_mean) / stats.log_lambda_std
    lam_r = (
        (sample.log_reciprocal_successive_minima - stats.log_lambda_recip_mean)
        / stats.log_lambda_recip_std
    )
    return np.concatenate([sell, lam, lam_r]).astype(np.float32)


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def compute_stats(samples: list[CrystalSample]) -> DatasetStats:
    """Normalization statistics (from training samples) for the lattice-flow targets."""
    for s in samples:
        if s.selling_log1p is None:
            attach_selling_targets([s], verbose=False)
        if s.log_successive_minima is None:
            attach_lattice_invariants([s], verbose=False)

    log_dens = np.array([s.log_density for s in samples], dtype=np.float64)
    log_density_mean = float(log_dens.mean())
    log_density_std = float(log_dens.std()) if log_dens.std() > 1e-8 else 1.0

    sell = np.stack([s.selling_log1p for s in samples if s.selling_log1p is not None])
    selling_mean, selling_std = _mean_std(sell)
    lam = np.stack([s.log_successive_minima for s in samples if s.log_successive_minima is not None])
    log_lambda_mean, log_lambda_std = _mean_std(lam)
    lam_r = np.stack(
        [s.log_reciprocal_successive_minima for s in samples if s.log_reciprocal_successive_minima is not None]
    )
    log_lambda_recip_mean, log_lambda_recip_std = _mean_std(lam_r)

    halls = sorted({s.hall_number for s in samples})
    hall_to_idx = {h: i for i, h in enumerate(halls)}
    idx_to_hall = {i: h for h, i in hall_to_idx.items()}
    axis_permutation_priors: dict[int, list[float]] = {}
    for hall_number in halls:
        counts = np.zeros(len(AXIS_PERMUTATIONS), dtype=np.float64)
        hall_samples = [sample for sample in samples if sample.hall_number == hall_number]
        if crystal_system_from_hall(hall_number) == "orthorhombic":
            for sample in hall_samples:
                ranks = canonical_axis_rank_permutation(
                    axis_rank_permutation(sample.cellpar),
                    hall_number,
                )
                counts[axis_permutation_index(ranks)] += 1.0
        if counts.sum() == 0.0:
            counts[axis_permutation_index((0, 1, 2))] = 1.0
        axis_permutation_priors[hall_number] = (counts / counts.sum()).tolist()

    zprimes = sorted({s.zprime for s in samples})
    zprime_to_idx = {z: i for i, z in enumerate(zprimes)}

    return DatasetStats(
        log_density_mean=log_density_mean,
        log_density_std=log_density_std,
        selling_mean=selling_mean.astype(np.float32),
        selling_std=selling_std.astype(np.float32),
        selling_dim=sell.shape[1],
        log_lambda_mean=log_lambda_mean.astype(np.float32),
        log_lambda_std=log_lambda_std.astype(np.float32),
        log_lambda_dim=lam.shape[1],
        log_lambda_recip_mean=log_lambda_recip_mean.astype(np.float32),
        log_lambda_recip_std=log_lambda_recip_std.astype(np.float32),
        log_lambda_recip_dim=lam_r.shape[1],
        lattice_flow_dim=int(sell.shape[1] + lam.shape[1] + lam_r.shape[1]),
        hall_to_idx=hall_to_idx,
        idx_to_hall=idx_to_hall,
        axis_permutation_priors=axis_permutation_priors,
        zprime_values=zprimes,
        zprime_to_idx=zprime_to_idx,
        n_node_features=NODE_FEATURE_DIM,
        n_edge_features=len(BOND_TYPES),
        n_halls=len(halls),
        n_zprimes=len(zprimes),
    )


class CrystalGraphDataset(Dataset):
    """PyTorch dataset returning PyG graphs for lattice-flow GNN training."""

    def __init__(
        self,
        samples: list[CrystalSample],
        stats: DatasetStats,
        target: str = "lattice_flow",
        with_log_density: bool = True,
        lazy_graphs: bool = False,
        graph_store: object | None = None,
        graph_cache_size: int = 8192,
    ):
        if target != "lattice_flow":
            raise ValueError(f"Unsupported target {target!r}; expected 'lattice_flow'")
        self.samples = samples
        self.stats = stats
        self.target = target
        self.with_log_density = with_log_density
        self.graph_store = graph_store
        self._graph_cache = None
        # Per-sample target tensors are identical every epoch; memoize them so a
        # step does not redo the NumPy target/axis-mask work for 128 samples.
        self._static_cache: dict[int, dict] = {}
        if graph_store is None and (lazy_graphs or any(s.graph is None for s in samples)):
            from cellnet.graph import SmilesGraphCache

            self._graph_cache = SmilesGraphCache(maxsize=graph_cache_size)

    def _sample_graph(self, sample: CrystalSample, idx: int, graph=None):
        if graph is not None:
            return graph
        if sample.graph is not None:
            return sample.graph
        if self.graph_store is not None:
            return self.graph_store.get(sample, idx)
        assert self._graph_cache is not None
        return self._graph_cache.get(sample.smiles)

    def _graphs_for_indices(self, indices: list[int]) -> dict[int, object]:
        graphs: dict[int, object] = {}
        pending: list[tuple[int, CrystalSample]] = []
        for idx in indices:
            sample = self.samples[idx]
            if sample.graph is not None:
                graphs[idx] = sample.graph
            else:
                pending.append((idx, sample))
        if pending and self.graph_store is not None and hasattr(
            self.graph_store, "get_graphs_for_dataset_indices"
        ):
            graphs.update(self.graph_store.get_graphs_for_dataset_indices(pending))
        elif pending:
            for idx, sample in pending:
                graphs[idx] = self._sample_graph(sample, idx)
        return graphs

    def _static_fields(self, idx: int) -> dict:
        cached = self._static_cache.get(idx)
        if cached is not None:
            return cached
        s = self.samples[idx]
        y = lattice_flow_target(s, self.stats)
        if crystal_system_from_hall(s.hall_number) == "orthorhombic":
            axis_mask = axis_permutation_equivalence_mask(s.cellpar, s.hall_number)
        else:
            # Non-orthorhombic samples contribute zero classification loss.
            axis_mask = np.ones(len(AXIS_PERMUTATIONS), dtype=bool)
        fields = {
            "y": torch.from_numpy(y.astype(np.float32)).view(1, -1),
            "hall": torch.tensor([self.stats.hall_to_idx[s.hall_number]], dtype=torch.long),
            "zprime": torch.tensor([self.stats.zprime_to_idx[s.zprime]], dtype=torch.long),
            "axis_permutation_mask": torch.from_numpy(axis_mask).view(1, -1),
            "cellpar": torch.from_numpy(s.cellpar.astype(np.float32)).view(1, -1),
        }
        if self.with_log_density:
            fields["log_density"] = torch.tensor(
                [(s.log_density - self.stats.log_density_mean) / self.stats.log_density_std],
                dtype=torch.float32,
            )
        self._static_cache[idx] = fields
        return fields

    def _make_data(self, idx: int, graph=None):
        from torch_geometric.data import Data

        s = self.samples[idx]
        graph = self._sample_graph(s, idx, graph=graph)
        data = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
        )
        for key, value in self._static_fields(idx).items():
            setattr(data, key, value)
        data.csd_code = s.csd_code
        data.smiles = s.smiles
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self._make_data(idx)

    def __getitems__(self, indices: list[int]) -> list:
        graphs = self._graphs_for_indices(indices)
        return [self._make_data(idx, graph=graphs.get(idx)) for idx in indices]
