"""Load HEM.db structures via PyXtal and export tabular summaries."""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from pyxtal.db import database
from torch.utils.data import Dataset
from tqdm import tqdm

from cellnet.features import smiles_features
from cellnet.graph import BOND_TYPES, NODE_FEATURE_DIM, smiles_to_graph
from cellnet.reciprocal import (
    cellpar_to_reciprocal_metric,
    log_transform_metric,
)
from cellnet.sequential import (
    FREE_PARAM_DIM,
    cellpar_to_padded_free,
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
class RecpConfig:
    """Legacy reciprocal-grid config kept for loading older stats.json checkpoints."""

    dmax: float = 6.0
    res: float = 0.1
    nmax: int = 6
    lmax: int = 4
    stack_sites: int = 1
    p_dim: int = 0
    rdf_dim: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "RecpConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class CrystalSample:
    """Single crystal structure record."""

    id: int
    csd_code: str
    smiles: str
    hall_number: int
    zprime: float
    spg_num: int
    space_group: str
    l_type: str
    cellpar: np.ndarray  # [a, b, c, alpha, beta, gamma]
    reciprocal_metric: np.ndarray  # 6 independent components of G*
    packing: np.ndarray | None = None  # [log(V), log(a/b), log(b/c), alpha, beta, gamma]
    free_padded: np.ndarray | None = None  # symmetry-reduced params in fixed 6 slots
    free_mask: np.ndarray | None = None
    log_density: float = 0.0
    log_successive_minima: np.ndarray | None = None  # (λ₁, λ₂, λ₃) in log-Å
    log_reciprocal_successive_minima: np.ndarray | None = None  # (λ*₁, λ*₂, λ*₃) in log(Å⁻¹)
    selling_log1p: np.ndarray | None = None  # signed-log1p(Delaunay Selling), shape (6,)
    features: np.ndarray | None = None
    graph: object | None = None  # torch_geometric.data.Data
    lattice_weight: float = 1.0
    lattice_shape_bin: int = 0


@dataclass
class DatasetStats:
    """Normalization and label-encoding statistics."""

    feature_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_std: np.ndarray = field(default_factory=lambda: np.ones(0))
    metric_mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    metric_std: np.ndarray = field(default_factory=lambda: np.ones(6))
    cellpar_mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    cellpar_std: np.ndarray = field(default_factory=lambda: np.ones(6))
    packing_mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    packing_std: np.ndarray = field(default_factory=lambda: np.ones(6))
    free_mean: np.ndarray = field(default_factory=lambda: np.zeros(FREE_PARAM_DIM))
    free_std: np.ndarray = field(default_factory=lambda: np.ones(FREE_PARAM_DIM))
    log_density_mean: float = 0.0
    log_density_std: float = 1.0
    powerspec_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    powerspec_std: np.ndarray = field(default_factory=lambda: np.ones(0))
    recp_config: RecpConfig = field(default_factory=RecpConfig)
    powerspec_dim: int = 0
    powerspec_p_dim: int = 0
    powerspec_rdf_dim: int = 0
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
    n_features: int = 0
    n_node_features: int = NODE_FEATURE_DIM
    n_edge_features: int = len(BOND_TYPES)
    n_halls: int = 0
    n_zprimes: int = 0

    def save(self, path: Path) -> None:
        data = {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "metric_mean": self.metric_mean.tolist(),
            "metric_std": self.metric_std.tolist(),
            "cellpar_mean": self.cellpar_mean.tolist(),
            "cellpar_std": self.cellpar_std.tolist(),
            "packing_mean": self.packing_mean.tolist(),
            "packing_std": self.packing_std.tolist(),
            "free_mean": self.free_mean.tolist(),
            "free_std": self.free_std.tolist(),
            "log_density_mean": self.log_density_mean,
            "log_density_std": self.log_density_std,
            "powerspec_mean": self.powerspec_mean.tolist(),
            "powerspec_std": self.powerspec_std.tolist(),
            "recp_config": self.recp_config.to_dict(),
            "powerspec_dim": self.powerspec_dim,
            "powerspec_p_dim": self.powerspec_p_dim,
            "powerspec_rdf_dim": self.powerspec_rdf_dim,
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
            "n_features": self.n_features,
            "n_node_features": self.n_node_features,
            "n_edge_features": self.n_edge_features,
            "n_halls": self.n_halls,
            "n_zprimes": self.n_zprimes,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "DatasetStats":
        data = json.loads(path.read_text())
        stats = cls(
            feature_mean=np.array(data["feature_mean"], dtype=np.float32),
            feature_std=np.array(data["feature_std"], dtype=np.float32),
            metric_mean=np.array(data["metric_mean"], dtype=np.float32),
            metric_std=np.array(data["metric_std"], dtype=np.float32),
            cellpar_mean=np.array(data["cellpar_mean"], dtype=np.float32),
            cellpar_std=np.array(data["cellpar_std"], dtype=np.float32),
            packing_mean=np.array(data.get("packing_mean", [0.0] * 6), dtype=np.float32),
            packing_std=np.array(data.get("packing_std", [1.0] * 6), dtype=np.float32),
            free_mean=np.array(data.get("free_mean", [0.0] * FREE_PARAM_DIM), dtype=np.float32),
            free_std=np.array(data.get("free_std", [1.0] * FREE_PARAM_DIM), dtype=np.float32),
            log_density_mean=float(data.get("log_density_mean", 0.0)),
            log_density_std=float(data.get("log_density_std", 1.0)),
            powerspec_mean=np.array(data.get("powerspec_mean", []), dtype=np.float32),
            powerspec_std=np.array(data.get("powerspec_std", []), dtype=np.float32),
            recp_config=RecpConfig.from_dict(data["recp_config"]) if "recp_config" in data else RecpConfig(),
            powerspec_dim=data.get("powerspec_dim", 0),
            powerspec_p_dim=data.get("powerspec_p_dim", 0),
            powerspec_rdf_dim=data.get("powerspec_rdf_dim", 0),
            selling_mean=np.array(data.get("selling_mean", [0.0] * 6), dtype=np.float32),
            selling_std=np.array(data.get("selling_std", [1.0] * 6), dtype=np.float32),
            selling_dim=int(data.get("selling_dim", 6)),
            log_lambda_mean=np.array(data.get("log_lambda_mean", [0.0] * 3), dtype=np.float32),
            log_lambda_std=np.array(data.get("log_lambda_std", [1.0] * 3), dtype=np.float32),
            log_lambda_dim=int(data.get("log_lambda_dim", 3)),
            log_lambda_recip_mean=np.array(data.get("log_lambda_recip_mean", [0.0] * 3), dtype=np.float32),
            log_lambda_recip_std=np.array(data.get("log_lambda_recip_std", [1.0] * 3), dtype=np.float32),
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
            n_features=data["n_features"],
            n_node_features=data.get("n_node_features", NODE_FEATURE_DIM),
            n_edge_features=data.get("n_edge_features", len(BOND_TYPES)),
            n_halls=data["n_halls"],
            n_zprimes=data["n_zprimes"],
        )
        return stats


def _load_pyxtal_record(db, code: str) -> tuple[object, object]:
    """Return (row, pyxtal) for a CSD code, expanding special Wyckoff sites."""
    row = db.get_row(code=code)
    xtal = db.get_pyxtal(code=code)
    if xtal.has_special_site():
        xtal = xtal.to_subgroup()
    return row, xtal


def load_hem_database(
    db_path: str | Path,
    max_samples: int | None = None,
    verbose: bool = True,
) -> list[CrystalSample]:
    """Load HEM.db records via PyXtal (get_pyxtal + to_subgroup)."""
    db = database(str(db_path))
    codes = db.get_all_codes()
    if max_samples is not None:
        codes = codes[:max_samples]

    samples: list[CrystalSample] = []
    skipped = 0

    for code in tqdm(codes, desc="Loading HEM.db", disable=not verbose):
        try:
            row, xtal = _load_pyxtal_record(db, code)
            cellpar = np.array(xtal.lattice.get_para(degree=True), dtype=np.float64)
            rec_metric = cellpar_to_reciprocal_metric(cellpar)

            samples.append(
                CrystalSample(
                    id=int(row.id),
                    csd_code=row.csd_code,
                    smiles=row.mol_smi,
                    hall_number=int(xtal.group.hall_number),
                    zprime=float(row.Zprime),
                    spg_num=int(xtal.group.number),
                    space_group=row.space_group,
                    l_type=row.l_type,
                    cellpar=cellpar,
                    reciprocal_metric=rec_metric,
                )
            )
        except Exception:
            skipped += 1

    if verbose:
        print(f"Loaded {len(samples)} samples via PyXtal (skipped={skipped})")

    return samples


def load_hem_csv(
    csv_path: str | Path,
    max_samples: int | None = None,
    verbose: bool = True,
) -> list[CrystalSample]:
    """Load crystal records from an exported HEM_structures.csv file."""
    csv_path = Path(csv_path)
    samples: list[CrystalSample] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break

            cellpar = np.array([float(v) for v in row["cell_parameters"].split(",")], dtype=np.float64)
            rec_metric = cellpar_to_reciprocal_metric(cellpar)

            samples.append(
                CrystalSample(
                    id=i,
                    csd_code=row["csd_code"],
                    smiles=row["smiles"],
                    hall_number=int(row["hall_number"]),
                    zprime=float(row["zprime"]),
                    spg_num=int(row["spg_num"]),
                    space_group="",
                    l_type="",
                    cellpar=cellpar,
                    reciprocal_metric=rec_metric,
                )
            )

    if verbose:
        print(f"Loaded {len(samples)} samples from {csv_path}")

    return samples


def export_hem_csv(
    db_path: str | Path,
    csv_path: str | Path,
    max_samples: int | None = None,
    verbose: bool = True,
) -> int:
    """Export HEM.db metadata to CSV using PyXtal structure parsing."""
    db = database(str(db_path))
    codes = db.get_all_codes()
    if max_samples is not None:
        codes = codes[:max_samples]

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "csd_code",
        "smiles",
        "spg_num",
        "hall_number",
        "zprime",
        "cell_parameters",
    ]
    n_written = 0

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for code in tqdm(codes, desc="Loading HEM.db", disable=not verbose):
            try:
                row, xtal = _load_pyxtal_record(db, code)
                cellpar = xtal.lattice.get_para(degree=True)
                writer.writerow(
                    {
                        "csd_code": row.csd_code,
                        "smiles": row.mol_smi,
                        "spg_num": int(xtal.group.number),
                        "hall_number": int(xtal.group.hall_number),
                        "zprime": float(row.Zprime),
                        "cell_parameters": ",".join(f"{v:.6f}" for v in cellpar),
                    }
                )
                n_written += 1
            except Exception:
                continue

    if verbose:
        print(f"Wrote {n_written} rows to {csv_path}")

    return n_written


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


def featurize_samples(
    samples: list[CrystalSample],
    fp_bits: int = 2048,
    fp_radius: int = 2,
    verbose: bool = True,
) -> list[CrystalSample]:
    """Attach SMILES features to each sample; drop invalid SMILES."""
    valid: list[CrystalSample] = []
    skipped = 0
    for s in samples:
        feat = smiles_features(s.smiles, fp_bits=fp_bits, fp_radius=fp_radius)
        if feat is None:
            skipped += 1
            continue
        s.features = feat
        valid.append(s)
    if verbose:
        print(f"Featurized {len(valid)} samples (skipped invalid SMILES={skipped})")
    return valid


def attach_sequential_targets(samples: list[CrystalSample]) -> None:
    """Compute symmetry-reduced free parameters and log-density targets."""
    for s in samples:
        free, mask = cellpar_to_padded_free(s.cellpar, s.hall_number)
        s.free_padded = free
        s.free_mask = mask
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


def compute_stats(
    samples: list[CrystalSample],
    use_graph: bool = False,
    use_sequential: bool = False,
    use_log_density: bool = False,
    use_selling: bool = False,
    use_log_lambda: bool = False,
) -> DatasetStats:
    """Compute normalization stats and label encodings from training samples."""
    if use_graph:
        feat_mean = np.zeros(0, dtype=np.float32)
        feat_std = np.ones(0, dtype=np.float32)
        n_features = 0
    else:
        features = np.stack([s.features for s in samples])
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0
        n_features = features.shape[1]

    metrics_log = np.stack([log_transform_metric(s.reciprocal_metric) for s in samples])
    cellpars = np.stack([s.cellpar for s in samples])

    metric_mean = metrics_log.mean(axis=0)
    metric_std = metrics_log.std(axis=0)
    metric_std[metric_std < 1e-8] = 1.0

    cellpar_mean = cellpars.mean(axis=0)
    cellpar_std = cellpars.std(axis=0)
    cellpar_std[cellpar_std < 1e-8] = 1.0

    packing_mean = np.zeros(6, dtype=np.float32)
    packing_std = np.ones(6, dtype=np.float32)

    if use_sequential:
        for s in samples:
            if s.free_padded is None:
                attach_sequential_targets([s])
        free_arr = np.stack([s.free_padded for s in samples])
        mask_arr = np.stack([s.free_mask for s in samples])
        free_mean = np.zeros(FREE_PARAM_DIM, dtype=np.float32)
        free_std = np.ones(FREE_PARAM_DIM, dtype=np.float32)
        for j in range(FREE_PARAM_DIM):
            active = free_arr[:, j][mask_arr[:, j]]
            if active.size > 0:
                free_mean[j] = active.mean()
                std = active.std()
                free_std[j] = std if std > 1e-8 else 1.0
        log_dens = np.array([s.log_density for s in samples], dtype=np.float64)
        log_density_mean = float(log_dens.mean())
        log_density_std = float(log_dens.std()) if log_dens.std() > 1e-8 else 1.0
    elif use_log_density:
        free_mean = np.zeros(FREE_PARAM_DIM, dtype=np.float32)
        free_std = np.ones(FREE_PARAM_DIM, dtype=np.float32)
        for s in samples:
            if s.log_density == 0.0 and s.free_padded is None:
                s.log_density = compute_log_density(s.cellpar, s.smiles, s.zprime, s.hall_number)
        log_dens = np.array([s.log_density for s in samples], dtype=np.float64)
        log_density_mean = float(log_dens.mean())
        log_density_std = float(log_dens.std()) if log_dens.std() > 1e-8 else 1.0
    else:
        free_mean = np.zeros(FREE_PARAM_DIM, dtype=np.float32)
        free_std = np.ones(FREE_PARAM_DIM, dtype=np.float32)
        log_density_mean = 0.0
        log_density_std = 1.0

    powerspec_mean = np.zeros(0, dtype=np.float32)
    powerspec_std = np.ones(0, dtype=np.float32)
    powerspec_dim = 0
    powerspec_p_dim = 0
    powerspec_rdf_dim = 0
    config = RecpConfig()

    if use_selling:
        for s in samples:
            if s.selling_log1p is None:
                attach_selling_targets([s], verbose=False)
        sell = np.stack([s.selling_log1p for s in samples if s.selling_log1p is not None])
        selling_mean = sell.mean(axis=0)
        selling_std = sell.std(axis=0)
        selling_std[selling_std < 1e-8] = 1.0
        selling_dim = sell.shape[1]
    else:
        selling_mean = np.zeros(6, dtype=np.float32)
        selling_std = np.ones(6, dtype=np.float32)
        selling_dim = 6

    if use_log_lambda:
        for s in samples:
            if s.log_successive_minima is None:
                attach_lattice_invariants([s], verbose=False)
        lam = np.stack([s.log_successive_minima for s in samples if s.log_successive_minima is not None])
        log_lambda_mean = lam.mean(axis=0)
        log_lambda_std = lam.std(axis=0)
        log_lambda_std[log_lambda_std < 1e-8] = 1.0
        log_lambda_dim = lam.shape[1]
        lam_r = np.stack(
            [s.log_reciprocal_successive_minima for s in samples if s.log_reciprocal_successive_minima is not None]
        )
        log_lambda_recip_mean = lam_r.mean(axis=0)
        log_lambda_recip_std = lam_r.std(axis=0)
        log_lambda_recip_std[log_lambda_recip_std < 1e-8] = 1.0
        log_lambda_recip_dim = lam_r.shape[1]
    else:
        log_lambda_mean = np.zeros(3, dtype=np.float32)
        log_lambda_std = np.ones(3, dtype=np.float32)
        log_lambda_dim = 3
        log_lambda_recip_mean = np.zeros(3, dtype=np.float32)
        log_lambda_recip_std = np.ones(3, dtype=np.float32)
        log_lambda_recip_dim = 3

    lattice_flow_dim = int(selling_dim + log_lambda_dim + log_lambda_recip_dim)

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
        feature_mean=feat_mean.astype(np.float32),
        feature_std=feat_std.astype(np.float32),
        metric_mean=metric_mean.astype(np.float32),
        metric_std=metric_std.astype(np.float32),
        cellpar_mean=cellpar_mean.astype(np.float32),
        cellpar_std=cellpar_std.astype(np.float32),
        packing_mean=packing_mean.astype(np.float32),
        packing_std=packing_std.astype(np.float32),
        free_mean=free_mean.astype(np.float32),
        free_std=free_std.astype(np.float32),
        log_density_mean=log_density_mean,
        log_density_std=log_density_std,
        powerspec_mean=powerspec_mean.astype(np.float32),
        powerspec_std=powerspec_std.astype(np.float32),
        recp_config=config,
        powerspec_dim=powerspec_dim,
        powerspec_p_dim=powerspec_p_dim,
        powerspec_rdf_dim=powerspec_rdf_dim,
        selling_mean=selling_mean.astype(np.float32),
        selling_std=selling_std.astype(np.float32),
        selling_dim=selling_dim,
        log_lambda_mean=log_lambda_mean.astype(np.float32),
        log_lambda_std=log_lambda_std.astype(np.float32),
        log_lambda_dim=log_lambda_dim,
        log_lambda_recip_mean=log_lambda_recip_mean.astype(np.float32),
        log_lambda_recip_std=log_lambda_recip_std.astype(np.float32),
        log_lambda_recip_dim=log_lambda_recip_dim,
        lattice_flow_dim=lattice_flow_dim,
        hall_to_idx=hall_to_idx,
        idx_to_hall=idx_to_hall,
        axis_permutation_priors=axis_permutation_priors,
        zprime_values=zprimes,
        zprime_to_idx=zprime_to_idx,
        n_features=n_features,
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
            "lattice_weight": torch.tensor([s.lattice_weight], dtype=torch.float32),
            "lattice_shape_bin": torch.tensor([s.lattice_shape_bin], dtype=torch.long),
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
