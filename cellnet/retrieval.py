"""Hall/Z′-filtered molecular-neighbor prior over lattice-shape bins."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors


def smiles_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def shape_retrieval_descriptors(smiles: str) -> np.ndarray | None:
    """Global 2D molecular extent/features that complement local fingerprints."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    distances = Chem.GetDistanceMatrix(mol)
    diameter = float(distances.max()) if distances.size else 0.0
    return np.asarray(
        [
            float(mol.GetNumHeavyAtoms()),
            diameter,
            float(Descriptors.MolWt(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.NumAromaticRings(mol)),
            float(Descriptors.FractionCSP3(mol)),
            float(Descriptors.TPSA(mol)),
        ],
        dtype=np.float32,
    )


class ShapeBinRetrievalIndex:
    """Hall/Z′-filtered molecular-neighbor prior over lattice-shape bins."""

    def __init__(
        self,
        smiles: list[str],
        csd_codes: list[str],
        hall_numbers: np.ndarray,
        zprimes: np.ndarray,
        shape_bins: np.ndarray,
        fingerprints: list,
        descriptors: np.ndarray | None = None,
        *,
        n_bins: int,
        k: int = 16,
    ):
        self.smiles = smiles
        self.csd_codes = csd_codes
        self.hall_numbers = np.asarray(hall_numbers, dtype=np.int32)
        self.zprimes = np.asarray(zprimes, dtype=np.float32)
        self.shape_bins = np.asarray(shape_bins, dtype=np.int64)
        self.fingerprints = fingerprints
        self.descriptors = (
            np.asarray(descriptors, dtype=np.float32) if descriptors is not None else None
        )
        if self.descriptors is not None and len(self.descriptors):
            self.descriptor_mean = self.descriptors.mean(axis=0)
            self.descriptor_std = self.descriptors.std(axis=0)
            self.descriptor_std[self.descriptor_std < 1e-6] = 1.0
        else:
            self.descriptor_mean = None
            self.descriptor_std = None
        self.n_bins = int(n_bins)
        self.k = int(k)
        self._groups: dict[tuple[int, float], np.ndarray] = {}
        self._exact: dict[tuple[str, int, float], np.ndarray] = {}
        grouped: dict[tuple[int, float], list[int]] = {}
        exact: dict[tuple[str, int, float], list[int]] = {}
        for idx, (smi, hall, zp) in enumerate(
            zip(self.smiles, self.hall_numbers, self.zprimes)
        ):
            key = (int(hall), round(float(zp), 6))
            grouped.setdefault(key, []).append(idx)
            exact.setdefault((smi, *key), []).append(idx)
        self._groups = {
            key: np.asarray(indices, dtype=np.int64) for key, indices in grouped.items()
        }
        self._exact = {
            key: np.asarray(indices, dtype=np.int64) for key, indices in exact.items()
        }

    @classmethod
    def build(cls, samples, *, n_bins: int, k: int = 16) -> "ShapeBinRetrievalIndex":
        smiles, codes, halls, zprimes, bins, fps, descriptors = [], [], [], [], [], [], []
        for sample in samples:
            fp = smiles_fingerprint(sample.smiles)
            desc = shape_retrieval_descriptors(sample.smiles)
            if fp is None or desc is None:
                continue
            smiles.append(sample.smiles)
            codes.append(sample.csd_code)
            halls.append(int(sample.hall_number))
            zprimes.append(float(sample.zprime))
            bins.append(int(sample.lattice_shape_bin))
            fps.append(fp)
            descriptors.append(desc)
        return cls(
            smiles,
            codes,
            np.asarray(halls),
            np.asarray(zprimes),
            np.asarray(bins),
            fps,
            np.stack(descriptors),
            n_bins=n_bins,
            k=k,
        )

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: Path) -> "ShapeBinRetrievalIndex":
        return pickle.loads(path.read_bytes())

    def query(
        self,
        smiles: str,
        hall_number: int,
        zprime: float,
    ) -> tuple[np.ndarray, float, list[tuple[str, float, int]]]:
        """
        Return (bin probabilities, confidence, neighbor diagnostics).

        Exact molecule/Hall/Z′ matches use all known packings. Otherwise,
        Tanimoto-weighted neighbors are restricted to the same Hall and Z′.
        """
        uniform = np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
        key = (int(hall_number), round(float(zprime), 6))
        exact_idx = self._exact.get((smiles, *key))
        if exact_idx is not None and len(exact_idx):
            counts = np.bincount(self.shape_bins[exact_idx], minlength=self.n_bins).astype(float)
            probs = counts / counts.sum()
            neighbors = [
                (self.csd_codes[idx], 1.0, int(self.shape_bins[idx])) for idx in exact_idx
            ]
            return probs, 1.0, neighbors

        group_idx = self._groups.get(key)
        qfp = smiles_fingerprint(smiles)
        if qfp is None or group_idx is None or len(group_idx) == 0:
            return uniform, 0.0, []

        group_fps = [self.fingerprints[int(idx)] for idx in group_idx]
        fp_sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(qfp, group_fps), dtype=np.float64
        )
        sims = fp_sims
        descriptors = getattr(self, "descriptors", None)
        qdesc = shape_retrieval_descriptors(smiles)
        if (
            descriptors is not None
            and qdesc is not None
            and self.descriptor_mean is not None
            and self.descriptor_std is not None
        ):
            qnorm = (qdesc - self.descriptor_mean) / self.descriptor_std
            dnorm = (descriptors[group_idx] - self.descriptor_mean) / self.descriptor_std
            descriptor_distance = np.mean((dnorm - qnorm) ** 2, axis=1)
            descriptor_sims = np.exp(-0.5 * descriptor_distance)
            sims = 0.55 * fp_sims + 0.45 * descriptor_sims
        order = np.argsort(sims)[::-1][: self.k]
        top_idx = group_idx[order]
        top_sims = sims[order]
        weights = np.clip(top_sims, 0.0, None) ** 3
        if float(weights.sum()) <= 1e-12:
            return uniform, 0.0, []
        counts = np.bincount(
            self.shape_bins[top_idx],
            weights=weights,
            minlength=self.n_bins,
        ).astype(np.float64)
        probs = (counts + 0.05) / (counts.sum() + 0.05 * self.n_bins)
        confidence = float(np.clip(top_sims[0], 0.0, 1.0))
        neighbors = [
            (self.csd_codes[int(idx)], float(sim), int(self.shape_bins[int(idx)]))
            for idx, sim in zip(top_idx, top_sims)
        ]
        return probs, confidence, neighbors

    def metadata(self) -> dict:
        return {
            "k": self.k,
            "n_bins": self.n_bins,
            "n_reference": len(self.smiles),
            "n_hall_zprime_groups": len(self._groups),
        }

    def save_metadata(self, path: Path) -> None:
        path.write_text(json.dumps(self.metadata(), indent=2))
