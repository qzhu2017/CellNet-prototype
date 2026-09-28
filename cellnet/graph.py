"""Molecular graph construction from SMILES."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import torch
from rdkit import Chem
from torch_geometric.data import Data

# Common elements in the HEM organic crystal dataset.
ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

NODE_FEATURE_DIM = len(ATOM_TYPES) + 7 + 5 + 1 + len(HYBRIDIZATION_TYPES) + 5


def _one_hot(value: int, choices: list[int]) -> list[float]:
    return [float(value == c) for c in choices]


def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_TYPES)
        + _one_hot(atom.GetTotalDegree(), list(range(7)))
        + _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + [float(atom.GetIsAromatic())]
        + _one_hot(int(atom.GetHybridization()), [int(h) for h in HYBRIDIZATION_TYPES])
        + _one_hot(atom.GetTotalNumHs(), list(range(5)))
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    return _one_hot(int(bond.GetBondType()), [int(b) for b in BOND_TYPES])


def smiles_to_graph(smiles: str) -> Data | None:
    """Convert SMILES to a PyG graph with atom and bond features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    if x.shape[0] == 0:
        return None

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(BOND_TYPES)), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _smiles_graph_pair(smiles: str) -> tuple[str, Data | None]:
    """Module-level worker so it can be pickled for ProcessPoolExecutor."""
    return smiles, smiles_to_graph(smiles)


class SmilesGraphCache:
    """LRU cache for SMILES → PyG graphs (maxsize<=0 keeps all entries)."""

    def __init__(self, maxsize: int = 8192):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Data] = OrderedDict()

    def get(self, smiles: str) -> Data:
        graph = self._cache.get(smiles)
        if graph is not None:
            self._cache.move_to_end(smiles)
            return graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Invalid SMILES for graph construction: {smiles!r}")
        self._cache[smiles] = graph
        if self.maxsize > 0:
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return graph

    def preload(self, smiles_list: list[str], workers: int = 1) -> int:
        """Graphify unique SMILES into the cache. Returns number newly added."""
        unique = list(dict.fromkeys(smiles_list))
        missing = [s for s in unique if s not in self._cache]
        if not missing:
            return 0
        if workers <= 1:
            for smiles in missing:
                self.get(smiles)
            return len(missing)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for smiles, graph in pool.map(_smiles_graph_pair, missing, chunksize=64):
                if graph is None:
                    raise ValueError(f"Invalid SMILES for graph construction: {smiles!r}")
                self._cache[smiles] = graph
                if self.maxsize > 0:
                    while len(self._cache) > self.maxsize:
                        self._cache.popitem(last=False)
        return len(missing)
