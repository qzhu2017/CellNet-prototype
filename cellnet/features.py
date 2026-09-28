"""SMILES featurization for molecular crystal prediction."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse SMILES; return None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Morgan (ECFP) fingerprint as float32 array."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def molecular_descriptors(smiles: str) -> np.ndarray | None:
    """RDKit molecular descriptors concatenated as a feature vector."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.RingCount(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
    ]
    return np.array(desc, dtype=np.float32)


def smiles_features(smiles: str, fp_bits: int = 2048, fp_radius: int = 2) -> np.ndarray | None:
    """Combined Morgan fingerprint + molecular descriptors."""
    fp = morgan_fingerprint(smiles, radius=fp_radius, n_bits=fp_bits)
    desc = molecular_descriptors(smiles)
    if fp is None or desc is None:
        return None
    return np.concatenate([fp, desc])


def normalize_descriptors(desc_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (normalized, mean, std) for descriptor columns."""
    mean = desc_matrix.mean(axis=0)
    std = desc_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    return (desc_matrix - mean) / std, mean, std
