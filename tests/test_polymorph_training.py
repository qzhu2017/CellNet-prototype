"""Tests for SMILES-group splits and polymorph training helpers."""

import numpy as np
import torch

from cellnet.data import DatasetStats
from cellnet.polymorph import SellingPolymorphBank
from cellnet.splits import split_by_smiles_group


class _Sample:
    def __init__(self, smiles, hall=115, zprime=1.0, idx=0):
        self.smiles = smiles
        self.hall_number = hall
        self.zprime = zprime
        self.selling_log1p = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32) + idx
        self.log_density = 0.5 + 0.01 * idx
        self.log_successive_minima = np.array([1.1, 2.1, 3.1], dtype=np.float32) + idx
        self.log_reciprocal_successive_minima = np.array([0.1, 0.2, 0.3], dtype=np.float32) + idx


def test_split_by_smiles_no_leakage():
    samples = [_Sample("A", idx=0), _Sample("A", idx=1), _Sample("B", idx=2), _Sample("C", idx=3)]
    train, val, test = split_by_smiles_group(samples, val_frac=0.25, test_frac=0.25, seed=0)
    all_idx = set(train + val + test)
    assert all_idx == {0, 1, 2, 3}

    def smiles_set(idxs):
        return {samples[i].smiles for i in idxs}

    assert smiles_set(train).isdisjoint(smiles_set(val))
    assert smiles_set(train).isdisjoint(smiles_set(test))
    assert smiles_set(val).isdisjoint(smiles_set(test))
    assert "A" in smiles_set(train) | smiles_set(val) | smiles_set(test)


def test_polymorph_bank_batch():
    stats = DatasetStats(
        selling_mean=np.zeros(6, dtype=np.float32),
        selling_std=np.ones(6, dtype=np.float32),
        log_density_mean=0.0,
        log_density_std=1.0,
        log_lambda_mean=np.zeros(3, dtype=np.float32),
        log_lambda_std=np.ones(3, dtype=np.float32),
        log_lambda_recip_mean=np.zeros(3, dtype=np.float32),
        log_lambda_recip_std=np.ones(3, dtype=np.float32),
        hall_to_idx={115: 0, 2: 1},
        idx_to_hall={0: 115, 1: 2},
        zprime_to_idx={1.0: 0},
        zprime_values=[1.0],
    )
    s0 = _Sample("POLY", idx=0)
    s1 = _Sample("POLY", idx=1)
    s2 = _Sample("UNIQ", idx=2)
    bank = SellingPolymorphBank([s0, s1, s2], stats)
    assert bank.n_groups == 1

    device = torch.device("cpu")
    lat, ld = bank.batch_lattice_tensors(
        ["POLY", "UNIQ"],
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        stats.idx_to_hall,
        stats.zprime_values,
        device,
        fallback_lattice=torch.zeros(2, 12),
        fallback_log_density=torch.zeros(2),
    )
    # POLY has two packings; UNIQ has one and is padded to the batch maximum.
    assert lat.shape == (2, 2, 12)
    assert ld.shape == (2, 2)


def test_polymorph_bank_missing_key_uses_fallback():
    stats = DatasetStats(
        selling_mean=np.zeros(6, dtype=np.float32),
        selling_std=np.ones(6, dtype=np.float32),
        log_density_mean=0.0,
        log_density_std=1.0,
        log_lambda_mean=np.zeros(3, dtype=np.float32),
        log_lambda_std=np.ones(3, dtype=np.float32),
        log_lambda_recip_mean=np.zeros(3, dtype=np.float32),
        log_lambda_recip_std=np.ones(3, dtype=np.float32),
        hall_to_idx={115: 0, 2: 1},
        idx_to_hall={0: 115, 1: 2},
        zprime_to_idx={1.0: 0},
        zprime_values=[1.0],
    )
    bank = SellingPolymorphBank([_Sample("KNOWN", idx=0)], stats)
    device = torch.device("cpu")
    fallback_lat = torch.ones(1, 12)
    fallback_ld = torch.tensor([0.42])
    lat, ld = bank.batch_lattice_tensors(
        ["MISSING"],
        torch.tensor([1]),
        torch.tensor([0]),
        stats.idx_to_hall,
        stats.zprime_values,
        device,
        fallback_lattice=fallback_lat,
        fallback_log_density=fallback_ld,
    )
    assert lat.shape == (1, 1, 12)
    assert ld.shape == (1, 1)
    assert torch.allclose(lat[0, 0], fallback_lat[0])
    assert torch.allclose(ld[0, 0], fallback_ld[0])
