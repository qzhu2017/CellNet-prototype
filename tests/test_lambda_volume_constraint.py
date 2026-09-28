"""Tests for λ₁λ₂λ₃ vs density-implied volume constraint."""

import numpy as np

from cellnet.hybrid_qrs import ConflictResolutionConfig, apply_lambda_volume_constraint
from cellnet.lattice_invariants import (
    LAMBDA_PRODUCT_UPPER_FRAC,
    constrain_log_lambda_product,
    lambda_product_from_log,
)
from cellnet.packing import cell_volume, packing_density, volume_from_density


def test_volume_from_density_roundtrip():
    cp = np.array([8.0, 12.0, 36.0, 90.0, 90.0, 90.0])
    rho = packing_density(cp, 78.11, 1.0, 115)  # benzene, MW 78.11
    v = volume_from_density(rho, 78.11, 1.0, 115)
    assert abs(v - cell_volume(cp)) / cell_volume(cp) < 1e-6


def test_constrain_log_lambda_product_scales_up():
    target_v = 100.0
    log_l = np.log([2.0, 3.0, 4.0])  # product 24 < V
    adj, clipped = constrain_log_lambda_product(log_l, target_v)
    assert clipped
    assert abs(lambda_product_from_log(adj) - target_v) < 1e-6


def test_constrain_log_lambda_product_scales_down():
    target_v = 100.0
    log_l = np.log([5.0, 6.0, 7.0])  # product 210 > 1.91 V
    adj, clipped = constrain_log_lambda_product(
        log_l, target_v, upper_frac=LAMBDA_PRODUCT_UPPER_FRAC
    )
    assert clipped
    assert abs(lambda_product_from_log(adj) - LAMBDA_PRODUCT_UPPER_FRAC * target_v) < 1e-5


def test_constrain_log_lambda_product_noop_in_band():
    target_v = 100.0
    log_l = np.log([4.5, 5.0, 5.0])  # product 112.5 ∈ [100, 191]
    adj, clipped = constrain_log_lambda_product(log_l, target_v)
    assert not clipped
    np.testing.assert_allclose(adj, log_l)


def test_apply_lambda_volume_constraint_disabled():
    cfg = ConflictResolutionConfig(lambda_volume_constraint=False)
    log_l = np.log([[2.0, 3.0, 4.0]])
    adj, _, clipped = apply_lambda_volume_constraint(log_l, 100.0, cfg)
    np.testing.assert_allclose(adj[0], log_l[0])
    assert clipped == [False]
