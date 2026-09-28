"""Unit tests for hybrid QRS conflict resolution helpers."""

import numpy as np

from cellnet.hybrid_qrs import (
    ConflictResolutionConfig,
    adaptive_hybrid_weights,
    pair_conflict_scores,
    selling_flow_uncertainty,
)


def test_selling_flow_uncertainty():
    samples = np.array([[0.0] * 6, [1.0] * 6, [2.0] * 6])
    global_spread, per_sample = selling_flow_uncertainty(samples)
    assert global_spread > 0
    assert per_sample.shape == (3,)
    assert per_sample[1] < per_sample[0]
    assert per_sample[1] < per_sample[2]


def test_adaptive_weights_downscale_lambda():
    cfg = ConflictResolutionConfig(enabled=True)
    _, w_lambda = adaptive_hybrid_weights(2.0, 2.0, 1.0, 1.0, 0.5, cfg)
    assert w_lambda < 2.0
    assert w_lambda >= 2.0 * cfg.w_lambda_min_frac


def test_adaptive_weights_disabled():
    cfg = ConflictResolutionConfig(enabled=False)
    w_s, w_l = adaptive_hybrid_weights(2.0, 2.0, 1.0, 1.0, 0.5, cfg)
    assert w_s == 2.0
    assert w_l == 2.0


def test_pair_conflict_scores():
    selling = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    lam = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    scores = pair_conflict_scores(selling, lam)
    assert scores.shape == (2,)
    assert scores[0] > scores[1]
