"""Tests for lattice-flow λ loss in physical log space."""

import torch

from cellnet.metrics import given_hz_lattice_flow_loss


def _make_outputs(pred_lam_norm, ref_lam_norm, mean, std):
    selling_dim = 6
    pred = torch.zeros(1, 12)
    target = torch.zeros(1, 12)
    pred[:, selling_dim : selling_dim + 3] = pred_lam_norm
    target[:, selling_dim : selling_dim + 3] = ref_lam_norm
    outputs = {
        "velocity": torch.zeros(1, 12),
        "velocity_target": torch.zeros(1, 12),
        "lattice_flow": pred,
        "log_density": torch.zeros(1),
    }
    return outputs, target, mean, std


def test_phys_log_loss_matches_physical_mse():
    mean = torch.tensor([2.0, 2.5, 3.0])
    std = torch.tensor([0.25, 0.25, 0.25])
    pred_norm = torch.tensor([[0.0, 0.0, 0.0]])
    ref_norm = torch.tensor([[0.4, 0.0, -0.4]])
    outputs, target, mean, std = _make_outputs(pred_norm, ref_norm, mean, std)

    _, metrics_phys = given_hz_lattice_flow_loss(
        outputs,
        target,
        torch.zeros(1),
        torch.zeros(6),
        torch.ones(6),
        mean,
        std,
        mean,
        std,
        w_flow=0.0,
        w_s=0.0,
        w_density=0.0,
        w_lambda=1.0,
        w_lambda_recip=0.0,
        lambda_loss_mode="phys_log",
    )
    pred_log = pred_norm * std + mean
    ref_log = ref_norm * std + mean
    expected = float(torch.mean((pred_log - ref_log) ** 2).item())
    assert abs(metrics_phys["lambda"] - expected) < 1e-5


def test_norm_mode_differs_from_phys_log():
    mean = torch.tensor([2.0, 2.5, 3.0])
    std = torch.tensor([0.25, 0.25, 0.25])
    pred_norm = torch.tensor([[0.0, 0.0, 0.0]])
    ref_norm = torch.tensor([[0.4, 0.0, -0.4]])
    outputs, target, mean, std = _make_outputs(pred_norm, ref_norm, mean, std)
    common = dict(
        outputs=outputs,
        target_lattice=target,
        target_log_density=torch.zeros(1),
        selling_mean=torch.zeros(6),
        selling_std=torch.ones(6),
        log_lambda_mean=mean,
        log_lambda_std=std,
        log_lambda_recip_mean=mean,
        log_lambda_recip_std=std,
        w_flow=0.0,
        w_s=0.0,
        w_density=0.0,
        w_lambda=1.0,
        w_lambda_recip=0.0,
    )
    _, m_norm = given_hz_lattice_flow_loss(**common, lambda_loss_mode="norm")
    _, m_phys = given_hz_lattice_flow_loss(**common, lambda_loss_mode="phys_log")
    assert m_norm["lambda"] != m_phys["lambda"]


def test_axis_permutation_loss_sums_equivalent_class_probability():
    mean = torch.zeros(3)
    std = torch.ones(3)
    outputs, target, mean, std = _make_outputs(
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        mean,
        std,
    )
    outputs["axis_permutation_logits"] = torch.log(
        torch.tensor([[0.2, 0.1, 0.3, 0.1, 0.2, 0.1]])
    )
    target_mask = torch.tensor([[True, False, True, False, False, False]])

    loss, metrics = given_hz_lattice_flow_loss(
        outputs,
        target,
        torch.zeros(1),
        torch.zeros(6),
        torch.ones(6),
        mean,
        std,
        mean,
        std,
        w_flow=0.0,
        w_s=0.0,
        w_density=0.0,
        w_lambda=0.0,
        w_lambda_recip=0.0,
        target_axis_permutation_mask=target_mask,
        w_axis_permutation=1.0,
    )

    expected = -torch.log(torch.tensor(0.5))
    assert torch.allclose(loss, expected)
    assert abs(metrics["axis_permutation"] - float(expected)) < 1e-6


def test_axis_permutation_loss_ignores_all_allowed_mask():
    mean = torch.zeros(3)
    std = torch.ones(3)
    outputs, target, mean, std = _make_outputs(
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        mean,
        std,
    )
    outputs["axis_permutation_logits"] = torch.randn(1, 6)

    loss, _ = given_hz_lattice_flow_loss(
        outputs,
        target,
        torch.zeros(1),
        torch.zeros(6),
        torch.ones(6),
        mean,
        std,
        mean,
        std,
        w_flow=0.0,
        w_s=0.0,
        w_density=0.0,
        w_lambda=0.0,
        w_lambda_recip=0.0,
        target_axis_permutation_mask=torch.ones(1, 6, dtype=torch.bool),
        w_axis_permutation=1.0,
    )

    assert abs(float(loss)) < 1e-6
