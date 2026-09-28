"""Tests for the conditional lattice-flow training loss."""

import torch

from cellnet.metrics import given_hz_conditional_lattice_flow_loss

SELLING_DIM = 6


def _outputs(pred_lattice: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = pred_lattice.shape[0]
    return {
        "velocity_s": torch.zeros(batch, 6),
        "velocity_target_s": torch.zeros(batch, 6),
        "velocity_lambda": torch.zeros(batch, 6),
        "velocity_target_lambda": torch.zeros(batch, 6),
        "lattice_flow": pred_lattice,
        "log_density": torch.zeros(batch),
    }


def _loss(outputs, target, lam_mean=None, lam_std=None, **kwargs):
    lam_mean = torch.zeros(3) if lam_mean is None else lam_mean
    lam_std = torch.ones(3) if lam_std is None else lam_std
    weights = dict(w_flow=0.0, w_flow_lambda=0.0, w_s=0.0, w_density=0.0, w_lambda=0.0, w_lambda_recip=0.0)
    weights.update(kwargs)
    return given_hz_conditional_lattice_flow_loss(
        outputs,
        target,
        torch.zeros(target.shape[0]),
        torch.zeros(6),
        torch.ones(6),
        lam_mean,
        lam_std,
        lam_mean,
        lam_std,
        **weights,
    )


def _lambda_case():
    mean = torch.tensor([2.0, 2.5, 3.0])
    std = torch.tensor([0.25, 0.25, 0.25])
    pred_norm = torch.tensor([[0.0, 0.0, 0.0]])
    ref_norm = torch.tensor([[0.4, 0.0, -0.4]])
    pred = torch.zeros(1, 12)
    target = torch.zeros(1, 12)
    pred[:, SELLING_DIM : SELLING_DIM + 3] = pred_norm
    target[:, SELLING_DIM : SELLING_DIM + 3] = ref_norm
    return pred, target, pred_norm, ref_norm, mean, std


def test_phys_log_loss_matches_physical_mse():
    pred, target, pred_norm, ref_norm, mean, std = _lambda_case()
    _, metrics = _loss(_outputs(pred), target, mean, std, w_lambda=1.0, lambda_loss_mode="phys_log")
    expected = torch.mean(((pred_norm * std + mean) - (ref_norm * std + mean)) ** 2)
    assert abs(metrics["lambda"] - float(expected)) < 1e-5


def test_norm_mode_differs_from_phys_log():
    pred, target, _, _, mean, std = _lambda_case()
    _, m_norm = _loss(_outputs(pred), target, mean, std, w_lambda=1.0, lambda_loss_mode="norm")
    _, m_phys = _loss(_outputs(pred), target, mean, std, w_lambda=1.0, lambda_loss_mode="phys_log")
    assert m_norm["lambda"] != m_phys["lambda"]


def test_velocity_losses_are_weighted_per_flow():
    outputs = _outputs(torch.zeros(1, 12))
    outputs["velocity_s"] = torch.ones(1, 6)          # Selling-flow velocity error = 1
    outputs["velocity_lambda"] = 2.0 * torch.ones(1, 6)  # λ-flow velocity error = 4
    target = torch.zeros(1, 12)
    loss, metrics = _loss(outputs, target, w_flow=0.5, w_flow_lambda=0.25)
    assert abs(metrics["flow_s"] - 1.0) < 1e-6
    assert abs(metrics["flow_lambda"] - 4.0) < 1e-6
    assert torch.allclose(loss, torch.tensor(0.5 * 1.0 + 0.25 * 4.0))


def test_axis_permutation_loss_sums_equivalent_class_probability():
    outputs = _outputs(torch.zeros(1, 12))
    outputs["axis_permutation_logits"] = torch.log(torch.tensor([[0.2, 0.1, 0.3, 0.1, 0.2, 0.1]]))
    mask = torch.tensor([[True, False, True, False, False, False]])
    loss, metrics = _loss(
        outputs, torch.zeros(1, 12), target_axis_permutation_mask=mask, w_axis_permutation=1.0
    )
    expected = -torch.log(torch.tensor(0.5))
    assert torch.allclose(loss, expected)
    assert abs(metrics["axis_permutation"] - float(expected)) < 1e-6


def test_axis_permutation_loss_ignores_all_allowed_mask():
    outputs = _outputs(torch.zeros(1, 12))
    outputs["axis_permutation_logits"] = torch.randn(1, 6)
    loss, _ = _loss(
        outputs,
        torch.zeros(1, 12),
        target_axis_permutation_mask=torch.ones(1, 6, dtype=torch.bool),
        w_axis_permutation=1.0,
    )
    assert abs(float(loss)) < 1e-6


def test_polymorph_loss_takes_minimum_over_known_packings():
    outputs = _outputs(torch.zeros(1, 12))
    target = torch.full((1, 12), 0.5)
    weights = dict(w_s=1.0, w_lambda=1.0, w_lambda_recip=1.0, w_density=1.0)
    # One reference matches the prediction exactly, the other does not.
    poly = torch.stack([torch.zeros(1, 12), torch.full((1, 12), 4.0)], dim=1)
    poly_ld = torch.tensor([[0.0, 5.0]])
    loss_match, _ = _loss(outputs, target, polymorph_lattice=poly, polymorph_log_density=poly_ld, **weights)
    loss_far, _ = _loss(
        outputs,
        target,
        polymorph_lattice=torch.full((1, 1, 12), 4.0),
        polymorph_log_density=torch.tensor([[5.0]]),
        **weights,
    )
    assert float(loss_match) < 1e-6
    assert float(loss_far) > 1.0
