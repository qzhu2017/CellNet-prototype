"""Tests for conditional Selling → λ lattice flow."""

import torch

from cellnet.metrics import given_hz_conditional_lattice_flow_loss
from cellnet.models import GivenHZConditionalLatticeFlowGNN


def _make_batch(batch: int = 2):
    node_dim, edge_dim = 32, 4
    selling_dim, lambda_dim = 6, 6
    data_x = torch.randn(batch * 3, node_dim)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
    from torch_geometric.data import Batch

    data = Batch(
        x=data_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    hall = torch.tensor([0, 1])
    zprime = torch.tensor([0, 0])
    x1 = torch.randn(batch, selling_dim + lambda_dim)
    log_density = torch.randn(batch)
    return data, hall, zprime, x1, log_density


def test_conditional_model_forward_and_k_samples():
    model = GivenHZConditionalLatticeFlowGNN(
        node_dim=32,
        edge_dim=4,
        selling_dim=6,
        lambda_dim=6,
        n_halls=4,
        n_zprimes=2,
        hidden_dim=64,
        gnn_layers=2,
        flow_steps=4,
    )
    data, hall, zprime, x1, _ = _make_batch()
    out = model(data, hall, zprime, x1=x1)
    assert out["lattice_flow"].shape == (2, 12)
    assert out["velocity_s"].shape == (2, 6)
    assert out["velocity_lambda"].shape == (2, 6)

    cond = model._condition(data, hall, zprime)
    one = model.sample_lattice(cond)
    assert one.shape == (2, 12)

    k_samples = []
    for _ in range(3):
        s = model.sample_selling(cond)
        lam = model.sample_lambda(cond, s)
        k_samples.append(torch.cat([s, lam], dim=-1))
    stacked = torch.stack(k_samples, dim=1)
    assert stacked.shape == (2, 3, 12)


def test_conditional_loss_runs():
    model = GivenHZConditionalLatticeFlowGNN(
        node_dim=32,
        edge_dim=4,
        selling_dim=6,
        lambda_dim=6,
        n_halls=4,
        n_zprimes=2,
        hidden_dim=64,
        gnn_layers=2,
    )
    data, hall, zprime, x1, log_density = _make_batch()
    out = model(data, hall, zprime, x1=x1)
    mean6 = torch.zeros(6)
    std6 = torch.ones(6)
    mean3 = torch.zeros(3)
    std3 = torch.ones(3)
    loss, metrics = given_hz_conditional_lattice_flow_loss(
        out,
        x1,
        log_density,
        mean6,
        std6,
        mean3,
        std3,
        mean3,
        std3,
        w_flow=0.5,
        w_flow_lambda=0.5,
        w_s=0.5,
        w_lambda=1.0,
        w_lambda_recip=1.0,
        w_density=0.1,
        lambda_loss_mode="phys_log",
    )
    assert loss.ndim == 0
    assert "flow_s" in metrics
    assert "flow_lambda" in metrics
