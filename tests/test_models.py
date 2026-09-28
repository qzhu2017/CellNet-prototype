"""Tests for the conditional lattice-flow model and the shipped checkpoint."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from cellnet.gnn import MolecularGNN
from cellnet.lattice_conf_pipeline import sample_flow_batch
from cellnet.metrics import given_hz_conditional_lattice_flow_loss
from cellnet.models import GivenHZConditionalLatticeFlowGNN

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "cellnet_flow" / "best.pt"


def _make_batch(batch: int = 2):
    node_dim, edge_dim = 32, 4
    selling_dim, lambda_dim = 6, 6
    data_x = torch.randn(batch * 3, node_dim)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
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


def test_multiscale_pooling_retains_graph_size_signal():
    node_dim, edge_dim = 4, 2

    def graph(n_atoms):
        return Data(
            x=torch.ones(n_atoms, node_dim),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, edge_dim)),
        )

    batch = Batch.from_data_list([graph(2), graph(5)])
    outs = {}
    for pooling in ("mean", "multiscale"):
        torch.manual_seed(7)
        gnn = MolecularGNN(
            node_dim, edge_dim, hidden_dim=8, n_layers=1, dropout=0.0, out_dim=8, pooling=pooling
        ).eval()
        outs[pooling] = gnn(batch)
    assert torch.allclose(outs["mean"][0], outs["mean"][1], atol=1e-6)
    assert not torch.allclose(outs["multiscale"][0], outs["multiscale"][1], atol=1e-6)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="shipped checkpoint not present")
def test_shipped_checkpoint_samples_reproducible_physical_cells():
    kwargs = dict(smiles="NC(N)=O", hall_number=6, zprime=1.0, k=4, device=torch.device("cpu"), seed=3)
    first = sample_flow_batch(CHECKPOINT, **kwargs)
    second = sample_flow_batch(CHECKPOINT, **kwargs)

    assert first.all_selling.shape == (4, 6)
    assert first.all_log_lambda.shape == (4, 3)
    assert first.all_log_lambda_recip.shape == (4, 3)
    np.testing.assert_array_equal(first.all_selling, second.all_selling)
    np.testing.assert_array_equal(first.all_log_lambda, second.all_log_lambda)

    # The flow does not enforce λ₁ ≤ λ₂ ≤ λ₃; only the physical range is checked here.
    lam = np.exp(first.all_log_lambda)
    assert np.all((lam > 2.0) & (lam < 30.0)), lam
    assert 1.0 < first.target_rho < 2.0, first.target_rho  # urea is 1.32 g/cm^3
