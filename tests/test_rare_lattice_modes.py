from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from cellnet.gnn import MolecularGNN
from cellnet.metrics import _weighted_sample_mean
from cellnet.models import GivenHZLatticeFlowGNN
from cellnet.sequential import sample_k_given_hz_lattice_flow
from cellnet.lattice_conf_pipeline import (
    LatticeQRSRecord,
    UniqueCellRecord,
    _lattice_qrs_candidate_indices,
    select_diverse_unique_cells,
)
from cellnet.rare_shapes import RareShapeWeighter
from cellnet.retrieval import ShapeBinRetrievalIndex


def _sample(score: float, hall: int = 6):
    return SimpleNamespace(
        hall_number=hall,
        zprime=1.0,
        log_successive_minima=np.array([0.0, 0.2, score], dtype=np.float64),
        lattice_weight=1.0,
    )


def test_rare_shape_weights_are_conditional_and_tail_aware():
    train = [_sample(score) for score in np.linspace(0.3, 1.3, 101)]
    weighter = RareShapeWeighter.fit(train, max_weight=3.0, power=2.0)
    middle = weighter.weight(_sample(0.8))
    low_tail = weighter.weight(_sample(0.25))
    high_tail = weighter.weight(_sample(1.4))
    assert low_tail > middle
    assert high_tail > middle
    assigned = weighter.assign(train, n_bins=8)
    assert np.isclose(assigned.mean(), 1.0)
    assert train[0].lattice_shape_bin == 0
    assert train[-1].lattice_shape_bin == 7


def test_weighted_flow_mean_emphasizes_rare_sample():
    losses = torch.tensor([1.0, 9.0])
    unweighted = _weighted_sample_mean(losses, None)
    weighted = _weighted_sample_mean(losses, torch.tensor([1.0, 3.0]))
    assert weighted > unweighted
    assert torch.isclose(weighted, torch.tensor(7.0))


def test_shape_retrieval_exact_match_returns_observed_bin():
    samples = [
        SimpleNamespace(
            smiles="CCO",
            csd_code="A",
            hall_number=6,
            zprime=1.0,
            lattice_shape_bin=7,
        ),
        SimpleNamespace(
            smiles="CCN",
            csd_code="B",
            hall_number=6,
            zprime=1.0,
            lattice_shape_bin=2,
        ),
    ]
    index = ShapeBinRetrievalIndex.build(samples, n_bins=8, k=2)
    probs, confidence, neighbors = index.query("CCO", 6, 1.0)
    assert confidence == 1.0
    assert int(probs.argmax()) == 7
    assert probs[7] == 1.0
    assert neighbors[0][0] == "A"


def test_extreme_lattice_qrs_draws_are_skipped():
    lam = np.array(
        [
            [5.0, 10.0, 25.0],
            [3.98, 15.95, 36.67],
            [6.0, 8.0, 24.0],
        ]
    )
    selected, skipped = _lattice_qrs_candidate_indices(np.log(lam), 8.0)
    assert selected == [0, 2]
    assert skipped[0][0] == 1
    assert skipped[0][1] > 9.0

    selected, skipped = _lattice_qrs_candidate_indices(np.log(lam), 0.0)
    assert selected == [0, 1, 2]
    assert skipped == []


def test_multiscale_pooling_retains_graph_size_signal():
    node_dim, edge_dim = 4, 2
    graph_small = Data(
        x=torch.ones(2, node_dim),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, edge_dim)),
    )
    graph_large = Data(
        x=torch.ones(5, node_dim),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, edge_dim)),
    )
    batch = Batch.from_data_list([graph_small, graph_large])

    torch.manual_seed(7)
    mean_model = MolecularGNN(
        node_dim, edge_dim, hidden_dim=8, n_layers=1, dropout=0.0, out_dim=8, pooling="mean"
    ).eval()
    torch.manual_seed(7)
    multi_model = MolecularGNN(
        node_dim,
        edge_dim,
        hidden_dim=8,
        n_layers=1,
        dropout=0.0,
        out_dim=8,
        pooling="multiscale",
    ).eval()

    mean_out = mean_model(batch)
    multi_out = multi_model(batch)
    assert torch.allclose(mean_out[0], mean_out[1], atol=1e-6)
    assert not torch.allclose(multi_out[0], multi_out[1], atol=1e-6)


def test_joint_flow_uses_shape_latent_for_training_and_sampling():
    node_dim, edge_dim = 4, 2
    graphs = [
        Data(
            x=torch.randn(3, node_dim),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, edge_dim)),
        )
        for _ in range(2)
    ]
    batch = Batch.from_data_list(graphs)
    model = GivenHZLatticeFlowGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        lattice_flow_dim=12,
        n_halls=2,
        n_zprimes=1,
        hidden_dim=32,
        gnn_layers=1,
        flow_steps=2,
        n_shape_bins=8,
        shape_emb_dim=4,
        predict_axis_permutation=True,
    )
    hall = torch.tensor([0, 1])
    zprime = torch.tensor([0, 0])
    target = torch.randn(2, 12)
    shape_bin = torch.tensor([0, 7])
    out = model(batch, hall, zprime, x1=target, shape_bin=shape_bin)
    assert out["shape_logits"].shape == (2, 8)
    assert torch.equal(out["shape_bin"], shape_bin)
    assert out["axis_permutation_logits"].shape == (2, 6)
    samples, density = sample_k_given_hz_lattice_flow(
        model, batch, hall, zprime, k=8
    )
    assert samples.shape == (2, 8, 12)
    assert density.shape == (2,)
    assert model._last_axis_permutation_logits.shape == (2, 8, 6)
    assert set(model._last_sampled_shape_bins[0].tolist()) == set(range(8))
    prior = torch.nn.functional.one_hot(torch.tensor(7), num_classes=8).float()
    sample_k_given_hz_lattice_flow(
        model,
        batch,
        hall,
        zprime,
        k=8,
        shape_prior=prior,
        shape_prior_weight=1.0,
    )
    assert torch.allclose(model._last_shape_probabilities[0], prior)


def _record(idx: int, cellpar: list[float], loss: float, support: int = 1):
    qrs = [
        LatticeQRSRecord(
            flow_idx=idx + j,
            cellpar=np.asarray(cellpar, dtype=np.float64),
            qrs_loss=loss + 0.01 * j,
            density=1.2,
            lambda_mse=0.0,
            lambda_recip_mse=0.0,
        )
        for j in range(support)
    ]
    unique = UniqueCellRecord(
        dedup_idx=idx,
        source_flow_indices=[record.flow_idx for record in qrs],
        cellpar=np.asarray(cellpar, dtype=np.float64),
    )
    return unique, qrs


def test_diverse_selection_keeps_quality_and_rare_shape():
    specs = [
        ([8.0, 9.0, 10.0, 90.0, 90.0, 90.0], 0.10),
        ([8.1, 9.1, 10.1, 90.0, 90.0, 90.0], 0.11),
        ([7.9, 8.9, 10.2, 90.0, 90.0, 90.0], 0.12),
        ([5.8, 7.4, 24.7, 90.0, 92.0, 90.0], 0.20),
    ]
    unique_cells = []
    records = []
    for idx, (cellpar, loss) in enumerate(specs):
        unique, recs = _record(idx, cellpar, loss)
        unique_cells.append(unique)
        records.extend(recs)

    selected = select_diverse_unique_cells(unique_cells, records, max_cells=2)
    selected_ids = {cell.dedup_idx for cell in selected}
    assert 0 in selected_ids  # best-QRS seed
    assert 3 in selected_ids  # distinct elongated shape
