"""Inference helpers for the auxiliary space-group + density model (cellnet/aux_model.py)."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from cellnet.aux_model import SpaceGroupDensityNet, sample_mixture


@lru_cache(maxsize=2)
def load_aux(checkpoint: str):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    v = ck["vocab"]
    classes = [(int(h), float(z)) for h, z in v["classes"]]
    halls = [int(h) for h in v["halls"]]
    zprimes = [float(z) for z in v["zprimes"]]
    model = SpaceGroupDensityNet(
        node_dim=ck["node_dim"], edge_dim=ck["edge_dim"], n_classes=len(classes),
        n_halls=len(halls), n_zprimes=len(zprimes), n_mix=ck.get("n_mix", 3),
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return model, classes, halls, zprimes, float(ck["ld_mean"]), float(ck["ld_std"])


def _graph_batch(smiles: str):
    from torch_geometric.data import Batch

    from cellnet.graph import smiles_to_graph

    return Batch.from_data_list([smiles_to_graph(smiles)])


def predict_space_groups(checkpoint: str | Path, smiles: str) -> list[tuple[int, float, float]]:
    """All (Hall, Z′, probability), most probable first."""
    model, classes, *_ = load_aux(str(checkpoint))
    with torch.no_grad():
        p = torch.softmax(model.sg_logits(model.encode(_graph_batch(smiles))), -1)[0].numpy()
    order = np.argsort(-p)
    return [(classes[i][0], classes[i][1], float(p[i])) for i in order]


def sample_densities(
    checkpoint: str | Path, smiles: str, hall_number: int, zprime: float, k: int,
    seed: int | None = None,
) -> np.ndarray:
    """k density draws (g/cm³) from p(ρ | molecule, Hall, Z′)."""
    model, classes, halls, zprimes, ld_mean, ld_std = load_aux(str(checkpoint))
    # Unseen Hall/Z′ fall back to the nearest known value (the flow model does the same for Z′).
    h = halls.index(hall_number) if hall_number in halls else 0
    z = min(range(len(zprimes)), key=lambda i: abs(zprimes[i] - float(zprime)))
    gen = torch.Generator().manual_seed(int(seed) if seed is not None else 0)
    with torch.no_grad():
        hv = model.encode(_graph_batch(smiles))
        logit, mu, ls = model.density_params(hv, torch.tensor([h]), torch.tensor([z]))
        draws = sample_mixture(logit, mu, ls, k, generator=gen)[0].numpy()
    return np.exp(draws * ld_std + ld_mean)
