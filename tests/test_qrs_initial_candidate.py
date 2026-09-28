"""Tests for retaining exact lattice-QRS initialization candidates."""

import numpy as np

from cellnet.lattice_invariants import selling_parameters
from cellnet.packing import molecular_weight, packing_density
from cellnet.qrs import QRSConfig, qrs_search_cellpar
from cellnet.sequential import (
    compute_log_reciprocal_successive_minima,
    compute_log_successive_minima,
)


def test_qrs_can_retain_exact_hexagonal_lambda_seed():
    hall = 452
    zprime = 1.0
    smiles = "CC(N)=O"
    seed = np.array([11.3, 11.3, 12.73, 90.0, 90.0, 120.0])
    rho = packing_density(seed, molecular_weight(smiles), zprime, hall)
    cfg = QRSConfig(
        n_stages=0,
        w_lambda=2.0,
        w_lambda_recip=0.5,
        w_selling=0.0,
        w_density=2.0,
        evaluate_initial_candidate=True,
    )

    result = qrs_search_cellpar(
        smiles=smiles,
        hall_number=hall,
        zprime=zprime,
        target_rho=rho,
        config=cfg,
        target_log_lambdas=compute_log_successive_minima(seed),
        target_log_lambdas_recip=compute_log_reciprocal_successive_minima(seed),
        target_selling=None,
        initial_cellpar=seed,
    )

    assert result.stage == -1
    assert result.loss < 1e-12
    assert np.allclose(result.cellpar, seed)
    assert np.isfinite(selling_parameters(result.cellpar)).all()
