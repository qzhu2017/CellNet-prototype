"""Model loading and K-sampling helpers used by validate_flow_k100.py.

(The original module was missing from the repository; these wrap the same inference code
the CSP pipeline uses: cellnet.inference.load_flow_qrs_model and
cellnet.sequential.sample_k_given_hz_conditional_lattice_flow.)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellnet.inference import load_flow_qrs_model  # noqa: E402
from cellnet.sequential import sample_k_given_hz_conditional_lattice_flow  # noqa: E402


def load_model(checkpoint, device):
    """Return (model, stats, kind, model_name)."""
    model, stats, model_args, kind = load_flow_qrs_model(checkpoint, device)
    model.eval()
    name = (model_args or {}).get("model", "given_hz_conditional_lattice_flow_gnn")
    return model, stats, kind, name


def sample_k(model, batch, hall, zprime, k, kind):
    """Return (samples_norm [B, k, D], pred_log_density_norm [B], None)."""
    if kind != "lattice_flow":
        raise ValueError(f"unsupported checkpoint kind {kind!r}")
    import torch

    with torch.no_grad():
        samples_norm, pred_ld = sample_k_given_hz_conditional_lattice_flow(model, batch, hall, zprime, k=k)
    return samples_norm, pred_ld.reshape(-1), None
