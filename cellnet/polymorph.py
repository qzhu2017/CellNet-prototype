"""Polymorph target banks for best-of-K training losses."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from cellnet.data import lattice_flow_target

PackingKey = tuple[str, int, float]


class SellingPolymorphBank:
    """
    Train-set polymorph targets keyed by (SMILES, Hall, Z′).

    Used to build padded [B, K, …] target tensors for best-of-K Selling flow loss.
    """

    def __init__(self, train_samples, stats):
        self.groups: dict[PackingKey, list[tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]]] = defaultdict(list)
        for s in train_samples:
            if s.selling_log1p is None:
                continue
            key = (s.smiles, int(s.hall_number), float(s.zprime))
            sell = ((s.selling_log1p - stats.selling_mean) / stats.selling_std).astype(np.float32)
            ld = float((s.log_density - stats.log_density_mean) / stats.log_density_std)
            lam = None
            lat = None
            if getattr(s, "log_successive_minima", None) is not None:
                lam = (
                    (s.log_successive_minima - stats.log_lambda_mean) / stats.log_lambda_std
                ).astype(np.float32)
            if getattr(s, "log_reciprocal_successive_minima", None) is not None:
                try:
                    lat = lattice_flow_target(s, stats).astype(np.float32)
                except Exception:
                    lat = None
            self.groups[key].append((sell, ld, lam, lat))

        self.n_groups = sum(1 for v in self.groups.values() if len(v) > 1)
        self.n_polymorph_structures = sum(len(v) for v in self.groups.values() if len(v) > 1)

        # Packed per-key lattice targets, built once so a training step is one
        # NumPy gather plus a single host→device copy (see batch_lattice_tensors).
        self._lat_dim = 12
        for entries in self.groups.values():
            found = next((lat.shape[0] for *_rest, lat in entries if lat is not None), None)
            if found is not None:
                self._lat_dim = int(found)
                break
        self._packed: dict[PackingKey, tuple[np.ndarray, np.ndarray]] = {}
        for key, entries in self.groups.items():
            lat_rows = np.zeros((len(entries), self._lat_dim), dtype=np.float32)
            ld_rows = np.zeros(len(entries), dtype=np.float32)
            for j, (sell, ld, lam, lat) in enumerate(entries):
                ld_rows[j] = ld
                if lat is not None:
                    lat_rows[j] = lat
                elif sell is not None and lam is not None:
                    lat_rows[j, : sell.shape[0]] = sell
                    lat_rows[j, sell.shape[0] : sell.shape[0] + lam.shape[0]] = lam
            self._packed[key] = (lat_rows, ld_rows)

    def batch_lattice_tensors(
        self,
        smiles_list: list[str],
        hall_idx: torch.Tensor,
        zprime_idx: torch.Tensor,
        idx_to_hall: dict[int, int],
        zprime_values: list[float],
        device: torch.device,
        fallback_lattice: torch.Tensor | None = None,
        fallback_log_density: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build padded polymorph lattice targets: (lattice [B,K,12], log_density [B,K]).

        Groups shorter than K are padded by repeating their last entry, so a
        min-over-K loss is unaffected. Samples absent from the bank (e.g. a
        validation split) take ``fallback_*`` in every K slot.

        The whole batch is assembled in NumPy and moved to ``device`` in two
        copies; the previous per-entry ``.item()`` / ``.to(device)`` calls cost
        hundreds of host↔device syncs per step on CUDA.
        """
        batch = len(smiles_list)
        hall_list = hall_idx.detach().cpu().tolist()
        zp_list = zprime_idx.detach().cpu().tolist()
        keys: list[PackingKey] = [
            (smiles_list[i], idx_to_hall[int(hall_list[i])], float(zprime_values[int(zp_list[i])]))
            for i in range(batch)
        ]

        packed = [self._packed.get(k) for k in keys]
        max_k = max([p[1].shape[0] for p in packed if p is not None] + [1])
        lat_dim = int(fallback_lattice.shape[-1]) if fallback_lattice is not None else self._lat_dim

        lat_np = np.zeros((batch, max_k, lat_dim), dtype=np.float32)
        ld_np = np.zeros((batch, max_k), dtype=np.float32)
        missing: list[int] = []
        for i, entry in enumerate(packed):
            if entry is None:
                missing.append(i)
                continue
            lat_rows, ld_rows = entry
            n = ld_rows.shape[0]
            lat_np[i, :n, : lat_rows.shape[1]] = lat_rows
            ld_np[i, :n] = ld_rows
            if n < max_k:
                lat_np[i, n:, : lat_rows.shape[1]] = lat_rows[-1]
                ld_np[i, n:] = ld_rows[-1]

        lat_out = torch.from_numpy(lat_np).to(device)
        ld_out = torch.from_numpy(ld_np).to(device)
        if missing:
            if fallback_lattice is None or fallback_log_density is None:
                raise KeyError(f"No polymorph bank entry for {keys[missing[0]]}")
            idx = torch.as_tensor(missing, device=device, dtype=torch.long)
            lat_out[idx] = fallback_lattice[idx].to(lat_out.dtype).unsqueeze(1)
            ld_out[idx] = fallback_log_density[idx].to(ld_out.dtype).unsqueeze(1)
        return lat_out, ld_out
