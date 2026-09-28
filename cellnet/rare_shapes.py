"""Label-density weights for rare, scale-free lattice shapes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


ShapeGroup = tuple[int, float]


def lattice_anisotropy_score(sample) -> float | None:
    """Return log(λ3/λ1), a scale-free direct-lattice anisotropy score."""
    log_lam = getattr(sample, "log_successive_minima", None)
    if log_lam is None:
        return None
    values = np.asarray(log_lam, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return None
    return float(values[-1] - values[0])


@dataclass(frozen=True)
class RareShapeWeighter:
    """Empirical conditional-CDF weighting within each (Hall, Z′) group."""

    group_scores: dict[ShapeGroup, np.ndarray]
    pooled_scores: np.ndarray
    max_weight: float = 3.0
    power: float = 2.0
    min_group_size: int = 32

    @classmethod
    def fit(
        cls,
        samples,
        *,
        max_weight: float = 3.0,
        power: float = 2.0,
        min_group_size: int = 32,
    ) -> "RareShapeWeighter":
        grouped: dict[ShapeGroup, list[float]] = defaultdict(list)
        pooled: list[float] = []
        for sample in samples:
            score = lattice_anisotropy_score(sample)
            if score is None:
                continue
            grouped[(int(sample.hall_number), float(sample.zprime))].append(score)
            pooled.append(score)
        arrays = {key: np.sort(values) for key, values in grouped.items()}
        return cls(
            group_scores=arrays,
            pooled_scores=np.sort(np.asarray(pooled, dtype=np.float64)),
            max_weight=max(float(max_weight), 1.0),
            power=max(float(power), 0.0),
            min_group_size=max(int(min_group_size), 1),
        )

    def quantile(self, sample) -> float:
        score = lattice_anisotropy_score(sample)
        if score is None:
            return 0.5
        group = self.group_scores.get((int(sample.hall_number), float(sample.zprime)))
        reference = group if group is not None and len(group) >= self.min_group_size else self.pooled_scores
        if len(reference) == 0:
            return 0.5
        left = int(np.searchsorted(reference, score, side="left"))
        right = int(np.searchsorted(reference, score, side="right"))
        return float((0.5 * (left + right) + 0.5) / (len(reference) + 1.0))

    def weight(self, sample) -> float:
        if self.max_weight <= 1.0:
            return 1.0
        quantile = self.quantile(sample)
        # Both unusually compact and unusually elongated shapes receive
        # weight; no absolute Å threshold is encoded.
        tail_strength = min(1.0, 2.0 * abs(quantile - 0.5))
        return float(1.0 + (self.max_weight - 1.0) * tail_strength**self.power)

    def shape_bin(self, sample, n_bins: int) -> int:
        if n_bins <= 1:
            return 0
        return min(int(self.quantile(sample) * n_bins), n_bins - 1)

    def assign(self, samples, *, n_bins: int = 0) -> np.ndarray:
        weights = np.asarray([self.weight(sample) for sample in samples], dtype=np.float64)
        if len(weights):
            weights /= max(float(weights.mean()), 1e-12)
        for sample, weight in zip(samples, weights):
            sample.lattice_weight = float(weight)
            sample.lattice_shape_bin = self.shape_bin(sample, n_bins)
        return weights
