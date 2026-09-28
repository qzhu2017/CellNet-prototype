"""Dataset splits that keep polymorphic SMILES in a single partition."""

from __future__ import annotations

import random
from collections import defaultdict


def split_by_smiles_group(
    samples,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset indices by SMILES so all polymorphs of a molecule share one split.

    Groups are assigned greedily to test, then val, then train to approximate ``val_frac``
    and ``test_frac`` by structure count.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s.smiles].append(i)

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)

    n = len(samples)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx: list[int] = []
    val_idx: list[int] = []
    train_idx: list[int] = []

    for _smiles, idxs in group_items:
        if len(test_idx) < n_test:
            test_idx.extend(idxs)
        elif len(val_idx) < n_val:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    return train_idx, val_idx, test_idx
