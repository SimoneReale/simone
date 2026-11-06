"""Label-free selection balancing predicted rank with molecular diversity."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .representations import MolecularInputs


def select_diverse(inputs: MolecularInputs, scores: np.ndarray, k: int = 50,
                   diversity_weight: float = 0.25, pool_size: int | None = None) -> np.ndarray:
    """Greedily select unique chemical entities using binary Morgan Tanimoto.

    Utility is percentile rank(score) minus diversity_weight times maximum
    similarity to already selected candidates. A weight of zero gives top-k.
    Percentile ranks make the tradeoff independent of classifier score scales.
    Score ties are resolved by stable structure identifier. If source rows
    contain a structure more than once, its highest-scored row represents it;
    exact duplicate ties retain their first row, but chemical selection is
    independent of input row ordering. No labels enter this procedure.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(inputs) or not np.isfinite(scores).all():
        raise ValueError("Selection requires one finite score per molecule")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k <= 0:
        raise ValueError("Selection k must be a positive integer")
    if not np.isfinite(diversity_weight) or not 0 <= diversity_weight <= 1:
        raise ValueError("diversity_weight must be finite and in [0, 1]")
    if pool_size is not None and (not isinstance(pool_size, (int, np.integer)) or isinstance(pool_size, bool) or pool_size < k):
        raise ValueError("Selection pool_size must be an integer at least k")
    if "ecfp" not in inputs.matrices:
        raise ValueError("Diversity selection requires binary Morgan 'ecfp' features")
    # Lexicographic sorting is stable across source order for distinct entities.
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], str(inputs.structure_ids[i]), i))
    unique, seen = [], set()
    for idx in order:
        identifier = inputs.structure_ids[idx]
        if identifier not in seen:
            unique.append(idx)
            seen.add(identifier)
    if len(unique) < k:
        raise ValueError(f"Cannot select {k} compounds: only {len(unique)} unique molecular structures")
    unique_array = np.asarray(unique, dtype=int)
    ranked = (rankdata(scores[unique_array], method="average") - 1) / max(1, len(unique_array) - 1)
    count = len(unique_array) if pool_size is None else min(pool_size, len(unique_array))
    candidates = unique_array[:count]
    rank_scores = ranked[:count]
    fingerprints = inputs.matrices["ecfp"][candidates].astype(np.float32)
    if fingerprints.nnz and np.any(fingerprints.data != 1):
        raise ValueError("Diversity selection expects binary Morgan fingerprints")
    counts = np.asarray(fingerprints.sum(axis=1)).ravel()
    maximum_similarity = np.zeros(count, dtype=float)
    available = np.ones(count, dtype=bool)
    selected = []
    stable_ids = np.asarray([str(identifier) for identifier in inputs.structure_ids[candidates]])
    for _ in range(k):
        utilities = rank_scores - diversity_weight * maximum_similarity
        utilities[~available] = -np.inf
        best_value = utilities.max()
        tied = np.flatnonzero(available & np.isclose(utilities, best_value, rtol=0, atol=1e-12))
        chosen = tied[np.argmin(stable_ids[tied])]
        selected.append(int(candidates[chosen]))
        available[chosen] = False
        intersection = (fingerprints @ fingerprints[chosen].T).toarray().ravel()
        union = counts + counts[chosen] - intersection
        similarity = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
        maximum_similarity = np.maximum(maximum_similarity, similarity)
    return np.asarray(selected, dtype=int)


def selection_metrics(y: np.ndarray, indices: np.ndarray) -> dict[str, int | float | None]:
    """Evaluate an already-fixed selection; this function does not select rows."""
    y, indices = np.asarray(y), np.asarray(indices)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Selection metrics require a vector of binary activity labels")
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Selected row indices must be a one-dimensional integer vector")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(y)):
        raise ValueError("Selected row indices must be unique and within the labeled set")
    hits = int(y[indices].sum())
    positives = int(y.sum())
    return {"selected": len(indices), "hits": hits, "precision": hits / len(indices) if len(indices) else None,
            "recall": hits / positives if positives else None}
