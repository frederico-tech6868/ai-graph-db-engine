"""Vector utilities: cosine similarity and top-k search.

Pure Python implementations are always available. If ``numpy`` is installed
it is used transparently for a speed boost, but it is *not* required.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .exceptions import DimensionMismatchError

try:  # optional acceleration
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy nearly always present here
    _np = None  # type: ignore
    _HAS_NUMPY = False


def _validate_pair(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise DimensionMismatchError(
            f"vector dimension mismatch: {len(a)} != {len(b)}"
        )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors in the range ``[-1, 1]``.

    Returns ``0.0`` if either vector is empty or has zero magnitude.
    Normalises the vectors before computing the dot product.
    """
    if a is None or b is None:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    _validate_pair(a, b)

    if _HAS_NUMPY:
        va = _np.asarray(a, dtype=float)
        vb = _np.asarray(b, dtype=float)
        na = float(_np.linalg.norm(va))
        nb = float(_np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(va, vb) / (na * nb))

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def top_k_similar(
    query_vec: Sequence[float],
    candidates: Sequence[Tuple[str, Sequence[float]]],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the ``k`` most similar candidates to ``query_vec``.

    Args:
        query_vec: The query embedding.
        candidates: Iterable of ``(id, vector)`` tuples.
        k: Maximum number of results.

    Returns:
        A list of ``(id, score)`` sorted by descending similarity.
    """
    scored: List[Tuple[str, float]] = []
    for cid, vec in candidates:
        if vec is None:
            continue
        try:
            score = cosine_similarity(query_vec, vec)
        except DimensionMismatchError:
            # skip candidates with mismatched dimensions
            continue
        scored.append((cid, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    if k is not None and k >= 0:
        return scored[:k]
    return scored
