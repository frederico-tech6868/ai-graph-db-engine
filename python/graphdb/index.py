"""In-memory indexes kept in sync with the store.

``LabelIndex`` provides O(1) lookup of node ids by label.
``PropertyIndex`` provides O(1) lookup by ``(label, property_key, value)``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from .core import Node

# Only hashable property values participate in the property index.
_HASHABLE = (str, int, float, bool)


class LabelIndex:
    """Maps ``label -> set(node_id)``."""

    def __init__(self) -> None:
        self._by_label: Dict[str, Set[str]] = {}

    def add(self, node: Node) -> None:
        self._by_label.setdefault(node.label, set()).add(node.id)

    def remove(self, node: Node) -> None:
        ids = self._by_label.get(node.label)
        if ids is not None:
            ids.discard(node.id)
            if not ids:
                del self._by_label[node.label]

    def get(self, label: str) -> Set[str]:
        return set(self._by_label.get(label, set()))

    def labels(self) -> List[str]:
        return list(self._by_label.keys())

    def clear(self) -> None:
        self._by_label.clear()


class PropertyIndex:
    """Maps ``(label, property_key, value) -> set(node_id)``."""

    def __init__(self) -> None:
        self._idx: Dict[Tuple[str, str, object], Set[str]] = {}

    def _keys(self, node: Node) -> Iterable[Tuple[str, str, object]]:
        for key, value in node.properties.items():
            if isinstance(value, _HASHABLE):
                yield (node.label, key, value)

    def add(self, node: Node) -> None:
        for k in self._keys(node):
            self._idx.setdefault(k, set()).add(node.id)

    def remove(self, node: Node) -> None:
        for k in self._keys(node):
            ids = self._idx.get(k)
            if ids is not None:
                ids.discard(node.id)
                if not ids:
                    del self._idx[k]

    def get(self, label: str, key: str, value: object) -> Set[str]:
        return set(self._idx.get((label, key, value), set()))

    def clear(self) -> None:
        self._idx.clear()
