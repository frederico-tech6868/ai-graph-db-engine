"""Core data model: Node, Edge and property typing.

Both :class:`Node` and :class:`Edge` are dataclasses that automatically
generate a UUID (string form) when no id is supplied.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .exceptions import InvalidPropertyError

# The set of value types allowed inside a property dict.
PropertyValue = Union[str, int, float, bool, list]
Properties = Dict[str, PropertyValue]

# Allowed primitive types for property values.
_ALLOWED_TYPES = (str, int, float, bool, list)


def new_id() -> str:
    """Return a fresh UUID4 string."""
    return str(uuid.uuid4())


def validate_properties(properties: Optional[Properties]) -> Properties:
    """Validate & normalise a property dict.

    Property values must be one of ``str``, ``int``, ``float``, ``bool`` or
    ``list``. ``list`` values must themselves contain only primitive types.
    Returns a new dict (never ``None``).
    """
    if properties is None:
        return {}
    if not isinstance(properties, dict):
        raise InvalidPropertyError(
            f"properties must be a dict, got {type(properties).__name__}"
        )
    validated: Properties = {}
    for key, value in properties.items():
        if not isinstance(key, str):
            raise InvalidPropertyError(f"property key must be str, got {key!r}")
        # NOTE: bool is a subclass of int, both are allowed so ordering is fine.
        if not isinstance(value, _ALLOWED_TYPES):
            raise InvalidPropertyError(
                f"property {key!r} has unsupported type {type(value).__name__}; "
                "allowed: str, int, float, bool, list"
            )
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, (str, int, float, bool)):
                    raise InvalidPropertyError(
                        f"list property {key!r} contains unsupported item "
                        f"type {type(item).__name__}"
                    )
        validated[key] = value
    return validated


@dataclass
class Node:
    """A graph node.

    Attributes:
        id: Unique identifier (UUID string). Auto-generated if omitted.
        label: The text label / type of the node (e.g. ``"User"``).
        properties: Typed property dict.
        embedding: Optional embedding vector (list of floats) for similarity.
    """

    label: str = ""
    properties: Properties = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    id: str = field(default_factory=new_id)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_id()
        self.properties = validate_properties(self.properties)
        if self.embedding is not None:
            self.embedding = [float(x) for x in self.embedding]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "properties": dict(self.properties),
            "embedding": list(self.embedding) if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(
            id=data.get("id") or new_id(),
            label=data.get("label", ""),
            properties=data.get("properties") or {},
            embedding=data.get("embedding"),
        )


@dataclass
class Edge:
    """A directed graph edge.

    Attributes:
        id: Unique identifier (UUID string). Auto-generated if omitted.
        src_id: Source node id.
        dst_id: Destination node id.
        label: The text label / type of the edge (e.g. ``"FOLLOWS"``).
        properties: Typed property dict.
        weight: Numeric edge weight (default ``1.0``).
    """

    src_id: str = ""
    dst_id: str = ""
    label: str = ""
    properties: Properties = field(default_factory=dict)
    weight: float = 1.0
    id: str = field(default_factory=new_id)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_id()
        self.properties = validate_properties(self.properties)
        self.weight = float(self.weight)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "label": self.label,
            "properties": dict(self.properties),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            id=data.get("id") or new_id(),
            src_id=data.get("src_id", ""),
            dst_id=data.get("dst_id", ""),
            label=data.get("label", ""),
            properties=data.get("properties") or {},
            weight=data.get("weight", 1.0),
        )
