"""graphdb: a from-scratch in-memory property graph database with
label-scoped vector similarity.

Public API::

    from graphdb import GraphStore, Node, Edge, GraphQuery
"""

from .core import Edge, Node, validate_properties
from .exceptions import (
    DimensionMismatchError,
    DuplicateIdError,
    EdgeNotFoundError,
    GraphDBError,
    InvalidPropertyError,
    NodeNotFoundError,
    PersistenceError,
)
from .index import LabelIndex, PropertyIndex
from .query import GraphQuery, bfs, dfs, find_path
from .similarity import SimilarMatch, SimilarityScanner
from .store import AddEdgeResult, GraphStore
from .vector import cosine_similarity, top_k_similar

__version__ = "0.1.0"

__all__ = [
    "Node",
    "Edge",
    "validate_properties",
    "GraphStore",
    "AddEdgeResult",
    "GraphQuery",
    "bfs",
    "dfs",
    "find_path",
    "SimilarMatch",
    "SimilarityScanner",
    "LabelIndex",
    "PropertyIndex",
    "cosine_similarity",
    "top_k_similar",
    "GraphDBError",
    "NodeNotFoundError",
    "EdgeNotFoundError",
    "DuplicateIdError",
    "InvalidPropertyError",
    "PersistenceError",
    "DimensionMismatchError",
]
