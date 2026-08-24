"""Type stubs for the compiled Rust extension module ``graphdb_rs``.

These describe the Python API exposed by the PyO3 bindings in ``src/lib.rs``.
"""

from typing import Optional

__version__: str

class PyNode:
    id: str
    label: str
    properties: dict
    embedding: Optional[list[float]]
    created_at: int

class PyEdge:
    id: str
    src_id: str
    dst_id: str
    label: str
    properties: dict
    weight: float

class PySimilarMatch:
    existing_edge_id: str
    src_similarity: float
    dst_similarity: float
    combined_score: float

class PyAddEdgeResult:
    edge: PyEdge
    similar_edges: list[PySimilarMatch]
    was_flagged: bool

class PyGraphStore:
    def __init__(self, path: Optional[str] = None) -> None: ...
    def add_node(
        self,
        label: str,
        properties: dict = ...,
        embedding: Optional[list[float]] = None,
    ) -> PyNode: ...
    def get_node(self, id: str) -> PyNode: ...
    def delete_node(self, id: str) -> None: ...
    def nodes_by_label(self, label: str) -> list[PyNode]: ...
    def all_nodes(self) -> list[PyNode]: ...
    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        label: str,
        properties: dict = ...,
        weight: float = 1.0,
        similarity_threshold: float = 0.85,
    ) -> PyAddEdgeResult: ...
    def get_edge(self, id: str) -> PyEdge: ...
    def delete_edge(self, id: str) -> None: ...
    def edges_from(self, node_id: str) -> list[PyEdge]: ...
    def edges_to(self, node_id: str) -> list[PyEdge]: ...
    def search_similar_nodes(
        self, query: list[float], label: Optional[str] = None, k: int = 5
    ) -> list[tuple[str, float]]: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...

def bfs(
    store: PyGraphStore,
    start_id: str,
    edge_label: Optional[str] = None,
    max_depth: int = 5,
) -> list[str]: ...
def dfs(
    store: PyGraphStore,
    start_id: str,
    edge_label: Optional[str] = None,
    max_depth: int = 5,
) -> list[str]: ...
def find_path(
    store: PyGraphStore, src_id: str, dst_id: str
) -> Optional[list[str]]: ...
