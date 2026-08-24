"""Custom exceptions for the graphdb engine."""


class GraphDBError(Exception):
    """Base class for all graphdb errors."""


class NodeNotFoundError(GraphDBError):
    """Raised when a node id cannot be found in the store."""


class EdgeNotFoundError(GraphDBError):
    """Raised when an edge id cannot be found in the store."""


class DuplicateIdError(GraphDBError):
    """Raised when adding an entity whose id already exists."""


class InvalidPropertyError(GraphDBError):
    """Raised when a property value has an unsupported type."""


class PersistenceError(GraphDBError):
    """Raised when saving/loading the graph fails."""


class DimensionMismatchError(GraphDBError):
    """Raised when two vectors have differing dimensions."""
