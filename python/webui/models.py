"""Pydantic request/response models for the web API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- nodes
class NodeCreate(BaseModel):
    label: str = Field(..., description="Text label / type of the node, e.g. 'User'")
    properties: Dict[str, Any] = Field(default_factory=dict)
    text: Optional[str] = Field(
        default=None,
        description="If provided, it is embedded and stored as the node embedding.",
    )


class NodeUpdate(BaseModel):
    properties: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------- edges
class EdgeCreate(BaseModel):
    src_id: str
    dst_id: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
    similarity_threshold: float = 0.85


# ------------------------------------------------------------------- search
class SearchRequest(BaseModel):
    text: str
    label: Optional[str] = None
    k: int = 5


# ------------------------------------------------------------------- memory
class RememberRequest(BaseModel):
    text: str
    memory_type: str = "observation"
    entities: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None


class RecallRequest(BaseModel):
    query: str
    k: int = 5
    memory_type: Optional[str] = None


# ---------------------------------------------------- orchestration (MCP/A2A)
class AgentCreate(BaseModel):
    agent_id: str
    name: Optional[str] = None
    description: str = ""
    skills: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)


class MCPCallRequest(BaseModel):
    agent_id: str
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ResourceReadRequest(BaseModel):
    agent_id: str
    uri: str


class A2AShareRequest(BaseModel):
    sender_id: str
    text: str
    topics: List[str] = Field(default_factory=list)
    memory_type: Optional[str] = None
    recipients: Optional[List[str]] = None


class A2ASendRequest(BaseModel):
    sender_id: str
    recipient_id: str
    content: Dict[str, Any] = Field(default_factory=dict)
    type: str = "text"


class A2APreviewRequest(BaseModel):
    topics: List[str] = Field(default_factory=list)
    text: str = ""
