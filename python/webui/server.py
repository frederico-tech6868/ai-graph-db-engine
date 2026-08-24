"""FastAPI application exposing the graph DB engine over REST + a SPA."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from graphdb.exceptions import (
    EdgeNotFoundError,
    NodeNotFoundError,
)

from .models import (
    A2APreviewRequest,
    A2ASendRequest,
    A2AShareRequest,
    AgentCreate,
    EdgeCreate,
    MCPCallRequest,
    NodeCreate,
    NodeUpdate,
    RecallRequest,
    RememberRequest,
    ResourceReadRequest,
    SearchRequest,
)
from .service import GraphService

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Graph DB Engine — Web UI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = GraphService()


# --------------------------------------------------------------------- health
@app.get("/api/health")
def health():
    return {"status": "ok", "backend": "python"}


@app.get("/api/stats")
def stats():
    return service.stats()


@app.get("/api/graph")
def graph(label: Optional[str] = None, limit: int = 200):
    return service.graph(label=label, limit=limit)


# --------------------------------------------------------------------- nodes
@app.get("/api/nodes")
def list_nodes(label: Optional[str] = None, limit: int = 200):
    return service.list_nodes(label=label, limit=limit)


@app.get("/api/nodes/{node_id}")
def get_node(node_id: str):
    node = service.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="node not found")
    return node


@app.post("/api/nodes")
def create_node(body: NodeCreate):
    try:
        return service.create_node(body.label, body.properties, body.text)
    except Exception as exc:  # invalid property etc.
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/nodes/{node_id}")
def update_node(node_id: str, body: NodeUpdate):
    try:
        return service.update_node(node_id, body.properties)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail="node not found")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/nodes/{node_id}")
def delete_node(node_id: str):
    try:
        service.delete_node(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail="node not found")
    return {"deleted": node_id}


# --------------------------------------------------------------------- edges
@app.get("/api/edges")
def list_edges():
    return service.list_edges()


@app.post("/api/edges")
def create_edge(body: EdgeCreate):
    try:
        return service.create_edge(
            body.src_id,
            body.dst_id,
            body.label,
            body.properties,
            body.weight,
            body.similarity_threshold,
        )
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/edges/{edge_id}")
def delete_edge(edge_id: str):
    try:
        service.delete_edge(edge_id)
    except EdgeNotFoundError:
        raise HTTPException(status_code=404, detail="edge not found")
    return {"deleted": edge_id}


# --------------------------------------------------------------------- search
@app.post("/api/search")
def search(body: SearchRequest):
    return service.search(body.text, body.label, body.k)


# ------------------------------------------------------------------ traversal
@app.get("/api/traverse/{node_id}")
def traverse(node_id: str, edge_label: Optional[str] = None, max_depth: int = 2):
    if service.store.get_node_or_none(node_id) is None:
        raise HTTPException(status_code=404, detail="node not found")
    return service.traverse(node_id, edge_label, max_depth)


@app.get("/api/path")
def path(src_id: str, dst_id: str):
    return {"path": service.path(src_id, dst_id)}


# --------------------------------------------------------------------- memory
@app.get("/api/memory/stats")
def memory_stats():
    return service.memory_stats()


@app.post("/api/memory/remember")
def remember(body: RememberRequest):
    return service.remember(body.text, body.memory_type, body.entities, body.session_id)


@app.post("/api/memory/recall")
def recall(body: RecallRequest):
    return service.recall(body.query, body.k, body.memory_type)


@app.post("/api/memory/reflect")
def reflect():
    return {"reflection": service.reflect()}


@app.get("/api/memory/entities")
def entities():
    return service.entities()


# ------------------------------------------------------- orchestration (MCP/A2A)
@app.get("/api/agents")
def list_agents():
    return service.list_agents()


@app.post("/api/agents")
def create_agent(body: AgentCreate):
    if not body.agent_id.strip():
        raise HTTPException(status_code=400, detail="agent_id is required")
    return service.create_agent(
        body.agent_id.strip(),
        name=body.name,
        description=body.description,
        skills=body.skills,
        interests=body.interests,
    )


@app.get("/api/agents/{agent_id}/tools")
def agent_tools(agent_id: str):
    try:
        return service.agent_tools(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")


@app.get("/api/agents/{agent_id}/resources")
def agent_resources(agent_id: str):
    try:
        return service.agent_resources(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")


@app.get("/api/agents/{agent_id}/inbox")
def agent_inbox(agent_id: str):
    try:
        return service.a2a_inbox(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")


@app.post("/api/mcp/call")
def mcp_call(body: MCPCallRequest):
    try:
        return service.mcp_call(body.agent_id, body.tool, body.arguments)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")


@app.post("/api/mcp/resource")
def mcp_resource(body: ResourceReadRequest):
    try:
        return service.mcp_read_resource(body.agent_id, body.uri)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")


@app.get("/api/mcp/log")
def mcp_log():
    return service.mcp_call_log()


@app.post("/api/a2a/share")
def a2a_share(body: A2AShareRequest):
    try:
        return service.a2a_share(
            body.sender_id, body.text, body.topics, body.memory_type, body.recipients
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")


@app.post("/api/a2a/send")
def a2a_send(body: A2ASendRequest):
    try:
        return service.a2a_send(body.sender_id, body.recipient_id, body.content, body.type)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")


@app.post("/api/a2a/preview")
def a2a_preview(body: A2APreviewRequest):
    return service.a2a_preview(body.topics, body.text)


@app.get("/api/a2a/messages")
def a2a_messages(limit: int = 50):
    return service.a2a_messages(limit=limit)


@app.post("/api/seed_agents")
def seed_agents():
    return service.seed_agents()


# ----------------------------------------------------------------- admin
@app.post("/api/save")
def save():
    return {"saved": service.save()}


@app.post("/api/load")
def load():
    service.load()
    return service.stats()


@app.post("/api/reset")
def reset():
    service.reset()
    return service.stats()


@app.post("/api/seed")
def seed():
    return service.seed()


# ----------------------------------------------------------------- static SPA
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
