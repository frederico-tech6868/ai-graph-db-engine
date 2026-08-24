"""A lightweight A2A (Agent-to-Agent) protocol over a shared graph memory.

A2A is the "Agent -> Agent" protocol: agents advertise who they are and what
they care about (an :class:`AgentCard`), exchange :class:`A2AMessage` messages,
and — crucially for this project — **share memories of interest**.

Sharing is *interest-routed*: when an agent publishes a memory tagged with one
or more topics, the :class:`A2ABus` delivers it only to agents whose declared
interests match, either by explicit topic overlap or by semantic similarity
between the memory's embedding and the agent's interest vectors.

Everything lives in the *same* :class:`~graphdb.store.GraphStore` shared by all
agents, so published memories, provenance edges and topics are all visible in
the web UI and searchable with the label-scoped vector engine. Runs fully
in-process and offline.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from graphdb.core import Edge, Node

from .schema import (
    INTERESTED_IN,
    PUBLISHED,
    SHARED_WITH,
    TAGGED,
    TOPIC,
)

if TYPE_CHECKING:  # pragma: no cover
    from ai_memory.memory import AgentMemory


# --------------------------------------------------------------------- models
@dataclass
class AgentCard:
    """Public description an agent advertises to peers (A2A discovery)."""

    agent_id: str
    name: str
    description: str = ""
    skills: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "skills": list(self.skills),
            "interests": list(self.interests),
        }


@dataclass
class A2AMessage:
    """A message exchanged between two agents."""

    sender_id: str
    recipient_id: str
    type: str  # e.g. "memory_share", "text", "request"
    content: Dict[str, object]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "type": self.type,
            "content": self.content,
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------- agent
class A2AAgent:
    """An agent participating in A2A, backed by a graph-based memory."""

    def __init__(self, memory: "AgentMemory", card: AgentCard) -> None:
        self.memory = memory
        self.card = card
        self.inbox: List[A2AMessage] = []

    @property
    def agent_id(self) -> str:
        return self.card.agent_id

    @property
    def node_id(self) -> str:
        return self.memory._agent_node.id

    def receive(self, message: A2AMessage) -> None:
        self.inbox.append(message)

    def unread(self) -> List[A2AMessage]:
        return list(self.inbox)


# ---------------------------------------------------------------------- bus
class A2ABus:
    """Routes A2A messages and interest-matched shared memories between agents.

    All agents must be backed by the *same* shared ``GraphStore`` and embedder.
    """

    def __init__(self, embedder, interest_threshold: float = 0.35) -> None:
        self.embedder = embedder
        self.interest_threshold = interest_threshold
        self._agents: Dict[str, A2AAgent] = {}
        self.history: List[A2AMessage] = []

    # ---------------------------------------------------------- registration
    def register(self, agent: A2AAgent) -> None:
        self._agents[agent.agent_id] = agent
        self._sync_interest_topics(agent)

    def agents(self) -> List[A2AAgent]:
        return list(self._agents.values())

    def get(self, agent_id: str) -> A2AAgent:
        if agent_id not in self._agents:
            raise KeyError(f"unknown agent: {agent_id}")
        return self._agents[agent_id]

    # --------------------------------------------------------- direct message
    def send(
        self,
        sender_id: str,
        recipient_id: str,
        content: Dict[str, object],
        type: str = "text",
    ) -> A2AMessage:
        """Send a direct message from one agent to another."""
        self.get(sender_id)  # validates sender
        recipient = self.get(recipient_id)
        msg = A2AMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            type=type,
            content=content,
        )
        recipient.receive(msg)
        self.history.append(msg)
        return msg

    # ------------------------------------------------------- memory sharing
    def share_memory(
        self,
        sender_id: str,
        text: str,
        topics: Optional[List[str]] = None,
        memory_type=None,
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """Publish a memory and route it to interested agents.

        Returns a summary describing which agents received the memory and why.
        """
        from ai_memory.schema import MemoryType, REMEMBERS

        sender = self.get(sender_id)
        topics = [t.strip() for t in (topics or []) if t.strip()]
        mt = memory_type or MemoryType.OBSERVATION

        # 1. Persist the memory once, authored by the sender.
        result = sender.memory.remember(text=text, memory_type=mt)
        mem_node = result.memory_node
        mem_vec = mem_node.embedding

        store = sender.memory.store

        # 2. Provenance: sender PUBLISHED this memory.
        store.add_edge(Edge(src_id=sender.node_id, dst_id=mem_node.id, label=PUBLISHED))

        # 3. Tag the memory with topic nodes.
        topic_nodes = [self._ensure_topic(store, t) for t in topics]
        for tn in topic_nodes:
            store.add_edge(Edge(src_id=mem_node.id, dst_id=tn.id, label=TAGGED))

        # 4. Determine recipients.
        if recipients is not None:
            targets = [self.get(r) for r in recipients if r != sender_id]
            matched = [(a, "explicit", 1.0) for a in targets]
        else:
            matched = self._interested_agents(sender_id, topics, mem_vec)

        # 5. Deliver: SHARED_WITH edge, REMEMBERS edge, inbox message.
        delivered: List[Dict[str, object]] = []
        for agent, reason, score in matched:
            store.add_edge(Edge(src_id=mem_node.id, dst_id=agent.node_id, label=SHARED_WITH))
            store.add_edge(
                Edge(src_id=agent.node_id, dst_id=mem_node.id, label=REMEMBERS)
            )
            msg = A2AMessage(
                sender_id=sender_id,
                recipient_id=agent.agent_id,
                type="memory_share",
                content={
                    "memory_id": mem_node.id,
                    "text": text,
                    "topics": topics,
                    "match_reason": reason,
                    "match_score": round(float(score), 4),
                },
            )
            agent.receive(msg)
            self.history.append(msg)
            delivered.append(
                {
                    "agent_id": agent.agent_id,
                    "reason": reason,
                    "score": round(float(score), 4),
                }
            )

        return {
            "memory_id": mem_node.id,
            "sender_id": sender_id,
            "topics": topics,
            "was_duplicate": result.was_duplicate,
            "delivered_to": delivered,
        }

    def interested_agents(self, topics: List[str], text: str) -> List[Dict[str, object]]:
        """Preview which agents would receive a memory (no side effects)."""
        vec = self.embedder.embed(text)
        return [
            {"agent_id": a.agent_id, "reason": reason, "score": round(float(s), 4)}
            for a, reason, s in self._interested_agents(None, topics, vec)
        ]

    # ------------------------------------------------------------- internals
    def _interested_agents(self, exclude_id, topics, mem_vec):
        topic_set = {t.lower() for t in topics}
        out = []
        for agent in self._agents.values():
            if exclude_id is not None and agent.agent_id == exclude_id:
                continue
            interests = agent.card.interests
            if not interests:
                continue
            # a) explicit topic overlap
            overlap = topic_set & {i.lower() for i in interests}
            if overlap:
                out.append((agent, f"topic:{sorted(overlap)[0]}", 1.0))
                continue
            # b) semantic similarity between memory and interest vectors
            best = 0.0
            best_interest = None
            for interest in interests:
                sim = _cosine(mem_vec, self.embedder.embed(interest))
                if sim > best:
                    best, best_interest = sim, interest
            if best >= self.interest_threshold:
                out.append((agent, f"semantic:{best_interest}", best))
        return out

    def _ensure_topic(self, store, name: str) -> Node:
        matches = store.find_nodes(label=TOPIC, name=name)
        if matches:
            return matches[0]
        node = Node(label=TOPIC, properties={"name": name}, embedding=self.embedder.embed(name))
        store.add_node(node)
        return node

    def _sync_interest_topics(self, agent: A2AAgent) -> None:
        """Materialise INTERESTED_IN edges for an agent's declared interests."""
        store = agent.memory.store
        for interest in agent.card.interests:
            topic = self._ensure_topic(store, interest)
            store.add_edge(Edge(src_id=agent.node_id, dst_id=topic.id, label=INTERESTED_IN))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


__all__ = [
    "AgentCard",
    "A2AMessage",
    "A2AAgent",
    "A2ABus",
]
