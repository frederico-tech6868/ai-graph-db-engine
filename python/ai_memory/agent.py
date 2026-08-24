"""GraphAgent: an LLM-backed conversational agent with graph memory.

The agent runs a full memory pipeline on every turn: recall relevant memories,
build a context window, call the LLM (or a deterministic mock when no ``llm_fn``
is supplied), then persist both the user message and the assistant reply as
memory nodes -- extracting and linking simple named entities along the way.

Everything works fully offline with :class:`~ai_memory.embedder.LocalEmbedder`
and the built-in mock LLM.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from .memory import AgentMemory
from .prompts import SYSTEM_PROMPT_WITH_MEMORY
from .schema import MemoryType

# Words that get capitalised but are not entities.
_STOPWORDS = {
    "I", "I'm", "The", "A", "An", "My", "Your", "We", "You", "He", "She",
    "They", "It", "This", "That", "What", "When", "Where", "Who", "Why",
    "How", "Do", "Does", "Is", "Are", "Was", "Were", "Can", "Could", "Would",
    "Should", "Will", "And", "But", "Or", "So", "If", "There", "Here",
    "Hello", "Hi", "Hey", "Yes", "No", "Ok", "Okay", "Please", "Thanks",
    "Understood", "Sure", "Got", "Noted", "Regarding", "Here", "Also",
}

# Matches sequences of Capitalised words (a naive proper-noun detector).
_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")


class GraphAgent:
    """A conversational agent backed by :class:`AgentMemory`."""

    def __init__(
        self,
        agent_id: str,
        memory: AgentMemory,
        llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.memory = memory
        self.llm_fn = llm_fn
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self._current_session: Optional[str] = None

    # ---------------------------------------------------------------- chat
    def chat(self, user_message: str, session_id: Optional[str] = None) -> str:
        """Run one conversational turn and return the assistant response."""
        # Resolve the session (create one lazily).
        if session_id is None:
            if self._current_session is None:
                self._current_session = self.memory.start_session()
            session_id = self._current_session
        else:
            self._current_session = session_id

        # 1-3) Recall relevant memories and build the context window.
        recalled = self.memory.recall(user_message, k=8)
        context = self.memory.recall_engine.build_context_window(recalled)

        # 4) Construct the message list.
        agent_name = self.system_prompt or self.agent_id
        system_content = SYSTEM_PROMPT_WITH_MEMORY(agent_name, context)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        # 5) Call the LLM (or mock).
        if self.llm_fn is not None:
            response = self.llm_fn(messages)
        else:
            response = self._mock_response(user_message, recalled)

        # 6-9) Persist user & assistant messages as memories with entities.
        user_entities = self._extract_entities(user_message)
        self.memory.remember(
            user_message,
            memory_type=MemoryType.OBSERVATION,
            entities=user_entities,
            session_id=session_id,
            metadata={"role": "user"},
        )
        self.memory.remember(
            response,
            memory_type=MemoryType.OBSERVATION,
            entities=self._extract_entities(response),
            session_id=session_id,
            metadata={"role": "assistant"},
        )

        # Update short-term conversation history.
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})

        # 10) Return the response.
        return response

    # --------------------------------------------------------------- helpers
    def _mock_response(self, user_message: str, recalled) -> str:
        """Deterministic offline response referencing recalled memories."""
        if recalled:
            snippets = "; ".join(rm.context_snippet for rm in recalled[:3])
            return (
                f"[mock] I recall {len(recalled)} relevant memory/memories: "
                f"{snippets}. Regarding \"{user_message}\", I've noted it."
            )
        return (
            f"[mock] I don't have prior memories about that yet. "
            f"I've stored your message: \"{user_message}\"."
        )

    def _extract_entities(self, text: str) -> List[str]:
        """Very small NER: capitalised words/phrases minus stopwords."""
        found: List[str] = []
        seen = set()
        for match in _ENTITY_RE.finditer(text or ""):
            # Drop possessive suffix like "Alice's" -> "Alice".
            phrase = re.sub(r"'s\b", "", match.group(1)).strip()
            if not phrase:
                continue
            # Strip leading stopword tokens ("What ML" -> "ML", "My Alice" -> "Alice").
            tokens = phrase.split()
            while tokens and tokens[0] in _STOPWORDS:
                tokens.pop(0)
            phrase = " ".join(tokens)
            if not phrase or phrase in _STOPWORDS:
                continue
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                found.append(phrase)
        return found

    # ---------------------------------------------------------- persistence
    def load_memory(self, path: str) -> None:
        """Load the underlying graph from disk."""
        self.memory.store.load(path)

    def save_memory(self, path: str) -> None:
        """Save the underlying graph to disk."""
        self.memory.store.save(path)

    def memory_stats(self) -> Dict[str, object]:
        return self.memory.stats()


__all__ = ["GraphAgent"]
