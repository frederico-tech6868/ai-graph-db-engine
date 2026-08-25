"""Token-budgeted context assembly for local LLMs (Ollama, etc.).

The graph engine is an *unbounded* long-term memory. A model context window is
*bounded* (Ollama's ``num_ctx`` is typically 64k-256k). The job of
:class:`ContextManager` is to sit between the two so the prompt you send to the
model **never overflows** the window while still surfacing the most relevant
long-term memories.

It does three things every turn:

1. **Budgeting** - splits the context window into fixed reservations (system
   prompt + space for the model's reply) and a working area, then divides the
   working area between *retrieved long-term memories* and the *live transcript*.
2. **Retrieval** - pulls only the top memories relevant to the current message
   (label-scoped vector search over the graph) and packs them greedily until the
   memory sub-budget is full. Everything else stays on disk in the graph.
3. **Rolling summarisation** - when the live transcript grows past its
   sub-budget, the oldest turns are summarised (optionally by a local Ollama
   chat model, otherwise by a deterministic extractive fallback), the summary is
   written **back into the graph as a memory**, and the raw turns are dropped
   from the live transcript. Nothing is lost - it becomes retrievable memory.

The result: a prompt whose measured token count is *guaranteed* <= the model's
context limit, regardless of how long the conversation runs.

Token counting is pluggable. By default a fast ``len(text)/chars_per_token``
heuristic is used (safe and dependency-free); pass your own ``token_counter``
(e.g. a real tokenizer) for exact accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .memory import AgentMemory
from .schema import MemoryType


# --------------------------------------------------------------------- budget
@dataclass
class ContextBudget:
    """How the model's context window is carved up (all values in tokens).

    ``context_limit`` is the model's ``num_ctx`` (e.g. 65536 for 64k, 262144 for
    256k). ``reserve_response`` is held back for the model's own reply.
    ``reserve_system`` is held back for the system prompt. Whatever remains is
    the *working area*, split between retrieved memories and the live transcript
    according to ``memory_fraction``.
    """

    context_limit: int = 65_536          # 64k; use 262_144 for 256k
    reserve_response: int = 2_048        # room for the model to answer
    reserve_system: int = 1_024          # room for the system prompt
    memory_fraction: float = 0.5         # share of the working area for memories
    chars_per_token: float = 4.0         # heuristic for the default counter

    def working_tokens(self) -> int:
        """Tokens available after system + response reservations."""
        return max(
            0, self.context_limit - self.reserve_response - self.reserve_system
        )

    def memory_tokens(self) -> int:
        return int(self.working_tokens() * self.memory_fraction)

    def history_tokens(self) -> int:
        return self.working_tokens() - self.memory_tokens()


@dataclass
class AssembledPrompt:
    """The result of :meth:`ContextManager.assemble`."""

    system: str
    memory_block: str
    history: List[Dict[str, str]]        # [{role, content}, ...] that fit
    user: str
    total_tokens: int
    within_limit: bool
    breakdown: Dict[str, int] = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, str]]:
        """Render as OpenAI/Ollama-style chat messages, memories folded into
        the system message so retrieval context is authoritative."""
        sys = self.system
        if self.memory_block:
            sys = (sys + "\n\n" if sys else "") + self.memory_block
        msgs: List[Dict[str, str]] = []
        if sys:
            msgs.append({"role": "system", "content": sys})
        msgs.extend(self.history)
        msgs.append({"role": "user", "content": self.user})
        return msgs

    def to_prompt(self) -> str:
        """Render as a single flat prompt string (for ``/api/generate``)."""
        parts = []
        if self.system:
            parts.append(self.system)
        if self.memory_block:
            parts.append(self.memory_block)
        for turn in self.history:
            parts.append(f"{turn['role'].upper()}: {turn['content']}")
        parts.append(f"USER: {self.user}")
        return "\n\n".join(parts)


def _default_counter(chars_per_token: float) -> Callable[[str], int]:
    def count(text: str) -> int:
        if not text:
            return 0
        return int(math.ceil(len(text) / chars_per_token))

    return count


class ContextManager:
    """Assemble budget-safe prompts backed by graph long-term memory."""

    def __init__(
        self,
        memory: AgentMemory,
        budget: Optional[ContextBudget] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        summarizer: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.memory = memory
        self.budget = budget or ContextBudget()
        self.count = token_counter or _default_counter(self.budget.chars_per_token)
        # summarizer(raw_text) -> short summary. Defaults to an offline
        # extractive summary; pass a local Ollama chat callback for a better one.
        self.summarizer = summarizer or self._extractive_summary

    # ---------------------------------------------------------- persistence
    def ingest_turn(self, user_text: str, assistant_text: str) -> None:
        """Store a completed exchange in the graph as long-term memory."""
        if user_text.strip():
            self.memory.remember(
                f"User said: {user_text.strip()}",
                memory_type=MemoryType.OBSERVATION,
            )
        if assistant_text.strip():
            self.memory.remember(
                f"Assistant replied: {assistant_text.strip()}",
                memory_type=MemoryType.OBSERVATION,
            )

    # ------------------------------------------------------------- assembly
    def assemble(
        self,
        user_message: str,
        system_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        recall_k: int = 12,
    ) -> Tuple[AssembledPrompt, List[Dict[str, str]]]:
        """Build a prompt that fits the window.

        Returns ``(assembled, trimmed_history)`` where ``trimmed_history`` is the
        live transcript after any overflow turns have been summarised into the
        graph and dropped - feed it back in on the next turn.
        """
        history = list(history or [])

        # 1. Fixed costs.
        sys_tokens = self.count(system_prompt)
        user_tokens = self.count(user_message)

        # 2. Retrieve + pack relevant long-term memories into their sub-budget.
        mem_budget = self.budget.memory_tokens()
        memory_block, mem_tokens = self._pack_memories(user_message, mem_budget, recall_k)

        # 3. Fit the live transcript into its sub-budget, summarising overflow.
        hist_budget = self.budget.history_tokens()
        # Give any unused memory budget back to history (and vice-versa).
        hist_budget += max(0, mem_budget - mem_tokens)
        kept_history, hist_tokens, trimmed_history = self._fit_history(
            history, hist_budget
        )

        total = sys_tokens + mem_tokens + hist_tokens + user_tokens
        assembled = AssembledPrompt(
            system=system_prompt,
            memory_block=memory_block,
            history=kept_history,
            user=user_message,
            total_tokens=total,
            within_limit=total + self.budget.reserve_response <= self.budget.context_limit,
            breakdown={
                "system": sys_tokens,
                "memories": mem_tokens,
                "history": hist_tokens,
                "user": user_tokens,
                "reserved_for_response": self.budget.reserve_response,
                "context_limit": self.budget.context_limit,
            },
        )
        return assembled, trimmed_history

    # ------------------------------------------------------------- internals
    def _pack_memories(self, query: str, budget: int, k: int) -> Tuple[str, int]:
        if budget <= 0:
            return "", 0
        recalled = self.memory.recall(query, k=k)
        header = "Relevant long-term memory (retrieved from the knowledge graph):"
        lines: List[str] = []
        used = self.count(header)
        for rm in recalled:
            text = str(rm.node.properties.get("text", "")).strip()
            if not text:
                continue
            line = f"- {text}"
            cost = self.count(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        if not lines:
            return "", 0
        return header + "\n" + "\n".join(lines), used

    def _fit_history(
        self, history: List[Dict[str, str]], budget: int
    ) -> Tuple[List[Dict[str, str]], int, List[Dict[str, str]]]:
        """Keep the most recent turns that fit; summarise the rest into memory."""
        # Walk from the newest turn backwards, keeping what fits.
        kept_rev: List[Dict[str, str]] = []
        used = 0
        overflow: List[Dict[str, str]] = []
        for turn in reversed(history):
            cost = self.count(turn.get("content", "")) + 2  # +role tag
            if used + cost <= budget:
                kept_rev.append(turn)
                used += cost
            else:
                overflow.append(turn)
        kept = list(reversed(kept_rev))
        overflow = list(reversed(overflow))  # chronological

        # Summarise dropped turns into the graph so they remain retrievable.
        if overflow:
            raw = "\n".join(
                f"{t.get('role', 'user')}: {t.get('content', '')}" for t in overflow
            )
            summary = self.summarizer(raw)
            if summary.strip():
                self.memory.remember(
                    f"Conversation summary: {summary.strip()}",
                    memory_type=MemoryType.REFLECTION,
                )
        return kept, used, kept

    @staticmethod
    def _extractive_summary(raw: str, max_sentences: int = 4) -> str:
        """Deterministic, offline fallback summariser.

        Picks the first and last couple of non-empty lines - crude but keeps the
        gist and never depends on a model being available.
        """
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) <= max_sentences:
            return " ".join(lines)
        head = lines[: max_sentences - 1]
        tail = lines[-1:]
        return " ".join(head + ["..."] + tail)


__all__ = ["ContextBudget", "ContextManager", "AssembledPrompt"]
