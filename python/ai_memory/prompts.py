"""Prompt templates for the memory-backed agent.

These are plain functions returning strings so they work with any LLM backend
(OpenAI, local, or the offline mock used in tests/demo).
"""

from __future__ import annotations


def SYSTEM_PROMPT_WITH_MEMORY(agent_name: str, memory_context: str) -> str:
    """System prompt that injects recalled long-term memory as context."""
    return (
        f"You are {agent_name}, an AI assistant with a persistent long-term "
        "memory. You remember facts about the user and past conversations "
        "across sessions.\n\n"
        "Use the relevant memories below to inform your answer. If a memory "
        "answers the user's question, rely on it. If nothing is relevant, "
        "answer normally and do not invent memories.\n\n"
        "=== RELEVANT LONG-TERM MEMORIES ===\n"
        f"{memory_context}\n"
        "=== END MEMORIES ===\n"
    )


def REFLECTION_PROMPT(recent_memories_text: str) -> str:
    """Prompt asking the LLM to synthesise a reflection from recent memories."""
    return (
        "Below are recent memories from your interactions. Synthesise a concise, "
        "higher-level reflection: identify recurring themes, important entities, "
        "and any insights or patterns. Keep it to a few sentences.\n\n"
        "=== RECENT MEMORIES ===\n"
        f"{recent_memories_text}\n"
        "=== END ===\n\n"
        "Reflection:"
    )


def ENTITY_EXTRACTION_PROMPT(text: str) -> str:
    """Prompt asking the LLM to extract named entities from a piece of text."""
    return (
        "Extract the named entities (people, places, organisations, products, "
        "concepts) from the text below. Return them as a comma-separated list "
        "with no extra commentary.\n\n"
        f"Text: {text}\n\n"
        "Entities:"
    )


__all__ = [
    "SYSTEM_PROMPT_WITH_MEMORY",
    "REFLECTION_PROMPT",
    "ENTITY_EXTRACTION_PROMPT",
]
