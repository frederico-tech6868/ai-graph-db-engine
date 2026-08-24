"""Full working example: one project agent per AI layer of project orchestration.

This program turns the "AI Layer" workflow described in ``aiLayers.md`` into a
runnable multi-agent orchestration on top of the graph engine.

Each **layer** of project orchestration becomes a **project agent** with:

* an :class:`~protocols.a2a.AgentCard` (name / description / skills / interests),
* its own :class:`~ai_memory.memory.AgentMemory` node in a *shared* graph,
* an MCP server (Agent -> Tools & Data) exposing ``remember_fact`` /
  ``recall_memories`` / ``reflect`` etc. over that memory, and
* A2A membership so it publishes "memories of interest" that are routed to the
  downstream layers that care about them.

The layers (faithful to ``aiLayers.md``)::

    1. Context & Priming      -> priming-agent
    2. Build the Layer        -> rules-agent        (create-rules / create-prd)
    3. Slicing & Parallelism  -> slicing-agent       (spec / worktrees)
    4. PIV: Plan              -> planner-agent       (plan-feature)
    5. PIV: Implement         -> executor-agent      (execute)
    6. PIV: Validate          -> validator-agent     (validate)
    7. Review                 -> review-agent        (code-review / fix)
    8. Commit                 -> commit-agent        (commit)
    9. System Evolution       -> evolution-agent     (rca / system-review)

The program then drives a complete Plan -> Implement -> Validate loop for a
sample feature ("Add OAuth login"), letting each agent consume the upstream
layer's shared memory (via A2A interest routing), do its work (via its MCP
tools), and publish its own output for the next layer.

Everything runs fully offline with the deterministic ``LocalEmbedder`` -- no
API keys, no network.

Run it::

    python examples/example_project_layers.py
"""

from __future__ import annotations

import os
import sys

# --- make the repo importable when run as a plain script from any cwd --------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ai_memory.embedder import LocalEmbedder  # noqa: E402
from ai_memory.schema import MemoryType  # noqa: E402
from graphdb.store import GraphStore  # noqa: E402
from protocols import Orchestrator  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Declarative spec: one project agent per AI layer.
#    `interests` are the topics an agent wants to hear about; `produces` are the
#    topics it tags its own output with. Overlap between one layer's `produces`
#    and the next layer's `interests` is what makes A2A route memory downstream.
# ---------------------------------------------------------------------------
LAYERS = [
    {
        "id": "priming-agent",
        "name": "Context & Priming",
        "description": "Loads codebase context (prime / prime-backend / prime-frontend).",
        "skills": ["prime", "prime-backend", "prime-frontend"],
        "interests": ["codebase", "context", "architecture"],
        "produces": ["codebase", "context", "architecture"],
    },
    {
        "id": "rules-agent",
        "name": "Build the Layer",
        "description": "Derives CLAUDE.md rules and PRDs (create-rules / create-prd).",
        "skills": ["create-rules", "create-prd"],
        "interests": ["codebase", "context", "requirements"],
        "produces": ["rules", "prd", "requirements"],
    },
    {
        "id": "slicing-agent",
        "name": "Slicing & Parallelism",
        "description": "Slices a PRD into PIV-sized tickets with a dependency graph (spec / worktrees).",
        "skills": ["spec", "new-worktrees", "merge-worktrees"],
        "interests": ["prd", "requirements", "rules"],
        "produces": ["tickets", "epics", "plan"],
    },
    {
        "id": "planner-agent",
        "name": "PIV: Plan",
        "description": "Writes a context-rich, one-pass implementation plan (plan-feature).",
        "skills": ["plan-feature"],
        "interests": ["tickets", "plan", "requirements"],
        "produces": ["plan", "feature", "implementation"],
    },
    {
        "id": "executor-agent",
        "name": "PIV: Implement",
        "description": "Builds strictly from the approved plan (execute).",
        "skills": ["execute"],
        "interests": ["plan", "feature", "implementation"],
        "produces": ["implementation", "code", "diff"],
    },
    {
        "id": "validator-agent",
        "name": "PIV: Validate",
        "description": "Runs tests / type-check / lint / build before a PR (validate).",
        "skills": ["validate"],
        "interests": ["implementation", "code", "tests"],
        "produces": ["validation", "tests", "quality"],
    },
    {
        "id": "review-agent",
        "name": "Review",
        "description": "First-pass review on a diff/PR and applies fixes (code-review / fix).",
        "skills": ["code-review", "code-review-fix"],
        "interests": ["implementation", "code", "validation", "quality"],
        "produces": ["review", "quality", "code"],
    },
    {
        "id": "commit-agent",
        "name": "Commit",
        "description": "Structured commit at the end of a loop (commit).",
        "skills": ["commit"],
        "interests": ["review", "quality", "validation"],
        "produces": ["git", "commit"],
    },
    {
        "id": "evolution-agent",
        "name": "System Evolution",
        "description": "Root-causes bugs and tightens the AI layer (rca / system-review / execution-report).",
        "skills": ["rca", "system-review", "execution-report"],
        "interests": ["commit", "git", "review", "rca", "metrics"],
        "produces": ["rca", "evolution", "rules", "metrics"],
    },
]


# ---------------------------------------------------------------------------
# 2. A tiny helper to print MCP call results readably.
# ---------------------------------------------------------------------------
def _mcp_text(envelope) -> str:
    """Pull the human-readable text out of an MCP tool-call envelope."""
    try:
        return envelope["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return str(envelope)


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# 3. Build the orchestration: register every layer as an agent.
# ---------------------------------------------------------------------------
def build_orchestration() -> Orchestrator:
    store = GraphStore()
    embedder = LocalEmbedder()
    # interest_threshold kept low because LocalEmbedder similarity is coarse;
    # topic overlap is the primary (reliable) routing signal.
    orch = Orchestrator(store=store, embedder=embedder, interest_threshold=0.35)

    for spec in LAYERS:
        orch.create_agent(
            agent_id=spec["id"],
            name=spec["name"],
            description=spec["description"],
            skills=spec["skills"],
            interests=spec["interests"],
        )
    return orch


def _produces(agent_id: str):
    for spec in LAYERS:
        if spec["id"] == agent_id:
            return spec["produces"]
    return []


# ---------------------------------------------------------------------------
# 4. One reusable "layer step": an agent records its output via its MCP tool,
#    then publishes it to the bus tagged with the topics it produces. A2A routes
#    that memory to whichever downstream layers registered matching interests.
# ---------------------------------------------------------------------------
def run_layer(orch: Orchestrator, agent_id: str, output: str, memory_type=MemoryType.FACT):
    spec = next(s for s in LAYERS if s["id"] == agent_id)

    # (a) What did this layer already receive from upstream layers?
    inbox = orch.inbox(agent_id)
    received = [m["content"].get("text", "") for m in inbox if m["type"] == "memory_share"]

    print(f"\n--- Layer: {spec['name']}  ({agent_id}) ---")
    print(f"    skills   : {', '.join(spec['skills'])}")
    if received:
        print(f"    consumed : {len(received)} upstream memory(ies) of interest")
        for r in received:
            print(f"               - {r}")
    else:
        print("    consumed : (nothing yet - this is an entry layer)")

    # (b) Do the work: persist output through the agent's own MCP tool.
    call = orch.mcp_call(
        agent_id,
        "remember_fact",
        {"text": output, "memory_type": memory_type.value},
    )
    print(f"    MCP tool : remember_fact -> {_mcp_text(call)}")

    # (c) Publish the output as a shared memory of interest for downstream layers.
    topics = _produces(agent_id)
    result = orch.a2a_share(sender_id=agent_id, text=output, topics=topics)
    routed = [d["agent_id"] for d in result["delivered_to"]]
    print(f"    A2A share: topics={topics}")
    print(f"    routed to: {routed if routed else '(no interested downstream layer)'}")
    return result


# ---------------------------------------------------------------------------
# 5. Drive a full end-to-end project run through every layer.
# ---------------------------------------------------------------------------
def main() -> None:
    orch = build_orchestration()
    feature = "Add OAuth login (Google + GitHub) to the web app"

    banner("PROJECT ORCHESTRATION - one agent per AI layer")
    print(f"Feature under development: {feature}")
    print(f"Registered layers/agents : {len(LAYERS)}")

    banner("END-TO-END PIV RUN (Plan -> Implement -> Validate + surrounding layers)")

    # Layer 1: Context & Priming
    run_layer(
        orch,
        "priming-agent",
        "Primed repo context: Flask API + React SPA, sessions via cookies, "
        "auth handled in app/auth/. Key modules: app/auth, app/api, web/src/auth.",
        memory_type=MemoryType.OBSERVATION,
    )

    # Layer 2: Build the Layer (rules + PRD)
    run_layer(
        orch,
        "rules-agent",
        f"PRD for '{feature}': support Google + GitHub OAuth2, store provider "
        "tokens encrypted, add /auth/callback route. CLAUDE.md rule: never log "
        "raw tokens; all auth code lives under app/auth/.",
    )

    # Layer 3: Slicing & Parallelism
    run_layer(
        orch,
        "slicing-agent",
        "Sliced PRD into 3 PIV tickets: [T1] OAuth provider config, "
        "[T2] /auth/callback + token exchange (depends on T1), "
        "[T3] frontend 'Sign in with...' buttons (depends on T2).",
    )

    # Layer 4: PIV - Plan
    run_layer(
        orch,
        "planner-agent",
        "Plan for T2: add OAuthClient in app/auth/oauth.py, exchange code->token, "
        "persist encrypted token via app/auth/store.py, wire /auth/callback in "
        "app/api/routes.py, add unit tests in tests/test_oauth.py.",
        memory_type=MemoryType.PLAN,
    )

    # Layer 5: PIV - Implement
    run_layer(
        orch,
        "executor-agent",
        "Implemented T2 per plan: app/auth/oauth.py (OAuthClient.exchange_code), "
        "app/auth/store.py (encrypted TokenStore), /auth/callback route. "
        "Diff: +214 / -6 across 4 files.",
    )

    # Layer 6: PIV - Validate
    run_layer(
        orch,
        "validator-agent",
        "Validation: pytest 42 passed, mypy clean, ruff clean, build OK. "
        "New tests test_oauth.py cover success + CSRF-state mismatch paths.",
    )

    # Layer 7: Review
    run_layer(
        orch,
        "review-agent",
        "Code review of T2 diff: request token-refresh handling + a check that "
        "'state' is verified before exchange. Applied both fixes.",
    )

    # Layer 8: Commit
    run_layer(
        orch,
        "commit-agent",
        "Committed T2: 'feat(auth): add OAuth2 login for Google + GitHub' "
        "(5 files, tests green). Ready to open PR.",
    )

    # Layer 9: System Evolution
    run_layer(
        orch,
        "evolution-agent",
        "System review: the missing 'state' check was a class of bug. New rule "
        "added to CLAUDE.md + regression test so OAuth flows always verify state.",
        memory_type=MemoryType.REFLECTION,
    )

    # -------------------------------------------------------------------
    # 6. Show the orchestration state that emerged in the shared graph.
    # -------------------------------------------------------------------
    banner("AGENTS (per-layer memory ownership)")
    for a in orch.agents():
        print(
            f"  {a['name']:<22} id={a['agent_id']:<16} "
            f"owned_memories={a['owned_memories']:<3} inbox={a['inbox']}"
        )

    banner("MCP: Agent -> Tools & Data (executor-agent as example)")
    print("  Tools exposed by executor-agent:")
    for t in orch.tools("executor-agent"):
        print(f"    - {t['name']}: {t['description']}")
    print("\n  Resources exposed by executor-agent:")
    for r in orch.resources("executor-agent"):
        print(f"    - {r['uri']} ({r['name']})")
    stats_res = orch.read_resource("executor-agent", "memory://stats")
    print(f"\n  read_resource memory://stats -> {stats_res['contents'][0]['text']}")

    banner("MCP: recall + reflect on the validator's memory")
    recall = orch.mcp_call("validator-agent", "recall_memories", {"query": "tests", "k": 3})
    print(f"  recall_memories('tests') -> {_mcp_text(recall)}")
    reflect = orch.mcp_call("evolution-agent", "reflect", {"recent_k": 10})
    print(f"  reflect() -> {_mcp_text(reflect)}")

    banner("A2A: message feed (interest-routed memory shares)")
    for m in orch.messages(limit=40):
        if m["type"] == "memory_share":
            topics = ", ".join(m["content"].get("topics", []))
            reason = m["content"].get("match_reason")
            print(
                f"  {m['sender_id']:<16} -> {m['recipient_id']:<16} "
                f"[{reason}] topics=({topics})"
            )

    banner("SHARED GRAPH STATS")
    print(
        f"  nodes: {len(orch.store.all_nodes())}   "
        f"edges: {len(orch.store.all_edges())}"
    )
    print(
        "  Every layer's output lives in ONE graph, cross-linked by "
        "PUBLISHED / TAGGED / SHARED_WITH / REMEMBERS edges."
    )
    print("\nDone. Each AI layer ran as an autonomous agent, used its MCP tools, and")
    print("shared memories of interest with downstream layers over A2A.\n")


if __name__ == "__main__":
    main()
