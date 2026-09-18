#!/usr/bin/env python3
"""Example: JEPA-Needle integration — Needle agents with JEPA-GraphRAG retrieval.

This demonstrates :class:`JEPANeedleAgent` and :class:`JEPAOrchestrator`, which
combine two subsystems already shipped with the engine:

  * Needle2 embedded function-calling agents (ai_memory.needle_agent)
  * JEPA-GraphRAG retrieval (ai_memory.jepa_graphrag)

Everything here runs WITHOUT ``cactus-needle`` installed. Only live inference
(``group.run(...)``) requires it, and that section is skipped automatically if
needle is unavailable.

    pip install cactus-needle          # for inference
    pip install 'cactus-needle[train]' # for fine-tuning
"""

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_memory.jepa_needle import (  # noqa: E402
    JEPANeedleAgent,
    JEPAOrchestrator,
    JEPA_TOOL_SCHEMAS,
)
from ai_memory.document_loader import DocumentLoader  # noqa: E402


LEGAL_TEXT = (
    "The termination clause allows either party to end the contract with "
    "notice. Liability is limited under the indemnification section. The "
    "governing law is the state of California for this agreement. Disputes "
    "are resolved through binding arbitration under the arbitration rules.\n"
)

TECH_TEXT = (
    "The API server exposes REST endpoints over HTTP. Authentication uses "
    "JWT bearer tokens in the request header. The database layer caches "
    "queries for low latency responses. Horizontal scaling is handled by a "
    "load balancer in front of stateless application servers.\n"
)


def _write(dir_: Path, name: str, text: str) -> str:
    p = dir_ / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="jepa_needle_"))
    # chunk_overlap MUST be < chunk_size, and small chunks surface multiple nodes
    loader = DocumentLoader(chunk_size=120, chunk_overlap=20, min_chunk_len=20)

    print("=" * 70)
    print("1. Build two JEPA-Needle knowledge groups")
    print("=" * 70)
    legal = JEPANeedleAgent("legal", loader=loader, default_search_mode="hybrid")
    legal.ingest([_write(tmp, "legal.txt", LEGAL_TEXT)])
    tech = JEPANeedleAgent("tech", loader=loader, default_search_mode="hybrid")
    tech.ingest([_write(tmp, "tech.txt", TECH_TEXT)])
    print(legal)
    print(tech)

    print("\n" + "=" * 70)
    print("2. JEPA-enhanced tool schemas exposed to Needle")
    print("=" * 70)
    for s in JEPA_TOOL_SCHEMAS:
        print(f"  - {s['name']}")

    print("\n" + "=" * 70)
    print("3. Multi-mode retrieval (local / global / latent / hybrid)")
    print("=" * 70)
    for mode in ("local", "global", "latent", "hybrid"):
        out = legal.search("termination clause and liability", k=3, mode=mode)
        top = out["results"][0]["text"][:60] if out["results"] else "(none)"
        print(f"  [{mode:7}] {out['count']} hits | top: {top!r}")

    print("\n" + "=" * 70)
    print("4. Community-aware retrieval")
    print("=" * 70)
    comms = legal.search_communities("contract obligations", k=3)
    for c in comms["communities"]:
        print(f"  community {c['community_id']} (size {c['size']}, "
              f"score {c['score']:.3f})")

    print("\n" + "=" * 70)
    print("5. Train the Graph-JEPA world model (self-supervised, VICReg)")
    print("=" * 70)
    history = legal.train_world_model(epochs=3)
    for i, loss in enumerate(history, 1):
        print(f"  epoch {i}: total={loss['total']:.4f} sim={loss['sim']:.4f} "
              f"var={loss['var']:.4f} cov={loss['cov']:.4f}")

    print("\n" + "=" * 70)
    print("6. JEPA-GraphRAG content-relevance routing across groups")
    print("=" * 70)
    orch = JEPAOrchestrator([legal, tech])
    for q in (
        "JWT bearer token authentication",
        "contract termination and liability",
        "REST API database caching",
        "governing law and arbitration",
    ):
        print(f"  {q!r:45} -> {orch.route(q).name}")

    print("\n" + "=" * 70)
    print("7. Live Needle inference")
    print("=" * 70)
    try:
        import needle  # noqa: F401
        # Needs a fine-tuned .cact archive loaded via legal.load_weights(...).
        result = orch.run("What are the termination clauses?")
        print("  routed_to:", result.get("routed_to"))
        print("  response :", result)
    except ImportError:
        print("  (skipped — install cactus-needle and load .cact weights to run "
              "the full agentic loop)")

    print("\nDone.")


if __name__ == "__main__":
    main()
