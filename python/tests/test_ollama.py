"""Tests for local Ollama integration and the project-layers example.

The Ollama tests spin up a tiny mock HTTP server that mimics Ollama's
``/api/embeddings`` and ``/api/chat`` endpoints, so they run fully offline and
never require a real Ollama install.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ai_memory.embedder import LocalEmbedder, OllamaEmbedder, get_embedder, ollama_chat


class _MockOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/embeddings":
            text = body.get("prompt", "")
            vec = [float((sum(bytearray(text.encode())) % 97) + i) / 100 for i in range(8)]
            out = {"embedding": vec}
        elif self.path == "/api/chat":
            out = {"message": {"role": "assistant", "content": "hello from mock"}}
        else:
            out = {}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def mock_ollama():
    srv = HTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{srv.server_address[1]}"
    yield host
    srv.shutdown()


def test_ollama_embedder_dim_and_embed(mock_ollama):
    emb = OllamaEmbedder(model="nomic-embed-text", host=mock_ollama)
    assert emb.dim == 8
    v = emb.embed("hello world")
    assert len(v) == 8
    assert all(isinstance(x, float) for x in v)


def test_ollama_embed_batch(mock_ollama):
    emb = OllamaEmbedder(host=mock_ollama)
    out = emb.embed_batch(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 8 for v in out)


def test_ollama_chat(mock_ollama):
    reply = ollama_chat("why?", host=mock_ollama, model="llama3.2")
    assert reply == "hello from mock"


def test_ollama_embedder_unreachable_raises():
    # Nothing is listening on this port -> constructor must raise RuntimeError.
    with pytest.raises(RuntimeError):
        OllamaEmbedder(host="http://127.0.0.1:1")


def test_get_embedder_falls_back_without_ollama(monkeypatch):
    monkeypatch.delenv("USE_OLLAMA", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    emb = get_embedder()
    assert isinstance(emb, LocalEmbedder)


def test_project_layers_example_runs(capsys):
    # Smoke test: the full layered-agents program runs end to end.
    import examples.example_project_layers as ex

    ex.main()
    out = capsys.readouterr().out
    assert "PROJECT ORCHESTRATION" in out
    assert "System Evolution" in out
    assert "SHARED GRAPH STATS" in out
