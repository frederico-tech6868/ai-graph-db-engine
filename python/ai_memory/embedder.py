"""Text -> embedding.

Two implementations are provided:

* :class:`OpenAIEmbedder` uses the ``openai`` package when it is installed and
  ``OPENAI_API_KEY`` is set. It calls the embeddings endpoint (default model
  ``text-embedding-3-small``).
* :class:`OllamaEmbedder` talks to a **local Ollama** server (default
  ``http://localhost:11434``) using only the standard library, so you get real
  neural embeddings from a locally-hosted model (e.g. ``nomic-embed-text``)
  with no cloud API and no extra Python dependencies.
* :class:`LocalEmbedder` is a pure-Python, dependency-free, *deterministic*
  fallback. It hashes character n-grams of each token into a fixed-size vector
  with tf-idf-like weighting and L2-normalises the result. It requires no ML
  model and no API key, and produces stable vectors across runs/processes
  (using :mod:`hashlib` rather than the salted built-in ``hash``).

Use :func:`get_embedder` to obtain the best available embedder automatically.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    """Abstract embedder interface."""

    #: Dimensionality of vectors this embedder produces.
    dim: int = 0

    def embed(self, text: str) -> List[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _char_ngrams(token: str, n: int = 3) -> List[str]:
    """Return the character n-grams of ``token`` (padded)."""
    if len(token) <= n:
        return [token]
    padded = f"^{token}$"
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def _stable_hash(text: str) -> int:
    """A deterministic, process-independent hash (unlike built-in ``hash``)."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16)


class LocalEmbedder(Embedder):
    """Deterministic pure-Python embedder based on hashed char n-grams.

    The algorithm:

    1. Lower-case and tokenize the text on whitespace/punctuation.
    2. For each token compute a set of features: the token itself plus its
       character 3-grams (this gives fuzzy sub-word matching).
    3. Hash each feature to a bucket ``0..dim-1`` and accumulate a signed,
       tf-idf-like weight (term frequency dampened with ``1 + log(tf)`` and an
       inverse-length factor so long generic tokens don't dominate).
    4. L2-normalise the resulting vector.

    The output is fully deterministic, so tests never need an API key.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vec

        # Term frequencies for a tf-idf-like weight.
        tf: dict = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        for tok, count in tf.items():
            # tf component (sublinear) times a mild inverse-length idf proxy.
            tf_weight = 1.0 + math.log(count)
            idf_proxy = 1.0 + 1.0 / math.sqrt(len(tok))
            weight = tf_weight * idf_proxy
            for gram in _char_ngrams(tok) + [tok]:
                h = _stable_hash(gram)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                vec[idx] += sign * weight

        return _l2_normalize(vec)


class OpenAIEmbedder(Embedder):
    """Embedder backed by the OpenAI embeddings API.

    Only usable when the ``openai`` package is importable and an API key is
    available. Raises ``RuntimeError`` on construction otherwise.
    """

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None) -> None:
        try:
            import openai  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError("openai package is not installed") from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        from openai import OpenAI

        self.model = model
        self.dim = self._DIMS.get(model, 1536)
        self._client = OpenAI(api_key=key)

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - network
        # Replace empty strings with a single space; the API rejects "".
        cleaned = [t if t else " " for t in texts]
        resp = self._client.embeddings.create(model=self.model, input=cleaned)
        return [list(item.embedding) for item in resp.data]


class OllamaEmbedder(Embedder):
    """Embedder backed by a **local Ollama** server.

    Uses only the Python standard library (``urllib``) to POST to Ollama's
    embeddings endpoint, so no extra packages are required. Point it at any
    embedding model you have pulled locally, e.g.::

        ollama pull nomic-embed-text

    Parameters
    ----------
    model:
        Ollama model name (default ``nomic-embed-text``, or the
        ``OLLAMA_EMBED_MODEL`` env var).
    host:
        Base URL of the Ollama server (default ``http://localhost:11434`` or the
        ``OLLAMA_HOST`` env var).
    timeout:
        Per-request timeout in seconds.

    The vector dimensionality is discovered automatically on construction by
    embedding a short probe string. A ``RuntimeError`` is raised if the server
    is unreachable or the model is not available (mirroring
    :class:`OpenAIEmbedder`), so callers can fall back gracefully.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        base = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = base.rstrip("/")
        self.timeout = timeout

        # Probe once to validate connectivity and learn the dimensionality.
        try:
            probe = self._embed_one("dimension probe")
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                f"could not reach Ollama at {self.host} with model "
                f"'{self.model}': {exc}"
            ) from exc
        if not probe:
            raise RuntimeError(
                f"Ollama model '{self.model}' returned an empty embedding"
            )
        self.dim = len(probe)

    def _embed_one(self, text: str) -> List[float]:
        import json
        import urllib.request

        payload = json.dumps(
            {"model": self.model, "prompt": text if text else " "}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [float(x) for x in data.get("embedding", [])]

    def embed(self, text: str) -> List[float]:
        return self._embed_one(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


def ollama_chat(
    prompt: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
    system: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """Send a one-shot prompt to a **local Ollama** chat model and return text.

    Standard-library only. Handy for giving agents a local LLM (e.g. to write
    richer reflections) without any cloud dependency::

        ollama pull llama3.2
        ollama_chat("Summarise the plan", model="llama3.2")

    Uses ``OLLAMA_CHAT_MODEL`` / ``OLLAMA_HOST`` env vars as defaults.
    """
    import json
    import urllib.request

    mdl = model or os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")
    base = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps(
        {"model": mdl, "messages": messages, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("message") or {}).get("content", "")


def get_embedder(prefer_openai: bool = True, prefer_ollama: bool = False) -> Embedder:
    """Return the best available embedder.

    Selection order:

    1. :class:`OllamaEmbedder` when ``prefer_ollama`` is true (or the
       ``USE_OLLAMA`` env var is set) and a local Ollama server is reachable.
    2. :class:`OpenAIEmbedder` when ``prefer_openai`` is true and both the
       ``openai`` package and ``OPENAI_API_KEY`` are available.
    3. :class:`LocalEmbedder` (always-available deterministic fallback).
    """
    if prefer_ollama or os.environ.get("USE_OLLAMA"):
        try:
            return OllamaEmbedder()
        except Exception:
            pass
    if prefer_openai and os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except Exception:
            pass
    return LocalEmbedder()


__all__ = [
    "Embedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "ollama_chat",
    "get_embedder",
]
