"""
providers.py - the "engines" behind the chatbot.

The chatbot needs two engines:
  1. an EMBEDDER  (turns text into numbers so we can search by meaning)
  2. an LLM       (writes the final answer)

Each engine has several options, so the project runs with OR without a Gemini key:

  EMBEDDER   gemini  - Gemini API (needs key)
             chroma  - free MiniLM model that ChromaDB downloads once (~80 MB, no key)   <- default without key
             local   - tiny built-in fallback (no download at all)

  LLM        gemini  - Gemini API (needs key)
             ollama  - free open-source model running on YOUR computer (e.g. Qwen), no key
             none    - no AI: we show the best-matching part of the notes ("notes-only" mode)

Everything is chosen automatically. You can force a choice in the .env file
(EMBEDDING_BACKEND=..., LLM_BACKEND=...).
"""
import hashlib
import math
import os
import re
from collections import Counter

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)  # values in .env win over anything already set in Windows


def _clean(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


GEMINI_KEY = _clean(os.getenv("GEMINI_API_KEY"))
HAS_KEY = bool(GEMINI_KEY) and not GEMINI_KEY.lower().startswith("paste")  # ignore the placeholder

GEMINI_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# ------------------------------------------------------------------ Gemini client
_gemini_client = None


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        if not HAS_KEY:
            raise RuntimeError("This option needs GEMINI_API_KEY in your .env file.")
        from google import genai

        _gemini_client = genai.Client(api_key=GEMINI_KEY)
    return _gemini_client


# ================================================================== EMBEDDERS
_embedder = _clean(os.getenv("EMBEDDING_BACKEND")).lower() or ("gemini" if HAS_KEY else "chroma")

_STOP = {"a", "an", "the", "is", "are", "do", "does", "i", "my", "me", "you", "can", "to", "of", "in",
         "on", "for", "and", "or", "it", "how", "what", "when", "which", "will", "be", "at", "with",
         "should", "take", "long", "usually", "much", "many"}


_LOCAL_DIM = 4096


def _local_embed(texts: list[str]) -> list[list[float]]:
    """Tiny bag-of-words embedding (no download, works anywhere, less smart than a neural model)."""
    out = []
    for text in texts:
        words = [w[:6] if len(w) > 6 else (w[:-1] if len(w) > 3 and w.endswith("s") else w)
                 for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
        counts = Counter(words + [f"{a}_{b}" for a, b in zip(words, words[1:])])
        vec = [0.0] * _LOCAL_DIM
        for feature, n in counts.items():
            vec[int(hashlib.md5(feature.encode()).hexdigest(), 16) % _LOCAL_DIM] += 1 + math.log(n)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


_chroma_ef = None


def _chroma_embed(texts: list[str]) -> list[list[float]]:
    """Free neural embeddings (MiniLM) - ChromaDB downloads the model the first time."""
    global _chroma_ef
    if _chroma_ef is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _chroma_ef = DefaultEmbeddingFunction()
    return [[float(x) for x in vec] for vec in _chroma_ef(texts)]


def _gemini_embed(texts: list[str], task_type: str) -> list[list[float]]:
    from google.genai import types

    result = _gemini().models.embed_content(
        model=GEMINI_EMBED_MODEL, contents=texts, config=types.EmbedContentConfig(task_type=task_type)
    )
    return [e.values for e in result.embeddings]


def embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    global _embedder
    if _embedder == "gemini":
        return _gemini_embed(texts, task_type)
    if _embedder == "chroma":
        try:
            return _chroma_embed(texts)
        except Exception as e:  # e.g. no internet for the one-time model download
            print(f"WARNING: could not load the free MiniLM model ({type(e).__name__}). "
                  f"Using the simple 'local' embeddings instead.")
            _embedder = "local"
    return _local_embed(texts)


def embedder_name() -> str:
    return _embedder


def min_score() -> float:
    """How similar a note must be to count as 'found' in notes-only mode (override with MIN_SCORE)."""
    default = {"gemini": 0.0, "chroma": 0.25, "local": 0.075}.get(_embedder, 0.15)
    return float(os.getenv("MIN_SCORE", default))


# ================================================================== LLMs
_llm = None


def _ollama_ready() -> bool:
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.5).json()
    except Exception:
        return False
    names = [m.get("name", "") for m in tags.get("models", [])]
    if any(n == OLLAMA_MODEL or n == f"{OLLAMA_MODEL}:latest" for n in names):
        return True
    print(f"Ollama is running but model '{OLLAMA_MODEL}' is not downloaded. Run:  ollama pull {OLLAMA_MODEL}")
    return False


def llm_backend() -> str:
    """Decide once: gemini / ollama / none."""
    global _llm
    if _llm is None:
        forced = _clean(os.getenv("LLM_BACKEND")).lower()
        if forced in {"gemini", "ollama", "none"}:
            _llm = forced
        elif HAS_KEY:
            _llm = "gemini"
        elif _ollama_ready():
            _llm = "ollama"
        else:
            _llm = "none"
    return _llm


def generate_json(system: str, user: str, schema_model) -> str:
    """Ask the LLM for an answer that matches `schema_model` (a Pydantic class). Returns JSON text."""
    backend = llm_backend()
    if backend == "gemini":
        from google.genai import types

        resp = _gemini().models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema_model,
                temperature=0,
            ),
        )
        return resp.text
    if backend == "ollama":
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "format": schema_model.model_json_schema(),  # forces valid JSON in our shape
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=300,  # the first call loads the model into memory and can be slow
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    raise RuntimeError("No LLM backend is active (notes-only mode).")
