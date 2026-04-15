"""Ollama API wrapper with retry logic, timeout handling, and embedding caching."""
import logging
import hashlib
from typing import List
import requests
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)
OLLAMA_BASE = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
REASON_MODEL = "llama3.2"
TIMEOUT = 60
_embed_cache: dict = {}


def _cache_key(text: str, model: str) -> str:
    return hashlib.md5(f"{model}:{text}".encode()).hexdigest()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    reraise=True,
)
def _post(endpoint: str, payload: dict) -> dict:
    resp = requests.post(f"{OLLAMA_BASE}{endpoint}", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _tfidf_fallback(text: str) -> np.ndarray:
    dim = 768
    vec = np.zeros(dim, dtype=np.float32)
    text = text.lower()
    for i in range(len(text) - 2):
        idx = hash(text[i:i+3]) % dim
        vec[idx] += 1.0
    return vec


def embed_text(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    key = _cache_key(text, model)
    if key in _embed_cache:
        return _embed_cache[key]
    try:
        data = _post("/api/embeddings", {"model": model, "prompt": text})
        vec = np.array(data["embedding"], dtype=np.float32)
    except Exception as e:
        logger.warning(f"Ollama embed failed ({e}), using fallback")
        vec = _tfidf_fallback(text)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    _embed_cache[key] = vec
    return vec


def embed_batch(texts: List[str], model: str = EMBED_MODEL) -> List[np.ndarray]:
    return [embed_text(t, model) for t in texts]


def generate(prompt: str, system: str = "", model: str = REASON_MODEL) -> str:
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    try:
        data = _post("/api/generate", payload)
        return data.get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama generate failed: {e}")
        return f"[LLM unavailable: {e}]"


def is_ollama_running() -> bool:
    try:
        return requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def list_models() -> List[str]:
    try:
        data = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5).json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
