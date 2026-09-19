"""Manual RAG over the policy knowledge base.

Chunking, embedding, and cosine search are implemented directly on numpy --
no vector database and no orchestration framework. The embedding backend is
pluggable so the pipeline works with Gemini embeddings, a local
sentence-transformers model, or an offline TF-IDF fallback.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.config import CACHE_DIR, KNOWLEDGE_BASE_DIR, get_settings

logger = logging.getLogger(__name__)

_POINT_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
_TOKEN_RE = re.compile(r"[a-z0-9₹]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "for", "from", "has", "have", "if", "in", "is", "it", "its", "may", "must",
    "not", "of", "on", "or", "that", "the", "their", "then", "there", "this",
    "to", "under", "was", "were", "what", "when", "which", "with", "within",
}


@dataclass(frozen=True)
class Chunk:
    """One atomic policy point plus the provenance needed to cite it."""

    text: str
    source: str
    title: str
    point: int


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float


# --- chunking -----------------------------------------------------------

def load_chunks(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> list[Chunk]:
    """Chunk each policy file by numbered policy point, not by fixed width.

    The knowledge base is authored as numbered rules, so one rule is the
    natural retrieval unit: self-contained, and citable on its own.
    """
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        title = path.stem.replace("_", " ").title()
        body_lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                continue
            body_lines.append(line)

        for line in body_lines:
            match = _POINT_RE.match(line)
            if not match:
                continue
            number, text = int(match.group(1)), match.group(2).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    text=f"{title} - point {number}: {text}",
                    source=path.name,
                    title=title,
                    point=number,
                )
            )

    if not chunks:
        raise RuntimeError(f"No policy chunks found in {kb_dir}")
    return chunks


def chunks_fingerprint(chunks: Sequence[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.source.encode())
        digest.update(chunk.text.encode())
    return digest.hexdigest()[:16]


# --- embedding backends -------------------------------------------------

class Embedder:
    """Interface every embedding backend implements."""

    name: str = "base"
    cacheable: bool = True

    def fit(self, documents: Sequence[str]) -> None:  # optional hook
        return None

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _stem(token: str) -> str:
    """Very small suffix stripper -- enough to unify delivered/delivery/deliver."""
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 4 and token.endswith("y"):
        token = token[:-1]
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [
        _stem(token)
        for token in _TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


class TfidfEmbedder(Embedder):
    """Offline lexical fallback: TF-IDF vectors compared with cosine similarity.

    The knowledge base is ~30 short, vocabulary-dense rules, so a lexical
    representation retrieves them reliably without any network or model
    download. Used when no API key is configured and for the test suite.
    """

    name = "tfidf"
    cacheable = False

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0)

    def fit(self, documents: Sequence[str]) -> None:
        tokenized = [tokenize(doc) for doc in documents]
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                vocab.setdefault(token, len(vocab))

        doc_freq = np.zeros(len(vocab))
        for tokens in tokenized:
            for token in set(tokens):
                doc_freq[vocab[token]] += 1

        n_docs = max(len(documents), 1)
        self._vocab = vocab
        self._idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1.0

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self._vocab))
        for token in tokenize(text):
            index = self._vocab.get(token)
            if index is not None:
                vector[index] += 1.0
        return vector * self._idf

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        if not self._vocab:
            self.fit(documents)
        return _l2_normalize(np.vstack([self._vectorize(doc) for doc in documents]))

    def embed_query(self, query: str) -> np.ndarray:
        return _l2_normalize(self._vectorize(query))


class GeminiEmbedder(Embedder):
    """Gemini embedding API -- no local model download, genuinely semantic."""

    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model
        self.name = model.split("/")[-1]

    def _embed(self, content, task_type: str) -> np.ndarray:
        response = self._genai.embed_content(
            model=self._model, content=content, task_type=task_type
        )
        embedding = response["embedding"]
        return _l2_normalize(np.asarray(embedding, dtype=np.float32))

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        return self._embed(list(documents), "retrieval_document")

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query, "retrieval_query")


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers model, used when the package is installed."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = f"st-{model_name}"

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        return _l2_normalize(
            np.asarray(self._model.encode(list(documents)), dtype=np.float32)
        )

    def embed_query(self, query: str) -> np.ndarray:
        return _l2_normalize(np.asarray(self._model.encode(query), dtype=np.float32))


def build_embedder(backend: Optional[str] = None) -> Embedder:
    """Select a backend, degrading gracefully rather than failing the app."""
    settings = get_settings()
    backend = (backend or settings.embedder_backend).lower()

    def try_gemini() -> Optional[Embedder]:
        if not settings.gemini_api_key:
            return None
        try:
            return GeminiEmbedder(settings.gemini_api_key, settings.gemini_embedding_model)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Gemini embedder unavailable (%s)", exc)
            return None

    def try_sentence_transformers() -> Optional[Embedder]:
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("sentence-transformers unavailable (%s)", exc)
            return None

    if backend == "gemini":
        return try_gemini() or TfidfEmbedder()
    if backend in {"sentence-transformers", "st"}:
        return try_sentence_transformers() or TfidfEmbedder()
    if backend == "tfidf":
        return TfidfEmbedder()

    # auto: prefer Gemini, then a local model, then the lexical fallback.
    return try_gemini() or try_sentence_transformers() or TfidfEmbedder()


# --- index --------------------------------------------------------------

class KnowledgeBase:
    """Chunked policy corpus with its embedding matrix and cosine search."""

    def __init__(
        self,
        chunks: Optional[list[Chunk]] = None,
        embedder: Optional[Embedder] = None,
        use_cache: bool = True,
    ) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        self.embedder = embedder or build_embedder()
        self._use_cache = use_cache
        self.matrix: np.ndarray = np.zeros((0, 0))

    @property
    def backend(self) -> str:
        return self.embedder.name

    def _cache_path(self) -> Path:
        key = f"{self.embedder.name}-{chunks_fingerprint(self.chunks)}"
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", key)
        return CACHE_DIR / f"kb-{safe}.npz"

    def build(self) -> "KnowledgeBase":
        """Embed every chunk once, reusing a disk cache when one is valid."""
        texts = [chunk.text for chunk in self.chunks]
        cache_path = self._cache_path()

        if self._use_cache and self.embedder.cacheable and cache_path.exists():
            try:
                cached = np.load(cache_path)
                matrix = cached["matrix"]
                if matrix.shape[0] == len(texts):
                    self.matrix = matrix
                    logger.info("Loaded KB embeddings from cache %s", cache_path.name)
                    return self
            except Exception as exc:  # pragma: no cover - corrupt cache
                logger.warning("Ignoring unreadable embedding cache: %s", exc)

        self.embedder.fit(texts)
        self.matrix = self.embedder.embed_documents(texts)

        if self._use_cache and self.embedder.cacheable:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, matrix=self.matrix)
            except OSError as exc:  # pragma: no cover - read-only fs
                logger.warning("Could not write embedding cache: %s", exc)
        return self

    def search(self, query: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        """Top-k chunks by cosine similarity (vectors are pre-normalized)."""
        if self.matrix.size == 0:
            self.build()
        k = k or get_settings().top_k
        k = max(1, min(k, len(self.chunks)))

        query_vector = self.embedder.embed_query(query)
        scores = self.matrix @ query_vector
        top_indices = np.argsort(-scores)[:k]
        return [
            RetrievedChunk(
                text=self.chunks[i].text,
                source=self.chunks[i].source,
                score=float(scores[i]),
            )
            for i in top_indices
        ]


_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Process-wide singleton; built once at API startup."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase().build()
        logger.info(
            "Knowledge base ready: %d chunks, backend=%s",
            len(_knowledge_base.chunks),
            _knowledge_base.backend,
        )
    return _knowledge_base


def reset_knowledge_base() -> None:
    """Test hook."""
    global _knowledge_base
    _knowledge_base = None
