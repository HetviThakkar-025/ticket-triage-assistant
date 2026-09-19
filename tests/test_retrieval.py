"""RAG sanity checks: chunking, cosine search, and citation provenance."""

import numpy as np
import pytest

from src.decision import build_prompt, fallback_decision, parse_decision
from src.models import Action
from src.retrieval import KnowledgeBase, TfidfEmbedder, load_chunks


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    # The lexical backend keeps this test offline and deterministic.
    return KnowledgeBase(embedder=TfidfEmbedder(), use_cache=False).build()


def test_chunks_are_one_policy_point_each():
    chunks = load_chunks()
    assert len(chunks) >= 24
    sources = {chunk.source for chunk in chunks}
    assert sources == {
        "cancellations.md",
        "damaged_goods.md",
        "defective_products.md",
        "returns.md",
        "shipping.md",
        "wrong_item.md",
    }
    # Every chunk carries its policy title and point number for citation.
    assert all(" - point " in chunk.text for chunk in chunks)
    assert all(chunk.point > 0 for chunk in chunks)


def test_embeddings_are_normalised(kb):
    norms = np.linalg.norm(kb.matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert kb.matrix.shape[0] == len(kb.chunks)


@pytest.mark.parametrize(
    "query,expected_source",
    [
        ("My parcel has still not arrived and it was dispatched 9 days ago.", "shipping.md"),
        ("My order arrived damaged yesterday.", "damaged_goods.md"),
        ("I ordered strawberry but received chocolate.", "wrong_item.md"),
        ("I changed my mind about this unopened product.", "returns.md"),
        ("I want to cancel my order before it ships.", "cancellations.md"),
        ("The device stopped working after two days.", "defective_products.md"),
    ],
)
def test_retrieval_surfaces_the_governing_policy(kb, query, expected_source):
    results = kb.search(query, k=8)
    assert results[0].score > 0
    assert expected_source in {result.source for result in results}


def test_results_are_ranked_by_descending_score(kb):
    results = kb.search("damaged order worth 5000 rupees", k=5)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_prompt_includes_ticket_and_cited_excerpts(kb):
    message = "My parcel was dispatched 9 days ago and has not arrived."
    prompt = build_prompt(message, kb.search(message, k=4))
    assert message in prompt
    assert "[shipping.md]" in prompt
    assert "NEEDS_MORE_INFORMATION" in prompt


def test_parse_decision_accepts_fenced_json_and_filters_sources():
    raw = """```json
    {"action": "APPROVE_RETURN", "confidence": 0.9, "reason": "Returns policy point 1.",
     "sources": ["returns.md", "hallucinated.md"]}
    ```"""
    decision = parse_decision(raw, allowed_sources=["returns.md"])
    assert decision.action is Action.APPROVE_RETURN
    assert decision.sources == ["returns.md"]


@pytest.mark.parametrize(
    "raw",
    [
        "I think you should approve the return.",
        '{"action": "SEND_FLOWERS", "confidence": 0.9, "reason": "x", "sources": []}',
        '{"action": "APPROVE_RETURN", "confidence": 7, "reason": "x", "sources": []}',
        '{"action": "APPROVE_RETURN", "confidence": 0.5, "sources": []}',
    ],
)
def test_parse_decision_rejects_malformed_output(raw):
    with pytest.raises(ValueError):
        parse_decision(raw, allowed_sources=["returns.md"])


def test_fallback_is_a_safe_needs_more_information():
    decision = fallback_decision()
    assert decision.action is Action.NEEDS_MORE_INFORMATION
    assert decision.confidence == 0.0
