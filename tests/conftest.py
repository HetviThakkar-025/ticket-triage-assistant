"""Shared fixtures: isolated database, and a stubbed decision engine.

The unit tests never call Gemini -- the decision engine is replaced through
FastAPI's dependency_overrides, so the suite runs offline and for free.
Live model behaviour is covered separately by tests/evaluate.py.
"""

import os
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("JWT_SECRET", "test-secret-not-used-in-production")
os.environ["EMBEDDER_BACKEND"] = "tfidf"
os.environ["WARM_KNOWLEDGE_BASE"] = "0"

from src.decision import DecisionEngine, DecisionResult, get_decision_engine  # noqa: E402
from src.models import Action, LLMDecision  # noqa: E402


class StubDecisionEngine(DecisionEngine):
    """Deterministic stand-in for the Gemini pipeline."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def decide(self, message: str) -> DecisionResult:
        self.calls.append(message)
        return DecisionResult(
            decision=LLMDecision(
                action=Action.NEEDS_MORE_INFORMATION,
                confidence=0.4,
                reason="Stubbed decision for tests.",
                sources=["returns.md"],
            ),
            retrieved=[],
            backend="stub",
        )


@pytest.fixture()
def stub_engine() -> StubDecisionEngine:
    return StubDecisionEngine()


@pytest.fixture()
def client(tmp_path, monkeypatch, stub_engine) -> Iterator[TestClient]:
    monkeypatch.setenv("TICKET_DB_PATH", str(tmp_path / "test.db"))
    from src.api import app

    app.dependency_overrides[get_decision_engine] = lambda: stub_engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def register_and_login(client: TestClient, email: str, password: str = "password123") -> str:
    """Register a user and return their bearer token."""
    response = client.post("/register", json={"email": email, "password": password})
    assert response.status_code == 201, response.text
    response = client.post("/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
