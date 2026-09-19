"""FastAPI application: auth, ticket submission, and ticket history."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status

from src import database
from src.auth import create_access_token, get_current_user, hash_password, verify_password
from src.decision import DecisionEngine, get_decision_engine
from src.models import (
    LoginRequest,
    RegisterRequest,
    TicketRequest,
    TicketResponse,
    TokenResponse,
    UserResponse,
)
from src.retrieval import get_knowledge_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    if os.getenv("WARM_KNOWLEDGE_BASE", "1") == "1":
        try:
            kb = get_knowledge_base()
            logger.info("Embedded %d policy chunks (%s)", len(kb.chunks), kb.backend)
        except Exception as exc:  # pragma: no cover - startup resilience
            logger.warning("Knowledge base warm-up skipped: %s", exc)
    yield


app = FastAPI(
    title="Ticket Triage Assistant",
    description="Policy-grounded support-ticket decisions (RAG + Gemini).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# --- auth ---------------------------------------------------------------

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> Any:
    email = payload.email.lower()
    try:
        user = database.create_user(email, hash_password(payload.password))
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    return UserResponse(id=user["id"], email=user["email"], created_at=user["created_at"])


@app.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> Any:
    user = database.get_user_by_email(payload.email.lower())
    # Same error for unknown email and wrong password: no account enumeration.
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(user["id"], user["email"])
    return TokenResponse(access_token=token, expires_in=expires_in)


@app.get("/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user)) -> Any:
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        created_at=current_user["created_at"],
    )


# --- tickets ------------------------------------------------------------

@app.post("/tickets", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
def create_ticket(
    payload: TicketRequest,
    current_user: dict = Depends(get_current_user),
    engine: DecisionEngine = Depends(get_decision_engine),
) -> Any:
    result = engine.decide(payload.message)
    decision = result.decision
    ticket = database.create_ticket_with_decision(
        user_id=current_user["id"],
        message=payload.message,
        action=decision.action.value,
        reason=decision.reason,
        confidence=decision.confidence,
        sources=decision.sources,
    )
    return TicketResponse(**ticket)


@app.get("/tickets", response_model=list[TicketResponse])
def list_tickets(current_user: dict = Depends(get_current_user)) -> Any:
    return [TicketResponse(**t) for t in database.list_tickets_for_user(current_user["id"])]


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: int, current_user: dict = Depends(get_current_user)) -> Any:
    ticket = database.get_ticket_for_user(ticket_id, current_user["id"])
    if ticket is None:
        # 404 for both "no such ticket" and "someone else's ticket": a 403 would
        # confirm the row exists and leak the ID space to other users.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return TicketResponse(**ticket)
