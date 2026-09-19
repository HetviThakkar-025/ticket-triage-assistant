"""Pydantic schemas shared by the API and the decision pipeline."""

from enum import Enum

from pydantic import BaseModel, EmailStr, Field, field_validator


class Action(str, Enum):
    """Closed vocabulary of triage actions derived from the policy knowledge base."""

    APPROVE_RETURN = "APPROVE_RETURN"
    REJECT_OPENED_ITEM = "REJECT_OPENED_ITEM"
    REJECT_FOOD_RETURN = "REJECT_FOOD_RETURN"
    REJECT_OUTSIDE_WINDOW = "REJECT_OUTSIDE_WINDOW"
    APPROVE_REFUND_OR_REPLACEMENT = "APPROVE_REFUND_OR_REPLACEMENT"
    REQUEST_PHOTOS = "REQUEST_PHOTOS"
    APPROVE_REPLACEMENT = "APPROVE_REPLACEMENT"
    REQUEST_DEFECT_EVIDENCE = "REQUEST_DEFECT_EVIDENCE"
    REPLACE_CORRECT_ITEM = "REPLACE_CORRECT_ITEM"
    WAIT_AND_TRACK = "WAIT_AND_TRACK"
    OPEN_SHIPPING_INVESTIGATION = "OPEN_SHIPPING_INVESTIGATION"
    OFFER_REPLACEMENT_OR_REFUND = "OFFER_REPLACEMENT_OR_REFUND"
    CANCEL_AND_REFUND = "CANCEL_AND_REFUND"
    CANNOT_CANCEL_AFTER_DISPATCH = "CANNOT_CANCEL_AFTER_DISPATCH"
    NEEDS_MORE_INFORMATION = "NEEDS_MORE_INFORMATION"


class LLMDecision(BaseModel):
    """Strict contract the model must satisfy before anything is persisted."""

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reason must not be blank")
        return cleaned

    @field_validator("sources")
    @classmethod
    def _dedupe_sources(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for item in value:
            name = str(item).strip()
            if name and name not in seen:
                seen.append(name)
        return seen


# --- auth ---------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: int
    email: str
    created_at: str


# --- tickets ------------------------------------------------------------

class TicketRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be blank")
        return cleaned


class TicketResponse(BaseModel):
    id: int
    message: str
    created_at: str
    action: Action
    confidence: float
    reason: str
    sources: list[str]
