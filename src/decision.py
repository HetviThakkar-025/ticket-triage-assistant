"""Policy-grounded decision pipeline: retrieve -> prompt Gemini -> validate."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from pydantic import ValidationError

from src.config import get_settings
from src.models import Action, LLMDecision
from src.retrieval import KnowledgeBase, RetrievedChunk, get_knowledge_base

logger = logging.getLogger(__name__)

# What each action means, described without restating the numeric policy
# thresholds -- those must come from the retrieved excerpts, not from memory.
ACTION_GUIDE = """
- APPROVE_RETURN: change-of-mind return that the returns policy allows.
- REJECT_OPENED_ITEM: change-of-mind return refused because the item was opened.
- REJECT_FOOD_RETURN: change-of-mind return refused because the item is a food product.
- REJECT_OUTSIDE_WINDOW: the report falls outside the time window the relevant policy allows.
- APPROVE_REFUND_OR_REPLACEMENT: damaged order that qualifies without further evidence.
- REQUEST_PHOTOS: damaged order where the policy requires photographs before approval.
- APPROVE_REPLACEMENT: functional defect that qualifies for replacement without further evidence.
- REQUEST_DEFECT_EVIDENCE: functional defect where the policy requires evidence before approval.
- REPLACE_CORRECT_ITEM: wrong item or wrong flavour that qualifies for a replacement.
- WAIT_AND_TRACK: delivery is late but still inside the window where the customer should keep tracking.
- OPEN_SHIPPING_INVESTIGATION: delivery is late enough that a shipping investigation is required.
- OFFER_REPLACEMENT_OR_REFUND: delivery is so late that a replacement or refund should be offered.
- CANCEL_AND_REFUND: the order can still be cancelled for a full refund.
- CANNOT_CANCEL_AFTER_DISPATCH: cancellation is refused because the order already shipped.
- NEEDS_MORE_INFORMATION: the message lacks a fact the applicable policy needs.
""".strip()

SYSTEM_PROMPT = f"""You are a support-ticket triage engine for an e-commerce company.
You decide the correct next action for a customer's ticket using ONLY the company
policy excerpts supplied with each request.

HARD RULES
1. Decide using the supplied policy excerpts only. Never invent, recall, or round a
   threshold, time window, or rupee amount that is not written in the excerpts.
2. Extract the following from the customer's message before deciding. Each one is
   either stated in the message or UNKNOWN -- never assume a default:
     - order value in rupees
     - days since delivery
     - days since dispatch
     - product type (food or non-food)
     - opened or unopened
     - order status (not yet dispatched / dispatched / delivered)
   Relative wording counts as stated: "yesterday" is 1 day, "today" is 0 days.
3. Identify which policy governs the ticket, then apply that policy's rules in order.
4. If a fact the governing policy needs is UNKNOWN and you cannot determine
   eligibility without it, the action is NEEDS_MORE_INFORMATION. Never guess an
   eligibility outcome that the message does not support.

CLASSIFICATION GUIDANCE
- A broken, cracked, crushed, or cosmetically damaged item is a damaged-goods issue.
- An item that powers on but malfunctions, or stops working, is a defective-product issue.
- A damaged or defective item is NOT a change-of-mind return, even if the customer
  uses the word "return".
- Receiving a different item, variant, or flavour than ordered is a wrong-item issue.
- A cancellation request is governed by the cancellation policy, which depends on
  whether the order has been dispatched.
- "Has not arrived yet" is a shipping/delivery issue measured from the dispatch date.

ALLOWED ACTIONS (return exactly one of these strings)
{ACTION_GUIDE}

CONFIDENCE
- 0.85-1.0: every fact the governing policy needs is present and one rule applies directly.
- 0.6-0.84: the governing policy is clear but some wording had to be interpreted.
- 0.0-0.5: NEEDS_MORE_INFORMATION, or the governing policy is genuinely ambiguous.

OUTPUT
Return a single JSON object and nothing else -- no prose, no markdown fences:
{{"action": "<one allowed action>", "confidence": <number between 0 and 1>,
  "reason": "<1-3 sentences citing the specific policy point applied>",
  "sources": ["<policy filename from the excerpts>"]}}
"sources" must contain only filenames that appear in the POLICY EXCERPTS block."""

FALLBACK_REASON = (
    "The automated policy check could not produce a validated decision for this "
    "ticket, so it has been routed for human review."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_RETRY_DELAY_RE = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)", re.IGNORECASE)
MAX_BACKOFF_SECONDS = 30.0


@dataclass
class DecisionResult:
    decision: LLMDecision
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    backend: str = "unknown"


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n".join(f"[{chunk.source}] {chunk.text}" for chunk in chunks)


def build_prompt(message: str, chunks: Sequence[RetrievedChunk]) -> str:
    return f"""{SYSTEM_PROMPT}

POLICY EXCERPTS
{format_context(chunks)}

CUSTOMER TICKET
\"\"\"{message}\"\"\"

Extract the facts, apply the governing policy, then output the JSON object."""


def fallback_decision(reason: str = FALLBACK_REASON) -> LLMDecision:
    return LLMDecision(
        action=Action.NEEDS_MORE_INFORMATION,
        confidence=0.0,
        reason=reason,
        sources=[],
    )


def parse_decision(raw: str, allowed_sources: Sequence[str]) -> LLMDecision:
    """Parse and validate model output, raising ValueError on anything malformed."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise ValueError("response contained no JSON object")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"response was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("response JSON was not an object")

    try:
        decision = LLMDecision.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"response did not match the required schema: {exc}") from exc

    # Keep citations honest: drop anything that was not actually retrieved.
    permitted = set(allowed_sources)
    decision.sources = [s for s in decision.sources if s in permitted]
    return decision


class DecisionEngine:
    """Interface the API depends on (overridden in tests with a stub)."""

    def decide(self, message: str) -> DecisionResult:
        raise NotImplementedError


class GeminiDecisionEngine(DecisionEngine):
    """Retrieval-augmented Gemini call with one repair retry and a safe fallback."""

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        model_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        self._genai = genai
        self._model_name = model_name or settings.gemini_model
        self._model = genai.GenerativeModel(self._model_name)
        self._kb = knowledge_base or get_knowledge_base()
        self._retry_delay_seconds = 2.0

    def _backoff_for(self, error_text: str) -> float:
        match = _RETRY_DELAY_RE.search(error_text)
        if match:
            return min(float(match.group(1)) + 1.0, MAX_BACKOFF_SECONDS)
        return self._retry_delay_seconds

    def _generate(self, prompt: str) -> str:
        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "max_output_tokens": 2048,
            },
        )
        return (response.text or "").strip()

    def decide(self, message: str) -> DecisionResult:
        chunks = self._kb.search(message)
        allowed_sources = [chunk.source for chunk in chunks]
        prompt = build_prompt(message, chunks)

        last_error = ""
        for attempt in (1, 2):
            try:
                raw = self._generate(prompt)
                decision = parse_decision(raw, allowed_sources)
                if not decision.sources:
                    # A grounded decision should cite something; fall back to the
                    # files that were actually retrieved rather than citing nothing.
                    decision.sources = sorted(set(allowed_sources[:3]))
                return DecisionResult(decision, list(chunks), self._model_name)
            except ValueError as exc:
                last_error = str(exc)
                logger.warning("Malformed model output on attempt %d: %s", attempt, exc)
                prompt = (
                    f"{prompt}\n\nYour previous response was rejected: {last_error}\n"
                    "Respond again with ONLY the JSON object described above."
                )
            except Exception as exc:  # network / API failure (e.g. rate limit)
                last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Gemini call failed on attempt %d: %s", attempt, last_error)
                # Back off before retrying: an immediate second call would hit
                # the same rate limit and waste the retry. Gemini reports a
                # suggested retry_delay on 429s -- honour it when present.
                if attempt == 1:
                    time.sleep(self._backoff_for(last_error))

        logger.error("Falling back to NEEDS_MORE_INFORMATION (%s)", last_error)
        # Keep a short hint in the stored reason so a fallback is distinguishable
        # from a genuine "insufficient information" verdict; the full error is
        # in the logs rather than in the customer-facing text.
        hint = last_error.splitlines()[0][:160] if last_error else "unknown error"
        return DecisionResult(
            fallback_decision(f"{FALLBACK_REASON} (details: {hint})"),
            list(chunks),
            self._model_name,
        )


_engine: Optional[DecisionEngine] = None


def get_decision_engine() -> DecisionEngine:
    """FastAPI dependency; overridden in tests via app.dependency_overrides."""
    global _engine
    if _engine is None:
        _engine = GeminiDecisionEngine()
    return _engine


def reset_decision_engine() -> None:
    global _engine
    _engine = None
