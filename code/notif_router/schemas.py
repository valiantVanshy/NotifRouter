"""Domain types shared by the router, the agent, the API and the CLI."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Action(str, Enum):
    NOTIFY = "notify"
    DIGEST = "digest"
    MUTE = "mute"


class MessageType(str, Enum):
    PERSONAL = "personal"
    URGENT = "urgent"
    EVENT = "event"
    PAYMENT = "payment"
    BUSINESS_UPDATE = "business_update"
    PROMOTION = "promotion"
    GREETING = "greeting"
    FORWARD = "forward"
    SPAM = "spam"
    SCAM = "scam"
    UNKNOWN = "unknown"


class ConversationType(str, Enum):
    PERSONAL = "personal"
    GROUP = "group"
    BUSINESS = "business"


class DecisionSource(str, Enum):
    GUARDRAIL = "guardrail"  # deterministic safety rule, no LLM involved
    AGENT = "agent"  # tool-calling LLM agent
    HEURISTIC = "heuristic"  # offline fallback when the LLM is unavailable


class IncomingMessage(BaseModel):
    # Shown as the ready-to-run example in the API docs (/docs).
    model_config = {"json_schema_extra": {"examples": [{
        "message_id": "demo_1",
        "user_id": "u_009",
        "conversation_type": "personal",
        "sender_user_id": "u_050",
        "created_at": "2026-07-20 10:00",
        "message_text": "Your wallet KYC failed. Reply with the OTP to keep payments active.",
        "forwarded_count": 0,
    }]}}

    message_id: str
    user_id: str
    conversation_type: ConversationType
    group_id: str | None = None
    business_id: str | None = None
    sender_user_id: str | None = None
    created_at: str = Field(description="Local time, format YYYY-MM-DD HH:MM")
    message_text: str | None = None
    media_type: str | None = Field(default=None, description="image, voice or empty")
    media_id: str | None = None
    forwarded_count: int = 0

    @field_validator("group_id", "business_id", "sender_user_id", "message_text",
                     "media_type", "media_id", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("forwarded_count", mode="before")
    @classmethod
    def _forwarded_default(cls, value):
        return 0 if value in (None, "") else value


class RoutingDecision(BaseModel):
    message_id: str
    action: Action
    message_type: MessageType
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_message_ids: list[str] = Field(default_factory=list)
    source: DecisionSource
    trace: list[dict] = Field(default_factory=list, description="Agent tool calls, for debugging")

    def evidence_as_csv(self) -> str:
        return ";".join(self.evidence_message_ids) if self.evidence_message_ids else "none"
