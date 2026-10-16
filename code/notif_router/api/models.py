"""Request and response bodies for the REST API."""

from pydantic import BaseModel, Field

from notif_router.schemas import Action, IncomingMessage, MessageType, RoutingDecision


class DecisionResponse(BaseModel):
    decision_id: int
    decision: RoutingDecision


class BatchRouteRequest(BaseModel):
    messages: list[IncomingMessage] = Field(min_length=1, max_length=100)


class BatchRouteResponse(BaseModel):
    results: list[DecisionResponse]


class FeedbackRequest(BaseModel):
    correct_action: Action | None = None
    correct_message_type: MessageType | None = None
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    feedback_id: int
    decision_id: int


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str
