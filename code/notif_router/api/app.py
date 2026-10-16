"""FastAPI application.

Endpoints
  GET  /health                              liveness + whether the agent or offline mode is active
  POST /v1/route                            route one message
  POST /v1/route/batch                      route up to 100 messages
  GET  /v1/decisions                        recent decisions (optionally for one user)
  GET  /v1/decisions/{message_id}           latest decision for a message
  POST /v1/decisions/{decision_id}/feedback user correction, stored for later tuning
  GET  /v1/stats                            decision counts and feedback volume
"""

import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader

from notif_router import __version__
from notif_router.api.models import (
    BatchRouteRequest,
    BatchRouteResponse,
    DecisionResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from notif_router.config import Settings, get_settings
from notif_router.logging_config import configure_logging
from notif_router.router import NotificationRouter
from notif_router.schemas import IncomingMessage

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(settings: Settings | None = None, router: NotificationRouter | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = router or NotificationRouter.from_settings(settings)
        yield

    app = FastAPI(
        title="notif-router",
        version=__version__,
        description="AI agent that decides whether a WhatsApp message should notify, digest or mute.",
        lifespan=lifespan,
    )

    def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
        if settings.api_key and not (provided and secrets.compare_digest(provided, settings.api_key)):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key header")

    def get_router(request: Request) -> NotificationRouter:
        return request.app.state.router

    protected = [Depends(require_api_key)]

    @app.get("/health", response_model=HealthResponse)
    def health(r: NotificationRouter = Depends(get_router)) -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, mode=r.mode)

    @app.post("/v1/route", response_model=DecisionResponse, dependencies=protected)
    async def route_message(message: IncomingMessage, r: NotificationRouter = Depends(get_router)):
        # The agent makes blocking network calls; keep the event loop free.
        decision_id, decision = await run_in_threadpool(r.route, message)
        return DecisionResponse(decision_id=decision_id, decision=decision)

    @app.post("/v1/route/batch", response_model=BatchRouteResponse, dependencies=protected)
    async def route_batch(body: BatchRouteRequest, r: NotificationRouter = Depends(get_router)):
        results = []
        for message in body.messages:
            decision_id, decision = await run_in_threadpool(r.route, message)
            results.append(DecisionResponse(decision_id=decision_id, decision=decision))
        return BatchRouteResponse(results=results)

    @app.get("/v1/decisions", response_model=list[DecisionResponse], dependencies=protected)
    def list_decisions(
        user_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        r: NotificationRouter = Depends(get_router),
    ):
        return [DecisionResponse(decision_id=i, decision=d) for i, d in r.repository.recent(limit, user_id)]

    @app.get("/v1/decisions/{message_id}", response_model=DecisionResponse, dependencies=protected)
    def get_decision(message_id: str, r: NotificationRouter = Depends(get_router)):
        found = r.repository.latest_for_message(message_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No decision for message '{message_id}'")
        return DecisionResponse(decision_id=found[0], decision=found[1])

    @app.post("/v1/decisions/{decision_id}/feedback", response_model=FeedbackResponse,
              status_code=status.HTTP_201_CREATED, dependencies=protected)
    def add_feedback(decision_id: int, body: FeedbackRequest, r: NotificationRouter = Depends(get_router)):
        if body.correct_action is None and body.correct_message_type is None and not body.comment:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Feedback is empty")
        feedback_id = r.repository.add_feedback(
            decision_id,
            body.correct_action.value if body.correct_action else None,
            body.correct_message_type.value if body.correct_message_type else None,
            body.comment,
        )
        if feedback_id is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Decision {decision_id} not found")
        return FeedbackResponse(feedback_id=feedback_id, decision_id=decision_id)

    @app.get("/v1/stats", dependencies=protected)
    def stats(r: NotificationRouter = Depends(get_router)) -> dict:
        return {"mode": r.mode, **r.repository.stats()}

    return app
