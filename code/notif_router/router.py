"""NotificationRouter: the single entry point used by both the API and the CLI.

Flow for each message:
  1. Safety guardrails. If one fully decides the message, stop here (no LLM cost).
  2. Decision cache. Identical input + model + prompt version reuses the stored result.
  3. Tool-calling agent (when an OpenAI key is configured).
  4. Heuristic fallback if the agent is disabled or fails, so a request never errors out.
Every decision is persisted with its source, latency and tool trace.
"""

import hashlib
import json
import logging
import time

import openai

from notif_router.agent import AgentError, LLMClient, OpenAIClient, RouterAgent
from notif_router.agent.llm import is_out_of_credit
from notif_router.agent.prompts import PROMPT_VERSION
from notif_router.config import Settings
from notif_router.context import ContextStore
from notif_router.guardrails import check_guardrails, enforce_consistency
from notif_router.heuristics import heuristic_decision
from notif_router.media import MediaAnalyzer
from notif_router.schemas import DecisionSource, IncomingMessage, RoutingDecision
from notif_router.storage import Database, DecisionRepository

logger = logging.getLogger(__name__)


def _input_hash(message: IncomingMessage, guardrail_rule: str | None) -> str:
    payload = message.model_dump(mode="json")
    payload["_guardrail"] = guardrail_rule
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class NotificationRouter:
    def __init__(
        self,
        store: ContextStore,
        repository: DecisionRepository,
        llm: LLMClient | None = None,
        agent_max_steps: int = 6,
    ):
        self.store = store
        self.repository = repository
        self.llm = llm
        self.media = MediaAnalyzer(store, repository, llm)
        self.agent = RouterAgent(llm, store, self.media, agent_max_steps) if llm else None

    @classmethod
    def from_settings(cls, settings: Settings, offline: bool = False) -> "NotificationRouter":
        database = Database(settings.database_url)
        database.create_tables()
        llm = None if offline or not settings.llm_enabled else OpenAIClient(settings)
        if llm is None:
            logger.warning("LLM disabled: running in offline heuristic mode")
        return cls(
            store=ContextStore.from_dataset(settings.dataset_dir),
            repository=DecisionRepository(database),
            llm=llm,
            agent_max_steps=settings.agent_max_steps,
        )

    def _model_chain(self) -> list[str]:
        return list(getattr(self.llm, "models", None) or [self.llm.model])

    def _disable_llm(self, why: str, log_context: dict) -> None:
        logger.error(f"{why}; switching to offline mode", extra=log_context)
        self.agent = None
        self.media.llm = None

    @property
    def mode(self) -> str:
        return "agent" if self.agent else "offline"

    def route(self, message: IncomingMessage, use_cache: bool = True) -> tuple[int, RoutingDecision]:
        started = time.perf_counter()
        verdict = check_guardrails(message, self.store)
        input_hash = _input_hash(message, verdict.rule if verdict else None)
        model_used: str | None = None

        log_context = {"context": {"message_id": message.message_id}}

        decision: RoutingDecision | None = None
        if verdict is not None and verdict.message_type is not None:
            decision = RoutingDecision(
                message_id=message.message_id,
                action=verdict.action,
                message_type=verdict.message_type,
                reason=verdict.reason,
                confidence=verdict.confidence,
                evidence_message_ids=verdict.evidence_message_ids,
                source=DecisionSource.GUARDRAIL,
                trace=[{"guardrail": verdict.rule}],
            )
        elif self.agent is not None:
            cached = self.repository.find_cached(input_hash, self._model_chain(), PROMPT_VERSION) if use_cache else None
            if cached is not None:
                decision, model_used = cached
            else:
                try:
                    decision = self.agent.route(message, locked=verdict)
                    model_used = self.llm.model  # read after the call: a fallback model may have taken over
                except AgentError:
                    logger.warning("agent gave no valid decision", extra=log_context)
                except (openai.AuthenticationError, openai.PermissionDeniedError):
                    # A bad key will not fix itself: stop calling OpenAI and serve heuristics.
                    self._disable_llm("OpenAI rejected the API key", log_context)
                except openai.RateLimitError as exc:
                    if is_out_of_credit(exc):
                        self._disable_llm("OpenAI account has no credit left", log_context)
                    else:  # the SDK already retried with backoff; serve this one from heuristics
                        logger.warning("OpenAI rate limit persisted after retries", extra=log_context)
                except Exception:
                    # Network/API failures must not fail the request; fall through to heuristics.
                    logger.exception("agent failed", extra=log_context)

        if decision is None:
            media_text = self.media.cached(message.media_id)
            decision = heuristic_decision(message, self.store, media_text=media_text, locked=verdict)

        decision = enforce_consistency(message, decision)

        latency_ms = int((time.perf_counter() - started) * 1000)
        decision_id = self.repository.save_decision(
            decision,
            user_id=message.user_id,
            input_hash=input_hash,
            model=model_used if decision.source is DecisionSource.AGENT else None,
            prompt_version=PROMPT_VERSION,
            latency_ms=latency_ms,
        )
        logger.info(
            "message routed",
            extra={"context": {
                "message_id": message.message_id, "action": decision.action.value,
                "type": decision.message_type.value, "source": decision.source.value,
                "latency_ms": latency_ms,
            }},
        )
        return decision_id, decision
