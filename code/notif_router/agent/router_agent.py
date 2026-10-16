"""The routing agent: a hand-rolled tool-calling loop on top of chat completions.

The model investigates with read-only tools, then must call `submit_decision`.
Submissions are validated (allowed values, evidence it actually retrieved,
guardrail-locked action); a bad submission is returned to the model as a tool
error so it can correct itself instead of failing the request.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from notif_router.agent.llm import LLMClient
from notif_router.agent.prompts import SYSTEM_PROMPT, build_task_prompt
from notif_router.agent.tools import SUBMIT_DECISION, TOOL_SCHEMAS, Toolbox
from notif_router.context import ContextStore
from notif_router.schemas import Action, DecisionSource, IncomingMessage, RoutingDecision

if TYPE_CHECKING:
    from notif_router.guardrails import GuardrailVerdict
    from notif_router.media import MediaAnalyzer

logger = logging.getLogger(__name__)

MAX_REASON_LENGTH = 400


class AgentError(RuntimeError):
    """The agent could not produce a valid decision within its step budget."""


class RouterAgent:
    def __init__(self, llm: LLMClient, store: ContextStore, media: MediaAnalyzer | None, max_steps: int = 6):
        self.llm = llm
        self.store = store
        self.media = media
        self.max_steps = max_steps

    def route(self, message: IncomingMessage, locked: GuardrailVerdict | None = None) -> RoutingDecision:
        toolbox = Toolbox(message=message, store=self.store, media=self.media)
        prefetched = self._prefetch_context(toolbox)
        task = build_task_prompt(message, locked.reason if locked else None)
        context = json.dumps(prefetched, default=str, separators=(",", ":"))[:8000]
        conversation: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{task}\n\n<gathered_context>\n{context}\n</gathered_context>"},
        ]
        trace: list[dict] = [{"step": 0, "prefetched": sorted(prefetched)}]
        tokens = 0

        for step in range(1, self.max_steps + 1):
            final_step = step == self.max_steps
            tool_choice = (
                {"type": "function", "function": {"name": SUBMIT_DECISION}} if final_step else "required"
            )
            turn = self.llm.chat(conversation, TOOL_SCHEMAS, tool_choice)
            tokens += turn.total_tokens
            conversation.append(turn.as_message())

            if not turn.tool_calls:
                conversation.append({"role": "user", "content": "Use the tools, then call submit_decision."})
                continue

            for call in turn.tool_calls:
                if call.name == SUBMIT_DECISION:
                    decision, error = self._validate_submission(call.arguments, message, toolbox, locked)
                    trace.append({"step": step, "tool": call.name, "ok": error is None})
                    if decision is not None:
                        decision.trace = [*trace, {"llm_calls": step, "tokens": tokens}]
                        return decision
                    result = {"error": error}
                else:
                    result = toolbox.execute(call.name, call.arguments)
                    trace.append({"step": step, "tool": call.name, "args": call.arguments[:200]})
                conversation.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str)[:6000],
                })

        raise AgentError(f"No valid decision for {message.message_id} after {self.max_steps} steps")

    @staticmethod
    def _prefetch_context(toolbox: Toolbox) -> dict:
        """Run the lookups the agent needs for almost every message up front.

        Each LLM round trip resends the whole conversation, so gathering the standard
        context in code (instead of 3-4 tool round trips) cuts tokens and latency roughly
        threefold. The tools stay available for follow-ups such as a different history query.
        """
        context = {
            "sender": toolbox.execute("get_sender_context", "{}"),
            "recipient": toolbox.execute("get_recipient_profile", "{}"),
        }
        if toolbox.message.media_type:
            context["media"] = toolbox.execute("analyze_media", "{}")
        # Search after media analysis so voice notes are matched on their transcript.
        context["similar_past_messages"] = toolbox.execute("search_message_history", "{}")
        return context

    @staticmethod
    def _validate_submission(
        raw_arguments: str,
        message: IncomingMessage,
        toolbox: Toolbox,
        locked: GuardrailVerdict | None,
    ) -> tuple[RoutingDecision | None, str | None]:
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            return None, "submit_decision arguments were not valid JSON."

        evidence = [str(e).strip() for e in arguments.get("evidence_message_ids") or [] if str(e).strip()]
        unknown = [e for e in evidence if e not in toolbox.seen_evidence_ids]
        if unknown:
            return None, (
                f"Evidence IDs {unknown} were not returned by search_message_history. "
                "Cite only IDs you retrieved, or search first."
            )
        if locked is not None and arguments.get("action") != locked.action.value:
            return None, f"A safety guardrail fixed the action to '{locked.action.value}'. Resubmit with it."

        reason = str(arguments.get("reason") or "").strip()
        if not reason:
            return None, "reason must be a non-empty explanation."

        try:
            decision = RoutingDecision(
                message_id=message.message_id,
                action=arguments.get("action"),
                message_type=arguments.get("message_type"),
                reason=reason[:MAX_REASON_LENGTH],
                confidence=round(float(arguments.get("confidence", 0.5)), 2),
                evidence_message_ids=evidence[:5],
                source=DecisionSource.AGENT,
            )
        except (ValidationError, TypeError, ValueError) as exc:
            return None, f"Invalid decision: {exc}"

        if decision.action is Action.MUTE and locked is not None and locked.confidence > decision.confidence:
            decision.confidence = locked.confidence
        return decision, None
