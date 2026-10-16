import httpx
import openai
import pytest

from notif_router.agent import AgentError, RouterAgent
from notif_router.guardrails import check_guardrails
from notif_router.router import NotificationRouter
from notif_router.schemas import Action, DecisionSource, IncomingMessage, MessageType

TANKER = IncomingMessage(message_id="m_1", user_id="u_1", conversation_type="group",
                         group_id="g_family", sender_user_id="u_mom",
                         created_at="2026-07-10 10:00",
                         message_text="Tanker arriving in 20 mins, keep the valve open!")

GOOD_SUBMIT = ("submit_decision", {
    "action": "notify", "message_type": "urgent",
    "reason": "A trusted family admin sent a time-critical request; the user replied to a similar one.",
    "confidence": 0.9, "evidence_message_ids": ["h_1"],
})


def test_agent_decides_in_one_call_from_prefetched_context(store, scripted_llm):
    llm = scripted_llm([[GOOD_SUBMIT]])
    decision = RouterAgent(llm, store, media=None).route(TANKER)

    assert decision.action is Action.NOTIFY
    assert decision.message_type is MessageType.URGENT
    assert decision.evidence_message_ids == ["h_1"]  # retrieved by the prefetch, so it may be cited
    assert decision.source is DecisionSource.AGENT
    assert len(llm.requests) == 1

    first_prompt = llm.requests[0][1]["content"]
    assert "<gathered_context>" in first_prompt and "h_1" in first_prompt
    assert decision.trace[0]["prefetched"] == ["recipient", "sender", "similar_past_messages"]
    assert decision.trace[-1]["llm_calls"] == 1


def test_agent_can_still_call_tools_for_follow_ups(store, scripted_llm):
    llm = scripted_llm([
        [("search_message_history", {"query": "valve", "same_channel_only": True})],
        [GOOD_SUBMIT],
    ])
    decision = RouterAgent(llm, store, media=None).route(TANKER)
    assert [t.get("tool") for t in decision.trace if "tool" in t] == [
        "search_message_history", "submit_decision"]


def test_agent_must_cite_only_retrieved_evidence(store, scripted_llm):
    hallucinated = ("submit_decision", {**GOOD_SUBMIT[1], "evidence_message_ids": ["made_up"]})
    llm = scripted_llm([[hallucinated], [("search_message_history", {})], [GOOD_SUBMIT]])
    decision = RouterAgent(llm, store, media=None).route(TANKER)

    tool_errors = [m["content"] for m in llm.requests[1] if m["role"] == "tool"]
    assert "not returned by search_message_history" in tool_errors[0]
    assert decision.evidence_message_ids == ["h_1"]


def test_agent_cannot_override_guardrail_action(store, scripted_llm):
    message = TANKER.model_copy(update={"group_id": "g_muted", "message_text": "Selling a cycle"})
    locked = check_guardrails(message, store)
    notify = ("submit_decision", {"action": "notify", "message_type": "promotion",
                                  "reason": "x", "confidence": 0.7, "evidence_message_ids": []})
    mute = ("submit_decision", {"action": "mute", "message_type": "promotion",
                                "reason": "Marketplace listing in a group the user muted.",
                                "confidence": 0.8, "evidence_message_ids": []})
    decision = RouterAgent(scripted_llm([[notify], [mute]]), store, media=None).route(message, locked)
    assert decision.action is Action.MUTE
    assert decision.message_type is MessageType.PROMOTION


def test_agent_gives_up_after_step_budget(store, scripted_llm):
    llm = scripted_llm([[("get_recipient_profile", {})]] * 10)
    with pytest.raises(AgentError):
        RouterAgent(llm, store, media=None, max_steps=3).route(TANKER)


def test_router_falls_back_to_heuristics_when_agent_fails(store, repository, scripted_llm):
    router = NotificationRouter(store, repository, llm=scripted_llm([]), agent_max_steps=2)
    _, decision = router.route(TANKER)
    assert decision.source is DecisionSource.HEURISTIC
    assert decision.action is Action.NOTIFY  # user replied to the similar tanker message


def _openai_error(error_class, status_code, body=None):
    response = httpx.Response(status_code, request=httpx.Request("POST", "https://api.openai.com"))
    return error_class("OpenAI error", response=response, body=body)


@pytest.mark.parametrize("error", [
    _openai_error(openai.AuthenticationError, 401),
    _openai_error(openai.RateLimitError, 429, {"code": "credit_balance_exhausted", "type": "insufficient_quota"}),
], ids=["bad_key", "no_credit"])
def test_router_stops_calling_openai_after_fatal_error(store, repository, scripted_llm, error):
    class FailingLLM(scripted_llm):
        def chat(self, messages, tools, tool_choice):
            self.requests.append(messages)
            raise error

    llm = FailingLLM([])
    router = NotificationRouter(store, repository, llm=llm)
    _, first = router.route(TANKER)
    _, second = router.route(TANKER.model_copy(update={"message_id": "m_2"}))

    assert first.source is DecisionSource.HEURISTIC and second.source is DecisionSource.HEURISTIC
    assert len(llm.requests) == 1
    assert router.mode == "offline"


def test_transient_rate_limit_falls_back_but_keeps_agent(store, repository, scripted_llm):
    class ThrottledLLM(scripted_llm):
        def chat(self, messages, tools, tool_choice):
            raise _openai_error(openai.RateLimitError, 429, {"code": "rate_limit_exceeded"})

    router = NotificationRouter(store, repository, llm=ThrottledLLM([]))
    _, decision = router.route(TANKER)
    assert decision.source is DecisionSource.HEURISTIC
    assert router.mode == "agent"


def test_explicitly_not_urgent_message_never_notifies(store, repository, scripted_llm):
    calm = TANKER.model_copy(update={"message_text": "Reached home, nothing urgent, talk tomorrow."})
    llm = scripted_llm([[("submit_decision", {**GOOD_SUBMIT[1], "message_type": "personal"})]])
    _, decision = NotificationRouter(store, repository, llm=llm).route(calm)
    assert decision.action is Action.DIGEST
    assert decision.trace[-1] == {"consistency_check": "not_urgent_downgrade"}


def test_direct_mention_keeps_notify_even_if_calm(store, repository, scripted_llm):
    calm = TANKER.model_copy(update={"message_text": "@u_1 no rush, but can you check the bill?"})
    llm = scripted_llm([[("submit_decision", {**GOOD_SUBMIT[1], "message_type": "personal"})]])
    _, decision = NotificationRouter(store, repository, llm=llm).route(calm)
    assert decision.action is Action.NOTIFY


def test_router_short_circuits_scams_without_calling_llm(store, repository, scripted_llm):
    llm = scripted_llm([])
    router = NotificationRouter(store, repository, llm=llm)
    scam = TANKER.model_copy(update={"message_text": "Account blocked. Share OTP to verify now."})
    _, decision = router.route(scam)
    assert decision.source is DecisionSource.GUARDRAIL
    assert decision.message_type is MessageType.SCAM
    assert llm.requests == []


def test_cache_reuses_decisions_from_any_model_in_the_chain(store, repository, scripted_llm):
    first = scripted_llm([[GOOD_SUBMIT]])
    first.model = "fallback-model"
    NotificationRouter(store, repository, llm=first).route(TANKER)

    second = scripted_llm([])
    second.model, second.models = "main-model", ["main-model", "fallback-model"]
    decision_id, decision = NotificationRouter(store, repository, llm=second).route(TANKER)

    assert second.requests == []  # served from cache, no LLM call
    assert decision.source is DecisionSource.AGENT
    assert repository.db.engine.connect().exec_driver_sql(
        "select model from decisions where id = ?", (decision_id,)).scalar() == "fallback-model"


def test_router_reuses_cached_agent_decision(store, repository, scripted_llm):
    llm = scripted_llm([[("search_message_history", {})], [GOOD_SUBMIT]])
    router = NotificationRouter(store, repository, llm=llm)
    router.route(TANKER)
    calls_after_first = len(llm.requests)
    _, again = router.route(TANKER)
    assert len(llm.requests) == calls_after_first
    assert again.source is DecisionSource.AGENT
