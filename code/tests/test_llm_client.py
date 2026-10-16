import json

import httpx
import openai

from notif_router.agent.llm import recover_failed_tool_call
from notif_router.agent.tools import TOOL_SCHEMAS

DECISION = {"action": "digest", "message_type": "greeting", "reason": "Harmless greeting.",
            "confidence": 0.8, "evidence_message_ids": []}


def _tool_use_failed(failed_generation: str) -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
    body = {"error": {"code": "tool_use_failed", "failed_generation": failed_generation}}
    return openai.BadRequestError("tool_use_failed", response=response, body=body)


def test_recovers_decision_written_under_wrong_tool_name():
    error = _tool_use_failed(json.dumps({"name": "json", "arguments": DECISION}))
    turn = recover_failed_tool_call(error, TOOL_SCHEMAS)
    assert turn.tool_calls[0].name == "submit_decision"
    assert json.loads(turn.tool_calls[0].arguments) == DECISION


def test_recovers_decision_written_as_plain_json():
    turn = recover_failed_tool_call(_tool_use_failed(json.dumps(DECISION)), TOOL_SCHEMAS)
    assert turn.tool_calls[0].name == "submit_decision"


def test_repairs_half_spelled_confidence():
    payload = json.dumps({"name": "submit_decision", "arguments": DECISION})
    broken = payload.replace('"confidence": 0.8', '"confidence": 0. eight')
    assert "0. eight" in broken
    turn = recover_failed_tool_call(_tool_use_failed(broken), TOOL_SCHEMAS)
    assert json.loads(turn.tool_calls[0].arguments)["confidence"] == 0.8


def test_does_not_guess_when_payload_is_ambiguous_or_garbage():
    assert recover_failed_tool_call(_tool_use_failed("not json"), TOOL_SCHEMAS) is None
    assert recover_failed_tool_call(_tool_use_failed(json.dumps({"query": "x"})), TOOL_SCHEMAS) is None
