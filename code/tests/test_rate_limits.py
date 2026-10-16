"""Rate-limit handling in the OpenAI-compatible client, without any network calls."""

import httpx
import openai
import pytest

from notif_router.agent import llm as llm_module
from notif_router.agent.llm import OpenAIClient, is_daily_limit
from notif_router.config import Settings

DAILY = ("Rate limit reached for model `big` on tokens per day (TPD): Limit 200000, "
         "Used 199000, Requested 2000. Please try again in 2m27.7s.")
PER_MINUTE = "Rate limit reached for model `big` on tokens per minute (TPM). Please try again in 1.5s."


def _rate_limit(message: str, headers: dict | None = None) -> openai.RateLimitError:
    response = httpx.Response(429, headers=headers or {},
                              request=httpx.Request("POST", "https://api.groq.com"))
    return openai.RateLimitError(message, response=response, body={"error": {"message": message}})


@pytest.fixture
def client(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    settings = Settings(_env_file=None, llm_api_key="test", model="big", fallback_models="small",
                        llm_rate_limit_max_wait_seconds=30)
    c = OpenAIClient(settings)
    c.sleeps = sleeps
    return c


def _failing_then_ok(errors):
    errors = list(errors)

    def request():
        if errors:
            raise errors.pop(0)
        return "ok"
    return request


def test_daily_quota_switches_to_fallback_model_without_waiting(client):
    assert client.model == "big"
    assert client._call(_failing_then_ok([_rate_limit(DAILY)]), can_switch_model=True) == "ok"
    assert client.model == "small"
    assert client.sleeps == []


def test_per_minute_limit_waits_as_instructed(client):
    assert client._call(_failing_then_ok([_rate_limit(PER_MINUTE)]), can_switch_model=True) == "ok"
    assert client.model == "big"
    assert client.sleeps == [2.0]  # 1.5s from the message + 0.5s safety margin


def test_retry_after_header_takes_priority(client):
    error = _rate_limit(PER_MINUTE, headers={"retry-after": "4"})
    client._call(_failing_then_ok([error]))
    assert client.sleeps == [4.0]


def test_gives_up_when_wait_budget_is_exceeded(client):
    with pytest.raises(openai.RateLimitError):
        client._call(_failing_then_ok([_rate_limit(PER_MINUTE, {"retry-after": "20"})] * 2))


def test_media_calls_never_switch_the_chat_model(client):
    client._call(_failing_then_ok([_rate_limit(DAILY.replace("2m27.7s", "3s"))]))
    assert client.model == "big"


def test_daily_limit_detection():
    assert is_daily_limit(_rate_limit(DAILY))
    assert not is_daily_limit(_rate_limit(PER_MINUTE))
