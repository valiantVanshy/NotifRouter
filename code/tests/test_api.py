import pytest
from fastapi.testclient import TestClient

from notif_router.api.app import create_app
from notif_router.config import Settings
from notif_router.router import NotificationRouter

MESSAGE = {"message_id": "m_api", "user_id": "u_1", "conversation_type": "group",
           "group_id": "g_family", "sender_user_id": "u_mom", "created_at": "2026-07-10 10:00",
           "message_text": "Tanker arriving in 20 mins, keep the valve open!"}


@pytest.fixture
def client(store, repository):
    settings = Settings(_env_file=None, api_key="secret", llm_api_key=None)
    app = create_app(settings, router=NotificationRouter(store, repository, llm=None))
    with TestClient(app) as test_client:
        yield test_client


AUTH = {"X-API-Key": "secret"}


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "offline"


def test_routes_require_api_key(client):
    assert client.post("/v1/route", json=MESSAGE).status_code == 401
    assert client.post("/v1/route", json=MESSAGE, headers={"X-API-Key": "wrong"}).status_code == 401


def test_route_then_fetch_and_give_feedback(client):
    routed = client.post("/v1/route", json=MESSAGE, headers=AUTH)
    assert routed.status_code == 200
    body = routed.json()
    assert body["decision"]["action"] in {"notify", "digest", "mute"}

    fetched = client.get("/v1/decisions/m_api", headers=AUTH)
    assert fetched.json()["decision_id"] == body["decision_id"]

    feedback = client.post(f"/v1/decisions/{body['decision_id']}/feedback", headers=AUTH,
                           json={"correct_action": "notify", "comment": "Needed this"})
    assert feedback.status_code == 201

    stats = client.get("/v1/stats", headers=AUTH).json()
    assert stats["feedback_count"] == 1


def test_validation_errors(client):
    bad = {**MESSAGE, "conversation_type": "carrier_pigeon"}
    assert client.post("/v1/route", json=bad, headers=AUTH).status_code == 422
    assert client.get("/v1/decisions/unknown", headers=AUTH).status_code == 404
    assert client.post("/v1/decisions/999/feedback", headers=AUTH,
                       json={"comment": "?"}).status_code == 404
    assert client.post("/v1/route/batch", json={"messages": []}, headers=AUTH).status_code == 422
