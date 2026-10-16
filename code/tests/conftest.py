import json
from collections import defaultdict
from pathlib import Path

import pytest

from notif_router.agent.llm import ChatTurn, ToolCall
from notif_router.config import DEFAULT_DATASET_DIR
from notif_router.context import ContextStore
from notif_router.storage import Database, DecisionRepository


@pytest.fixture
def store(tmp_path: Path) -> ContextStore:
    """A tiny hand-built world: one user, one family group, two businesses, some history."""
    s = ContextStore(dataset_dir=tmp_path)
    s.users["u_1"] = {"user_id": "u_1", "quiet_hours": "23:00-07:00", "opened_30d": 40,
                      "replied_30d": 10, "dismissed_30d": 5, "reported_30d": 0}
    s.groups["g_family"] = {"group_id": "g_family", "name": "Family", "type": "family",
                            "member_count": 8, "messages_30d": 120}
    s.memberships[("g_family", "u_1")] = {"role": "member", "joined_at": "2024-01-01", "sent_30d": 3,
                                          "read_30d": 50, "replied_30d": 6, "dismissed_30d": 1,
                                          "muted_by_user": False}
    s.memberships[("g_family", "u_mom")] = {"role": "admin", "joined_at": "2023-01-01", "sent_30d": 30,
                                            "read_30d": 90, "replied_30d": 20, "dismissed_30d": 0,
                                            "muted_by_user": False}
    s.groups["g_muted"] = {"group_id": "g_muted", "name": "Deals", "type": "marketplace",
                           "member_count": 300, "messages_30d": 900}
    s.memberships[("g_muted", "u_1")] = {"role": "member", "joined_at": "2024-01-01", "sent_30d": 0,
                                         "read_30d": 2, "replied_30d": 0, "dismissed_30d": 40,
                                         "muted_by_user": True}
    s.businesses["b_shop"] = {"business_id": "b_shop", "display_name": "Shop", "brand_name": "Shop",
                              "category": "ecommerce", "verified": True, "official_domain": "shop.in",
                              "sender_domain": "shop.in", "account_age_days": 900,
                              "messages_sent_30d": 1000, "user_reports_30d": 2,
                              "sender_domain_age_days": 800}
    s.businesses["b_fake"] = {"business_id": "b_fake", "display_name": "Shop Support",
                              "brand_name": "Shop", "category": "ecommerce", "verified": False,
                              "official_domain": "shop.in", "sender_domain": "shop-support.co",
                              "account_age_days": 10, "messages_sent_30d": 500,
                              "user_reports_30d": 40, "sender_domain_age_days": 5}
    s.relationships[("u_1", "b_shop")] = {"why_user_knows_account": "recent_order",
                                          "last_activity_at": "2026-07-01 10:00",
                                          "allows_promotions": False,
                                          "promotions_opted_out_at": "2026-06-01 10:00",
                                          "activity_count_180d": 5, "opened_30d": 4,
                                          "dismissed_30d": 1, "replied_30d": 1}
    s.history_by_user = defaultdict(list)
    history = [
        ("h_1", "group", "g_family", None, "u_mom", "2026-07-01 09:00",
         "Water tanker coming at 4 PM, keep the valve open", {"opened": True, "replied": True}),
        ("h_2", "business", None, "b_shop", None, "2026-07-02 09:00",
         "Your order 1234 has been packed", {"opened": True}),
        ("h_3", "group", "g_family", None, "u_mom", "2026-09-01 09:00",
         "Future message that must never be used", {"opened": True, "replied": True}),
    ]
    for mid, ctype, gid, bid, sender, created, text, reaction in history:
        row = {"message_id": mid, "user_id": "u_1", "conversation_type": ctype, "group_id": gid,
               "business_id": bid, "sender_user_id": sender, "created_at": created,
               "message_text": text, "media_type": None, "media_id": None, "forwarded_count": 0}
        s.history_by_id[mid] = row
        s.history_by_user["u_1"].append(row)
        s.reactions[("u_1", mid)] = {"opened": False, "replied": False, "dismissed": False,
                                     "muted_after": False, "reported": False,
                                     "reaction_minutes": None, **reaction}
    return s


@pytest.fixture
def repository(tmp_path: Path) -> DecisionRepository:
    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    database.create_tables()
    return DecisionRepository(database)


@pytest.fixture
def real_store() -> ContextStore:
    if not DEFAULT_DATASET_DIR.is_dir():
        pytest.skip("dataset/ folder not available")
    return ContextStore.from_dataset(DEFAULT_DATASET_DIR)


class ScriptedLLM:
    """Fake LLMClient that replays a fixed list of tool calls, one turn per chat() call."""

    model = "fake-model"

    def __init__(self, turns: list[list[tuple[str, dict]]]):
        self.turns = list(turns)
        self.requests: list[list[dict]] = []

    def chat(self, messages, tools, tool_choice):
        self.requests.append([dict(m) for m in messages])
        if not self.turns:
            return ChatTurn(content="I am done.")
        calls = [ToolCall(id=f"call_{len(self.requests)}_{i}", name=name, arguments=json.dumps(args))
                 for i, (name, args) in enumerate(self.turns.pop(0))]
        return ChatTurn(content=None, tool_calls=calls)

    def transcribe_audio(self, path):
        return "fake transcript"

    def describe_image(self, path):
        return "fake image text"


@pytest.fixture
def scripted_llm():
    return ScriptedLLM
