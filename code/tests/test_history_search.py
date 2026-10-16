from notif_router.context.history_search import search_history, text_similarity
from notif_router.schemas import IncomingMessage


def test_text_similarity_prefers_related_text():
    a = "Water tanker coming at 5 PM, keep the valve open"
    assert text_similarity(a, "Tanker at 4 PM, keep valve open") > text_similarity(a, "Happy birthday!")
    assert text_similarity(None, a) == 0.0


def test_search_ranks_same_chat_first_and_ignores_future_messages(store):
    message = IncomingMessage(message_id="m_1", user_id="u_1", conversation_type="group",
                              group_id="g_family", sender_user_id="u_mom",
                              created_at="2026-07-10 10:00",
                              message_text="Tanker arriving at 5, valve open please")
    hits = search_history(store, message, limit=5)
    ids = [h.message_id for h in hits]
    assert ids[0] == "h_1"
    assert hits[0].channel_match == "same_group_same_sender"
    assert hits[0].reaction == "replied"
    assert "h_3" not in ids  # created after the incoming message


def test_same_channel_only_filters_other_chats(store):
    message = IncomingMessage(message_id="m_2", user_id="u_1", conversation_type="business",
                              business_id="b_shop", created_at="2026-07-10 10:00",
                              message_text="Order packed")
    hits = search_history(store, message, same_channel_only=True)
    assert [h.message_id for h in hits] == ["h_2"]
