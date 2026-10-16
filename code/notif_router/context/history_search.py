"""Find past messages that help explain how this user will feel about a new one.

Ranking combines three things:
  * text similarity (token overlap + sequence ratio, so it works offline),
  * whether the past message came through the same chat / business / sender,
  * how the user reacted to it (reactions are what make history useful as evidence).
"""

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from notif_router.context.store import ContextStore, parse_timestamp
from notif_router.schemas import IncomingMessage

_TOKEN = re.compile(r"[a-z0-9@]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "it", "this",
    "that", "be", "you", "your", "i", "we", "me", "my", "at", "by", "with", "if", "so", "just",
    "can", "will", "pls", "please", "from", "as", "have", "has", "not", "no", "do", "all",
}


def _tokens(text: str | None) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1}


def text_similarity(a: str | None, b: str | None) -> float:
    """0..1 similarity that rewards shared vocabulary and near-duplicate wording."""
    if not a or not b:
        return 0.0
    ta, tb = _tokens(a), _tokens(b)
    overlap = len(ta & tb) / max(min(len(ta), len(tb)), 1) if ta and tb else 0.0
    sequence = SequenceMatcher(None, a.lower()[:400], b.lower()[:400]).ratio()
    return round(0.6 * overlap + 0.4 * sequence, 3)


def reaction_label(reaction: dict) -> str:
    """Collapse a reaction record into the single strongest signal."""
    if not reaction:
        return "no_record"
    if reaction.get("reported"):
        return "reported"
    if reaction.get("muted_after"):
        return "muted_chat_after"
    if reaction.get("replied"):
        return "replied"
    if reaction.get("dismissed"):
        return "dismissed"
    if reaction.get("opened"):
        return "opened_only"
    return "ignored"


@dataclass
class HistoryHit:
    message_id: str
    created_at: str | None
    conversation_type: str
    channel_match: str
    similarity: float
    score: float
    reaction: str
    forwarded_count: int
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


def _channel_match(message: IncomingMessage, past: dict) -> str:
    ctype = message.conversation_type.value
    if past["conversation_type"] != ctype:
        return "different_channel"
    if ctype == "group" and past["group_id"] == message.group_id:
        if message.sender_user_id and past["sender_user_id"] == message.sender_user_id:
            return "same_group_same_sender"
        return "same_group"
    if ctype == "business" and past["business_id"] == message.business_id:
        return "same_business"
    if ctype == "personal" and past["sender_user_id"] == message.sender_user_id:
        return "same_sender"
    if message.sender_user_id and past["sender_user_id"] == message.sender_user_id:
        return "same_sender_other_chat"
    return "same_channel_type"


_CHANNEL_BONUS = {
    "same_group_same_sender": 0.40,
    "same_business": 0.40,
    "same_sender": 0.40,
    "same_group": 0.25,
    "same_sender_other_chat": 0.20,
    "same_channel_type": 0.05,
    "different_channel": 0.0,
}


def search_history(
    store: ContextStore,
    message: IncomingMessage,
    query: str | None = None,
    limit: int = 5,
    same_channel_only: bool = False,
) -> list[HistoryHit]:
    """Rank the recipient's past messages by relevance to `message` (or a custom query)."""
    query_text = query if query is not None else (message.message_text or "")
    received_at = parse_timestamp(message.created_at)
    hits: list[HistoryHit] = []

    for past in store.history_by_user.get(message.user_id, []):
        channel = _channel_match(message, past)
        if same_channel_only and channel in {"different_channel", "same_channel_type"}:
            continue
        past_at = parse_timestamp(past["created_at"])
        if received_at and past_at and past_at > received_at:
            continue  # never use the future as evidence

        similarity = text_similarity(query_text, past["message_text"])
        reaction = reaction_label(store.reaction_for(message.user_id, past["message_id"]))
        score = 0.6 * similarity + _CHANNEL_BONUS[channel]
        if reaction in {"replied", "reported", "muted_chat_after", "dismissed"}:
            score += 0.05  # strong reactions make better evidence
        hits.append(HistoryHit(
            message_id=past["message_id"],
            created_at=past["created_at"],
            conversation_type=past["conversation_type"],
            channel_match=channel,
            similarity=similarity,
            score=round(score, 3),
            reaction=reaction,
            forwarded_count=past["forwarded_count"],
            text=(past["message_text"] or "(media only)")[:220],
        ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]
