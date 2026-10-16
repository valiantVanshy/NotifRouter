"""Rule-of-thumb router used when the LLM agent is unavailable.

It is deliberately simple and explainable: classify the content with keyword
patterns, then let the user's reaction to the most similar past message (if
any) override the category default. It keeps the service useful during an
OpenAI outage and gives the CLI an offline mode.
"""

import re

from notif_router.context import ContextStore, HistoryHit, search_history
from notif_router.guardrails import GuardrailVerdict, is_promotional
from notif_router.schemas import Action, DecisionSource, IncomingMessage, MessageType, RoutingDecision

_PATTERNS: list[tuple[MessageType, re.Pattern]] = [
    (MessageType.GREETING, re.compile(
        r"good (morning|night|evening)|blessed|happy (diwali|holi|new year|birthday|sunday)"
        r"|have a (nice|peaceful|great) day|good vibes|stay positive", re.I)),
    (MessageType.FORWARD, re.compile(
        r"\bfwd\b|forwarded|forward (this|to)|share (this|in all|with all)|as received", re.I)),
    (MessageType.PROMOTION, re.compile(
        r"\d+\s*% off|\boffer\b|cashback|discount|\bsale\b|\bselling\b|for sale|\bdeal\b|coupon"
        r"|limited (time|period)|shopping benefit", re.I)),
    (MessageType.URGENT, re.compile(
        r"\burgent|asap|right now|immediately|emergency|\bin \d+ ?min|\d+ ?mins? max|escalat"
        r"|can you come online|alert threshold|valve|leak|fire|ambulance|hospital", re.I)),
    (MessageType.PAYMENT, re.compile(
        r"\bpay (rs|inr|₹|the|your|now|by|before)|\b(bill|invoice|fee|fees|emi|dues?)\b"
        r"|token amount|rent due", re.I)),
    (MessageType.EVENT, re.compile(
        r"meeting|\bsync\b|review|circular|appointment|schedule|\bform\b|deadline|consent"
        r"|field trip|\bbus\b|\b\d{1,2}(:\d{2})? ?(am|pm)\b|tomorrow", re.I)),
    (MessageType.BUSINESS_UPDATE, re.compile(
        r"\b(order|delivery|delivered|shipment|packed|refund|statement|booking|ticket"
        r"|return pickup|account update|advisory)\b", re.I)),
]

_CALM = re.compile(r"nothing urgent|not urgent|no rush|no hurry|no pressure|whenever you", re.I)

_DEFAULT_ACTION = {
    MessageType.URGENT: Action.NOTIFY,
    MessageType.EVENT: Action.DIGEST,
    MessageType.PERSONAL: Action.DIGEST,
    MessageType.PAYMENT: Action.DIGEST,
    MessageType.BUSINESS_UPDATE: Action.DIGEST,
    MessageType.PROMOTION: Action.DIGEST,
    MessageType.GREETING: Action.DIGEST,
    MessageType.FORWARD: Action.MUTE,
    MessageType.SPAM: Action.MUTE,
    MessageType.SCAM: Action.MUTE,
    MessageType.UNKNOWN: Action.DIGEST,
}

_REACTION_ACTION = {
    "replied": Action.NOTIFY,
    "opened_only": Action.DIGEST,
    "dismissed": Action.MUTE,
    "muted_chat_after": Action.MUTE,
    "reported": Action.MUTE,
    "ignored": Action.MUTE,
}


def classify_content(message: IncomingMessage, text: str, store: ContextStore) -> MessageType:
    ctype = message.conversation_type.value
    if ctype == "business" and is_promotional(text):
        return MessageType.PROMOTION
    mentioned = bool(re.search(rf"@{re.escape(message.user_id)}\b", text))
    for message_type, pattern in _PATTERNS:
        if pattern.search(text):
            if message_type is MessageType.BUSINESS_UPDATE and ctype != "business":
                continue
            if message_type is MessageType.URGENT and _CALM.search(text):
                continue
            if mentioned and message_type in {MessageType.EVENT, MessageType.PAYMENT}:
                return MessageType.PERSONAL
            return message_type
    if mentioned:
        return MessageType.PERSONAL
    if message.forwarded_count >= 5:
        return MessageType.FORWARD
    if ctype == "business":
        return MessageType.BUSINESS_UPDATE
    if ctype == "personal" and message.sender_user_id:
        if store.sender_context(message.sender_user_id, message.user_id)["first_contact"]:
            return MessageType.UNKNOWN
    return MessageType.PERSONAL if text else MessageType.UNKNOWN


def _best_evidence(hits: list[HistoryHit], has_text: bool) -> HistoryHit | None:
    for hit in hits:
        close_channel = hit.channel_match not in {"different_channel", "same_channel_type"}
        if hit.similarity >= 0.35 or (close_channel and (hit.similarity >= 0.2 or not has_text)):
            return hit
    return None


def heuristic_decision(
    message: IncomingMessage,
    store: ContextStore,
    media_text: str | None = None,
    locked: GuardrailVerdict | None = None,
) -> RoutingDecision:
    text = " ".join(t for t in (message.message_text, media_text) if t)
    message_type = classify_content(message, text, store)
    hits = search_history(store, message, query=text, limit=5)
    evidence = _best_evidence(hits, has_text=bool(text))

    if locked is not None:
        return RoutingDecision(
            message_id=message.message_id,
            action=locked.action,
            message_type=locked.message_type or message_type,
            reason=locked.reason,
            confidence=locked.confidence,
            evidence_message_ids=locked.evidence_message_ids or ([evidence.message_id] if evidence else []),
            source=DecisionSource.HEURISTIC,
        )

    action = _DEFAULT_ACTION[message_type]
    reasons = [f"Content looks like a {message_type.value.replace('_', ' ')} message"]
    confidence = 0.62

    if evidence and evidence.reaction in _REACTION_ACTION:
        action = _REACTION_ACTION[evidence.reaction]
        reasons.append(
            f"the user {evidence.reaction.replace('_', ' ')} a similar earlier message "
            f"({evidence.channel_match.replace('_', ' ')})"
        )
        confidence = 0.74

    if message.conversation_type.value == "business" and message.business_id:
        business = store.business_context(message.business_id, message.user_id)
        if message_type is MessageType.PROMOTION and not business.get("has_prior_relationship"):
            if not business.get("verified"):
                message_type, action = MessageType.SPAM, Action.MUTE
                reasons.append("it is unsolicited marketing from an unverified business")

    if re.search(rf"@{re.escape(message.user_id)}\b", text) and action is not Action.MUTE:
        action = Action.NOTIFY
        reasons.append("the user is mentioned directly")

    profile = store.recipient_profile(message.user_id, message.created_at)
    if action is Action.NOTIFY and profile.get("in_quiet_hours_now") and message_type is not MessageType.URGENT:
        action = Action.DIGEST
        reasons.append("it arrived during the user's quiet hours and is not urgent")

    return RoutingDecision(
        message_id=message.message_id,
        action=action,
        message_type=message_type,
        reason="; ".join(reasons) + ".",
        confidence=confidence,
        evidence_message_ids=[evidence.message_id] if evidence else [],
        source=DecisionSource.HEURISTIC,
    )
