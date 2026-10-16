"""Deterministic safety guardrails that run before the agent.

These rules encode decisions that must never depend on an LLM's mood: scams,
spoofed businesses, prompt-injection attempts and explicit user preferences
(opted-out promotions, muted groups). A guardrail can either fully decide a
message, or lock the action to `mute` and let the agent pick the message type.
"""

import re
from dataclasses import dataclass, field

from notif_router.context import ContextStore, search_history
from notif_router.schemas import Action, IncomingMessage, MessageType, RoutingDecision

_INJECTION = re.compile(
    r"ignore (all )?(previous|prior|above)|routing (rule|override)|mark (this|it) as (notify|urgent)"
    r"|set action\s*=|system (note|prompt)|assistant instruction|internal metadata"
    r"|verified_business\s*=\s*true",
    re.IGNORECASE,
)
_SENSITIVE = re.compile(
    r"\b(otp|one[- ]time (pass)?code|password|passcode|pin|cvv|card (number|details)"
    r"|login code|\d[- ]digit (login )?code|kyc|bank details|account number|wallet and card)\b",
    re.IGNORECASE,
)
_PRESSURE_VERB = re.compile(
    r"\b(verify|confirm|shar(e|ing)|reply with|send|enter|update|submit|re-?activate|unlock"
    r"|batao|bata do|bhejo|bhej do)\b",  # includes common Hinglish ("tell", "send")
    re.IGNORECASE,
)
_ACCOUNT_THREAT = re.compile(
    r"\b(account|profile|service|access|wallet|card)\b[^.]{0,40}\b(block|lock|restrict|suspend|close|band|expire|deactivat)"
    r"|avoid (account|service) (lock|block|suspension)",
    re.IGNORECASE,
)
_SUSPICIOUS_LINK = re.compile(
    r"\b(bit\.ly|tinyurl\.com|cutt\.ly|is\.gd|goo\.gl|rb\.gy|t\.co|shorturl\.at)/|\b(this|the|below) link\b",
    re.IGNORECASE,
)
_SHORTENED_LINK = re.compile(r"\b(bit\.ly|tinyurl\.com|cutt\.ly|is\.gd|goo\.gl|rb\.gy|shorturl\.at)/", re.IGNORECASE)
_UPFRONT_PAYMENT = re.compile(
    r"\bpay\b[^.]{0,40}\b(fee|charge|token)\b|\b(processing|reactivation|release|unlock) fee\b"
    r"|scan the qr[^.]{0,40}send screenshot",
    re.IGNORECASE,
)
_REWARD_BAIT = re.compile(
    r"\b(loan|refund|benefits?|prize|reward|amount|lottery)\b[^.]{0,30}\b(approved|released|claim|won)"
    r"|\bclaim (your )?(benefits?|prize|reward|refund)",
    re.IGNORECASE,
)
_URGENCY = re.compile(
    r"\burgent|immediately|final reminder|today|tonight|before midnight|jaldi|time kam|limited window",
    re.IGNORECASE,
)
_SAFETY_ADVICE = re.compile(
    r"never (ask|share)|do not share|don't share|will not ask|won't ask", re.IGNORECASE
)
_PROMO = re.compile(
    r"\d+\s*% off|\boffer\b|cashback|discount|\bsale\b|coupon|use code|reply stop|deal\b|reward",
    re.IGNORECASE,
)
_TRANSACTIONAL = re.compile(
    r"\b(order|delivery|delivered|shipment|appointment|invoice|receipt|refund|booking|ticket)\b",
    re.IGNORECASE,
)


@dataclass
class GuardrailVerdict:
    rule: str
    action: Action
    message_type: MessageType | None  # None = action is locked, agent should choose the type
    reason: str
    confidence: float
    evidence_message_ids: list[str] = field(default_factory=list)


def looks_like_prompt_injection(text: str | None) -> bool:
    return bool(text and _INJECTION.search(text))


def asks_for_credentials(text: str | None) -> bool:
    if not text or _SAFETY_ADVICE.search(text):
        return False
    return bool(_SENSITIVE.search(text) and _PRESSURE_VERB.search(text))


def fraud_signals(text: str | None) -> list[str]:
    """Named fraud signals in `text`. One signal alone is common in legitimate messages;
    the combinations checked in `looks_like_fraud` are what scams look like."""
    if not text or _SAFETY_ADVICE.search(text):
        return []
    checks = {
        "credential request": asks_for_credentials(text),
        "account threat": bool(_ACCOUNT_THREAT.search(text)),
        "suspicious link": bool(_SUSPICIOUS_LINK.search(text)),
        "shortened link": bool(_SHORTENED_LINK.search(text)),
        "upfront payment": bool(_UPFRONT_PAYMENT.search(text)),
        "reward bait": bool(_REWARD_BAIT.search(text)),
        "time pressure": bool(_URGENCY.search(text)),
    }
    return [name for name, hit in checks.items() if hit]


def looks_like_fraud(signals: list[str]) -> bool:
    s = set(signals)
    return (
        "credential request" in s
        or ("account threat" in s and s & {"suspicious link", "upfront payment"})
        or ("reward bait" in s and s & {"suspicious link", "upfront payment"})
        or ({"shortened link", "time pressure"} <= s)
    )


_NOT_URGENT = re.compile(
    r"nothing urgent|not urgent|no rush|no hurry|no pressure|whenever you", re.IGNORECASE
)


def says_not_urgent(text: str | None) -> bool:
    """The sender explicitly says the message can wait."""
    return bool(text and _NOT_URGENT.search(text))


def enforce_consistency(message: IncomingMessage, decision: RoutingDecision) -> RoutingDecision:
    """Post-decision check applied to every decision (agent, cached or heuristic).

    If the sender explicitly says it is not urgent and the user is not @mentioned, the
    message must not interrupt: downgrade notify to digest. Returns the possibly-updated
    decision.
    """
    if (decision.action is Action.NOTIFY and says_not_urgent(message.message_text)
            and not _mentions(message.message_text, message.user_id)):
        return decision.model_copy(update={
            "action": Action.DIGEST,
            "reason": f"{decision.reason} Delivered in the digest because the sender says it is not urgent.",
            "confidence": min(decision.confidence, 0.8),
            "trace": [*decision.trace, {"consistency_check": "not_urgent_downgrade"}],
        })
    return decision


def is_promotional(text: str | None) -> bool:
    return bool(text and _PROMO.search(text) and not _TRANSACTIONAL.search(text))


def _mentions(text: str | None, user_id: str) -> bool:
    return bool(text and re.search(rf"@{re.escape(user_id)}\b", text))


def check_guardrails(message: IncomingMessage, store: ContextStore) -> GuardrailVerdict | None:
    """Return the first guardrail that fires, or None if the agent should decide freely."""
    text = message.message_text

    if looks_like_prompt_injection(text):
        return GuardrailVerdict(
            rule="prompt_injection",
            action=Action.MUTE,
            message_type=MessageType.SCAM,
            reason="The message tries to give instructions to the router instead of the user, "
                   "a common manipulation tactic in scam messages.",
            confidence=0.93,
            evidence_message_ids=_similar_evidence(message, store, min_similarity=0.3),
        )

    signals = fraud_signals(text)
    if looks_like_fraud(signals):
        if "credential request" in signals:
            reason = ("The message pressures the user to share or verify an OTP, password, PIN, "
                      "card or account details, which is a phishing pattern.")
        else:
            reason = f"The message combines common fraud tactics ({', '.join(signals)})."
        return GuardrailVerdict(
            rule="phishing_or_fraud",
            action=Action.MUTE,
            message_type=MessageType.SCAM,
            reason=reason,
            confidence=0.9,
            evidence_message_ids=_similar_evidence(message, store, min_similarity=0.3),
        )

    if message.conversation_type.value == "business" and message.business_id:
        business = store.business_context(message.business_id, message.user_id)
        if business.get("known_business"):
            if not business["verified"] and business["domain_mismatch"]:
                return GuardrailVerdict(
                    rule="spoofed_business_domain",
                    action=Action.MUTE,
                    message_type=MessageType.SCAM,
                    reason=f"Unverified sender is using '{business['sender_domain']}' instead of "
                           f"the official domain '{business['official_domain']}', which suggests "
                           "brand impersonation.",
                    confidence=0.92,
                )
            age = business["account_age_days"] or 0
            if not business["verified"] and age < 45 and (
                business["user_reports_30d"] > 30 or business["report_rate_30d"] >= 0.03
            ):
                return GuardrailVerdict(
                    rule="high_risk_new_business",
                    action=Action.MUTE,
                    message_type=MessageType.SCAM,
                    reason=f"Unverified business account only {age} days old with "
                           f"{business['user_reports_30d']} user reports in 30 days.",
                    confidence=0.85,
                )
            relationship = business.get("relationship") or {}
            if (relationship and not relationship["allows_promotions"]
                    and relationship["promotions_opted_out_at"] and is_promotional(text)):
                return GuardrailVerdict(
                    rule="promotions_opted_out",
                    action=Action.MUTE,
                    message_type=MessageType.PROMOTION,
                    reason="The user opted out of promotions from this business and this message "
                           "is marketing, not a transactional update.",
                    confidence=0.86,
                    evidence_message_ids=_similar_evidence(message, store, min_similarity=0.35),
                )

    if message.conversation_type.value == "group" and message.group_id:
        membership = store.memberships.get((message.group_id, message.user_id)) or {}
        if membership.get("muted_by_user") and not _mentions(text, message.user_id):
            return GuardrailVerdict(
                rule="muted_group",
                action=Action.MUTE,
                message_type=None,
                reason="The user has muted this group and the message does not mention them directly.",
                confidence=0.82,
                evidence_message_ids=_similar_evidence(message, store, min_similarity=0.3),
            )

    if message.forwarded_count >= 5:
        ignored = [
            hit.message_id
            for hit in search_history(store, message, limit=8)
            if hit.similarity >= 0.35 and hit.reaction in {"dismissed", "ignored", "muted_chat_after"}
        ]
        if len(ignored) >= 2:
            return GuardrailVerdict(
                rule="repeated_ignored_forward",
                action=Action.MUTE,
                message_type=None,
                reason=f"Forwarded {message.forwarded_count} times and the user ignored or "
                       f"dismissed {len(ignored)} similar forwards before.",
                confidence=0.84,
                evidence_message_ids=ignored[:3],
            )

    return None


def _similar_evidence(message: IncomingMessage, store: ContextStore, min_similarity: float) -> list[str]:
    hits = search_history(store, message, limit=3)
    return [h.message_id for h in hits if h.similarity >= min_similarity][:2]
