"""In-memory store of everything the router knows about users, chats and businesses.

The store is loaded once from the dataset CSVs and indexed for fast lookups. The
rest of the app only talks to this class, so swapping the CSVs for a real
database later means changing this file only.
"""

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: (value.strip() or None) if isinstance(value, str) else value
             for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flag(value) -> bool:
    return _to_int(value) == 1


def _in_quiet_hours(when: datetime | None, window: str | None) -> bool:
    """True if `when` falls inside a window like '22:00-07:00' (overnight windows supported)."""
    if when is None or not window or "-" not in window:
        return False
    try:
        start_raw, end_raw = window.split("-", 1)
        start = time.fromisoformat(start_raw.strip())
        end = time.fromisoformat(end_raw.strip())
    except ValueError:
        return False
    now = when.time()
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


@dataclass
class ContextStore:
    dataset_dir: Path
    users: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    memberships: dict[tuple[str, str], dict] = field(default_factory=dict)
    businesses: dict[str, dict] = field(default_factory=dict)
    relationships: dict[tuple[str, str], dict] = field(default_factory=dict)
    history_by_user: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    history_by_id: dict[str, dict] = field(default_factory=dict)
    reactions: dict[tuple[str, str], dict] = field(default_factory=dict)
    media_files: dict[str, Path] = field(default_factory=dict)
    daily_load: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_dataset(cls, dataset_dir: Path) -> "ContextStore":
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        store = cls(dataset_dir=dataset_dir)
        store._load()
        logger.info(
            "context store loaded",
            extra={"context": {"users": len(store.users), "history": len(store.history_by_id)}},
        )
        return store

    def _load(self) -> None:
        d = self.dataset_dir
        for row in _read_csv(d / "users.csv"):
            self.users[row["user_id"]] = {
                "user_id": row["user_id"],
                "quiet_hours": row["do_not_disturb_window"],
                "opened_30d": _to_int(row["messages_opened_30d"]) or 0,
                "replied_30d": _to_int(row["messages_replied_30d"]) or 0,
                "dismissed_30d": _to_int(row["notifications_dismissed_30d"]) or 0,
                "reported_30d": _to_int(row["messages_reported_30d"]) or 0,
            }
        for row in _read_csv(d / "groups.csv"):
            self.groups[row["group_id"]] = {
                "group_id": row["group_id"],
                "name": row["group_name"],
                "type": row["group_type"],
                "member_count": _to_int(row["member_count"]),
                "messages_30d": _to_int(row["messages_30d"]),
            }
        for row in _read_csv(d / "group_members.csv"):
            self.memberships[(row["group_id"], row["user_id"])] = {
                "role": row["role"],
                "joined_at": row["joined_at"],
                "sent_30d": _to_int(row["messages_sent_30d"]) or 0,
                "read_30d": _to_int(row["messages_read_30d"]) or 0,
                "replied_30d": _to_int(row["replies_sent_30d"]) or 0,
                "dismissed_30d": _to_int(row["notifications_dismissed_30d"]) or 0,
                "muted_by_user": _flag(row["group_muted_by_user"]),
            }
        for row in _read_csv(d / "business_accounts.csv"):
            self.businesses[row["business_id"]] = {
                "business_id": row["business_id"],
                "display_name": row["display_name"],
                "brand_name": row["brand_name"],
                "category": row["category"],
                "verified": _flag(row["verified"]),
                "official_domain": row["official_domain"],
                "sender_domain": row["domain_used_by_sender"],
                "account_age_days": _to_int(row["account_age_days"]),
                "messages_sent_30d": _to_int(row["messages_sent_30d"]) or 0,
                "user_reports_30d": _to_int(row["user_reports_30d"]) or 0,
                "sender_domain_age_days": _to_int(row["domain_used_by_sender_age_days"]),
            }
        for row in _read_csv(d / "user_business_history.csv"):
            self.relationships[(row["user_id"], row["business_id"])] = {
                "why_user_knows_account": row["why_user_knows_account"],
                "last_activity_at": row["last_activity_at"],
                "allows_promotions": _flag(row["allows_promotions"]),
                "promotions_opted_out_at": row["promotions_opted_out_at"],
                "activity_count_180d": _to_int(row["activity_count_180d"]) or 0,
                "opened_30d": _to_int(row["messages_opened_30d"]) or 0,
                "dismissed_30d": _to_int(row["messages_dismissed_30d"]) or 0,
                "replied_30d": _to_int(row["messages_replied_30d"]) or 0,
            }
        for row in _read_csv(d / "message_history.csv"):
            row["forwarded_count"] = _to_int(row["forwarded_count"]) or 0
            self.history_by_id[row["message_id"]] = row
            self.history_by_user[row["user_id"]].append(row)
        for row in _read_csv(d / "message_events.csv"):
            self.reactions[(row["user_id"], row["message_id"])] = {
                "opened": _flag(row["message_opened"]),
                "replied": _flag(row["message_replied"]),
                "dismissed": _flag(row["notification_dismissed"]),
                "muted_after": _flag(row["muted_after_message"]),
                "reported": _flag(row["message_reported"]),
                "reaction_minutes": _to_int(row["reaction_time_minutes"]),
            }
        for filename, id_column in (("images.csv", "image_id"), ("voice_notes.csv", "voice_note_id")):
            for row in _read_csv(d / filename):
                self.media_files[row[id_column]] = d / row["file_path"]
        for row in _read_csv(d / "daily_notification_summary.csv"):
            self.daily_load[row["user_id"]].append({
                "date": row["date"],
                "sent": _to_int(row["notifications_sent"]) or 0,
                "dismissed": _to_int(row["notifications_dismissed"]) or 0,
            })
        for entries in self.daily_load.values():
            entries.sort(key=lambda e: e["date"], reverse=True)

    # ----------------------------------------------------------------- lookups

    def recipient_profile(self, user_id: str, created_at: str | None = None) -> dict:
        user = self.users.get(user_id)
        if user is None:
            return {"user_id": user_id, "known_user": False}
        opened = max(user["opened_30d"], 1)
        recent = self.daily_load.get(user_id, [])[:7]
        sent = sum(day["sent"] for day in recent)
        dismissed = sum(day["dismissed"] for day in recent)
        return {
            **user,
            "known_user": True,
            "reply_rate": round(user["replied_30d"] / opened, 2),
            "dismiss_rate": round(user["dismissed_30d"] / opened, 2),
            "in_quiet_hours_now": _in_quiet_hours(parse_timestamp(created_at), user["quiet_hours"]),
            "notifications_last_7d": sent,
            "notification_dismiss_rate_7d": round(dismissed / sent, 2) if sent else None,
        }

    def group_context(self, group_id: str, user_id: str, sender_user_id: str | None) -> dict:
        group = self.groups.get(group_id)
        if group is None:
            return {"group_id": group_id, "known_group": False}
        membership = self.memberships.get((group_id, user_id))
        sender_membership = self.memberships.get((group_id, sender_user_id)) if sender_user_id else None
        return {
            **group,
            "known_group": True,
            "recipient_membership": membership,
            "sender_user_id": sender_user_id,
            "sender_role": sender_membership["role"] if sender_membership else "unknown",
        }

    def business_context(self, business_id: str, user_id: str) -> dict:
        business = self.businesses.get(business_id)
        if business is None:
            return {"business_id": business_id, "known_business": False}
        relationship = self.relationships.get((user_id, business_id))
        official = (business["official_domain"] or "").lower()
        used = (business["sender_domain"] or "").lower()
        sent = max(business["messages_sent_30d"], 1)
        return {
            **business,
            "known_business": True,
            "domain_mismatch": bool(official and used and official != used),
            "report_rate_30d": round(business["user_reports_30d"] / sent, 3),
            "relationship": relationship,
            "has_prior_relationship": relationship is not None,
        }

    def sender_context(self, sender_user_id: str, user_id: str) -> dict:
        """How this recipient has engaged with a person who messages them directly."""
        past = [
            m for m in self.history_by_user.get(user_id, [])
            if m["conversation_type"] == "personal" and m["sender_user_id"] == sender_user_id
        ]
        reactions = [self.reactions.get((user_id, m["message_id"]), {}) for m in past]
        return {
            "sender_user_id": sender_user_id,
            "relationship": "stranger: no earlier direct messages" if not past else "existing contact",
            "previous_direct_messages": len(past),
            "recipient_replied_to": sum(1 for r in reactions if r.get("replied")),
            "recipient_dismissed": sum(1 for r in reactions if r.get("dismissed")),
            "recipient_reported": sum(1 for r in reactions if r.get("reported")),
            "first_contact": not past,
        }

    def reaction_for(self, user_id: str, message_id: str) -> dict:
        return self.reactions.get((user_id, message_id), {})

    def media_path(self, media_id: str | None) -> Path | None:
        return self.media_files.get(media_id) if media_id else None
