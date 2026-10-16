"""Read/write helpers so the rest of the app never builds SQL queries itself."""

from sqlalchemy import func, select

from notif_router.schemas import RoutingDecision
from notif_router.storage.database import Database
from notif_router.storage.tables import DecisionRecord, FeedbackRecord, MediaAnalysisRecord


def _to_decision(record: DecisionRecord) -> RoutingDecision:
    return RoutingDecision(
        message_id=record.message_id,
        action=record.action,
        message_type=record.message_type,
        reason=record.reason,
        confidence=record.confidence,
        evidence_message_ids=list(record.evidence_message_ids or []),
        source=record.source,
        trace=list(record.trace or []),
    )


class DecisionRepository:
    def __init__(self, database: Database):
        self.db = database

    # ---------------------------------------------------------------- decisions

    def save_decision(
        self,
        decision: RoutingDecision,
        *,
        user_id: str,
        input_hash: str,
        model: str | None,
        prompt_version: str,
        latency_ms: int,
    ) -> int:
        with self.db.session() as session:
            record = DecisionRecord(
                message_id=decision.message_id,
                user_id=user_id,
                input_hash=input_hash,
                action=decision.action.value,
                message_type=decision.message_type.value,
                reason=decision.reason,
                confidence=decision.confidence,
                evidence_message_ids=decision.evidence_message_ids,
                source=decision.source.value,
                model=model,
                prompt_version=prompt_version,
                latency_ms=latency_ms,
                trace=decision.trace,
            )
            session.add(record)
            session.flush()
            return record.id

    def find_cached(
        self, input_hash: str, models: list[str], prompt_version: str
    ) -> tuple[RoutingDecision, str] | None:
        """Reuse an earlier agent decision for byte-identical input and prompt, made by any of
        `models` (the configured model chain). The most preferred model's decision wins.
        Returns (decision, model that made it)."""
        with self.db.session() as session:
            records = session.scalars(
                select(DecisionRecord)
                .where(DecisionRecord.input_hash == input_hash,
                       DecisionRecord.model.in_(models),
                       DecisionRecord.prompt_version == prompt_version,
                       DecisionRecord.source == "agent")
                .order_by(DecisionRecord.id.desc())
            ).all()
            if not records:
                return None
            best = min(records, key=lambda r: models.index(r.model))
            return _to_decision(best), best.model

    def latest_for_message(self, message_id: str) -> tuple[int, RoutingDecision] | None:
        with self.db.session() as session:
            record = session.scalars(
                select(DecisionRecord)
                .where(DecisionRecord.message_id == message_id)
                .order_by(DecisionRecord.id.desc())
                .limit(1)
            ).first()
            return (record.id, _to_decision(record)) if record else None

    def recent(self, limit: int = 50, user_id: str | None = None) -> list[tuple[int, RoutingDecision]]:
        with self.db.session() as session:
            query = select(DecisionRecord).order_by(DecisionRecord.id.desc()).limit(limit)
            if user_id:
                query = query.where(DecisionRecord.user_id == user_id)
            return [(r.id, _to_decision(r)) for r in session.scalars(query)]

    def stats(self) -> dict:
        with self.db.session() as session:
            rows = session.execute(
                select(DecisionRecord.action, func.count()).group_by(DecisionRecord.action)
            ).all()
            feedback_count = session.scalar(select(func.count()).select_from(FeedbackRecord))
            return {"decisions_by_action": dict(rows), "feedback_count": feedback_count or 0}

    # ----------------------------------------------------------------- feedback

    def add_feedback(
        self,
        decision_id: int,
        correct_action: str | None,
        correct_message_type: str | None,
        comment: str | None,
    ) -> int | None:
        with self.db.session() as session:
            if session.get(DecisionRecord, decision_id) is None:
                return None
            record = FeedbackRecord(
                decision_id=decision_id,
                correct_action=correct_action,
                correct_message_type=correct_message_type,
                comment=comment,
            )
            session.add(record)
            session.flush()
            return record.id

    # -------------------------------------------------------------- media cache

    def get_media_analysis(self, media_id: str) -> str | None:
        with self.db.session() as session:
            record = session.get(MediaAnalysisRecord, media_id)
            return record.content if record else None

    def save_media_analysis(self, media_id: str, media_type: str, content: str, model: str) -> None:
        with self.db.session() as session:
            session.merge(MediaAnalysisRecord(
                media_id=media_id, media_type=media_type, content=content, model=model,
            ))
