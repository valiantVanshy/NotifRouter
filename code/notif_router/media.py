"""Turns voice notes and images into text the router can reason about, with a DB cache."""

import logging

from notif_router.agent.llm import LLMClient
from notif_router.context import ContextStore
from notif_router.storage import DecisionRepository

logger = logging.getLogger(__name__)


class MediaAnalyzer:
    def __init__(self, store: ContextStore, repository: DecisionRepository, llm: LLMClient | None):
        self.store = store
        self.repository = repository
        self.llm = llm

    def cached(self, media_id: str | None) -> str | None:
        return self.repository.get_media_analysis(media_id) if media_id else None

    def analyze(self, media_type: str | None, media_id: str | None) -> str | None:
        """Return a transcript (voice) or description + OCR (image); None if unavailable."""
        if not media_type or not media_id:
            return None
        cached = self.cached(media_id)
        if cached is not None:
            return cached
        path = self.store.media_path(media_id)
        if self.llm is None or path is None or not path.exists():
            return None
        try:
            if media_type == "voice":
                content = self.llm.transcribe_audio(path)
            elif media_type == "image":
                content = self.llm.describe_image(path)
            else:
                return None
        except Exception:
            logger.exception("media analysis failed", extra={"context": {"media_id": media_id}})
            return None
        if content:
            self.repository.save_media_analysis(media_id, media_type, content, self.llm.model)
        return content or None
