"""Application state: the Recommender is loaded once at startup."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from reviewrec.ranking.recommender import Recommender
from reviewrec.store import ArtifactsMissingError

log = logging.getLogger("reviewrec.api")


def load_recommender() -> tuple[Recommender | None, str | None]:
    try:
        rec = Recommender.from_settings()
        # Warm up the embedding model so the first user request is not slow.
        rec.embedder.encode_query("warm up query")
        return rec, None
    except ArtifactsMissingError as exc:
        log.error("%s", exc)
        return None, str(exc)


def get_recommender(request: Request) -> Recommender:
    rec = getattr(request.app.state, "recommender", None)
    if rec is None:
        detail = getattr(request.app.state, "load_error", None) or "Recommender not loaded"
        raise HTTPException(status_code=503, detail=detail)
    return rec
