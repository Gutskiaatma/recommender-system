from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request

from backend.app.dependencies import get_recommender
from backend.app.schemas import AspectInfo, HealthResponse
from reviewrec.aspects.lexicon import get_lexicon
from reviewrec.config import get_settings
from reviewrec.ranking.recommender import Recommender

router = APIRouter(prefix="/api", tags=["meta"])

EXAMPLE_QUERIES = [
    "I want a smartphone with an excellent camera, long battery life, and good performance for gaming.",
    "Cheap phone under $150 with a battery that lasts all day",
    "Samsung phone with a big bright screen for watching videos",
    "Durable phone that survives drops, I don't care about the camera",
    "Dual SIM phone with good signal for international travel",
    "Small compact phone that fits in my pocket with a good camera",
]


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict:
    rec = getattr(request.app.state, "recommender", None)
    if rec is None:
        return {"status": "artifacts_missing", "detail": getattr(request.app.state, "load_error", None)}
    m = rec.store.manifest
    return {
        "status": "ok",
        "n_products": len(rec.store.products),
        "embedding_model": m.get("embedding_model"),
        "sentiment_backend": m.get("sentiment_backend"),
    }


@router.get("/aspects", response_model=list[AspectInfo])
def aspects() -> list[dict]:
    lex = get_lexicon()
    return [
        {"aspect": a, "label": lex.labels[a], "description": lex.descriptions[a], "example_terms": lex.terms[a][:6]}
        for a in lex.aspects
    ]


@router.get("/examples", response_model=list[str])
def examples() -> list[str]:
    return EXAMPLE_QUERIES


@router.get("/stats")
def stats(rec: Recommender = Depends(get_recommender)) -> dict:
    """Dataset statistics, pipeline manifest and (if present) evaluation results."""
    s = get_settings()
    eval_dir = s.path("paths.evaluation_dir") / "results"
    evaluation = {}
    for name in ("ranking_summary.json", "absa_summary.json", "lexicon_summary.json"):
        f = eval_dir / name
        if f.exists():
            evaluation[name.replace("_summary.json", "")] = json.loads(f.read_text(encoding="utf-8"))
    quality_file = s.processed_dir / "data_quality_report.json"
    return {
        "dataset": rec.store.stats,
        "manifest": rec.store.manifest,
        "data_quality": json.loads(quality_file.read_text(encoding="utf-8")) if quality_file.exists() else None,
        "evaluation": evaluation,
        "weights": {"alpha": rec.weights.alpha, "beta": rec.weights.beta, "gamma": rec.weights.gamma},
    }
