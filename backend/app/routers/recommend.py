from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from backend.app.dependencies import get_recommender
from backend.app.schemas import CompareRequest, CompareResponse, RecommendRequest, RecommendResponse
from reviewrec.explain.compare import compare_products
from reviewrec.ranking.recommender import Recommender

router = APIRouter(prefix="/api", tags=["recommendations"])


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, rec: Recommender = Depends(get_recommender)) -> dict:
    # Embedding + scoring is CPU work; run it off the event loop.
    return await run_in_threadpool(rec.recommend, req.query, req.method, req.top_k)


@router.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest, rec: Recommender = Depends(get_recommender)) -> dict:
    if len(set(req.product_ids)) != len(req.product_ids):
        raise HTTPException(status_code=422, detail="product_ids must be distinct")
    try:
        return await run_in_threadpool(compare_products, rec, req.product_ids, req.query, req.aspects)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
