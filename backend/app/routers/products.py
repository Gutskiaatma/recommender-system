from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.dependencies import get_recommender
from backend.app.schemas import ProductCard, ProductListItem
from reviewrec.ranking.recommender import Recommender

router = APIRouter(prefix="/api/products", tags=["products"])


@router.get("", response_model=list[ProductListItem])
def list_products(
    search: str | None = Query(None, max_length=100),
    limit: int = Query(20, ge=1, le=100),
    rec: Recommender = Depends(get_recommender),
) -> list[dict]:
    p = rec.store.products
    if search:
        mask = p["title"].str.contains(re.escape(search), case=False, regex=True) | p["brand"].str.contains(
            re.escape(search), case=False, regex=True
        )
        p = p[mask]
    p = p.sort_values("n_reviews", ascending=False).head(limit)
    return [
        {
            "product_id": r.product_id,
            "title": r.title,
            "brand": r.brand,
            "price": None if r.price != r.price else float(r.price),
            "n_reviews": int(r.n_reviews),
            "mean_rating": round(float(r.mean_rating), 3),
        }
        for r in p.itertuples(index=False)
    ]


@router.get("/{product_id}", response_model=ProductCard)
def get_product(product_id: str, rec: Recommender = Depends(get_recommender)) -> dict:
    idx = rec.store.product_row.get(product_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Unknown product id: {product_id}")
    card = rec.product_card(idx, list(rec.store.aspect_names), None, None)
    # Product page: show every aspect that has at least one mention.
    card["aspects"] = [a for a in card["aspects"] if a["n_mentions"] > 0]
    return card
