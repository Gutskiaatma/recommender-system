"""Pydantic request/response models (they also generate the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Method = Literal["tfidf", "semantic", "hybrid"]


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, examples=["Phone with an excellent camera and long battery life under $400"])
    method: Method = "hybrid"
    top_k: int = Field(10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("query is too short")
        return v


class CompareRequest(BaseModel):
    product_ids: list[str] = Field(..., min_length=2, max_length=3)
    query: str | None = Field(None, max_length=500)
    aspects: list[str] | None = None


class Snippet(BaseModel):
    text: str
    clause: str | None = None
    term: str | None = None
    polarity: str | None = None
    sentiment: float | None = None
    rating: int
    review_id: str
    query_similarity: float | None = None


class AspectSummary(BaseModel):
    aspect: str
    label: str
    score: float
    catalog_average: float
    n_mentions: int
    n_positive: int
    n_negative: int
    n_neutral: int
    positive_share: float | None
    confidence: float
    sufficient_evidence: bool
    snippets: list[Snippet] = []


class ScoreBreakdown(BaseModel):
    final: float
    relevance: float
    relevance_raw: float
    aspect: float | None
    quality: float


class ProductCard(BaseModel):
    product_id: str
    title: str
    brand: str
    price: float | None
    n_reviews: int
    mean_rating: float
    bayes_rating: float
    rating_distribution: dict[str, int]
    n_variants: int
    variant_titles: list[str]
    aspects: list[AspectSummary]
    reasons: list[str]
    evidence: list[Snippet]


class RankedProduct(ProductCard):
    rank: int
    scores: ScoreBreakdown


class ParsedQueryOut(BaseModel):
    text: str
    aspects: dict[str, float]
    matched_terms: dict[str, list[str]]
    ignored_aspects: list[str]
    brands: list[str]
    excluded_brands: list[str]
    price_min: float | None
    price_max: float | None


class RecommendResponse(BaseModel):
    query: str
    method: Method
    parsed: ParsedQueryOut
    weights: dict[str, float] | None
    notes: list[str]
    n_candidates: int
    results: list[RankedProduct]
    elapsed_ms: float


class CompareAspect(BaseModel):
    aspect: str
    label: str
    score: float
    catalog_average: float
    n_mentions: int
    n_positive: int
    n_negative: int
    n_neutral: int
    positive_share: float | None
    confidence: float
    sufficient_evidence: bool


class CompareProduct(BaseModel):
    product_id: str
    title: str
    brand: str
    price: float | None
    n_reviews: int
    mean_rating: float
    bayes_rating: float
    rating_distribution: dict[str, int]
    aspects: list[CompareAspect]
    match: dict[str, float | None] | None


class CompareResponse(BaseModel):
    query: str | None
    aspects: list[dict[str, str]]
    products: list[CompareProduct]
    best_by_aspect: dict[str, str | None]


class ProductListItem(BaseModel):
    product_id: str
    title: str
    brand: str
    price: float | None
    n_reviews: int
    mean_rating: float


class AspectInfo(BaseModel):
    aspect: str
    label: str
    description: str
    example_terms: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "artifacts_missing"]
    detail: str | None = None
    n_products: int | None = None
    embedding_model: str | None = None
    sentiment_backend: str | None = None
