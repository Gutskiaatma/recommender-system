"""API tests with FastAPI's TestClient against artifacts built from the synthetic dataset."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(built_artifacts):
    from backend.app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_products"] == 4


def test_recommend_returns_ranked_explained_results(client):
    r = client.post("/api/recommend", json={"query": "phone with an excellent camera", "method": "hybrid", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["parsed"]["aspects"] == {"camera": 1.5}
    results = body["results"]
    assert [x["rank"] for x in results] == [1, 2, 3]
    finals = [x["scores"]["final"] for x in results]
    assert finals == sorted(finals, reverse=True)
    top = results[0]
    assert top["aspects"][0]["aspect"] == "camera"
    # Every snippet must come from the dataset and carry its review id.
    for snip in top["aspects"][0]["snippets"]:
        assert snip["review_id"].startswith("r_")
        assert "camera" in snip["text"].lower() or snip["term"] in snip["text"].lower()


@pytest.mark.parametrize("method", ["tfidf", "semantic", "hybrid"])
def test_all_methods(client, method):
    r = client.post("/api/recommend", json={"query": "long battery life", "method": method, "top_k": 4})
    assert r.status_code == 200
    assert len(r.json()["results"]) >= 1


def test_price_filter(client):
    r = client.post("/api/recommend", json={"query": "good phone under $200", "top_k": 10})
    prices = [x["price"] for x in r.json()["results"]]
    assert prices and all(p <= 200 for p in prices)


def test_brand_exclusion(client):
    r = client.post("/api/recommend", json={"query": "phone with good battery, not Beta", "top_k": 10})
    assert all(x["brand"] != "Beta" for x in r.json()["results"])


def test_validation_errors(client):
    assert client.post("/api/recommend", json={"query": "  "}).status_code == 422
    assert client.post("/api/recommend", json={"query": "camera", "method": "magic"}).status_code == 422
    assert client.post("/api/recommend", json={"query": "camera", "top_k": 500}).status_code == 422


def test_compare_and_products(client):
    items = client.get("/api/products", params={"limit": 3}).json()
    assert len(items) == 3
    ids = [i["product_id"] for i in items]
    r = client.post("/api/compare", json={"product_ids": ids, "query": "great camera and battery"})
    assert r.status_code == 200
    body = r.json()
    assert [a["aspect"] for a in body["aspects"]] == ["camera", "battery"]
    assert all(p["match"] is not None for p in body["products"])
    detail = client.get(f"/api/products/{ids[0]}")
    assert detail.status_code == 200
    assert client.get("/api/products/does-not-exist").status_code == 404
    assert client.post("/api/compare", json={"product_ids": [ids[0], ids[0]]}).status_code == 422


def test_stats_and_aspects(client):
    stats = client.get("/api/stats").json()
    assert stats["dataset"]["n_products"] == 4
    aspects = client.get("/api/aspects").json()
    assert {a["aspect"] for a in aspects} >= {"camera", "battery", "performance"}
