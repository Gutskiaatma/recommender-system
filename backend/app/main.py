"""FastAPI application.

Run from the project root:
    uvicorn backend.app.main:app --reload --port 8000

Interactive API docs: http://localhost:8000/docs
If ``frontend/dist`` exists (after ``npm run build``) the UI is served at "/" too.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.dependencies import load_recommender
from backend.app.routers import meta, products, recommend
from reviewrec.config import PROJECT_ROOT, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommender, app.state.load_error = load_recommender()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Review-Based Product Recommender",
        version="1.0.0",
        description="Recommends phones from natural-language preferences using semantic retrieval, "
        "aspect-based sentiment analysis of customer reviews and Bayesian quality signals.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get("api.cors_origins", []),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(meta.router)
    app.include_router(recommend.router)
    app.include_router(products.router)

    dist = PROJECT_ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and Path(dist.resolve()) in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
