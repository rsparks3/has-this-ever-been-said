"""Has this ever been said? - FastAPI service.

Run:  uvicorn server:app --host 0.0.0.0 --port 8010
Then open http://localhost:8010

GET  /            the dashboard
GET  /api/info    model id + whether it has finished loading
POST /api/score   {"text": "..."} -> per-token probabilities (see scorer.py)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scorer import MAX_TOKENS, Scorer

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "docs"

scorer = Scorer()
app = FastAPI(title="Has this ever been said?", version="0.1")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def info() -> dict:
    return {"model_id": scorer.model_id, "ready": scorer.ready, "error": scorer.error,
            "max_tokens": MAX_TOKENS}


@app.post("/api/score")
def score(req: ScoreRequest) -> dict:
    if scorer.error:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {scorer.error}")
    if not scorer.ready:
        raise HTTPException(status_code=503, detail="Model is still loading; try again in a moment.")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")
    return scorer.score(text).as_dict()
