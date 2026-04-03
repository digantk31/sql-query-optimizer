"""
FastAPI application — HTTP wrapper around SQLQueryOptimizerEnv.

Endpoints:
  GET  /              — Interactive demo UI
  GET  /health        — Liveness probe
  GET  /tasks         — List available tasks
  POST /reset         — Start a new episode
  POST /step          — Submit an optimized query
  GET  /state         — Current episode state
  GET  /docs          — Swagger UI

Runs on port 7860 for Hugging Face Spaces compatibility.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from env import SQLQueryOptimizerEnv
from models import SQLAction, SQLObservation, StepResult, EnvironmentState

# ─────────────────────────────────────────────────────────── app setup ───────

app = FastAPI(
    title="SQL Query Optimizer Environment",
    description=(
        "An OpenEnv-compatible RL environment where AI agents learn to "
        "rewrite slow SQL queries for better performance."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the demo UI)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Single global environment instance (one session per Space pod)
_env = SQLQueryOptimizerEnv()

# ─────────────────────────────────────────────────────────── UI ──────────────


@app.get("/", include_in_schema=False)
def root():
    """Serve the interactive demo UI."""
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return JSONResponse({
        "name": "SQL Query Optimizer",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
    })


# ─────────────────────────────────────────────────────────── endpoints ───────


@app.get("/health", tags=["meta"])
def health():
    """Liveness probe — always returns 200 if the server is running."""
    return {"status": "ok", "environment": "sql-query-optimizer", "version": "1.0.0"}


@app.get("/tasks", tags=["meta"])
def list_tasks():
    """Return metadata for all available tasks."""
    return _env.list_tasks()


@app.post("/reset", response_model=SQLObservation, tags=["openenv"])
def reset(task_id: Optional[str] = Query(default=None, description="Task ID to start")):
    """
    Reset the environment and start a new episode.

    Valid task IDs: ``select_star_removal``, ``subquery_to_join``, ``aggregation_optimization``
    """
    try:
        obs = _env.reset(task_id)
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult, tags=["openenv"])
def step(action: SQLAction):
    """
    Submit an optimized SQL query and receive a graded reward.

    Body must contain ``optimized_query`` (string).
    Returns new observation, reward breakdown, and done flag.
    """
    try:
        result = _env.step(action)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=EnvironmentState, tags=["openenv"])
def state():
    """Return a lightweight snapshot of the current episode state."""
    return _env.state()


# ─────────────────────────────────────────────────────────── entrypoint ──────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
