"""
FastAPI application — HTTP wrapper around SQLQueryOptimizerEnv.
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
from models import SQLAction, SQLObservation, StepResult, EnvironmentState, SQLReward, RewardBreakdown

app = FastAPI(title="SQL Query Optimizer Environment", version="1.0.0",
              docs_url="/docs", redoc_url="/redoc")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

_env = SQLQueryOptimizerEnv()


def _clamp(v: float) -> float:
    """Force score strictly into (0.001, 0.999) — validator requires exclusive (0,1)."""
    return max(0.001, min(0.999, v))


def _clamp_result(result: StepResult) -> StepResult:
    """Clamp every score field in a StepResult before returning over HTTP."""
    bd = result.reward.breakdown
    clamped_bd = RewardBreakdown(
        validity=_clamp(bd.validity),
        correctness=_clamp(bd.correctness),
        performance=_clamp(bd.performance),
        style=_clamp(bd.style),
    )
    clamped_reward = SQLReward(
        value=_clamp(result.reward.value),
        breakdown=clamped_bd,
        feedback=result.reward.feedback,
    )
    return StepResult(
        observation=result.observation,
        reward=clamped_reward,
        done=result.done,
        info=result.info,
    )


@app.get("/", include_in_schema=False)
def root():
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return JSONResponse({"name": "SQL Query Optimizer", "version": "1.0.0", "docs": "/docs"})


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "environment": "sql-query-optimizer", "version": "1.0.0"}


@app.get("/tasks", tags=["meta"])
def list_tasks():
    return _env.list_tasks()


@app.post("/reset", response_model=SQLObservation, tags=["openenv"])
def reset(task_id: Optional[str] = Query(default=None)):
    try:
        return _env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult, tags=["openenv"])
def step(action: SQLAction):
    try:
        result = _env.step(action)
        return _clamp_result(result)   # ← clamp here before sending over HTTP
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=EnvironmentState, tags=["openenv"])
def state():
    return _env.state()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
