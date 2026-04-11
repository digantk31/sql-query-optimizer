"""
SQLQueryOptimizerEnv — OpenEnv-compatible Python class.
All reward scores are strictly clamped to (0.001, 0.999).
"""
from __future__ import annotations

import sqlite3
from typing import List, Optional

from db import create_database
from models import (
    EnvironmentState,
    ExecutionMetrics,
    RewardBreakdown,
    SQLAction,
    SQLObservation,
    SQLReward,
    StepResult,
)
from tasks import (
    SCHEMA_DDL,
    TASK_GRADERS,
    TASK_ORDER,
    TASKS,
    get_query_metrics,
)


def _clamp(v: float) -> float:
    """Force any score strictly into (0.001, 0.999) — never 0.0 or 1.0."""
    return max(0.001, min(0.999, float(v)))


class SQLQueryOptimizerEnv:

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._task_id: Optional[str] = None
        self._step: int = 0
        self._best_reward: float = 0.001
        self._best_query: str = ""
        self._done: bool = False
        self._last_feedback: str = ""
        self._last_reward: float = 0.001
        self._conn = create_database()

    def reset(self, task_id: Optional[str] = None) -> SQLObservation:
        if task_id is None:
            task_id = TASK_ORDER[0]
        if task_id not in TASKS:
            valid = list(TASKS.keys())
            raise ValueError(f"Unknown task_id '{task_id}'. Valid options: {valid}")
        self._task_id = task_id
        self._step = 0
        self._best_reward = 0.001
        self._done = False
        self._last_feedback = ""
        self._last_reward = 0.001
        task = TASKS[task_id]
        slow_query = task["slow_query"]
        self._best_query = slow_query
        return self._make_obs(slow_query)

    def step(self, action: SQLAction) -> StepResult:
        if self._task_id is None:
            raise RuntimeError("No active episode — call reset() first.")
        if self._done:
            raise RuntimeError("Episode is over — call reset() to start a new one.")
        task = TASKS[self._task_id]
        self._step += 1
        grader = TASK_GRADERS[self._task_id]
        raw_score, raw_bd, feedback = grader(action.optimized_query, self._conn)
        score = _clamp(raw_score)
        bd_dict = {k: _clamp(v) for k, v in raw_bd.items()}
        if score > self._best_reward:
            self._best_reward = score
            self._best_query = action.optimized_query
        self._last_feedback = feedback
        self._last_reward = score
        self._done = self._step >= task["max_steps"] or score >= 0.95
        reward = SQLReward(
            value=score,
            breakdown=RewardBreakdown(**bd_dict),
            feedback=feedback,
        )
        obs = self._make_obs(action.optimized_query)
        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={
                "best_reward": self._best_reward,
                "step": self._step,
                "task_id": self._task_id,
                "episode_done": self._done,
            },
        )

    def state(self) -> EnvironmentState:
        max_steps = TASKS[self._task_id]["max_steps"] if self._task_id else 0
        return EnvironmentState(
            task_id=self._task_id or "",
            step_number=self._step,
            max_steps=max_steps,
            best_reward=self._best_reward,
            done=self._done,
            current_query=self._best_query,
        )

    def list_tasks(self) -> List[dict]:
        return [
            {
                "task_id": tid,
                "name": t["name"],
                "difficulty": t["difficulty"],
                "max_steps": t["max_steps"],
                "description": t["description"],
            }
            for tid, t in TASKS.items()
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _make_obs(self, current_query: str) -> SQLObservation:
        task = TASKS[self._task_id]
        slow_metrics = get_query_metrics(task["slow_query"], self._conn)
        return SQLObservation(
            task_id=self._task_id,
            task_name=task["name"],
            difficulty=task["difficulty"],
            description=task["description"],
            schema_ddl=SCHEMA_DDL,
            slow_query=task["slow_query"],
            current_query=current_query,
            step_number=self._step,
            max_steps=task["max_steps"],
            slow_metrics=slow_metrics,
            last_feedback=self._last_feedback,
            last_reward=self._last_reward,
        )