"""
SQLQueryOptimizerEnv — OpenEnv-compatible Python class.

Implements the standard interface:
  reset(task_id?)  → SQLObservation
  step(action)     → StepResult
  state()          → EnvironmentState
  close()          → None
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


class SQLQueryOptimizerEnv:
    """
    An RL environment where agents learn to rewrite slow SQL queries.

    The environment:
      • Maintains a live SQLite database pre-populated with e-commerce data.
      • Exposes three tasks of increasing difficulty.
      • Grades each submitted query across four orthogonal dimensions.
      • Provides partial-credit rewards at every step (dense signal).

    Usage::

        env = SQLQueryOptimizerEnv()
        obs = env.reset("select_star_removal")
        result = env.step(SQLAction(optimized_query="SELECT user_id, username, email ..."))
        print(result.reward.value)
        env.close()
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._task_id: Optional[str] = None
        self._step: int = 0
        self._best_reward: float = 0.0
        self._best_query: str = ""
        self._done: bool = False
        self._last_feedback: str = ""
        self._last_reward: float = 0.0
        self._conn = create_database()

    # ──────────────────────────────────────────────────────────── public API ──

    def reset(self, task_id: Optional[str] = None) -> SQLObservation:
        """
        Start a new episode.

        Args:
            task_id: One of the task IDs returned by ``list_tasks()``.
                     Defaults to the first (easiest) task.

        Returns:
            Initial observation containing schema, slow query, and task description.
        """
        if task_id is None:
            task_id = TASK_ORDER[0]

        if task_id not in TASKS:
            valid = list(TASKS.keys())
            raise ValueError(f"Unknown task_id '{task_id}'. Valid options: {valid}")

        self._task_id = task_id
        self._step = 0
        self._best_reward = 0.0
        self._done = False
        self._last_feedback = ""
        self._last_reward = 0.0

        task = TASKS[task_id]
        slow_query = task["slow_query"]
        self._best_query = slow_query   # baseline is the slow query itself

        return self._make_obs(slow_query)

    def step(self, action: SQLAction) -> StepResult:
        """
        Submit an optimized query and receive a graded reward.

        Args:
            action: SQLAction with ``optimized_query`` field.

        Returns:
            StepResult with observation, reward, done flag, and info dict.

        Raises:
            RuntimeError: if called before reset() or after episode ends.
        """
        if self._task_id is None:
            raise RuntimeError("No active episode — call reset() first.")
        if self._done:
            raise RuntimeError("Episode is over — call reset() to start a new one.")

        task = TASKS[self._task_id]
        self._step += 1

        # Grade the submitted query
        grader = TASK_GRADERS[self._task_id]
        score, bd_dict, feedback = grader(action.optimized_query, self._conn)

        # Track best query seen so far
        if score > self._best_reward:
            self._best_reward = score
            self._best_query = action.optimized_query

        self._last_feedback = feedback
        self._last_reward = score

        # Episode ends when step budget exhausted or near-perfect score
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
        """Return a lightweight snapshot of the current episode state."""
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
        """Return metadata for all available tasks."""
        return [
            {
                "task_id":    tid,
                "name":       t["name"],
                "difficulty": t["difficulty"],
                "max_steps":  t["max_steps"],
                "description": t["description"],
            }
            for tid, t in TASKS.items()
        ]

    def close(self) -> None:
        """Release database resources."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ──────────────────────────────────────────────────────────── helpers ─────

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
