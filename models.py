"""
Typed Pydantic models for the SQL Query Optimizer environment.
Implements the OpenEnv spec: Observation, Action, Reward.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class ExecutionMetrics(BaseModel):
    """Describes query execution plan characteristics."""
    uses_index: bool = False
    full_scan_count: int = 0
    query_plan: List[str] = Field(default_factory=list)


class SQLObservation(BaseModel):
    """
    Full observation returned after reset() or step().

    Contains everything the agent needs to understand the task
    and formulate a better SQL query.
    """
    task_id: str
    task_name: str
    difficulty: str                   # "easy" | "medium" | "hard"
    description: str
    schema_ddl: str                   # Full DDL of available tables + indexes
    slow_query: str                   # The original slow query (never changes)
    current_query: str                # Agent's best query so far
    step_number: int
    max_steps: int
    slow_metrics: ExecutionMetrics    # Execution plan of the slow query
    last_feedback: str = ""           # Human-readable feedback from last step
    last_reward: float = 0.0          # Reward from last step


class SQLAction(BaseModel):
    """The action an agent takes: submit an optimized SQL query."""
    optimized_query: str = Field(
        ...,
        description="The rewritten SQL query to evaluate.",
        examples=["SELECT user_id, username, email FROM users WHERE is_active = 1"],
    )


class RewardBreakdown(BaseModel):
    """Granular reward components, each in [0, 1] range."""
    validity: float = Field(0.0, ge=0.0, le=1.0,
                            description="Can the query execute without error?")
    correctness: float = Field(0.0, ge=0.0, le=1.0,
                               description="Does it return the correct result set?")
    performance: float = Field(0.0, ge=0.0, le=1.0,
                               description="Does it use indexes / avoid full scans?")
    style: float = Field(0.0, ge=0.0, le=1.0,
                         description="Avoids anti-patterns like SELECT * and IN subqueries?")


class SQLReward(BaseModel):
    """Reward signal returned after each step()."""
    value: float = Field(..., ge=0.0, le=1.0, description="Total reward (weighted sum)")
    breakdown: RewardBreakdown
    feedback: str = Field(..., description="Human-readable explanation of the score")


class StepResult(BaseModel):
    """Full result returned by step()."""
    observation: SQLObservation
    reward: SQLReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Snapshot of current environment state, returned by state()."""
    task_id: str
    step_number: int
    max_steps: int
    best_reward: float
    done: bool
    current_query: str = ""
