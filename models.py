"""
Typed Pydantic models for the SQL Query Optimizer environment.
Scores are strictly in (0.0, 1.0) exclusive as required by OpenEnv validator.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List


class ExecutionMetrics(BaseModel):
    uses_index: bool = False
    full_scan_count: int = 0
    query_plan: List[str] = Field(default_factory=list)


class SQLObservation(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    description: str
    schema_ddl: str
    slow_query: str
    current_query: str
    step_number: int
    max_steps: int
    slow_metrics: ExecutionMetrics
    last_feedback: str = ""
    last_reward: float = 0.001


class SQLAction(BaseModel):
    optimized_query: str = Field(
        ...,
        description="The rewritten SQL query to evaluate.",
    )


class RewardBreakdown(BaseModel):
    validity:    float = Field(0.001, gt=0.0, lt=1.0)
    correctness: float = Field(0.001, gt=0.0, lt=1.0)
    performance: float = Field(0.001, gt=0.0, lt=1.0)
    style:       float = Field(0.001, gt=0.0, lt=1.0)

    @field_validator('validity', 'correctness', 'performance', 'style', mode='before')
    @classmethod
    def clamp_strict(cls, v):
        return max(0.001, min(0.999, float(v)))


class SQLReward(BaseModel):
    value:     float = Field(..., gt=0.0, lt=1.0)
    breakdown: RewardBreakdown
    feedback:  str = Field(...,)

    @field_validator('value', mode='before')
    @classmethod
    def clamp_value(cls, v):
        return max(0.001, min(0.999, float(v)))


class StepResult(BaseModel):
    observation: SQLObservation
    reward:      SQLReward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    task_id:      str
    step_number:  int
    max_steps:    int
    best_reward:  float = 0.001
    done:         bool
    current_query: str = ""
