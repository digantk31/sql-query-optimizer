#!/usr/bin/env python3
"""
Baseline inference script for SQL Query Optimizer Environment.

Runs an LLM agent against all three tasks and reports scores.

Required environment variables:
  API_BASE_URL  — Base URL for the LLM API
  MODEL_NAME    — Model identifier
  HF_TOKEN      — API key (no default — must be set)

Usage:
  set API_BASE_URL=https://api.openai.com/v1
  set MODEL_NAME=gpt-4o-mini
  set HF_TOKEN=sk-...
  python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

from openai import OpenAI

from env import SQLQueryOptimizerEnv
from models import SQLAction
from tasks import TASK_ORDER, TASKS

# ── Environment variables (hackathon required format) ─────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")          # no default — must be provided

TEMPERATURE  = 0.1
MAX_TOKENS   = 1024

SYSTEM_PROMPT = """\
You are an expert database engineer specialising in SQL query optimisation.
Your job is to rewrite slow SQL queries to be more efficient.

Rules you MUST follow:
1. The rewritten query MUST return the EXACT same result set as the original
   (same rows, same columns requested by the task).
2. Common optimisations to apply:
   - Replace SELECT * with only the required columns.
   - Convert IN (SELECT ...) subqueries to explicit JOINs.
   - Replace correlated subqueries in the SELECT list with JOIN + GROUP BY.
   - Exploit available indexes (listed in the schema comments).
3. Return ONLY the optimised SQL, no markdown fences, no explanation.
"""

# ── LLM helpers ───────────────────────────────────────────────────────────────

def build_user_message(obs_dict: dict, feedback: str) -> str:
    lines = [
        f"Task ({obs_dict['difficulty']}): {obs_dict['description']}",
        "",
        "=== Database Schema ===",
        obs_dict["schema_ddl"],
        "",
        "=== Slow Query to Optimise ===",
        obs_dict["slow_query"],
    ]
    if feedback:
        lines += [
            "",
            "=== Your Previous Attempt ===",
            obs_dict["current_query"],
            "",
            "=== Grader Feedback ===",
            feedback,
            "",
            "Please fix the issues described above and return an improved query.",
        ]
    else:
        lines += ["", "Write an optimised version of the slow query."]
    return "\n".join(lines)


def call_llm(client: OpenAI, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (response.choices[0].message.content or "").strip()
    for fence in ("```sql", "```SQL", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task(env: SQLQueryOptimizerEnv, client: OpenAI, task_id: str) -> float:
    task_meta = TASKS[task_id]

    # ── START log (required structured format) ────────────────────────────────
    print(f"[START] task_id={task_id} difficulty={task_meta['difficulty']} "
          f"max_steps={task_meta['max_steps']}")

    obs      = env.reset(task_id)
    obs_dict = obs.model_dump()
    feedback = ""
    best_reward: float = 0.0

    while True:
        step_num = obs.step_number + 1

        try:
            user_msg  = build_user_message(obs_dict, feedback)
            optimised = call_llm(client, user_msg)
        except Exception as exc:
            print(f"[STEP] task_id={task_id} step={step_num} reward=0.0 "
                  f"error=LLM_FAILURE detail={exc}")
            break

        result   = env.step(SQLAction(optimized_query=optimised))
        reward   = result.reward

        # ── STEP log (required structured format) ─────────────────────────────
        print(f"[STEP] task_id={task_id} step={step_num} "
              f"reward={reward.value:.4f} "
              f"validity={reward.breakdown.validity:.2f} "
              f"correctness={reward.breakdown.correctness:.2f} "
              f"performance={reward.breakdown.performance:.2f} "
              f"style={reward.breakdown.style:.2f} "
              f"done={result.done}")

        if reward.value > best_reward:
            best_reward = reward.value

        obs      = result.observation
        obs_dict = obs.model_dump()
        feedback = reward.feedback

        if result.done:
            break

    # ── END log (required structured format) ──────────────────────────────────
    print(f"[END] task_id={task_id} best_reward={best_reward:.4f}")
    return best_reward


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> Dict[str, float]:
    print(f"[INFO] SQL Query Optimizer Baseline Inference")
    print(f"[INFO] API_BASE_URL={API_BASE_URL}")
    print(f"[INFO] MODEL_NAME={MODEL_NAME}")

    if not HF_TOKEN:
        print("[WARNING] HF_TOKEN is not set — API calls will fail.")

    client = OpenAI(api_key=HF_TOKEN or "placeholder", base_url=API_BASE_URL)
    env    = SQLQueryOptimizerEnv()

    results: Dict[str, float] = {}
    t0 = time.time()

    try:
        for task_id in TASK_ORDER:
            results[task_id] = run_task(env, client, task_id)
    finally:
        env.close()

    elapsed = time.time() - t0
    overall = sum(results.values()) / len(results)

    print(f"\n[SUMMARY] overall_average={overall:.4f} elapsed_seconds={elapsed:.1f}")
    for task_id, score in results.items():
        print(f"[SUMMARY] task={task_id} score={score:.4f}")

    output = {
        "task_scores":     results,
        "overall_average": overall,
        "model":           MODEL_NAME,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open("baseline_results.json", "w") as fh:
        json.dump(output, fh, indent=2)
    print("[INFO] Results saved to baseline_results.json")

    return results


if __name__ == "__main__":
    results = main()
    if all(v == 0.0 for v in results.values()):
        sys.exit(1)