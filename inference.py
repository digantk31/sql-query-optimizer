#!/usr/bin/env python3
"""
Baseline inference script for SQL Query Optimizer Environment.

Runs an LLM agent against all three tasks and reports scores.

Required environment variables:
  API_BASE_URL  — Base URL for the LLM API  (e.g., https://api.openai.com/v1)
  MODEL_NAME    — Model identifier           (e.g., gpt-4o-mini)
  HF_TOKEN      — API key / Hugging Face token

Usage:
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export HF_TOKEN=sk-...
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

# ─────────────────────────────────────────────── configuration ───────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN: str     = os.environ.get("HF_TOKEN",     "")

TEMPERATURE: float = 0.1
MAX_TOKENS: int    = 1024

SYSTEM_PROMPT = """\
You are an expert database engineer specialising in SQL query optimisation.
Your job is to rewrite slow SQL queries to be more efficient.

Rules you MUST follow:
1. The rewritten query MUST return the EXACT same result set as the original
   (same rows, same columns requested by the task).
2. Common optimisations to apply:
   - Replace SELECT * with only the required columns.
   - Convert IN (SELECT …) subqueries to explicit JOINs.
   - Replace correlated subqueries in the SELECT list with JOIN + GROUP BY.
   - Exploit available indexes (listed in the schema comments).
3. Return ONLY the optimised SQL — no markdown fences, no explanation.
"""

# ──────────────────────────────────────────────── agent helpers ──────────────


def build_user_message(obs_dict: dict, feedback: str) -> str:
    """Construct the user turn for the LLM."""
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
        lines.append("")
        lines.append("Write an optimised version of the slow query.")

    return "\n".join(lines)


def call_llm(client: OpenAI, user_message: str) -> str:
    """Call the LLM and return the raw text response."""
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
    text = response.choices[0].message.content or ""
    # Strip markdown code fences if the model wraps its answer
    text = text.strip()
    for fence in ("```sql", "```SQL", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ─────────────────────────────────────────────── per-task runner ─────────────


def run_task(env: SQLQueryOptimizerEnv, client: OpenAI, task_id: str) -> float:
    """
    Run one full episode for the given task.

    Returns the best reward achieved during the episode.
    """
    print(f"\n{'═' * 62}")
    task_meta = TASKS[task_id]
    print(f"  Task   : {task_meta['name']}  [{task_meta['difficulty'].upper()}]")
    print(f"  Task ID: {task_id}")
    print("═" * 62)

    obs      = env.reset(task_id)
    obs_dict = obs.model_dump()
    feedback = ""
    best_reward: float = 0.0

    print(f"  Description: {obs.description[:120]}…")
    print(f"  Max steps  : {obs.max_steps}")
    print(f"\n  Slow query:\n  {obs.slow_query[:300]}")

    while True:
        step_num = obs.step_number + 1
        print(f"\n  ── Step {step_num} ──────────────────────────────────────────")

        # ── LLM generates an optimised query ──────────────────────────────
        try:
            user_msg       = build_user_message(obs_dict, feedback)
            optimised      = call_llm(client, user_msg)
        except Exception as exc:
            print(f"  [LLM ERROR] {exc}")
            break

        preview = optimised.replace("\n", " ")[:180]
        print(f"  Submitted: {preview}…" if len(optimised) > 180 else f"  Submitted: {preview}")

        # ── Submit to environment ──────────────────────────────────────────
        result   = env.step(SQLAction(optimized_query=optimised))
        reward   = result.reward

        print(f"  Reward   : {reward.value:.3f}  "
              f"(validity={reward.breakdown.validity:.2f} "
              f"correctness={reward.breakdown.correctness:.2f} "
              f"performance={reward.breakdown.performance:.2f} "
              f"style={reward.breakdown.style:.2f})")
        print(f"  Feedback : {reward.feedback}")

        if reward.value > best_reward:
            best_reward = reward.value

        obs      = result.observation
        obs_dict = obs.model_dump()
        feedback = reward.feedback

        if result.done:
            done_reason = (
                "perfect score" if reward.value >= 0.95 else "step budget exhausted"
            )
            print(f"\n  Episode complete ({done_reason}).")
            break

    print(f"\n  Best reward for '{task_id}': {best_reward:.3f}")
    return best_reward


# ─────────────────────────────────────────────────────────── main ────────────


def main() -> Dict[str, float]:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       SQL Query Optimizer — Baseline Inference           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  API : {API_BASE_URL[:54]:<54}║")
    print(f"║  Model: {MODEL_NAME[:53]:<53}║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not HF_TOKEN:
        print("\n[WARNING] HF_TOKEN is not set — API calls may fail.\n")

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

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  FINAL SCORES")
    print("═" * 62)
    for task_id, score in results.items():
        t = TASKS[task_id]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {t['name']:<38}  {bar}  {score:.3f}")

    overall = sum(results.values()) / len(results)
    print(f"\n  Overall average : {overall:.3f}")
    print(f"  Wall-clock time : {elapsed:.1f}s")
    print("═" * 62)

    # Persist results for automated evaluation
    output = {
        "task_scores":     results,
        "overall_average": overall,
        "model":           MODEL_NAME,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open("baseline_results.json", "w") as fh:
        json.dump(output, fh, indent=2)
    print("\n  Results saved → baseline_results.json")

    return results


if __name__ == "__main__":
    results = main()
    # Non-zero exit if every task scored 0 (likely a config problem)
    if all(v == 0.0 for v in results.values()):
        sys.exit(1)
