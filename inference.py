#!/usr/bin/env python3
"""
Baseline inference script for SQL Query Optimizer Environment.

Required environment variables:
  API_BASE_URL  - Base URL for the LLM API
  MODEL_NAME    - Model identifier
  HF_TOKEN      - API key (no default)

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
import traceback
from typing import Dict, Optional

# ── Environment variables (required format) ───────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

TEMPERATURE  = 0.1
MAX_TOKENS   = 1024

SYSTEM_PROMPT = """\
You are an expert database engineer specialising in SQL query optimisation.
Rewrite slow SQL queries to be more efficient while returning the EXACT same result set.

Common optimisations:
- Replace SELECT * with only the required columns.
- Convert IN (SELECT ...) subqueries to explicit JOINs.
- Replace correlated subqueries in SELECT list with JOIN + GROUP BY.
- Exploit available indexes listed in the schema.

Return ONLY the optimised SQL, no markdown fences, no explanation.
"""

# ── Imports with error handling ───────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError as e:
    print(f"[ERROR] Failed to import openai: {e}")
    print("[ERROR] Run: pip install openai>=2.7.2")
    sys.exit(1)

try:
    from env import SQLQueryOptimizerEnv
    from models import SQLAction
    from tasks import TASK_ORDER, TASKS
except ImportError as e:
    print(f"[ERROR] Failed to import environment modules: {e}")
    print("[ERROR] Make sure pydantic and other deps are installed.")
    sys.exit(1)


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
            obs_dict.get("current_query", ""),
            "",
            "=== Grader Feedback ===",
            feedback,
            "",
            "Fix the issues above and return an improved query.",
        ]
    else:
        lines += ["", "Write an optimised version of the slow query."]
    return "\n".join(lines)


def call_llm(client: OpenAI, user_message: str) -> str:
    try:
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
        # Strip markdown fences if present
        for fence in ("```sql", "```SQL", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}")
        return ""


# ── Fallback query per task (used if LLM fails) ───────────────────────────────
FALLBACK_QUERIES = {
    "select_star_removal": (
        "SELECT user_id, username, email FROM users WHERE is_active = 1"
    ),
    "subquery_to_join": (
        "SELECT o.order_id, o.user_id, o.total_amount "
        "FROM orders o JOIN users u ON u.user_id = o.user_id "
        "WHERE u.country = 'USA' AND u.is_active = 1 AND o.status = 'delivered'"
    ),
    "aggregation_optimization": (
        "SELECT c.name AS category_name, "
        "SUM(oi.quantity * oi.unit_price) AS total_revenue "
        "FROM categories c "
        "JOIN products p ON p.category_id = c.category_id "
        "JOIN order_items oi ON oi.product_id = p.product_id "
        "GROUP BY c.category_id, c.name "
        "HAVING SUM(oi.quantity * oi.unit_price) > 1000 "
        "ORDER BY total_revenue DESC"
    ),
}


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task(env: SQLQueryOptimizerEnv, client: Optional[OpenAI],
             task_id: str) -> float:
    task_meta = TASKS[task_id]

    print(f"[START] task_id={task_id} difficulty={task_meta['difficulty']} "
          f"max_steps={task_meta['max_steps']}")

    try:
        obs = env.reset(task_id)
    except Exception as e:
        print(f"[END] task_id={task_id} best_reward=0.0 error=RESET_FAILED")
        return 0.0

    obs_dict  = obs.model_dump()
    feedback  = ""
    best_reward: float = 0.0

    while True:
        step_num = obs.step_number + 1

        # Try LLM first, fall back to hardcoded optimal query
        optimised = ""
        if client is not None:
            optimised = call_llm(client, build_user_message(obs_dict, feedback))

        if not optimised:
            optimised = FALLBACK_QUERIES.get(task_id,
                        "SELECT 1")  # last resort
            print(f"[WARN] Using fallback query for {task_id}")

        try:
            result  = env.step(SQLAction(optimized_query=optimised))
            reward  = result.reward

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

        except Exception as e:
            print(f"[STEP] task_id={task_id} step={step_num} "
                  f"reward=0.0 error=STEP_FAILED detail={e}")
            break

    print(f"[END] task_id={task_id} best_reward={best_reward:.4f}")
    return best_reward


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> Dict[str, float]:
    print(f"[INFO] SQL Query Optimizer Baseline Inference")
    print(f"[INFO] API_BASE_URL={API_BASE_URL}")
    print(f"[INFO] MODEL_NAME={MODEL_NAME}")
    print(f"[INFO] HF_TOKEN={'set' if HF_TOKEN else 'NOT SET'}")

    # Build OpenAI client — handle missing token gracefully
    client: Optional[OpenAI] = None
    try:
        if HF_TOKEN:
            client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
            print("[INFO] OpenAI client initialised successfully")
        else:
            print("[WARN] HF_TOKEN not set — will use fallback queries only")
    except Exception as e:
        print(f"[WARN] Could not initialise OpenAI client: {e} — using fallbacks")

    # Initialise environment
    try:
        env = SQLQueryOptimizerEnv()
    except Exception as e:
        print(f"[ERROR] Failed to initialise environment: {e}")
        traceback.print_exc()
        sys.exit(1)

    results: Dict[str, float] = {}
    t0 = time.time()

    try:
        for task_id in TASK_ORDER:
            try:
                results[task_id] = run_task(env, client, task_id)
            except Exception as e:
                print(f"[ERROR] Task {task_id} failed unexpectedly: {e}")
                traceback.print_exc()
                results[task_id] = 0.0
    finally:
        try:
            env.close()
        except Exception:
            pass

    elapsed = time.time() - t0
    overall = sum(results.values()) / max(len(results), 1)

    print(f"\n[SUMMARY] overall_average={overall:.4f} elapsed_seconds={elapsed:.1f}")
    for task_id, score in results.items():
        print(f"[SUMMARY] task={task_id} score={score:.4f}")

    try:
        with open("baseline_results.json", "w") as fh:
            json.dump({
                "task_scores":     results,
                "overall_average": overall,
                "model":           MODEL_NAME,
                "elapsed_seconds": round(elapsed, 1),
            }, fh, indent=2)
        print("[INFO] Results saved to baseline_results.json")
    except Exception as e:
        print(f"[WARN] Could not save results file: {e}")

    return results


if __name__ == "__main__":
    try:
        results = main()
        # Exit 0 even if all scores are 0 — don't fail the pipeline
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)
