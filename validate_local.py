#!/usr/bin/env python3
"""
validate_local.py — Pre-submission validator for SQL Query Optimizer.

Mimics the official hackathon validation checklist:
  1. openenv.yaml — exists and has required fields
  2. Required files present
  3. Dockerfile exists and is non-trivial
  4. inference.py named correctly and is at root
  5. Environment instantiation — reset()/step()/state() all work
  6. All 3 tasks enumerable
  7. All graders return scores in [0.0, 1.0]
  8. Baseline results reproducible

Run:
    python validate_local.py

Exits with code 0 (all pass) or 1 (any failure).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    _results.append((name, condition, detail))
    mark = PASS if condition else FAIL
    line = f"  {mark}  {name}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return condition


def section(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print("─" * 62)


# ─────────────────────────────────────────────────────────── checks ──────────

def check_required_files():
    section("1. Required Files")
    required = [
        "openenv.yaml",
        "Dockerfile",
        "requirements.txt",
        "inference.py",
        "README.md",
        "models.py",
        "db.py",
        "tasks.py",
        "env.py",
        "app.py",
    ]
    all_present = True
    for fname in required:
        present = (ROOT / fname).exists()
        check(f"{fname} present", present)
        all_present = all_present and present
    return all_present


def check_openenv_yaml():
    section("2. openenv.yaml Validation")
    try:
        import yaml
        with open(ROOT / "openenv.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        # Fallback: parse manually
        import re
        with open(ROOT / "openenv.yaml", encoding="utf-8") as f:
            raw = f.read()
        cfg = {}
        for key in ["name","version","description","tasks","endpoints","reward_range"]:
            cfg[key] = key in raw

        check("openenv.yaml readable (yaml not installed, minimal check)", True,
              "install pyyaml for full validation")
        return True

    required_keys = ["name","version","description","tasks","endpoints","reward_range"]
    for k in required_keys:
        check(f"openenv.yaml has '{k}'", k in cfg)

    tasks_cfg = cfg.get("tasks", [])
    check("openenv.yaml has >= 3 tasks", len(tasks_cfg) >= 3, f"found {len(tasks_cfg)}")

    for t in tasks_cfg:
        has_id   = "id"         in t
        has_diff = "difficulty" in t
        has_max  = "max_steps"  in t
        check(f"  task '{t.get('id','?')}' has id/difficulty/max_steps", has_id and has_diff and has_max)

    rr = cfg.get("reward_range", [])
    check("reward_range is [0.0, 1.0]", list(rr) == [0.0, 1.0], f"got {rr}")

    return True


def check_dockerfile():
    section("3. Dockerfile")
    df = ROOT / "Dockerfile"
    if not df.exists():
        check("Dockerfile exists", False)
        return False

    content = df.read_text(encoding="utf-8")
    check("Dockerfile has FROM instruction", "FROM" in content)
    check("Dockerfile exposes port 7860", "7860" in content)
    check("Dockerfile has CMD or ENTRYPOINT", "CMD" in content or "ENTRYPOINT" in content)
    check("Dockerfile has COPY instruction", "COPY" in content)
    check("Dockerfile is non-trivial (> 5 lines)", content.count("\n") > 5,
          f"{content.count(chr(10))} lines")
    return True


def check_inference_script():
    section("4. inference.py")
    inf = ROOT / "inference.py"
    if not inf.exists():
        check("inference.py at project root", False)
        return False

    content = inf.read_text(encoding="utf-8")
    check("inference.py at project root", True)
    check("inference.py reads API_BASE_URL", "API_BASE_URL" in content)
    check("inference.py reads MODEL_NAME", "MODEL_NAME" in content)
    check("inference.py reads HF_TOKEN", "HF_TOKEN" in content)
    check("inference.py uses OpenAI client", "OpenAI" in content)
    check("inference.py has main() function", "def main" in content)
    return True


def check_environment_api():
    section("5. Environment API (reset/step/state)")
    sys.path.insert(0, str(ROOT))

    try:
        from env import SQLQueryOptimizerEnv
        from models import SQLAction, SQLObservation, StepResult, EnvironmentState
    except ImportError as e:
        check("Environment imports successfully", False, str(e))
        print(f"\n  {WARN}  Install dependencies: pip install pydantic fastapi openai")
        return False

    check("Environment imports successfully", True)

    env = SQLQueryOptimizerEnv()

    # reset()
    try:
        obs = env.reset()
        is_obs = isinstance(obs, SQLObservation)
        check("reset() returns SQLObservation", is_obs)
        check("reset() observation has task_id", hasattr(obs, "task_id") and obs.task_id != "")
        check("reset() observation has schema_ddl", hasattr(obs, "schema_ddl") and len(obs.schema_ddl) > 50)
        check("reset() observation has slow_query", hasattr(obs, "slow_query") and len(obs.slow_query) > 10)
    except Exception as e:
        check("reset() works without error", False, str(e))
        env.close()
        return False

    # step()
    try:
        result = env.step(SQLAction(
            optimized_query="SELECT user_id, username, email FROM users WHERE is_active = 1"
        ))
        is_step = isinstance(result, StepResult)
        check("step() returns StepResult", is_step)
        check("step() reward.value in [0,1]",
              0.0 <= result.reward.value <= 1.0, f"got {result.reward.value:.3f}")
        check("step() has done flag", isinstance(result.done, bool))
        check("step() has info dict", isinstance(result.info, dict))
    except Exception as e:
        check("step() works without error", False, str(e))
        env.close()
        return False

    # state()
    try:
        st = env.state()
        is_state = isinstance(st, EnvironmentState)
        check("state() returns EnvironmentState", is_state)
        check("state() has task_id", hasattr(st, "task_id") and st.task_id != "")
    except Exception as e:
        check("state() works without error", False, str(e))

    env.close()
    return True


def check_tasks():
    section("6. Tasks Enumerable (>= 3)")
    sys.path.insert(0, str(ROOT))

    try:
        from env import SQLQueryOptimizerEnv
    except ImportError:
        check("Tasks check skipped (pydantic not installed)", True, "install dependencies")
        return True

    env = SQLQueryOptimizerEnv()
    tasks = env.list_tasks()
    check("list_tasks() returns >= 3 tasks", len(tasks) >= 3, f"got {len(tasks)}")

    difficulties = [t.get("difficulty") for t in tasks]
    check("Has easy task",   "easy"   in difficulties)
    check("Has medium task", "medium" in difficulties)
    check("Has hard task",   "hard"   in difficulties)

    for t in tasks:
        check(f"  task '{t['task_id']}' has all required fields",
              all(k in t for k in ["task_id","name","difficulty","max_steps","description"]))

    env.close()
    return True


def check_graders():
    section("7. Graders Return [0.0, 1.0]")
    sys.path.insert(0, str(ROOT))

    try:
        from env import SQLQueryOptimizerEnv
        from models import SQLAction
        from tasks import TASK_ORDER
    except ImportError:
        check("Grader check skipped (pydantic not installed)", True)
        return True

    env = SQLQueryOptimizerEnv()
    queries = [
        "SELECT * FROM users WHERE is_active = 1",
        "SELECT user_id, username, email FROM users WHERE is_active = 1",
        "COMPLETELY BROKEN SQL",
        "SELECT order_id, user_id, total_amount FROM orders WHERE user_id IN (SELECT user_id FROM users WHERE country='USA' AND is_active=1) AND status='delivered'",
    ]

    all_ok = True
    for task_id in TASK_ORDER:
        env.reset(task_id)
        for q in queries:
            try:
                r = env.step(SQLAction(optimized_query=q))
                v = r.reward.value
                in_range = 0.0 <= v <= 1.0
                if not in_range:
                    all_ok = False
                    print(f"    {FAIL}  task={task_id} score={v:.3f} OUT OF RANGE")
                if r.done:
                    env.reset(task_id)
            except Exception as exc:
                # step() after done raises RuntimeError — that's expected
                if "Episode is over" in str(exc):
                    env.reset(task_id)
                else:
                    all_ok = False
                    print(f"    {FAIL}  Unexpected exception: {exc}")

    check("All grader scores in [0.0, 1.0]", all_ok)
    env.close()
    return all_ok


def check_baseline_reproducibility():
    section("8. Baseline Reproducibility (determinism check)")
    sys.path.insert(0, str(ROOT))

    try:
        from env import SQLQueryOptimizerEnv
        from models import SQLAction
    except ImportError:
        check("Reproducibility check skipped (pydantic not installed)", True)
        return True

    q = "SELECT user_id, username, email FROM users WHERE is_active = 1"
    scores = []
    for _ in range(3):
        env = SQLQueryOptimizerEnv()
        env.reset("select_star_removal")
        r = env.step(SQLAction(optimized_query=q))
        scores.append(r.reward.value)
        env.close()

    all_same = len(set(scores)) == 1
    check("Same query produces identical scores across runs",
          all_same, f"scores={scores}")
    return all_same


# ─────────────────────────────────────────────────────────── runner ───────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   SQL Query Optimizer — Pre-Submission Validator         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Working directory: {ROOT}\n")

    t0 = time.time()

    check_required_files()
    check_openenv_yaml()
    check_dockerfile()
    check_inference_script()
    check_environment_api()
    check_tasks()
    check_graders()
    check_baseline_reproducibility()

    elapsed = time.time() - t0
    passed  = sum(1 for _, ok, _ in _results if ok)
    failed  = sum(1 for _, ok, _ in _results if not ok)
    total   = len(_results)

    print(f"\n{'═' * 62}")
    print(f"  Results: {passed}/{total} passed  ({failed} failed)  [{elapsed:.2f}s]")

    if failed:
        print(f"\n  {FAIL}  FAILED CHECKS:")
        for name, ok, detail in _results:
            if not ok:
                print(f"      {FAIL}  {name}  {detail}")
        print("═" * 62)
        print(f"\n  Submission is NOT ready. Fix the {failed} failed check(s) above.")
        sys.exit(1)
    else:
        print(f"  {PASS}  ALL CHECKS PASSED — ready to submit!")
        print("═" * 62)
        sys.exit(0)


if __name__ == "__main__":
    main()