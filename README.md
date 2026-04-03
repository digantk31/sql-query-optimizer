# SQL Query Optimizer — OpenEnv Environment

An RL environment where AI agents learn to rewrite slow SQL queries for
correctness, performance, and style — against a real SQLite e-commerce database.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-brightgreen)]()
[![HF Space](https://img.shields.io/badge/🤗%20Space-live-blue)]()
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)]()

---

## Motivation

SQL query optimisation is a high-value, real-world DBA task performed
millions of times daily. Slow queries account for a significant fraction of
cloud database cost and application latency. This environment teaches agents to:

- Eliminate wasteful `SELECT *` patterns
- Replace correlated subqueries with efficient `JOIN`s
- Restructure aggregations to avoid O(n²) complexity

A well-trained agent could act as an autonomous SQL linter/rewriter in CI
pipelines or IDE plugins.

---

## Environment Description

The environment manages a live SQLite database pre-populated with synthetic
e-commerce data (deterministic, seed 42):

| Table        | Rows  | Description                          |
|--------------|-------|--------------------------------------|
| `users`      | 1 500 | Registered customers                 |
| `categories` | 10    | Two-level product taxonomy           |
| `products`   | 300   | Items for sale                       |
| `orders`     | 3 000 | Customer orders                      |
| `order_items`| 8 000 | Line items within each order         |

Seven indexes are pre-created. The agent's job is to write queries that
exploit them.

---

## Action Space

```json
{
  "optimized_query": "SELECT user_id, username, email FROM users WHERE is_active = 1"
}
```

A single SQL query string. The agent submits one query per step.

---

## Observation Space

| Field              | Type    | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `task_id`          | string  | Active task identifier                           |
| `task_name`        | string  | Human-readable name                              |
| `difficulty`       | string  | `easy` / `medium` / `hard`                       |
| `description`      | string  | Exact specification of required output           |
| `schema_ddl`       | string  | Full `CREATE TABLE` + `CREATE INDEX` statements  |
| `slow_query`       | string  | The original slow query (never changes)          |
| `current_query`    | string  | Agent's best query so far                        |
| `step_number`      | integer | Current step in the episode                      |
| `max_steps`        | integer | Budget for this task                             |
| `slow_metrics`     | object  | `uses_index`, `full_scan_count`, `query_plan`    |
| `last_feedback`    | string  | Grader's human-readable message from last step   |
| `last_reward`      | float   | Reward from last step                            |

---

## Reward Function

Every step yields a **dense, partial-credit** reward in `[0.0, 1.0]`:

```
reward = validity + correctness + performance + style
```

| Dimension     | Max   | Criteria                                              |
|---------------|-------|-------------------------------------------------------|
| `validity`    | 0.10  | Query executes without SQL error                      |
| `correctness` | 0.35–0.40 | Result set matches reference (checked by PK sets) |
| `performance` | 0.20–0.30 | Index access detected in `EXPLAIN QUERY PLAN`     |
| `style`       | 0.20–0.30 | No `SELECT *`, no correlated subqueries           |

Partial credit is given within each dimension, so agents receive a signal
even for partially-correct or partially-optimised queries.

---

## Tasks

### Task 1 — SELECT * Elimination *(easy, 5 steps)*

**Problem:** `SELECT * FROM users WHERE is_active = 1` fetches all 9 columns
when only 3 are needed.

**Goal:** Return `user_id, username, email` for all active users.

**Perfect score requires:**
- Valid SQL ✓
- Correct `user_id` set returned ✓
- No `SELECT *` ✓
- Index scan on `idx_users_active` ✓

**Baseline score:** ~0.70 (typical GPT-4o-mini first attempt)

---

### Task 2 — Correlated Subquery → JOIN *(medium, 6 steps)*

**Problem:**
```sql
SELECT order_id, user_id, total_amount
FROM   orders
WHERE  user_id IN (
    SELECT user_id FROM users
    WHERE  country = 'USA' AND is_active = 1
) AND status = 'delivered'
```
The `IN (SELECT …)` forces repeated inner-query execution.

**Goal:** Rewrite as a `JOIN`, returning same result set.

**Perfect score requires:**
- Correct delivered-order set ✓
- Uses `JOIN` not `IN (SELECT …)` ✓
- Index access on `idx_users_country` and `idx_orders_status` ✓

**Baseline score:** ~0.60

---

### Task 3 — Aggregation Optimization *(hard, 8 steps)*

**Problem:** Correlated subqueries in the `SELECT` list compute per-category
revenue in O(n²).

**Goal:** Rewrite using `JOIN` + `GROUP BY` to compute the same
`(category_name, total_revenue)` aggregation in O(n log n).

**Perfect score requires:**
- Exact `(category_name, total_revenue)` match ✓
- No correlated subqueries ✓
- `JOIN` + `GROUP BY` present ✓
- Index access detected ✓

**Baseline score:** ~0.45

---

## Baseline Scores (GPT-4o-mini)

| Task                      | Difficulty | Score |
|---------------------------|------------|-------|
| SELECT * Elimination      | easy       | 0.70  |
| Correlated Subquery→JOIN  | medium     | 0.60  |
| Aggregation Optimization  | hard       | 0.45  |
| **Overall Average**       |            | **0.58** |

---

## Setup

### Local (Python)

```bash
git clone https://huggingface.co/spaces/<your-username>/sql-query-optimizer
cd sql-query-optimizer

pip install -r requirements.txt

# Start the API server
python app.py
# → http://localhost:7860/docs

# Run the baseline agent
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py
```

### Docker

```bash
docker build -t sql-optimizer-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=sk-... \
  sql-optimizer-env
```

---

## API Reference

| Method | Path     | Description                        |
|--------|----------|------------------------------------|
| GET    | /health  | Liveness probe                     |
| GET    | /tasks   | List all tasks with metadata       |
| POST   | /reset   | Start new episode (`?task_id=...`) |
| POST   | /step    | Submit optimised query             |
| GET    | /state   | Current episode snapshot           |
| GET    | /docs    | Interactive Swagger UI             |

### Quick example

```python
import requests

BASE = "http://localhost:7860"

obs = requests.post(f"{BASE}/reset?task_id=subquery_to_join").json()
print(obs["slow_query"])

result = requests.post(f"{BASE}/step", json={
    "optimized_query": """
        SELECT o.order_id, o.user_id, o.total_amount
        FROM   orders o
        JOIN   users  u ON u.user_id = o.user_id
        WHERE  u.country  = 'USA'
          AND  u.is_active = 1
          AND  o.status    = 'delivered'
    """
}).json()

print(result["reward"]["value"])       # e.g. 0.90
print(result["reward"]["feedback"])    # "✓ Valid SQL | ✓ Correct result set | ✓ No IN (SELECT…) | ✓ JOIN + index access"
```

### Python class (no HTTP)

```python
from env import SQLQueryOptimizerEnv
from models import SQLAction

env = SQLQueryOptimizerEnv()
obs = env.reset("select_star_removal")
result = env.step(SQLAction(optimized_query="SELECT user_id, username, email FROM users WHERE is_active = 1"))
print(result.reward.value)   # 1.0
env.close()
```

---

## File Structure

```
sql-query-optimizer/
├── models.py        # Pydantic models (Observation, Action, Reward, …)
├── db.py            # SQLite database factory (seeded data)
├── tasks.py         # Task definitions + deterministic graders
├── env.py           # SQLQueryOptimizerEnv class (reset/step/state)
├── app.py           # FastAPI HTTP wrapper
├── inference.py     # Baseline LLM agent script
├── openenv.yaml     # OpenEnv metadata
├── Dockerfile       # Container config for HF Spaces
├── requirements.txt
└── README.md
```

---

## License

MIT © 2024 Digant Kathiriya
