---
title: SQL Query Optimizer
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - sql
  - reinforcement-learning
  - database
  - agent
  - real-world
license: mit
---

# SQL Query Optimizer — OpenEnv Environment

An RL environment where AI agents learn to rewrite slow SQL queries for
correctness, performance, and style — against a live SQLite e-commerce database.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-brightgreen)]()
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)]()
[![License MIT](https://img.shields.io/badge/license-MIT-green)]()

---

## Quick Start

```bash
# Reset to the easiest task
curl -X POST "$SPACE_URL/reset?task_id=select_star_removal"

# Submit an optimised query
curl -X POST "$SPACE_URL/step" \
     -H "Content-Type: application/json" \
     -d '{"optimized_query": "SELECT user_id, username, email FROM users WHERE is_active = 1"}'

# Check current episode state
curl "$SPACE_URL/state"
```

Interactive API docs: `$SPACE_URL/docs`

---

## Tasks

| ID | Name | Difficulty | Max Steps |
|----|------|------------|-----------|
| `select_star_removal` | SELECT * Elimination | Easy | 5 |
| `subquery_to_join` | Correlated Subquery → JOIN | Medium | 6 |
| `aggregation_optimization` | Aggregation Optimization | Hard | 8 |

## Reward Dimensions

Every step returns a dense, partial-credit reward:

| Dimension | Max | Signal |
|-----------|-----|--------|
| `validity` | 0.10 | Query runs without error |
| `correctness` | 0.35–0.40 | Same result set as reference |
| `performance` | 0.20–0.30 | Index usage in EXPLAIN plan |
| `style` | 0.20–0.30 | No SELECT *, no correlated subqueries |

## Baseline Scores (GPT-4o-mini)

| Task | Score |
|------|-------|
| SELECT * Elimination | 0.70 |
| Correlated Subquery → JOIN | 0.60 |
| Aggregation Optimization | 0.45 |
| **Overall** | **0.58** |

---

Full documentation in [README.md](README.md).
