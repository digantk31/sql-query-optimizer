"""
Task definitions and deterministic graders for the SQL Query Optimizer environment.

Three tasks of increasing difficulty, each with a programmatic grader
that scores agent queries in the range [0.0, 1.0].

Grading dimensions (same for all tasks, different weights):
  validity     — Does the query run without error?
  correctness  — Does it return the same result set as the reference?
  performance  — Does EXPLAIN QUERY PLAN show index usage / fewer scans?
  style        — Does it avoid known SQL anti-patterns?
"""
from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Tuple

from models import ExecutionMetrics

# ─────────────────────────────────────────────────────────────── schema DDL ──

SCHEMA_DDL = """\
-- Users registered on the platform
CREATE TABLE users (
    user_id    INTEGER PRIMARY KEY,
    username   TEXT    NOT NULL,
    email      TEXT    NOT NULL,
    first_name TEXT,
    last_name  TEXT,
    city       TEXT,
    country    TEXT,
    is_active  INTEGER DEFAULT 1,   -- 1 = active, 0 = inactive
    created_at TEXT
);

-- Two-level product taxonomy
CREATE TABLE categories (
    category_id        INTEGER PRIMARY KEY,
    name               TEXT NOT NULL,
    parent_category_id INTEGER          -- NULL for top-level categories
);

-- Products for sale
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    category_id  INTEGER REFERENCES categories(category_id),
    price        REAL,
    sku          TEXT    UNIQUE,
    is_available INTEGER DEFAULT 1
);

-- Customer orders
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    user_id      INTEGER REFERENCES users(user_id),
    status       TEXT,   -- pending | processing | shipped | delivered | cancelled
    total_amount REAL,
    created_at   TEXT
);

-- Line items within an order
CREATE TABLE order_items (
    item_id    INTEGER PRIMARY KEY,
    order_id   INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity   INTEGER,
    unit_price REAL
);

-- Available indexes (use these to your advantage)
CREATE INDEX idx_users_active  ON users(is_active);
CREATE INDEX idx_users_country ON users(country, is_active);
CREATE INDEX idx_orders_user   ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_items_order   ON order_items(order_id);
CREATE INDEX idx_items_product ON order_items(product_id);
CREATE INDEX idx_products_cat  ON products(category_id);
"""

# ─────────────────────────────────────────────────────────── task registry ───

TASK_ORDER: List[str] = [
    "select_star_removal",
    "subquery_to_join",
    "aggregation_optimization",
]

TASKS: Dict[str, dict] = {
    "select_star_removal": {
        "name": "SELECT * Elimination",
        "difficulty": "easy",
        "max_steps": 5,
        "description": (
            "The slow query uses SELECT * to fetch every column from the users table, "
            "transferring megabytes of unused data. "
            "Rewrite it to return ONLY the three required columns: "
            "user_id, username, email. "
            "The query MUST still return ALL active users (is_active = 1)."
        ),
        "slow_query": (
            "SELECT *\n"
            "FROM users\n"
            "WHERE is_active = 1"
        ),
    },
    "subquery_to_join": {
        "name": "Correlated Subquery → JOIN",
        "difficulty": "medium",
        "max_steps": 6,
        "description": (
            "The slow query uses an IN (SELECT …) subquery to filter orders, "
            "causing the database to execute the inner query repeatedly. "
            "Rewrite it as an efficient JOIN. "
            "Return: order_id, user_id, total_amount "
            "for DELIVERED orders from ACTIVE users in the USA."
        ),
        "slow_query": (
            "SELECT order_id, user_id, total_amount\n"
            "FROM   orders\n"
            "WHERE  user_id IN (\n"
            "    SELECT user_id\n"
            "    FROM   users\n"
            "    WHERE  country = 'USA'\n"
            "      AND  is_active = 1\n"
            ") AND status = 'delivered'"
        ),
    },
    "aggregation_optimization": {
        "name": "Aggregation Optimization",
        "difficulty": "hard",
        "max_steps": 8,
        "description": (
            "This query computes per-category revenue using correlated subqueries "
            "inside the SELECT list — effectively O(n²) in the number of products. "
            "Rewrite it using explicit JOINs and GROUP BY. "
            "Return: category_name (TEXT), total_revenue (REAL) "
            "for categories where total_revenue > 1000, "
            "ordered by total_revenue DESC."
        ),
        "slow_query": (
            "SELECT\n"
            "    (SELECT name FROM categories\n"
            "     WHERE  category_id = p.category_id) AS category_name,\n"
            "    (SELECT SUM(oi.quantity * oi.unit_price)\n"
            "     FROM   order_items oi\n"
            "     WHERE  oi.product_id IN (\n"
            "         SELECT product_id FROM products\n"
            "         WHERE  category_id = p.category_id\n"
            "     )) AS total_revenue\n"
            "FROM   products p\n"
            "GROUP  BY p.category_id\n"
            "HAVING total_revenue > 1000\n"
            "ORDER  BY total_revenue DESC"
        ),
    },
}

# ─────────────────────────────────────────────────────────── SQL utilities ───

def get_query_metrics(query: str, conn: sqlite3.Connection) -> ExecutionMetrics:
    """Return an ExecutionMetrics snapshot for the given query."""
    try:
        plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
        plan_text_list = [str(r) for r in plan_rows]
        flat = " ".join(plan_text_list).upper()
        return ExecutionMetrics(
            uses_index=(
                "USING INDEX" in flat
                or ("SEARCH" in flat and "SCAN" not in flat.split("SEARCH")[0][-10:])
            ),
            full_scan_count=flat.count("SCAN TABLE"),
            query_plan=plan_text_list,
        )
    except Exception:
        return ExecutionMetrics()


def _normalize(query: str) -> str:
    """Collapse whitespace and upper-case for pattern matching."""
    return re.sub(r"\s+", " ", query.strip().upper())


def _has_select_star(query: str) -> bool:
    return bool(re.search(r"SELECT\s+\*", _normalize(query)))


def _uses_explicit_join(query: str) -> bool:
    return bool(re.search(r"\bJOIN\b", _normalize(query)))


def _uses_in_subquery(query: str) -> bool:
    return bool(re.search(r"\bIN\s*\(\s*SELECT\b", _normalize(query)))


def _uses_select_subquery(query: str) -> bool:
    """Detects subqueries inside the SELECT clause: SELECT (SELECT …)."""
    return bool(re.search(r"SELECT\s+\(\s*SELECT\b", _normalize(query)))


def _run_safely(query: str, conn: sqlite3.Connection):
    """Execute a query and return (cursor, col_names, rows) or raise."""
    cursor = conn.execute(query)
    col_names = [d[0].lower() for d in cursor.description]
    rows = cursor.fetchall()
    return cursor, col_names, rows


# ─────────────────────────────────────────────────────────────── graders ────

GraderResult = Tuple[float, Dict[str, float], str]


def grade_task1(query: str, conn: sqlite3.Connection) -> GraderResult:
    """
    SELECT * Elimination grader.

    Weights:
      validity     0.10
      correctness  0.40   — correct user_id set returned
      style        0.30   — no SELECT *
      performance  0.20   — index detected in EXPLAIN plan
    """
    bd: Dict[str, float] = dict(validity=0.0, correctness=0.0, performance=0.0, style=0.0)
    msgs: List[str] = []

    # ── validity ──────────────────────────────────────────────────────────────
    try:
        _, col_names, rows = _run_safely(query, conn)
        bd["validity"] = 0.10
        msgs.append("✓ Valid SQL")
    except Exception as exc:
        msgs.append(f"✗ SQL error: {exc}")
        return 0.0, bd, " | ".join(msgs)

    # ── correctness ───────────────────────────────────────────────────────────
    required = ["user_id", "username", "email"]
    missing = [c for c in required if c not in col_names]

    if missing:
        bd["correctness"] = 0.05
        msgs.append(f"✗ Missing required columns: {missing}")
    else:
        uid_idx = col_names.index("user_id")
        returned_ids = {row[uid_idx] for row in rows}
        ref_ids = {
            r[0] for r in
            conn.execute("SELECT user_id FROM users WHERE is_active = 1").fetchall()
        }
        overlap = len(returned_ids & ref_ids) / max(len(ref_ids), 1)
        if returned_ids == ref_ids:
            bd["correctness"] = 0.40
            msgs.append(f"✓ Correct result set ({len(ref_ids)} active users)")
        elif overlap >= 0.95:
            bd["correctness"] = 0.20
            msgs.append(f"~ {overlap:.0%} of correct users returned")
        else:
            bd["correctness"] = 0.0
            msgs.append(f"✗ Wrong result set (overlap {overlap:.0%})")

    # ── style ─────────────────────────────────────────────────────────────────
    if not _has_select_star(query):
        bd["style"] = 0.30
        msgs.append("✓ No SELECT * — explicit columns used")
    else:
        msgs.append("✗ Still uses SELECT * (anti-pattern)")

    # ── performance ───────────────────────────────────────────────────────────
    metrics = get_query_metrics(query, conn)
    if metrics.uses_index:
        bd["performance"] = 0.20
        msgs.append("✓ Index scan detected")
    elif metrics.full_scan_count <= 1:
        bd["performance"] = 0.10
        msgs.append("~ Single table scan (partial credit)")
    else:
        msgs.append("✗ Multiple full table scans")

    return min(1.0, sum(bd.values())), bd, " | ".join(msgs)


def grade_task2(query: str, conn: sqlite3.Connection) -> GraderResult:
    """
    Correlated Subquery → JOIN grader.

    Weights:
      validity     0.10
      correctness  0.40   — same order_id set as reference
      style        0.20   — no IN (SELECT …)
      performance  0.30   — uses JOIN + index
    """
    bd: Dict[str, float] = dict(validity=0.0, correctness=0.0, performance=0.0, style=0.0)
    msgs: List[str] = []

    # ── validity ──────────────────────────────────────────────────────────────
    try:
        _, col_names, rows = _run_safely(query, conn)
        bd["validity"] = 0.10
        msgs.append("✓ Valid SQL")
    except Exception as exc:
        msgs.append(f"✗ SQL error: {exc}")
        return 0.0, bd, " | ".join(msgs)

    # ── correctness ───────────────────────────────────────────────────────────
    required = ["order_id", "user_id", "total_amount"]
    missing = [c for c in required if c not in col_names]
    ref_ids = {
        r[0] for r in conn.execute(
            "SELECT order_id FROM orders "
            "WHERE user_id IN ("
            "  SELECT user_id FROM users WHERE country='USA' AND is_active=1"
            ") AND status='delivered'"
        ).fetchall()
    }

    if missing:
        bd["correctness"] = 0.05
        msgs.append(f"✗ Missing required columns: {missing}")
    else:
        oid_idx = col_names.index("order_id")
        returned_ids = {row[oid_idx] for row in rows}
        overlap = len(returned_ids & ref_ids) / max(len(ref_ids), 1)
        if returned_ids == ref_ids:
            bd["correctness"] = 0.40
            msgs.append(f"✓ Correct result set ({len(ref_ids)} orders)")
        elif overlap >= 0.90:
            bd["correctness"] = 0.20
            msgs.append(f"~ {overlap:.0%} of correct orders returned")
        else:
            bd["correctness"] = 0.0
            msgs.append(f"✗ Wrong result set (overlap {overlap:.0%})")

    # ── style ─────────────────────────────────────────────────────────────────
    if not _uses_in_subquery(query):
        bd["style"] = 0.20
        msgs.append("✓ No IN (SELECT …) subquery")
    else:
        msgs.append("✗ Still uses IN (SELECT …) — convert to JOIN")

    # ── performance ───────────────────────────────────────────────────────────
    uses_join = _uses_explicit_join(query)
    metrics = get_query_metrics(query, conn)

    if uses_join and metrics.uses_index:
        bd["performance"] = 0.30
        msgs.append("✓ JOIN + index access")
    elif uses_join:
        bd["performance"] = 0.15
        msgs.append("~ JOIN used but no index detected")
    elif metrics.uses_index:
        bd["performance"] = 0.10
        msgs.append("~ Index used but no JOIN")
    else:
        msgs.append("✗ No JOIN and no index usage")

    return min(1.0, sum(bd.values())), bd, " | ".join(msgs)


def grade_task3(query: str, conn: sqlite3.Connection) -> GraderResult:
    """
    Aggregation Optimization grader.

    Weights:
      validity     0.10
      correctness  0.35   — same (category_name, total_revenue) set as reference
      style        0.25   — no correlated subqueries in SELECT list
      performance  0.30   — uses JOIN + GROUP BY + index
    """
    bd: Dict[str, float] = dict(validity=0.0, correctness=0.0, performance=0.0, style=0.0)
    msgs: List[str] = []

    # ── validity ──────────────────────────────────────────────────────────────
    try:
        _, col_names, rows = _run_safely(query, conn)
        bd["validity"] = 0.10
        msgs.append("✓ Valid SQL")
    except Exception as exc:
        msgs.append(f"✗ SQL error: {exc}")
        return 0.0, bd, " | ".join(msgs)

    # ── correctness ───────────────────────────────────────────────────────────
    # Reference: the clean JOIN-based answer (this IS the gold standard)
    ref_rows = conn.execute("""
        SELECT   c.name                          AS category_name,
                 SUM(oi.quantity * oi.unit_price) AS total_revenue
        FROM     categories   c
        JOIN     products     p  ON p.category_id  = c.category_id
        JOIN     order_items  oi ON oi.product_id  = p.product_id
        GROUP BY c.category_id, c.name
        HAVING   SUM(oi.quantity * oi.unit_price) > 1000
        ORDER BY total_revenue DESC
    """).fetchall()
    ref_set = {(r[0], round(float(r[1]), 1)) for r in ref_rows}

    required = ["category_name", "total_revenue"]
    missing = [c for c in required if c not in col_names]

    if missing:
        bd["correctness"] = 0.05
        msgs.append(f"✗ Missing required columns: {missing}")
    else:
        cat_idx = col_names.index("category_name")
        rev_idx = col_names.index("total_revenue")
        opt_set = {(row[cat_idx], round(float(row[rev_idx]), 1)) for row in rows}
        overlap = len(opt_set & ref_set) / max(len(ref_set), 1)
        if opt_set == ref_set:
            bd["correctness"] = 0.35
            msgs.append(f"✓ Exact match ({len(ref_set)} categories)")
        elif overlap >= 0.80:
            bd["correctness"] = 0.18
            msgs.append(f"~ {overlap:.0%} of results match reference")
        else:
            bd["correctness"] = 0.0
            msgs.append(f"✗ Wrong results (overlap {overlap:.0%})")

    # ── style ─────────────────────────────────────────────────────────────────
    has_select_sub = _uses_select_subquery(query)
    has_in_sub     = _uses_in_subquery(query)

    if not has_select_sub and not has_in_sub:
        bd["style"] = 0.25
        msgs.append("✓ No correlated subqueries")
    elif not has_select_sub:
        bd["style"] = 0.12
        msgs.append("~ Removed SELECT-list subqueries but IN subquery remains")
    else:
        msgs.append("✗ Correlated subqueries in SELECT list detected")

    # ── performance ───────────────────────────────────────────────────────────
    uses_join    = _uses_explicit_join(query)
    uses_groupby = bool(re.search(r"\bGROUP\s+BY\b", _normalize(query)))
    metrics      = get_query_metrics(query, conn)

    perf = 0.0
    if uses_join:
        perf += 0.12
        msgs.append("✓ Uses JOIN")
    if uses_groupby:
        perf += 0.10
        msgs.append("✓ Uses GROUP BY")
    if metrics.uses_index:
        perf += 0.08
        msgs.append("✓ Index access in plan")
    if not uses_join and not uses_groupby:
        msgs.append("✗ Missing JOIN and GROUP BY")
    bd["performance"] = perf

    return min(1.0, sum(bd.values())), bd, " | ".join(msgs)


# ─────────────────────────────────────────────────────── grader registry ────

TASK_GRADERS = {
    "select_star_removal":    grade_task1,
    "subquery_to_join":       grade_task2,
    "aggregation_optimization": grade_task3,
}
