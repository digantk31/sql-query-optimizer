"""
Task definitions and deterministic graders — v2 (4 tasks).

Tasks:
  1. select_star_removal      — easy    Remove SELECT *
  2. subquery_to_join         — medium  IN subquery → JOIN
  3. aggregation_optimization — hard    Correlated subqueries → GROUP BY
  4. cte_refactoring          — expert  Deeply nested subqueries → CTEs

All graders return (score: float, breakdown: dict, feedback: str).
Scores are always in [0.0, 1.0].
"""
from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Tuple

from models import ExecutionMetrics

# ── Schema DDL (shown to agent) ───────────────────────────────────────────────

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
    is_active  INTEGER DEFAULT 1,
    created_at TEXT
);

-- Two-level product taxonomy
CREATE TABLE categories (
    category_id        INTEGER PRIMARY KEY,
    name               TEXT NOT NULL,
    parent_category_id INTEGER
);

-- Product suppliers
CREATE TABLE suppliers (
    supplier_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT,
    rating      REAL
);

-- Products for sale
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    category_id  INTEGER REFERENCES categories(category_id),
    supplier_id  INTEGER REFERENCES suppliers(supplier_id),
    price        REAL,
    sku          TEXT UNIQUE,
    is_available INTEGER DEFAULT 1
);

-- Stock levels per product
CREATE TABLE inventory (
    inventory_id  INTEGER PRIMARY KEY,
    product_id    INTEGER UNIQUE REFERENCES products(product_id),
    stock_qty     INTEGER DEFAULT 0,
    reorder_level INTEGER DEFAULT 10,
    last_updated  TEXT
);

-- Customer orders
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    user_id      INTEGER REFERENCES users(user_id),
    status       TEXT,   -- pending|processing|shipped|delivered|cancelled
    total_amount REAL,
    coupon_id    INTEGER,
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

-- Product reviews
CREATE TABLE reviews (
    review_id   INTEGER PRIMARY KEY,
    product_id  INTEGER REFERENCES products(product_id),
    user_id     INTEGER REFERENCES users(user_id),
    rating      INTEGER CHECK(rating BETWEEN 1 AND 5),
    review_text TEXT,
    created_at  TEXT
);

-- Discount coupons
CREATE TABLE coupons (
    coupon_id    INTEGER PRIMARY KEY,
    code         TEXT UNIQUE,
    discount_pct REAL,
    is_active    INTEGER DEFAULT 1
);

-- Key indexes (exploit these for performance)
CREATE INDEX idx_users_active    ON users(is_active);
CREATE INDEX idx_users_country   ON users(country, is_active);
CREATE INDEX idx_orders_user     ON orders(user_id);
CREATE INDEX idx_orders_status   ON orders(status);
CREATE INDEX idx_orders_created  ON orders(created_at);
CREATE INDEX idx_items_order     ON order_items(order_id);
CREATE INDEX idx_items_product   ON order_items(product_id);
CREATE INDEX idx_products_cat    ON products(category_id);
CREATE INDEX idx_products_avail  ON products(is_available);
CREATE INDEX idx_reviews_product ON reviews(product_id);
CREATE INDEX idx_inventory_prod  ON inventory(product_id);
"""

# ── Task registry ─────────────────────────────────────────────────────────────

TASK_ORDER: List[str] = [
    "select_star_removal",
    "subquery_to_join",
    "aggregation_optimization",
    "cte_refactoring",
]

TASKS: Dict[str, dict] = {
    "select_star_removal": {
        "name": "SELECT * Elimination",
        "difficulty": "easy",
        "max_steps": 5,
        "description": (
            "The slow query uses SELECT * to fetch every column from the users table. "
            "Rewrite it to return ONLY the three required columns: user_id, username, email. "
            "The query MUST still return ALL active users (is_active = 1)."
        ),
        "slow_query": (
            "SELECT *\n"
            "FROM users\n"
            "WHERE is_active = 1"
        ),
    },
    "subquery_to_join": {
        "name": "Correlated Subquery to JOIN",
        "difficulty": "medium",
        "max_steps": 6,
        "description": (
            "The slow query uses IN (SELECT ...) to filter orders, causing repeated inner-query execution. "
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
            "inside the SELECT list, which is O(n^2). "
            "Rewrite it using explicit JOINs and GROUP BY. "
            "Return: category_name (TEXT), total_revenue (REAL) "
            "for categories where total_revenue > 10000, "
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
            "HAVING total_revenue > 10000\n"
            "ORDER  BY total_revenue DESC"
        ),
    },
    "cte_refactoring": {
        "name": "CTE Refactoring",
        "difficulty": "expert",
        "max_steps": 10,
        "description": (
            "This query identifies the top 5 customers by lifetime spend, "
            "using deeply nested subqueries that are hard to read and slow to execute. "
            "Rewrite it using Common Table Expressions (WITH clauses) for clarity and performance. "
            "Return: username, email, total_spend (REAL), order_count (INTEGER) "
            "for the top 5 active customers by total_spend, ordered by total_spend DESC."
        ),
        "slow_query": (
            "SELECT username, email, total_spend, order_count\n"
            "FROM (\n"
            "    SELECT\n"
            "        (SELECT username FROM users WHERE user_id = o.user_id) AS username,\n"
            "        (SELECT email    FROM users WHERE user_id = o.user_id) AS email,\n"
            "        SUM(o.total_amount) AS total_spend,\n"
            "        COUNT(*) AS order_count\n"
            "    FROM orders o\n"
            "    WHERE o.user_id IN (\n"
            "        SELECT user_id FROM users WHERE is_active = 1\n"
            "    )\n"
            "    GROUP BY o.user_id\n"
            ") ranked\n"
            "ORDER BY total_spend DESC\n"
            "LIMIT 5"
        ),
    },
}

# ── SQL utilities ─────────────────────────────────────────────────────────────

def get_query_metrics(query: str, conn: sqlite3.Connection) -> ExecutionMetrics:
    try:
        plan_rows  = conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
        plan_texts = [str(r) for r in plan_rows]
        flat       = " ".join(plan_texts).upper()
        return ExecutionMetrics(
            uses_index=(
                "USING INDEX" in flat or
                ("SEARCH" in flat and flat.count("SCAN TABLE") == 0)
            ),
            full_scan_count=flat.count("SCAN TABLE"),
            query_plan=plan_texts,
        )
    except Exception:
        return ExecutionMetrics()


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().upper())

def _has_select_star(q):  return bool(re.search(r"SELECT\s+\*",             _norm(q)))
def _uses_join(q):        return bool(re.search(r"\bJOIN\b",                _norm(q)))
def _uses_in_sub(q):      return bool(re.search(r"\bIN\s*\(\s*SELECT\b",    _norm(q)))
def _uses_sel_sub(q):     return bool(re.search(r"SELECT\s+\(\s*SELECT\b",  _norm(q)))
def _uses_cte(q):         return bool(re.search(r"^\s*WITH\b",              q.strip(), re.IGNORECASE))
def _uses_groupby(q):     return bool(re.search(r"\bGROUP\s+BY\b",          _norm(q)))

def _run(query: str, conn: sqlite3.Connection):
    cur  = conn.execute(query)
    cols = [d[0].lower() for d in cur.description]
    rows = cur.fetchall()
    return cols, rows

# ── Graders ───────────────────────────────────────────────────────────────────

GraderResult = Tuple[float, Dict[str, float], str]


def grade_task1(query: str, conn: sqlite3.Connection) -> GraderResult:
    bd   = dict(validity=0.0, correctness=0.0, performance=0.0, style=0.0)
    msgs = []

    try:
        cols, rows = _run(query, conn)
        bd["validity"] = 0.10
        msgs.append("✓ Valid SQL")
    except Exception as e:
        msgs.append(f"✗ SQL error: {e}")
        return 0.001, bd, " | ".join(msgs)

    required = ["user_id", "username", "email"]
    missing  = [c for c in required if c not in cols]
    if missing:
        bd["correctness"] = 0.05
        msgs.append(f"✗ Missing columns: {missing}")
    else:
        ref = {r[0] for r in conn.execute("SELECT user_id FROM users WHERE is_active=1").fetchall()}
        got = {row[cols.index("user_id")] for row in rows}
        if got == ref:
            bd["correctness"] = 0.40
            msgs.append(f"✓ Correct ({len(ref)} users)")
        elif len(got & ref) / max(len(ref), 1) >= 0.95:
            bd["correctness"] = 0.20
            msgs.append("~ Partial correctness")
        else:
            msgs.append("✗ Wrong result set")

    if not _has_select_star(query):
        bd["style"] = 0.30
        msgs.append("✓ No SELECT *")
    else:
        msgs.append("✗ Still uses SELECT *")

    m = get_query_metrics(query, conn)
    if m.uses_index:
        bd["performance"] = 0.20
        msgs.append("✓ Index scan")
    elif m.full_scan_count <= 1:
        bd["performance"] = 0.10
        msgs.append("~ Single scan")
    else:
        msgs.append("✗ Multiple full scans")

    return min(0.999, max(0.001, sum(bd.values()))), bd, " | ".join(msgs)


def grade_task2(query: str, conn: sqlite3.Connection) -> GraderResult:
    bd   = dict(validity=0.0, correctness=0.0, performance=0.0, style=0.0)
    msgs = []

    try:
        cols, rows = _run(query, conn)
        bd["validity"] = 0.10
        msgs.append("✓ Valid SQL")
    except Exception as e:
        msgs.append(f"✗ SQL error: {e}")
        return 0.001, bd, " | ".join(msgs)

    ref = {r[0] for r in conn.execute(
        "SELECT order_id FROM orders WHERE status='delivered' "
        "AND user_id IN (SELECT user_id FROM users WHERE country='USA' AND is_active=1)"
    ).fetchall()}

    required = ["order_id", "user_id", "total_amount"]
    missing  = [c for c in required if c not in cols]
    if missing:
        bd["correctness"] = 0.05
        msgs.append(f"✗ Missing columns: {missing}")
    else:
        got     = {row[cols.index("order_id")] for row in rows}
        overlap = len(got & ref) / max(len(ref), 1)
        if got == ref:
            bd["correctness"] = 0.40
            msgs.append(f"✓ Correct ({len(ref)} orders)")
        elif overlap >= 0.90:
            bd["correctness"] = 0.20
            msgs.append(f"~ {overlap:.0%} correct")
        else:
            msgs.append(f"✗ Wrong results ({overlap:.0%})")

    if not _uses_in_sub(query):
        bd["style"] = 0.20
        msgs.append("✓ No IN subquery")
    else:
        msgs.append("✗ Still uses IN subquery")

    uses_j = _uses_join(query)
    m      = get_query_metrics(query, conn)
    if uses_j and m.uses_index:
        bd["performance"] = 0.30
        msgs.append("✓ JOIN + index")
    elif uses_j:
        bd["performance"] = 0.15
        msgs.append("~ JOIN but no index")
    elif m.uses_index:
        bd["performance"] = 0.10
        msgs.append("~ Index but no JOIN")
    else:
        msgs.append("✗ No JOIN, no index")

    return min(0.999, max(0.001, sum(bd.values()))), bd, " | ".join(msgs)


def grade_task3(query: str, conn: sqlite3.Connection) -> GraderResult:
    bd   = dict(validity=0.0, correctness=0.0, performance=0.0, style=0.0)
    msgs = []

    try:
        cols, rows = _run(query, conn)
        bd["validity"] = 0.10
        msgs.append("✓ Valid SQL")
    except Exception as e:
        msgs.append(f"✗ SQL error: {e}")
        return 0.001, bd, " | ".join(msgs)

    ref_rows = conn.execute("""
        SELECT c.name, SUM(oi.quantity * oi.unit_price) AS total_revenue
        FROM categories c
        JOIN products p ON p.category_id = c.category_id
        JOIN order_items oi ON oi.product_id = p.product_id
        GROUP BY c.category_id, c.name
        HAVING SUM(oi.quantity * oi.unit_price) > 10000
        ORDER BY total_revenue DESC
    """).fetchall()
    ref_set = {(r[0], round(float(r[1]), 0)) for r in ref_rows}

    required = ["category_name", "total_revenue"]
    missing  = [c for c in required if c not in cols]
    if missing:
        bd["correctness"] = 0.05
        msgs.append(f"✗ Missing columns: {missing}")
    else:
        got_set = {(row[cols.index("category_name")], round(float(row[cols.index("total_revenue")]), 0)) for row in rows}
        overlap = len(got_set & ref_set) / max(len(ref_set), 1)
        if got_set == ref_set:
            bd["correctness"] = 0.35
            msgs.append(f"✓ Exact match ({len(ref_set)} categories)")
        elif overlap >= 0.80:
            bd["correctness"] = 0.18
            msgs.append(f"~ {overlap:.0%} match")
        else:
            msgs.append(f"✗ Wrong results ({overlap:.0%})")

    if not _uses_sel_sub(query) and not _uses_in_sub(query):
        bd["style"] = 0.25
        msgs.append("✓ No correlated subqueries")
    elif not _uses_sel_sub(query):
        bd["style"] = 0.12
        msgs.append("~ Partial: SELECT subquery removed")
    else:
        msgs.append("✗ Correlated subqueries in SELECT")

    perf = 0.0
    if _uses_join(query):  perf += 0.12; msgs.append("✓ JOIN")
    if _uses_groupby(query): perf += 0.10; msgs.append("✓ GROUP BY")
    m = get_query_metrics(query, conn)
    if m.uses_index: perf += 0.08; msgs.append("✓ Index")
    bd["performance"] = perf

    return min(0.999, max(0.001, sum(bd.values()))), bd, " | ".join(msgs)


def grade_task4(query: str, conn: sqlite3.Connection) -> GraderResult:
    """
    CTE Refactoring grader.

    Weights:
      validity     0.10
      correctness  0.35  — exact top-5 username+spend match
      style        0.30  — uses WITH (CTE), no nested subqueries
      performance  0.25  — JOIN + index
    """
    bd   = dict(validity=0.0, correctness=0.0, performance=0.0, style=0.0)
    msgs = []

    try:
        cols, rows = _run(query, conn)
        bd["validity"] = 0.10
        msgs.append("✓ Valid SQL")
    except Exception as e:
        msgs.append(f"✗ SQL error: {e}")
        return 0.001, bd, " | ".join(msgs)

    ref_rows = conn.execute("""
        WITH active_users AS (
            SELECT user_id, username, email FROM users WHERE is_active = 1
        ),
        user_spend AS (
            SELECT o.user_id,
                   SUM(o.total_amount) AS total_spend,
                   COUNT(*) AS order_count
            FROM orders o
            JOIN active_users au ON au.user_id = o.user_id
            GROUP BY o.user_id
        )
        SELECT au.username, au.email,
               us.total_spend, us.order_count
        FROM user_spend us
        JOIN active_users au ON au.user_id = us.user_id
        ORDER BY us.total_spend DESC
        LIMIT 5
    """).fetchall()
    ref_set = {(r[0], round(float(r[2]), 1)) for r in ref_rows}

    required = ["username", "email", "total_spend", "order_count"]
    missing  = [c for c in required if c not in cols]
    if missing:
        bd["correctness"] = 0.05
        msgs.append(f"✗ Missing columns: {missing}")
    elif len(rows) != 5:
        bd["correctness"] = 0.10
        msgs.append(f"✗ Expected 5 rows, got {len(rows)}")
    else:
        got_set = {(row[cols.index("username")], round(float(row[cols.index("total_spend")]), 1)) for row in rows}
        overlap = len(got_set & ref_set) / max(len(ref_set), 1)
        if got_set == ref_set:
            bd["correctness"] = 0.35
            msgs.append("✓ Exact top-5 match")
        elif overlap >= 0.80:
            bd["correctness"] = 0.18
            msgs.append(f"~ {overlap:.0%} of top-5 correct")
        else:
            msgs.append(f"✗ Wrong top-5 ({overlap:.0%})")

    uses_cte    = _uses_cte(query)
    has_nest_sub = _uses_sel_sub(query) or (query.upper().count("SELECT") > 2 and _uses_in_sub(query))

    if uses_cte and not has_nest_sub:
        bd["style"] = 0.30
        msgs.append("✓ CTE used, no nested subqueries")
    elif uses_cte:
        bd["style"] = 0.15
        msgs.append("~ CTE used but nested subqueries remain")
    else:
        msgs.append("✗ No CTE (WITH clause) found")

    perf = 0.0
    if _uses_join(query):    perf += 0.12; msgs.append("✓ JOIN")
    if _uses_groupby(query): perf += 0.08; msgs.append("✓ GROUP BY")
    m = get_query_metrics(query, conn)
    if m.uses_index:         perf += 0.05; msgs.append("✓ Index")
    bd["performance"] = perf

    return min(0.999, max(0.001, sum(bd.values()))), bd, " | ".join(msgs)


# ── Grader registry ───────────────────────────────────────────────────────────

TASK_GRADERS = {
    "select_star_removal":      grade_task1,
    "subquery_to_join":         grade_task2,
    "aggregation_optimization": grade_task3,
    "cte_refactoring":          grade_task4,
}

def _safe(fn):
    def w(q,conn):
        s,b,f=fn(q,conn)
        s=max(0.001,min(0.999,float(s)))
        b={k:max(0.001,min(0.999,float(v))) for k,v in b.items()}
        return s,b,f
    return w
TASK_GRADERS={k:_safe(v) for k,v in TASK_GRADERS.items()}
