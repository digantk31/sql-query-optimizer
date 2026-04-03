#!/usr/bin/env python3
"""
test_env.py — Unit + integration tests for SQL Query Optimizer Environment.

Run with:
    python test_env.py
    # or: python -m pytest test_env.py -v

Tests cover:
  - Database seeding (determinism, row counts)
  - All three graders (perfect answer, bad answer, partial answer, invalid SQL)
  - Environment lifecycle (reset, step, state, done flag, step budget)
  - Reward range guarantees [0.0, 1.0]
  - Determinism (same input → same score every time)
"""
from __future__ import annotations

import sys
import time

# ─────────────────────────────────────────────── compatibility shim ──────────
try:
    import pydantic  # noqa: F401
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    print("⚠  pydantic not installed — running grader-only tests via stdlib.\n")

# ─────────────────────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []

def check(name: str, condition: bool, detail: str = "") -> None:
    mark = PASS if condition else FAIL
    results.append((name, condition, detail))
    status = f"  {mark} {name}"
    if detail:
        status += f"  ({detail})"
    print(status)
    if not condition:
        # Don't raise immediately so all tests run
        pass


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  STDLIB-ONLY TESTS  (run regardless of pydantic)
# ══════════════════════════════════════════════════════════════════════════════

import sqlite3, random, re

def _make_db():
    """Inline version of create_database() for stdlib tests."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript("""
        CREATE TABLE users (user_id INTEGER PRIMARY KEY, username TEXT NOT NULL,
            email TEXT NOT NULL, first_name TEXT, last_name TEXT, city TEXT,
            country TEXT, is_active INTEGER DEFAULT 1, created_at TEXT);
        CREATE TABLE categories (category_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            parent_category_id INTEGER);
        CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            category_id INTEGER, price REAL, sku TEXT, is_available INTEGER DEFAULT 1);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER,
            status TEXT, total_amount REAL, created_at TEXT);
        CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER,
            product_id INTEGER, quantity INTEGER, unit_price REAL);
        CREATE INDEX idx_users_active  ON users(is_active);
        CREATE INDEX idx_users_country ON users(country, is_active);
        CREATE INDEX idx_orders_user   ON orders(user_id);
        CREATE INDEX idx_orders_status ON orders(status);
        CREATE INDEX idx_items_order   ON order_items(order_id);
        CREATE INDEX idx_items_product ON order_items(product_id);
        CREATE INDEX idx_products_cat  ON products(category_id);
    """)
    rng = random.Random(42)
    cats = [(1,"Electronics",None),(2,"Clothing",None),(3,"Books",None),(4,"Smartphones",1),
            (5,"Laptops",1),(6,"Tablets",1),(7,"T-Shirts",2),(8,"Jeans",2),(9,"Fiction",3),(10,"Non-Fiction",3)]
    conn.executemany("INSERT INTO categories VALUES (?,?,?)", cats)
    countries = ["USA","USA","USA","UK","Canada","Germany","France","Australia"]
    cities    = ["New York","Los Angeles","Chicago","London","Toronto","Berlin","Paris","Sydney"]
    users = [(i,f"user_{i}",f"user{i}@example.com",f"First{i}",f"Last{i}",
              rng.choice(cities),rng.choice(countries),1 if rng.random()>0.25 else 0,"2022-01-01")
             for i in range(1,1501)]
    conn.executemany("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?)", users)
    leaf = [4,5,6,7,8,9,10]
    products = [(i,f"Product_{i:04d}",rng.choice(leaf),round(rng.uniform(9.99,1499.99),2),f"SKU-{i:05d}",1)
                for i in range(1,301)]
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)
    statuses=["pending","processing","shipped","delivered","cancelled"]; w=[.10,.10,.20,.50,.10]
    orders = [(i,rng.randint(1,1500),rng.choices(statuses,weights=w)[0],round(rng.uniform(20,3000),2),"2024-01-01")
              for i in range(1,3001)]
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    items = [(i,rng.randint(1,3000),rng.randint(1,300),rng.randint(1,5),round(rng.uniform(9.99,1499.99),2))
             for i in range(1,8001)]
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    conn.commit()
    return conn


def test_database():
    section("Database Seeding")
    conn = _make_db()

    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    check("users table has 1500 rows", n == 1500, f"got {n}")

    n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    check("orders table has 3000 rows", n == 3000, f"got {n}")

    n = conn.execute("SELECT COUNT(*) FROM order_items").fetchone()[0]
    check("order_items table has 8000 rows", n == 8000, f"got {n}")

    n = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    check("categories table has 10 rows", n == 10, f"got {n}")

    n = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    check("products table has 300 rows", n == 300, f"got {n}")

    # Determinism: run twice and compare active user counts
    conn2 = _make_db()
    a1 = conn.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
    a2 = conn2.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
    check("database is deterministic (seed=42)", a1 == a2, f"both={a1}")

    # Indexes exist
    indexes = [r[1] for r in conn.execute("PRAGMA index_list(users)").fetchall()]
    check("idx_users_active index exists", "idx_users_active" in indexes)
    check("idx_users_country index exists", "idx_users_country" in indexes)

    conn.close(); conn2.close()


def _norm(q): return re.sub(r"\s+"," ",q.strip().upper())

def test_grader_helpers():
    section("Grader Helper Functions (regexp)")

    check("SELECT * detected", bool(re.search(r"SELECT\s+\*", _norm("SELECT * FROM users"))))
    check("SELECT col not flagged", not bool(re.search(r"SELECT\s+\*", _norm("SELECT id FROM users"))))

    check("IN (SELECT …) detected",
          bool(re.search(r"\bIN\s*\(\s*SELECT\b", _norm("WHERE id IN (SELECT id FROM t)"))))
    check("IN (1,2,3) not flagged",
          not bool(re.search(r"\bIN\s*\(\s*SELECT\b", _norm("WHERE id IN (1,2,3)"))))

    check("JOIN detected", bool(re.search(r"\bJOIN\b", _norm("FROM a JOIN b ON a.id=b.id"))))
    check("No JOIN detected", not bool(re.search(r"\bJOIN\b", _norm("FROM users WHERE id=1"))))

    check("SELECT (SELECT …) detected",
          bool(re.search(r"SELECT\s+\(\s*SELECT\b", _norm("SELECT (SELECT name FROM c WHERE c.id=p.id)"))))
    check("Normal SELECT not flagged",
          not bool(re.search(r"SELECT\s+\(\s*SELECT\b", _norm("SELECT name FROM cats"))))


def test_result_correctness():
    section("Result Correctness (SQL logic)")
    conn = _make_db()

    # Task 1: active user IDs
    ref = {r[0] for r in conn.execute("SELECT user_id FROM users WHERE is_active=1").fetchall()}
    opt = {r[0] for r in conn.execute(
        "SELECT user_id FROM users WHERE is_active=1").fetchall()}
    check("Task1: active user set reproducible", ref == opt, f"{len(ref)} users")

    # Task 2: delivered orders from USA active users
    q_slow = """SELECT order_id FROM orders WHERE user_id IN
                (SELECT user_id FROM users WHERE country='USA' AND is_active=1)
                AND status='delivered'"""
    q_join = """SELECT o.order_id FROM orders o JOIN users u ON u.user_id=o.user_id
                WHERE u.country='USA' AND u.is_active=1 AND o.status='delivered'"""
    ref2 = {r[0] for r in conn.execute(q_slow).fetchall()}
    opt2 = {r[0] for r in conn.execute(q_join).fetchall()}
    check("Task2: IN and JOIN produce identical order sets", ref2 == opt2, f"{len(ref2)} orders")

    # Task 3: category revenue
    q3 = """SELECT c.name, SUM(oi.quantity*oi.unit_price)
             FROM categories c JOIN products p ON p.category_id=c.category_id
             JOIN order_items oi ON oi.product_id=p.product_id
             GROUP BY c.category_id, c.name HAVING SUM(oi.quantity*oi.unit_price)>1000
             ORDER BY 2 DESC"""
    rows3 = conn.execute(q3).fetchall()
    check("Task3: categories with revenue > 1000 exist", len(rows3) > 0, f"{len(rows3)} categories")
    check("Task3: all revenues > 1000", all(r[1] > 1000 for r in rows3))

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC-DEPENDENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_env_lifecycle():
    section("Environment Lifecycle (pydantic required)")
    from env import SQLQueryOptimizerEnv
    from models import SQLAction

    env = SQLQueryOptimizerEnv()

    # reset() returns correct structure
    obs = env.reset("select_star_removal")
    check("reset() returns task_id", obs.task_id == "select_star_removal")
    check("reset() returns difficulty=easy", obs.difficulty == "easy")
    check("reset() step_number=0", obs.step_number == 0)
    check("reset() schema_ddl non-empty", len(obs.schema_ddl) > 100)
    check("reset() slow_query non-empty", len(obs.slow_query) > 10)
    check("reset() last_reward=0.0 on fresh episode", obs.last_reward == 0.0)

    # state() reflects reset
    st = env.state()
    check("state() task_id matches", st.task_id == "select_star_removal")
    check("state() step_number=0 after reset", st.step_number == 0)
    check("state() best_reward=0.0 after reset", st.best_reward == 0.0)
    check("state() done=False after reset", not st.done)

    # step() increments step counter
    result = env.step(SQLAction(optimized_query="SELECT user_id, username, email FROM users WHERE is_active=1"))
    check("step() increments step number", result.observation.step_number == 1)
    check("step() reward.value in [0,1]", 0.0 <= result.reward.value <= 1.0)
    check("step() returns info dict with best_reward", "best_reward" in result.info)

    # Episode ends after max_steps
    obs2 = env.reset("select_star_removal")  # max_steps=5
    for _ in range(4):
        env.step(SQLAction(optimized_query="SELECT * FROM users WHERE is_active=1"))
    r_last = env.step(SQLAction(optimized_query="SELECT * FROM users WHERE is_active=1"))
    check("Episode done after max_steps exhausted", r_last.done)

    # RuntimeError when stepping into done episode
    try:
        env.step(SQLAction(optimized_query="SELECT 1"))
        check("RuntimeError raised after done episode", False)
    except RuntimeError:
        check("RuntimeError raised after done episode", True)

    # list_tasks returns all three
    tasks = env.list_tasks()
    check("list_tasks() returns 3 tasks", len(tasks) == 3)
    ids = [t["task_id"] for t in tasks]
    check("all task IDs present", "select_star_removal" in ids and "aggregation_optimization" in ids)

    env.close()


def test_grader_task1():
    section("Grader: Task 1 — SELECT * Elimination")
    from env import SQLQueryOptimizerEnv
    from models import SQLAction

    env = SQLQueryOptimizerEnv()

    # Perfect answer
    env.reset("select_star_removal")
    r = env.step(SQLAction(optimized_query="SELECT user_id, username, email FROM users WHERE is_active = 1"))
    check("Perfect T1: reward >= 0.90", r.reward.value >= 0.90, f"got {r.reward.value:.3f}")
    check("Perfect T1: style=0.30 (no SELECT *)", r.reward.breakdown.style == 0.30)
    check("Perfect T1: validity > 0", r.reward.breakdown.validity > 0)
    check("Perfect T1: correctness > 0", r.reward.breakdown.correctness > 0)

    # SELECT * still present
    env.reset("select_star_removal")
    r_bad = env.step(SQLAction(optimized_query="SELECT * FROM users WHERE is_active = 1"))
    check("SELECT * T1: style=0.0", r_bad.reward.breakdown.style == 0.0)
    check("SELECT * T1: reward < perfect", r_bad.reward.value < r.reward.value)

    # Invalid SQL
    env.reset("select_star_removal")
    r_inv = env.step(SQLAction(optimized_query="NOT VALID SQL AT ALL !!!"))
    check("Invalid SQL T1: reward=0.0", r_inv.reward.value == 0.0)
    check("Invalid SQL T1: validity=0.0", r_inv.reward.breakdown.validity == 0.0)

    # Wrong result set (missing WHERE clause)
    env.reset("select_star_removal")
    r_wrong = env.step(SQLAction(optimized_query="SELECT user_id, username, email FROM users"))
    check("Wrong result T1: correctness < perfect", r_wrong.reward.breakdown.correctness < 0.40)

    env.close()


def test_grader_task2():
    section("Grader: Task 2 — Correlated Subquery → JOIN")
    from env import SQLQueryOptimizerEnv
    from models import SQLAction

    env = SQLQueryOptimizerEnv()

    perfect_q2 = """
        SELECT o.order_id, o.user_id, o.total_amount
        FROM   orders o
        JOIN   users  u ON u.user_id = o.user_id
        WHERE  u.country  = 'USA'
          AND  u.is_active = 1
          AND  o.status    = 'delivered'
    """

    env.reset("subquery_to_join")
    r = env.step(SQLAction(optimized_query=perfect_q2))
    check("Perfect T2: reward >= 0.80", r.reward.value >= 0.80, f"got {r.reward.value:.3f}")
    check("Perfect T2: style=0.20 (no IN subquery)", r.reward.breakdown.style == 0.20)
    check("Perfect T2: correctness = 0.40", r.reward.breakdown.correctness == 0.40)

    # Still uses IN (SELECT …)
    env.reset("subquery_to_join")
    r_in = env.step(SQLAction(optimized_query=(
        "SELECT order_id, user_id, total_amount FROM orders "
        "WHERE user_id IN (SELECT user_id FROM users WHERE country='USA' AND is_active=1) "
        "AND status='delivered'"
    )))
    check("IN subquery T2: style=0.0", r_in.reward.breakdown.style == 0.0)
    check("IN subquery T2: reward < JOIN reward", r_in.reward.value < r.reward.value)

    # Invalid SQL
    env.reset("subquery_to_join")
    r_inv = env.step(SQLAction(optimized_query="BROKEN QUERY"))
    check("Invalid SQL T2: reward=0.0", r_inv.reward.value == 0.0)

    env.close()


def test_grader_task3():
    section("Grader: Task 3 — Aggregation Optimization")
    from env import SQLQueryOptimizerEnv
    from models import SQLAction

    env = SQLQueryOptimizerEnv()

    perfect_q3 = """
        SELECT   c.name                          AS category_name,
                 SUM(oi.quantity * oi.unit_price) AS total_revenue
        FROM     categories   c
        JOIN     products     p  ON p.category_id = c.category_id
        JOIN     order_items  oi ON oi.product_id = p.product_id
        GROUP BY c.category_id, c.name
        HAVING   SUM(oi.quantity * oi.unit_price) > 1000
        ORDER BY total_revenue DESC
    """

    env.reset("aggregation_optimization")
    r = env.step(SQLAction(optimized_query=perfect_q3))
    check("Perfect T3: reward >= 0.90", r.reward.value >= 0.90, f"got {r.reward.value:.3f}")
    check("Perfect T3: style=0.25 (no correlated subqueries)", r.reward.breakdown.style == 0.25)
    check("Perfect T3: correctness = 0.35", r.reward.breakdown.correctness == 0.35)
    check("Perfect T3: performance > 0", r.reward.breakdown.performance > 0)

    # Correlated subquery still present
    env.reset("aggregation_optimization")
    r_sub = env.step(SQLAction(optimized_query=(
        "SELECT (SELECT name FROM categories WHERE category_id=p.category_id) AS category_name, "
        "SUM(oi.quantity*oi.unit_price) AS total_revenue "
        "FROM products p JOIN order_items oi ON oi.product_id=p.product_id "
        "GROUP BY p.category_id HAVING SUM(oi.quantity*oi.unit_price)>1000 ORDER BY total_revenue DESC"
    )))
    check("Correlated T3: style < 0.25", r_sub.reward.breakdown.style < 0.25)
    check("Correlated T3: reward < perfect", r_sub.reward.value < r.reward.value)

    env.close()


def test_reward_range():
    section("Reward Range Guarantee [0.0, 1.0]")
    from env import SQLQueryOptimizerEnv
    from models import SQLAction
    from tasks import TASK_ORDER

    env = SQLQueryOptimizerEnv()
    queries = [
        "SELECT * FROM users",
        "SELECT user_id FROM users WHERE is_active=1",
        "COMPLETELY INVALID",
        "SELECT 1",
        "SELECT o.order_id, o.user_id, o.total_amount FROM orders o JOIN users u ON u.user_id=o.user_id WHERE u.country='USA' AND u.is_active=1 AND o.status='delivered'",
        "SELECT c.name AS category_name, SUM(oi.quantity*oi.unit_price) AS total_revenue FROM categories c JOIN products p ON p.category_id=c.category_id JOIN order_items oi ON oi.product_id=p.product_id GROUP BY c.category_id,c.name HAVING SUM(oi.quantity*oi.unit_price)>1000 ORDER BY total_revenue DESC",
    ]

    all_in_range = True
    for task_id in TASK_ORDER:
        env.reset(task_id)
        for q in queries:
            r = env.step(SQLAction(optimized_query=q))
            v = r.reward.value
            if not (0.0 <= v <= 1.0):
                all_in_range = False
                print(f"    OUT OF RANGE: task={task_id} query={q[:40]} reward={v}")
            if r.done:
                env.reset(task_id)

    check("All rewards in [0.0, 1.0] across all tasks and queries", all_in_range)
    env.close()


def test_determinism():
    section("Determinism (same input → same output)")
    from env import SQLQueryOptimizerEnv
    from models import SQLAction

    q = "SELECT user_id, username, email FROM users WHERE is_active = 1"

    env1 = SQLQueryOptimizerEnv()
    env2 = SQLQueryOptimizerEnv()

    env1.reset("select_star_removal")
    env2.reset("select_star_removal")

    r1 = env1.step(SQLAction(optimized_query=q))
    r2 = env2.step(SQLAction(optimized_query=q))

    check("Same query → same reward across two env instances",
          r1.reward.value == r2.reward.value,
          f"{r1.reward.value} vs {r2.reward.value}")

    env1.close(); env2.close()


def test_episode_reset_cleans_state():
    section("Episode Reset Cleans State")
    from env import SQLQueryOptimizerEnv
    from models import SQLAction

    env = SQLQueryOptimizerEnv()

    # Run task1 to completion
    env.reset("select_star_removal")
    for _ in range(5):
        env.step(SQLAction(optimized_query="SELECT * FROM users WHERE is_active=1"))

    # Reset to task2 — state should be clean
    obs = env.reset("subquery_to_join")
    check("After reset: task_id switches", obs.task_id == "subquery_to_join")
    check("After reset: step_number=0", obs.step_number == 0)
    st = env.state()
    check("After reset: best_reward=0.0", st.best_reward == 0.0)
    check("After reset: done=False", not st.done)

    # Can step into the new episode normally
    r = env.step(SQLAction(optimized_query="SELECT order_id, user_id, total_amount FROM orders WHERE status='delivered'"))
    check("Can step after reset", r.reward.value >= 0.0)

    env.close()


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     SQL Query Optimizer — Test Suite                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    t0 = time.time()

    # Always run (no pydantic needed)
    test_database()
    test_grader_helpers()
    test_result_correctness()

    # Run pydantic-dependent tests only if available
    if HAS_PYDANTIC:
        import sys as _sys
        _sys.path.insert(0, ".")
        test_env_lifecycle()
        test_grader_task1()
        test_grader_task2()
        test_grader_task3()
        test_reward_range()
        test_determinism()
        test_episode_reset_cleans_state()
    else:
        print("\n  ⚠  Skipping pydantic tests (install pydantic + dependencies)")

    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total  = len(results)
    print(f"  Results: {passed}/{total} passed  ({failed} failed)  [{elapsed:.2f}s]")

    if failed:
        print("\n  FAILED TESTS:")
        for name, ok, detail in results:
            if not ok:
                print(f"    {FAIL} {name}  {detail}")
        print("═" * 60)
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED ✓")
        print("═" * 60)


if __name__ == "__main__":
    main()
