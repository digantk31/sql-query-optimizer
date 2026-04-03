"""
Database setup module.

Creates an in-memory SQLite e-commerce database with deterministic
seeded data used by all three tasks.
"""
import sqlite3
import random


def create_database() -> sqlite3.Connection:
    """
    Build and populate an in-memory SQLite database.

    Schema: users, categories, products, orders, order_items
    Data:   1500 users | 10 categories | 300 products
            3000 orders | 8000 order_items
    Seed:   42 (fully deterministic)
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)

    # ------------------------------------------------------------------ schema
    conn.executescript("""
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

        CREATE TABLE categories (
            category_id        INTEGER PRIMARY KEY,
            name               TEXT NOT NULL,
            parent_category_id INTEGER
        );

        CREATE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            name         TEXT    NOT NULL,
            category_id  INTEGER,
            price        REAL,
            sku          TEXT,
            is_available INTEGER DEFAULT 1
        );

        CREATE TABLE orders (
            order_id     INTEGER PRIMARY KEY,
            user_id      INTEGER REFERENCES users(user_id),
            status       TEXT,
            total_amount REAL,
            created_at   TEXT
        );

        CREATE TABLE order_items (
            item_id    INTEGER PRIMARY KEY,
            order_id   INTEGER REFERENCES orders(order_id),
            product_id INTEGER REFERENCES products(product_id),
            quantity   INTEGER,
            unit_price REAL
        );

        -- Indexes that an optimal query should exploit
        CREATE INDEX idx_users_active  ON users(is_active);
        CREATE INDEX idx_users_country ON users(country, is_active);
        CREATE INDEX idx_orders_user   ON orders(user_id);
        CREATE INDEX idx_orders_status ON orders(status);
        CREATE INDEX idx_items_order   ON order_items(order_id);
        CREATE INDEX idx_items_product ON order_items(product_id);
        CREATE INDEX idx_products_cat  ON products(category_id);
    """)

    # ------------------------------------------------------------------ seed
    rng = random.Random(42)

    # categories (2 levels)
    categories = [
        (1, "Electronics", None),
        (2, "Clothing",    None),
        (3, "Books",       None),
        (4, "Smartphones", 1),
        (5, "Laptops",     1),
        (6, "Tablets",     1),
        (7, "T-Shirts",    2),
        (8, "Jeans",       2),
        (9, "Fiction",     3),
        (10, "Non-Fiction", 3),
    ]
    conn.executemany("INSERT INTO categories VALUES (?,?,?)", categories)

    # users (1 500 rows)
    countries = ["USA", "USA", "USA", "UK", "Canada", "Germany", "France", "Australia"]
    cities    = ["New York", "Los Angeles", "Chicago", "London",
                 "Toronto", "Berlin", "Paris", "Sydney"]
    users = [
        (
            i, f"user_{i}", f"user{i}@example.com",
            f"First{i}", f"Last{i}",
            rng.choice(cities), rng.choice(countries),
            1 if rng.random() > 0.25 else 0,
            f"202{rng.randint(0,3)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        )
        for i in range(1, 1501)
    ]
    conn.executemany("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?)", users)

    # products (300 rows)
    leaf_cats = [4, 5, 6, 7, 8, 9, 10]
    products = [
        (
            i, f"Product_{i:04d}", rng.choice(leaf_cats),
            round(rng.uniform(9.99, 1499.99), 2),
            f"SKU-{i:05d}", 1 if rng.random() > 0.05 else 0,
        )
        for i in range(1, 301)
    ]
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)

    # orders (3 000 rows)
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    weights  = [0.10, 0.10, 0.20, 0.50, 0.10]
    orders = [
        (
            i, rng.randint(1, 1500),
            rng.choices(statuses, weights=weights)[0],
            round(rng.uniform(20.0, 3000.0), 2),
            f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        )
        for i in range(1, 3001)
    ]
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)

    # order_items (8 000 rows)
    items = [
        (
            i, rng.randint(1, 3000), rng.randint(1, 300),
            rng.randint(1, 5), round(rng.uniform(9.99, 1499.99), 2),
        )
        for i in range(1, 8001)
    ]
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)

    conn.commit()
    return conn
