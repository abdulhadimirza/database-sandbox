import sqlite3
import os
from typing import Generator
from contextlib import contextmanager
import urllib.request

# The database file will be stored in the root directory
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sandbox.db')

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager to yield a database connection and ensure it is closed properly.
    """
    conn = sqlite3.connect(DB_PATH)
    # Return rows as dictionary-like objects instead of tuples
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_readonly_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager to yield a read-only database connection.
    """
    safe_path = urllib.request.pathname2url(DB_PATH)
    uri = f'file:{safe_path}?mode=ro'
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db() -> None:
    """
    Initialize the database schema.
    This creates the necessary tables if they do not exist.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create Products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            )
        ''')
        
        # Create Orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Create Order Items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                price_at_time REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        ''')
        
        # Insert Mock Data if the database is empty
        cursor.execute('SELECT COUNT(*) as count FROM users')
        if cursor.fetchone()['count'] == 0:
            print("Inserting mock data...")
            
            # Users
            users = [
                ('Alice Smith', 'alice@example.com'),
                ('Bob Jones', 'bob@example.com'),
                ('Charlie Brown', 'charlie@example.com')
            ]
            cursor.executemany('INSERT INTO users (name, email) VALUES (?, ?)', users)
            
            # Products
            products = [
                ('Laptop', 1200.50, 10),
                ('Mouse', 25.00, 50),
                ('Keyboard', 75.00, 30),
                ('Monitor', 300.00, 15)
            ]
            cursor.executemany('INSERT INTO products (name, price, stock) VALUES (?, ?, ?)', products)
            
            # Orders
            orders = [
                (1, 'shipped'),
                (2, 'pending'),
                (1, 'delivered')
            ]
            cursor.executemany('INSERT INTO orders (user_id, status) VALUES (?, ?)', orders)
            
            # Order Items
            order_items = [
                (1, 1, 1, 1200.50), # Alice ordered 1 Laptop
                (1, 2, 2, 25.00),   # Alice ordered 2 Mice
                (2, 3, 1, 75.00),   # Bob ordered 1 Keyboard
                (3, 4, 2, 300.00)   # Alice (2nd order) ordered 2 Monitors
            ]
            cursor.executemany('INSERT INTO order_items (order_id, product_id, quantity, price_at_time) VALUES (?, ?, ?, ?)', order_items)
            
        conn.commit()

if __name__ == '__main__':
    print(f"Initializing database at {DB_PATH}")
    init_db()
    print("Database initialized successfully.")
