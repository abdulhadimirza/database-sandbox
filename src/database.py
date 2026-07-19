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
        
        # Example schema: Create a simple users or items table as a starting point
        # For a sandbox, this can be easily modified later
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sample_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

if __name__ == '__main__':
    print(f"Initializing database at {DB_PATH}")
    init_db()
    print("Database initialized successfully.")
