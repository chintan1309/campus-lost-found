"""
models.py — Database schema and connection logic for Campus Lost & Found.
Uses a single SQLite database with an `items` table.
"""

import sqlite3
import json
from datetime import datetime

DATABASE = 'lost_found.db'


def get_db():
    """Open a new database connection."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # allows dict-style column access
    return conn


def init_db():
    """Create the items table if it doesn't already exist."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            description TEXT,
            status      TEXT NOT NULL CHECK(status IN ('lost', 'found')),
            image_path  TEXT,
            embedding   TEXT,           -- stored as JSON string (list of floats)
            contact_info TEXT,
            timestamp   TEXT NOT NULL,
            resolved    INTEGER NOT NULL DEFAULT 0  -- 0 = active, 1 = resolved/claimed
        )
    ''')
    # Migrate existing DBs that don't have the resolved column yet
    try:
        cursor.execute('ALTER TABLE items ADD COLUMN resolved INTEGER NOT NULL DEFAULT 0')
    except Exception:
        pass  # column already exists
    conn.commit()
    conn.close()


def insert_item(title, description, status, image_path, embedding, contact_info):
    """
    Insert a new item into the database.
    Returns the newly created item's ID.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO items (title, description, status, image_path, embedding, contact_info, timestamp, resolved)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
    ''', (
        title,
        description,
        status,
        image_path,
        json.dumps(embedding),          # serialise numpy array → JSON string
        contact_info,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    item_id = cursor.lastrowid
    conn.close()
    return item_id


def get_item_by_id(item_id):
    """Fetch a single item by its primary key. Returns a Row object or None."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM items WHERE id = ?', (item_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_all_items():
    """Return all ACTIVE (non-resolved) items ordered by most recent first."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM items WHERE resolved = 0 ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_items_including_resolved():
    """Return every item including resolved ones (for stats)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM items ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_items_by_status(status):
    """
    Return all ACTIVE (non-resolved) items that match the given status.
    Used by the matching engine — resolved items are excluded so they
    don't surface as matches for new posts.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM items WHERE status = ? AND resolved = 0', (status,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def resolve_item(item_id):
    """
    Mark an item as resolved (owner got their item back / item was claimed).
    Resolved items are hidden from the homepage and excluded from AI matching.
    Returns True if a row was updated, False if item_id not found.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE items SET resolved = 1 WHERE id = ?', (item_id,))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def get_resolved_count():
    """Return the number of successfully resolved items (for the stats strip)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM items WHERE resolved = 1')
    count = cursor.fetchone()[0]
    conn.close()
    return count
