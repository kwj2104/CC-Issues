"""SQLite schema DDL and connection helpers."""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- keys: snapshot_ts, repo, row_count, api_total_count, tool_version, malformed_count

CREATE TABLE IF NOT EXISTS issues (
  number          INTEGER PRIMARY KEY,
  title           TEXT NOT NULL,
  body            TEXT NOT NULL DEFAULT '',
  state           TEXT NOT NULL,
  created_at      TEXT NOT NULL,   -- ISO8601 UTC
  updated_at      TEXT NOT NULL,
  comments        INTEGER NOT NULL,
  reactions_total INTEGER NOT NULL,
  reactions_plus1 INTEGER NOT NULL,
  author_association TEXT,
  html_url        TEXT NOT NULL,
  locked          INTEGER NOT NULL DEFAULT 0,
  active_lock_reason TEXT
);

CREATE TABLE IF NOT EXISTS issue_labels (
  number INTEGER NOT NULL REFERENCES issues(number),
  label  TEXT NOT NULL,            -- lowercased
  PRIMARY KEY (number, label)
);
CREATE INDEX IF NOT EXISTS idx_labels_label ON issue_labels(label);

CREATE TABLE IF NOT EXISTS features (
  number INTEGER PRIMARY KEY REFERENCES issues(number),
  age_days REAL NOT NULL,          -- (snapshot_ts - created_at)/86400, floor 1.0
  f_reactions REAL NOT NULL,       -- log2(1+reactions_total)
  f_comments  REAL NOT NULL,       -- log2(1+comments)
  f_velocity  REAL NOT NULL,       -- log2(1 + 30*(reactions_total+comments)/age_days)
  f_severity  REAL NOT NULL,       -- per section 5, capped
  f_demand    REAL NOT NULL,       -- per section 5
  is_junk            INTEGER NOT NULL,  -- abandoned empty report (all junk conds hold)
  maintainer_authored INTEGER NOT NULL, -- author is OWNER/MEMBER/COLLABORATOR (flag only)
  in_pool  INTEGER NOT NULL,       -- window/carve-out predicate
  eligible INTEGER NOT NULL        -- in_pool AND labels AND lock reason AND NOT junk
);

CREATE TABLE IF NOT EXISTS clusters (
  number INTEGER PRIMARY KEY REFERENCES issues(number),
  cluster_id INTEGER NOT NULL,     -- singletons get their own id
  cluster_size INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS score_runs (
  run_id TEXT PRIMARY KEY,         -- sha256(snapshot_ts + canonical_config_json)[:12]
  created_ts TEXT NOT NULL,
  weights_json TEXT NOT NULL,
  config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
  run_id TEXT NOT NULL REFERENCES score_runs(run_id),
  number INTEGER NOT NULL REFERENCES issues(number),
  score REAL NOT NULL,
  c_reactions REAL NOT NULL, c_comments REAL NOT NULL, c_velocity REAL NOT NULL,
  c_severity REAL NOT NULL, c_demand REAL NOT NULL, c_cluster REAL NOT NULL,
  rank INTEGER,                    -- rank among eligible, 1-based; NULL if ineligible
  selected INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id, number)
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection with row access by name and foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    migrate_schema(conn)
    conn.commit()


# Columns added after the initial release. CREATE TABLE IF NOT EXISTS will not
# add them to a pre-existing table, so patch them in idempotently. Any NOT NULL
# addition needs a DEFAULT (SQLite requirement on a non-empty table); the
# features columns are fully repopulated by run_features regardless.
_ADDED_COLUMNS = {
    "issues": [
        ("locked", "INTEGER NOT NULL DEFAULT 0"),
        ("active_lock_reason", "TEXT"),
    ],
    "features": [
        ("is_junk", "INTEGER NOT NULL DEFAULT 0"),
        ("maintainer_authored", "INTEGER NOT NULL DEFAULT 0"),
    ],
}


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Add any columns introduced after a DB was first created (idempotent)."""
    for table, columns in _ADDED_COLUMNS.items():
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
    )


def get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else default
