"""SQLite log of predictions. Parameterised queries only; no user data is stored."""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(os.environ.get("PRAEMONITUS_DB", Path(__file__).parent / "predictions.db"))
SOURCES = ("streamlit", "cli", "batch")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  model_id TEXT NOT NULL,
  source TEXT NOT NULL,
  input_json TEXT NOT NULL,
  probability REAL NOT NULL,
  threshold REAL NOT NULL,
  prediction INTEGER NOT NULL,
  out_of_range INTEGER NOT NULL,
  latency_ms REAL NOT NULL
)
"""

COLUMNS = [
    "id",
    "created_at",
    "model_id",
    "source",
    "input_json",
    "probability",
    "threshold",
    "prediction",
    "out_of_range",
    "latency_ms",
]


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """One connection per operation (Streamlit re-runs the script on every interaction)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(SCHEMA)


def save_prediction(
    *,
    model_id: str,
    source: str,
    inputs: dict,
    probability: float,
    threshold: float,
    prediction: int,
    out_of_range: bool,
    latency_ms: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Insert one prediction and return its id. Time is stored in UTC, ISO 8601."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect(db_path)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO predictions (created_at, model_id, source, input_json, probability,"
            " threshold, prediction, out_of_range, latency_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                created_at,
                model_id,
                source,
                json.dumps(inputs, sort_keys=True),
                float(probability),
                float(threshold),
                int(prediction),
                int(bool(out_of_range)),
                float(latency_ms),
            ),
        )
        return int(cursor.lastrowid)


def get_history(
    limit: int = 50,
    prediction: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Most recent predictions first, optionally only one decision (0 or 1)."""
    init_db(db_path)
    query = "SELECT * FROM predictions"
    params: list = []
    if prediction is not None:
        query += " WHERE prediction = ?"
        params.append(int(prediction))
    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with closing(_connect(db_path)) as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def history_csv_text(db_path: str | Path = DEFAULT_DB_PATH) -> str:
    """Full history as CSV text, oldest first."""
    init_db(db_path)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id").fetchall()
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([row[c] for c in COLUMNS])
    return buffer.getvalue()


def export_csv(path: str | Path, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Write the full history to a CSV file and return the number of rows written."""
    text = history_csv_text(db_path)
    Path(path).write_text(text, encoding="utf-8")
    return text.count("\n") - 1
