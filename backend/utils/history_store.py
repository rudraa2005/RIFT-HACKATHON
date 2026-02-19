"""
Persistent history store for upload runs.

Stores each processed upload in SQLite so the frontend History page
can load real run records and the corresponding JSON report.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class HistoryStore:
    """SQLite-backed history persistence for processed runs."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        backend_root = Path(__file__).resolve().parents[1]
        self._db_path = db_path or (backend_root / "data" / "history.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    records INTEGER NOT NULL DEFAULT 0,
                    risk_score REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'Analyzed',
                    file_size_bytes INTEGER NOT NULL DEFAULT 0,
                    report_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _compute_avg_risk(self, report: Dict[str, Any]) -> float:
        accounts = report.get("suspicious_accounts", [])
        if not accounts:
            return 0.0
        scores = []
        for account in accounts:
            score = account.get("suspicion_score")
            if isinstance(score, (int, float)):
                scores.append(float(score))
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 2)

    def record_run(self, filename: str, file_size_bytes: int, report: Dict[str, Any]) -> int:
        summary = report.get("summary", {})
        run_records = int(summary.get("total_accounts_analyzed", 0) or 0)
        risk_score = self._compute_avg_risk(report)
        uploaded_at = datetime.now(timezone.utc).isoformat()

        payload = json.dumps(report, ensure_ascii=False)

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO run_history (
                    filename,
                    uploaded_at,
                    records,
                    risk_score,
                    status,
                    file_size_bytes,
                    report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    uploaded_at,
                    run_records,
                    risk_score,
                    "Analyzed",
                    int(file_size_bytes),
                    payload,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, uploaded_at, records, risk_score, status, file_size_bytes
                FROM run_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_run_report(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT report_json FROM run_history WHERE id = ?",
                (int(run_id),),
            ).fetchone()

        if row is None:
            return None
        try:
            return json.loads(row["report_json"])
        except json.JSONDecodeError:
            return None
