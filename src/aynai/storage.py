import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "aynai.db"


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the database and migrate its schema when necessary."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                is_available INTEGER NOT NULL,
                status_code INTEGER,
                response_time_ms REAL NOT NULL,
                error TEXT,
                checked_at TEXT NOT NULL,
                is_anomaly INTEGER,
                baseline_ms REAL,
                mad_ms REAL,
                anomaly_reason TEXT
            )
        """)

        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(monitoring_results)"
            )
        }

        migrations = {
            "is_anomaly": "INTEGER",
            "baseline_ms": "REAL",
            "mad_ms": "REAL",
            "anomaly_reason": "TEXT",
        }

        for column_name, column_type in migrations.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE monitoring_results "
                    f"ADD COLUMN {column_name} {column_type}"
                )

        connection.commit()
    finally:
        connection.close()



def save_result(
    result: dict,
    db_path: str | Path = DEFAULT_DB_PATH,
    anomaly: dict | None = None,
) -> int:
    """Save one monitoring result to the database."""
    init_db(db_path)

    checked_at = datetime.now(timezone.utc).isoformat()

    connection = sqlite3.connect(db_path)

    try:

        cursor = connection.execute(
            """
            INSERT INTO monitoring_results (
                url,
                is_available,
                status_code,
                response_time_ms,
                error,
                checked_at,
                is_anomaly,
                baseline_ms,
                mad_ms,
                anomaly_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["url"],
                int(result["is_available"]),
                result["status_code"],
                result["response_time_ms"],
                result["error"],
                checked_at,
                (
                    int(anomaly["is_anomaly"])
                    if anomaly is not None
                    else None
                ),
                anomaly["baseline_ms"] if anomaly is not None else None,
                anomaly["mad_ms"] if anomaly is not None else None,
                anomaly["reason"] if anomaly is not None else None,
            ),
        )

        connection.commit()
        return cursor.lastrowid
    finally:
        connection.close()


def get_results(
    url: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """Retrieve monitoring results, optionally filtered by URL."""
    init_db(db_path)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    try:
        if url is None:
            cursor = connection.execute("""
                SELECT *
                FROM monitoring_results
                ORDER BY id DESC
                """)
        else:
            cursor = connection.execute(
                """
                SELECT *
                FROM monitoring_results
                WHERE url = ?
                ORDER BY id DESC
                """,
                (url,),
            )

        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()
