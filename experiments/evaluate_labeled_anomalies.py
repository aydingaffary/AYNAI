
"""Evaluate MAD predictions against independently reviewed latency labels."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aynai.anomaly import detect_latency_anomaly
from aynai.storage import get_results

DB_PATH = ROOT / "data" / "aynai.db"
MIN_HISTORY = 5
HISTORY_LIMIT = 100


def load_monitoring_data(db_path: str | Path) -> pd.DataFrame:
    """Load successful measurements, including unlabeled records."""
    rows = get_results(db_path=db_path)
    data = pd.DataFrame(rows)

    if data.empty:
        return data

    data["checked_at"] = pd.to_datetime(
        data["checked_at"], utc=True, errors="coerce"
    )
    data["response_time_ms"] = pd.to_numeric(
        data["response_time_ms"], errors="coerce"
    )

    valid_measurements = (
        data["response_time_ms"].notna()
        & data["response_time_ms"].map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        )
        & (data["response_time_ms"] >= 0)
        & (data["is_available"] == 1)
        & data["checked_at"].notna()
    )

    data = data.loc[valid_measurements].copy()
    return data.sort_values(["url", "checked_at", "id"])


def evaluate_labeled_records(data: pd.DataFrame) -> dict:
    """Evaluate labeled records using only prior measurements."""
    if data.empty:
        return {
            "status": "no_data",
            "labeled_records": 0,
            "evaluated_records": 0,
        }

    labeled_count = int(data["latency_anomaly_label"].notna().sum())

    if labeled_count == 0:
        return {
            "status": "insufficient_labels",
            "labeled_records": 0,
            "evaluated_records": 0,
        }

    actual = []
    predicted = []

    for _, service in data.groupby("url", sort=False):
        service = service.sort_values(["checked_at", "id"])
        history = []

        for _, row in service.iterrows():
            latency = float(row["response_time_ms"])
            label = row["latency_anomaly_label"]

            if pd.notna(label) and len(history) >= MIN_HISTORY:
                result = detect_latency_anomaly(
                    latency,
                    history[-HISTORY_LIMIT:],
                )
                actual.append(int(label))
                predicted.append(int(result["is_anomaly"]))

            # Unlabeled measurements also contribute to future history.
            history.append(latency)

    if not actual:
        return {
            "status": "insufficient_history",
            "labeled_records": labeled_count,
            "evaluated_records": 0,
            "note": "No labeled records had enough prior history.",
        }

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])

    result = {
        "status": (
            "ok" if len(set(actual)) == 2 else "single_class_labels"
        ),
        "labeled_records": labeled_count,
        "evaluated_records": len(actual),
        "evaluated_normal_labels": actual.count(0),
        "evaluated_anomaly_labels": actual.count(1),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["actual_normal", "actual_anomaly"],
    }

    if len(set(actual)) == 1:
        result["note"] = (
            "Only one class is present in evaluated labels; "
            "interpret metrics cautiously."
        )

    return result


def main() -> None:
    """Run the labeled MAD evaluation."""
    data = load_monitoring_data(DB_PATH)
    result = evaluate_labeled_records(data)

    print("AYNAI - MAD evaluation against reviewed latency labels")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
