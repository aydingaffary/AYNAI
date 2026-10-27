"""Compare MAD and Isolation Forest using the same historical labeled records."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

from aynai.anomaly import detect_latency_anomaly
from aynai.storage import DEFAULT_DB_PATH, get_results

MIN_HISTORY = 5
MAX_HISTORY = 100
RANDOM_STATE = 42


def evaluate_labeled_detectors(
    data: pd.DataFrame,
    min_history: int = MIN_HISTORY,
    max_history: int = MAX_HISTORY,
) -> dict:
    """Evaluate both detectors chronologically on the same labeled records."""
    required_columns = {
        "id",
        "url",
        "checked_at",
        "response_time_ms",
        "is_available",
        "latency_anomaly_label",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if min_history < 5:
        raise ValueError("min_history must be at least 5")
    if max_history < min_history:
        raise ValueError("max_history must be >= min_history")

    records = data.copy()
    records["checked_at"] = pd.to_datetime(
        records["checked_at"], utc=True, errors="coerce"
    )
    records = records.sort_values(["url", "checked_at", "id"])

    predictions: dict[str, list[int]] = {
        "mad": [],
        "isolation_forest": [],
    }
    actual: list[int] = []
    skipped_insufficient_history = 0

    for _, service_records in records.groupby("url", sort=False):
        history: list[float] = []

        for _, row in service_records.iterrows():
            latency = row["response_time_ms"]
            is_available = row["is_available"]

            # Only successful measurements can enter the training history.
            valid_measurement = (
                pd.notna(latency)
                and pd.notna(is_available)
                and int(is_available) == 1
                and float(latency) >= 0
            )

            label = row["latency_anomaly_label"]

            if pd.notna(label) and int(label) in (0, 1):
                if valid_measurement and len(history) >= min_history:
                    prior_history = history[-max_history:]

                    mad_result = detect_latency_anomaly(
                        float(latency),
                        prior_history,
                    )
                    mad_prediction = int(mad_result["is_anomaly"])

                    model = IsolationForest(
                        n_estimators=100,
                        contamination="auto",
                        random_state=RANDOM_STATE,
                    )
                    model.fit(pd.DataFrame({"response_time_ms": prior_history}))
                    if_prediction = int(
                        model.predict(
                            pd.DataFrame({"response_time_ms": [float(latency)]})
                        )[0]
                        == -1
                    )

                    actual.append(int(label))
                    predictions["mad"].append(mad_prediction)
                    predictions["isolation_forest"].append(if_prediction)
                else:
                    skipped_insufficient_history += 1

            if valid_measurement:
                history.append(float(latency))

    if not actual:
        return {
            "status": "insufficient_history",
            "evaluated_records": 0,
            "skipped_insufficient_history": skipped_insufficient_history,
        }

    results = {
        "status": "ok",
        "evaluated_records": len(actual),
        "skipped_insufficient_history": skipped_insufficient_history,
        "actual_normal_labels": sum(value == 0 for value in actual),
        "actual_anomaly_labels": sum(value == 1 for value in actual),
        "minimum_history": min_history,
        "maximum_history": max_history,
    }

    for name, predicted in predictions.items():
        tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
        fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
        tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
        fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_matrix_labels": [
                "actual_normal",
                "actual_anomaly",
            ],
        }

    return results


def main() -> None:
    """Load stored records and print evaluation results."""
    data = pd.DataFrame(get_results(db_path=Path(DEFAULT_DB_PATH)))
    if data.empty:
        print(json.dumps({"status": "no_data"}, indent=2))
        return

    results = evaluate_labeled_detectors(data)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
