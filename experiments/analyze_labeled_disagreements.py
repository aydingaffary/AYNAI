"""Analyze disagreements between MAD and Isolation Forest on labeled records."""

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


def analyze_labeled_disagreements(
    data: pd.DataFrame,
    min_history: int = MIN_HISTORY,
    max_history: int = MAX_HISTORY,
) -> dict:
    """Report labeled records where MAD and Isolation Forest disagree."""
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

    disagreements: list[dict] = []
    evaluated_records = 0
    skipped_insufficient_history = 0

    for url, service_records in records.groupby("url", sort=False):
        history: list[float] = []

        for _, row in service_records.iterrows():
            latency = row["response_time_ms"]
            available = row["is_available"]
            label = row["latency_anomaly_label"]

            valid_measurement = (
                pd.notna(latency)
                and pd.notna(available)
                and int(available) == 1
                and float(latency) >= 0
                and pd.notna(row["checked_at"])
            )

            if pd.notna(label) and int(label) in (0, 1):
                if valid_measurement and len(history) >= min_history:
                    prior_history = history[-max_history:]
                    current_latency = float(latency)

                    mad_result = detect_latency_anomaly(
                        current_latency,
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
                            pd.DataFrame({"response_time_ms": [current_latency]})
                        )[0]
                        == -1
                    )

                    evaluated_records += 1
                    actual = int(label)

                    if mad_prediction != if_prediction:
                        if mad_prediction == 1 and if_prediction == 0:
                            disagreement_type = "mad_only"
                        else:
                            disagreement_type = "isolation_forest_only"

                        disagreements.append(
                            {
                                "id": int(row["id"]),
                                "url": url,
                                "checked_at": row["checked_at"].isoformat(),
                                "response_time_ms": current_latency,
                                "actual_label": actual,
                                "mad_prediction": mad_prediction,
                                "isolation_forest_prediction": if_prediction,
                                "disagreement_type": disagreement_type,
                                "history_count": len(prior_history),
                                "history_median_ms": float(
                                    pd.Series(prior_history).median()
                                ),
                                "mad_baseline_ms": mad_result["baseline_ms"],
                                "mad_ms": mad_result["mad_ms"],
                                "label_notes": (
                                    row["label_notes"]
                                    if "label_notes" in records.columns
                                    and pd.notna(row["label_notes"])
                                    else None
                                ),
                            }
                        )
                else:
                    skipped_insufficient_history += 1

            if valid_measurement:
                history.append(float(latency))

    return {
        "status": "ok" if evaluated_records else "insufficient_history",
        "evaluated_records": evaluated_records,
        "skipped_insufficient_history": skipped_insufficient_history,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "note": (
            "Labels are preliminary human annotations. Disagreements do not "
            "establish which detector is correct or identify the cause of latency."
        ),
    }


def main() -> None:
    """Load stored monitoring results and print disagreement analysis."""
    data = pd.DataFrame(get_results(db_path=Path(DEFAULT_DB_PATH)))
    if data.empty:
        print(json.dumps({"status": "no_data"}, indent=2))
        return

    result = analyze_labeled_disagreements(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
