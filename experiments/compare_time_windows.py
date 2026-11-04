
"""Compare time-based history windows for MAD anomaly detection."""

import math

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from experiments.evaluate_labeled_anomalies import (
    DB_PATH,
    MIN_HISTORY,
    load_monitoring_data,
)
from aynai.anomaly import detect_latency_anomaly


TIME_WINDOWS_MINUTES = (2, 5, 10, 15, 30)
THRESHOLDS = (2.0, 2.5, 3.0, 3.5)

HISTORY_LIMIT = 50


def evaluate_time_window(
    data: pd.DataFrame,
    window_minutes: int,
    threshold: float = 2.0,
    history_limit: int = HISTORY_LIMIT,
    min_history: int = MIN_HISTORY,
) -> dict:
    """Evaluate MAD using only recent prior measurements."""

    if (
        isinstance(window_minutes, bool)
        or not isinstance(window_minutes, int)
        or window_minutes <= 0
    ):
        raise ValueError("window_minutes must be a positive integer")

    if (
        isinstance(history_limit, bool)
        or not isinstance(history_limit, int)
        or history_limit < 1
    ):
        raise ValueError("history_limit must be a positive integer")

    if (
        isinstance(min_history, bool)
        or not isinstance(min_history, int)
        or min_history < MIN_HISTORY
        or min_history > history_limit
    ):
        raise ValueError(
            f"min_history must be >= {MIN_HISTORY} "
            "and <= history_limit"
        )

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("threshold must be a positive finite number")

    total_labeled = (
        int(data["latency_anomaly_label"].notna().sum())
        if not data.empty
        else 0
    )

    actual = []
    predicted = []

    for _, service in data.groupby("url", sort=False):
        service = service.sort_values(["checked_at", "id"])
        history = []

        for _, row in service.iterrows():
            latency = row["response_time_ms"]
            label = row["latency_anomaly_label"]
            is_available = row["is_available"]
            checked_at = row["checked_at"]

            valid_measurement = (
                pd.notna(latency)
                and math.isfinite(float(latency))
                and float(latency) >= 0
                and pd.notna(is_available)
                and int(is_available) == 1
                and pd.notna(checked_at)
            )

            if not valid_measurement:
                continue

            latency = float(latency)

            cutoff = checked_at - pd.Timedelta(
                minutes=window_minutes
            )

            recent_history = [
                value
                for timestamp, value in history
                if timestamp >= cutoff
            ][-history_limit:]

            # Predict before adding the current observation.
            if (
                pd.notna(label)
                and len(recent_history) >= min_history
            ):
                result = detect_latency_anomaly(
                    latency,
                    recent_history,
                    threshold=threshold,
                )
                actual.append(int(label))
                predicted.append(int(result["is_anomaly"]))

            history.append((checked_at, latency))

    evaluated = len(actual)
    coverage = (
        round(evaluated / total_labeled * 100, 2)
        if total_labeled
        else 0.0
    )

    base_result = {
        "window_minutes": window_minutes,
        "history_limit": history_limit,
        "min_history": min_history,
        "threshold": threshold,
        "total_labeled_records": total_labeled,
        "evaluated_records": evaluated,
        "coverage_percent": coverage,
    }

    if not actual:
        return {
            **base_result,
            "status": "no_evaluable_labels",
        }

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        **base_result,
        "actual_normal": actual.count(0),
        "actual_anomaly": actual.count(1),
        "predicted_anomalies": sum(predicted),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": round(
            precision_score(actual, predicted, zero_division=0), 3
        ),
        "recall": round(
            recall_score(actual, predicted, zero_division=0), 3
        ),
        "f1": round(
            f1_score(actual, predicted, zero_division=0), 3
        ),
        "status": (
            "ok" if len(set(actual)) == 2 else "single_class_labels"
        ),
    }


def main() -> None:
    data = load_monitoring_data(DB_PATH)

    if data.empty:
        print("No monitoring data found.")
        return

    print("AYNAI - Time-based MAD history comparison")
    print("Time windows:", TIME_WINDOWS_MINUTES)
    print("Thresholds:", THRESHOLDS)
    print("Maximum history:", HISTORY_LIMIT)
    print("Minimum history:", MIN_HISTORY)
    print("Database records will not be modified.")

    results = [
        evaluate_time_window(
            data,
            window_minutes=window,
            threshold=threshold,
        )
        for window in TIME_WINDOWS_MINUTES
        for threshold in THRESHOLDS
    ]

    results_df = pd.DataFrame(results)

    print("\nResults:")
    print(
        results_df.sort_values(
            ["f1", "coverage_percent"],
            ascending=False,
            na_position="last",
        ).to_string(index=False)
    )

    print(
        "\nWarning: exploratory results only. "
        "The labeled dataset is small, covers only two services, "
        "and may not be independent of the labeling process."
    )


if __name__ == "__main__":
    main()
