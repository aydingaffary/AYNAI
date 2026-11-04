
"""Compare MAD thresholds and history limits against reviewed labels."""

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


HISTORY_LIMITS = (5, 10, 20, 50, 100)
THRESHOLDS = (2.0, 2.5, 3.0, 3.5)


def evaluate_configuration(
    data: pd.DataFrame,
    threshold: float,
    history_limit: int,
) -> dict:
    """Evaluate one MAD configuration using prior measurements only."""

    if (
        isinstance(history_limit, bool)
        or not isinstance(history_limit, int)
        or history_limit < MIN_HISTORY
    ):
        raise ValueError(
            f"history_limit must be an integer >= {MIN_HISTORY}"
        )

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("threshold must be a positive finite number")

    actual = []
    predicted = []

    if data.empty:
        return {
            "history_limit": history_limit,
            "threshold": threshold,
            "evaluated_records": 0,
            "status": "no_data",
        }

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

            # Predict before adding the current measurement to history.
            if pd.notna(label) and len(history) >= MIN_HISTORY:
                result = detect_latency_anomaly(
                    latency,
                    history[-history_limit:],
                    threshold=threshold,
                )
                actual.append(int(label))
                predicted.append(int(result["is_anomaly"]))

            # Unlabeled observations may contribute to future history.
            history.append(latency)

    if not actual:
        return {
            "history_limit": history_limit,
            "threshold": threshold,
            "evaluated_records": 0,
            "status": "no_evaluable_labels",
        }

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "history_limit": history_limit,
        "threshold": threshold,
        "evaluated_records": len(actual),
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

    labeled_count = int(data["latency_anomaly_label"].notna().sum())

    print("AYNAI - MAD history and threshold comparison")
    print(f"Total labeled records: {labeled_count}")
    print("History limits:", HISTORY_LIMITS)
    print("Thresholds:", THRESHOLDS)
    print("No source code or database records will be modified.")

    results = [
        evaluate_configuration(data, threshold, history_limit)
        for history_limit in HISTORY_LIMITS
        for threshold in THRESHOLDS
    ]

    results_df = pd.DataFrame(results)

    print("\nResults:")
    print(
        results_df.sort_values(
            ["f1", "recall"],
            ascending=False,
        ).to_string(index=False)
    )

    print(
        "\nWarning: results are exploratory. The labeled sample is "
        "small and may not be fully independent of the labeling process. "
        "Do not select a production configuration from this comparison alone."
    )


if __name__ == "__main__":
    main()
