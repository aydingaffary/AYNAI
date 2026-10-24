"""Compare MAD thresholds against independently reviewed latency labels."""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from evaluate_labeled_anomalies import (
    DB_PATH,
    HISTORY_LIMIT,
    MIN_HISTORY,
    load_monitoring_data,
)
from aynai.anomaly import detect_latency_anomaly

THRESHOLDS = (2.0, 2.5, 3.0, 3.5)


def evaluate_threshold(data: pd.DataFrame, threshold: float) -> dict:
    """Evaluate one threshold using only measurements preceding each label."""
    actual = []
    predicted = []
    evaluated_records = 0

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
                    threshold=threshold,
                )
                actual.append(int(label))
                predicted.append(int(result["is_anomaly"]))
                evaluated_records += 1

            # Unlabeled records remain part of the historical baseline.
            history.append(latency)

    if not actual:
        return {
            "threshold": threshold,
            "evaluated_records": 0,
            "status": "no_evaluable_labels",
        }

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "evaluated_records": evaluated_records,
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

    print("AYNAI - Experimental MAD threshold comparison")
    print(f"Total labeled records: {labeled_count}")
    print("Thresholds:", ", ".join(map(str, THRESHOLDS)))
    print("No source code or database records will be modified.")

    results = [
        evaluate_threshold(data, threshold)
        for threshold in THRESHOLDS
    ]

    print("\nResults:")
    print(pd.DataFrame(results).to_string(index=False))
    print(
        "\nWarning: results are exploratory because the labeled sample "
        "is very small. Do not select a production threshold from this "
        "comparison alone."
    )


if __name__ == "__main__":
    main()
