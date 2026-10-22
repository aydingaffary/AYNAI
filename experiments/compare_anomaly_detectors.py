"""Compare MAD and Isolation Forest on chronological latency data.

This experiment has no verified ground-truth anomaly labels.
Its output measures detector agreement, not predictive accuracy.
"""

import sqlite3
from pathlib import Path
from statistics import median

import pandas as pd
from sklearn.ensemble import IsolationForest

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "aynai.db"
TRAIN_RATIO = 0.8
MIN_RECORDS = 10
MIN_HISTORY = 5
HISTORY_LIMIT = 100


def load_data() -> pd.DataFrame:
    """Load successful monitoring records from the SQLite database."""
    with sqlite3.connect(DB_PATH) as connection:
        data = pd.read_sql_query(
            """
            SELECT id, url, response_time_ms, is_available, checked_at
            FROM monitoring_results
            WHERE response_time_ms IS NOT NULL
              AND is_available = 1
            """,
            connection,
        )

    if data.empty:
        raise ValueError("No successful monitoring records found")

    data["checked_at"] = pd.to_datetime(data["checked_at"], utc=True)
    return data


def analyze_mad(
    current_ms: float,
    history: list[float],
    threshold: float = 3.5,
) -> dict:
    """Evaluate latency against historical median and MAD."""
    if len(history) < MIN_HISTORY:
        return {
            "is_anomaly": False,
            "baseline_ms": None,
            "mad_ms": None,
            "robust_z": None,
        }

    baseline = median(history)
    mad = median([abs(value - baseline) for value in history])

    if mad == 0:
        is_anomaly = (
            current_ms > baseline and current_ms > 1.5 * baseline
        )
        robust_z = None
    else:
        robust_z = 0.6745 * (current_ms - baseline) / mad
        is_anomaly = robust_z > threshold

    return {
        "is_anomaly": is_anomaly,
        "baseline_ms": baseline,
        "mad_ms": mad,
        "robust_z": robust_z,
    }


def split_chronologically(
    service: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a service's records into earlier training and later test data."""
    service = service.sort_values(
        ["checked_at", "id"]
    ).reset_index(drop=True)

    split = int(len(service) * TRAIN_RATIO)
    return service.iloc[:split], service.iloc[split:]


def get_historical_latency(
    service: pd.DataFrame,
    current_row: pd.Series,
) -> list[float]:
    """Return up to HISTORY_LIMIT successful measurements before a record."""
    earlier = service[
        (service["checked_at"] < current_row["checked_at"])
        | (
            (service["checked_at"] == current_row["checked_at"])
            & (service["id"] < current_row["id"])
        )
    ]

    return (
        earlier["response_time_ms"]
        .tail(HISTORY_LIMIT)
        .astype(float)
        .tolist()
    )


def evaluate_service(
    url: str,
    service: pd.DataFrame,
) -> dict | None:
    """Fit Isolation Forest and compare its predictions with historical MAD."""
    if len(service) < MIN_RECORDS:
        print(f"\nSkipping {url}: insufficient records")
        return None

    train, test = split_chronologically(service)

    if len(train) < MIN_HISTORY or test.empty:
        print(f"\nSkipping {url}: insufficient train/test data")
        return None

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )
    model.fit(train[["response_time_ms"]])

    results = test.copy()
    results["if_anomaly"] = (
        model.predict(test[["response_time_ms"]]) == -1
    )
    results["anomaly_score"] = model.decision_function(
        test[["response_time_ms"]]
    )

    mad_results = [
        analyze_mad(
            float(row["response_time_ms"]),
            get_historical_latency(service, row),
        )
        for _, row in test.iterrows()
    ]

    for column, key in [
        ("mad_anomaly", "is_anomaly"),
        ("baseline_ms", "baseline_ms"),
        ("mad_ms", "mad_ms"),
        ("robust_z", "robust_z"),
    ]:
        results[column] = [result[key] for result in mad_results]

    both = results["if_anomaly"] & results["mad_anomaly"]
    mad_only = results["mad_anomaly"] & ~results["if_anomaly"]
    if_only = results["if_anomaly"] & ~results["mad_anomaly"]

    summary = {
        "url": url,
        "training_records": len(train),
        "test_records": len(test),
        "mad_count": int(results["mad_anomaly"].sum()),
        "if_count": int(results["if_anomaly"].sum()),
        "overlap_count": int(both.sum()),
        "mad_only_count": int(mad_only.sum()),
        "if_only_count": int(if_only.sum()),
    }

    print_service_summary(summary)

    if summary["if_only_count"]:
        print_if_only_records(results, if_only)

    return summary


def print_service_summary(summary: dict) -> None:
    """Print detector counts for one service."""
    print(f"\nService: {summary['url']}")
    print(f"Training records: {summary['training_records']}")
    print(f"Test records: {summary['test_records']}")
    print(f"MAD anomalies: {summary['mad_count']}")
    print(f"Isolation Forest anomalies: {summary['if_count']}")
    print(f"Detected by both: {summary['overlap_count']}")
    print(f"MAD only: {summary['mad_only_count']}")
    print(f"Isolation Forest only: {summary['if_only_count']}")


def print_if_only_records(
    results: pd.DataFrame,
    if_only: pd.Series,
) -> None:
    """Print records flagged by Isolation Forest but not by MAD."""
    columns = [
        "id",
        "response_time_ms",
        "baseline_ms",
        "mad_ms",
        "robust_z",
        "anomaly_score",
        "checked_at",
    ]
    print("\nIsolation Forest-only records:")
    print(
        results.loc[if_only, columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )


def print_overall_summary(summaries: list[dict]) -> None:
    """Print aggregate detector counts across evaluated services."""
    if not summaries:
        print("\nNo services could be evaluated.")
        return

    print("\nOverall test summary")
    print(f"Test records: {sum(item['test_records'] for item in summaries)}")
    print(f"MAD anomalies: {sum(item['mad_count'] for item in summaries)}")
    print(f"Isolation Forest anomalies: {sum(item['if_count'] for item in summaries)}")
    print(
        "Detected by both: "
        f"{sum(item['overlap_count'] for item in summaries)}"
    )
    print(
        "Isolation Forest only: "
        f"{sum(item['if_only_count'] for item in summaries)}"
    )


def main() -> None:
    """Run the chronological comparison experiment."""
    data = load_data()

    print("AYNAI - MAD vs Isolation Forest: disagreement analysis")
    print("Chronological holdout; no verified anomaly labels")

    summaries = [
        summary
        for url, service in data.groupby("url")
        if (summary := evaluate_service(url, service)) is not None
    ]

    print_overall_summary(summaries)
    print(
        "\nDetector disagreements are not verified false positives. "
        "Ground-truth anomaly labels are unavailable."
    )


if __name__ == "__main__":
    main()