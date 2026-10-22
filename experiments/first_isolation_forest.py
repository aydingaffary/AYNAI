
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "aynai.db"
TRAIN_RATIO = 0.8
MIN_RECORDS = 10


def main() -> None:
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

    print("AYNAI - Isolation Forest chronological evaluation")
    print(f"Total successful records: {len(data)}")
    print(f"Training ratio: {TRAIN_RATIO:.0%}")
    print(f"Test ratio: {1 - TRAIN_RATIO:.0%}")

    for url, service_data in data.groupby("url"):
        service_data = service_data.sort_values("checked_at").reset_index(
            drop=True
        )

        if len(service_data) < MIN_RECORDS:
            print(f"\nSkipping {url}: only {len(service_data)} records")
            continue

        split_index = int(len(service_data) * TRAIN_RATIO)
        train_data = service_data.iloc[:split_index]
        test_data = service_data.iloc[split_index:]

        if train_data.empty or test_data.empty:
            print(f"\nSkipping {url}: insufficient train/test data")
            continue

        features_train = train_data[["response_time_ms"]]
        features_test = test_data[["response_time_ms"]]

        model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
        )
        model.fit(features_train)

        predictions = model.predict(features_test)
        scores = model.decision_function(features_test)

        results = test_data.copy()
        results["ml_prediction"] = predictions
        results["anomaly_score"] = scores

        anomalies = results[results["ml_prediction"] == -1].sort_values(
            "anomaly_score"
        )

        print(f"\nService: {url}")
        print(f"Training records: {len(train_data)}")
        print(f"Test records: {len(test_data)}")
        print(
            f"Training period: {train_data['checked_at'].min()} "
            f"to {train_data['checked_at'].max()}"
        )
        print(
            f"Test period: {test_data['checked_at'].min()} "
            f"to {test_data['checked_at'].max()}"
        )
        print(f"Predicted anomalies: {len(anomalies)}")
        print(
            f"Predicted anomaly rate: "
            f"{len(anomalies) / len(test_data):.1%}"
        )

        if not anomalies.empty:
            print("Most unusual test records:")
            print(
                anomalies[
                    ["response_time_ms", "anomaly_score", "checked_at"]
                ]
                .head(5)
                .to_string(index=False)
            )

    print(
        "\nNote: Predictions are unsupervised and are not verified anomaly labels."
    )
    print(
        "This is a chronological holdout experiment, not a measurement "
        "of model accuracy."
    )


if __name__ == "__main__":
    main()
