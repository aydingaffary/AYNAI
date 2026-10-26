"""Review latency measurements and record independent human labels."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aynai.storage import DEFAULT_DB_PATH, get_results, update_labels

DEFAULT_BATCH_SIZE = 10
HISTORY_SIZE = 5


def load_review_candidates(db_path: Path) -> tuple[list[dict], list[dict]]:
    """Load successful measurements and separate labeled records."""
    records = get_results(db_path=db_path)
    records.sort(key=lambda row: (row["checked_at"], row["id"]))

    successful = [
        row
        for row in records
        if row["is_available"] == 1
        and row["response_time_ms"] is not None
    ]

    unlabeled = [
        row
        for row in successful
        if row["latency_anomaly_label"] is None
    ]

    return successful, unlabeled


def format_history(
    successful: list[dict],
    current: dict,
) -> list[dict]:
    """Return up to five earlier successful measurements for this URL."""
    earlier = [
        row
        for row in successful
        if row["url"] == current["url"]
        and (row["checked_at"], row["id"])
        < (current["checked_at"], current["id"])
    ]
    return earlier[-HISTORY_SIZE:]


def review_batch(
    db_path: Path = DEFAULT_DB_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Interactively label up to ten previously unlabeled records."""
    successful, candidates = load_review_candidates(db_path)

    if not candidates:
        print("No unlabeled successful measurements remain.")
        return

    eligible = [
        row
        for row in candidates
        if len(format_history(successful, row)) >= HISTORY_SIZE
    ]

    if not eligible:
        print(
            "No unlabeled records have at least five "
            "earlier successful measurements."
        )
        return

    batch = eligible[:batch_size]

    print(f"Reviewing {len(batch)} records.")
    print("\nLabeling guidelines:")
    print("  0 = Normal: latency appears consistent with prior measurements.")
    print("  1 = Anomaly: latency is clearly and unusually elevated.")
    print("  s = Skip: evidence is insufficient to make a confident decision.")
    print("  q = Quit: stop reviewing without labeling the current record.")
    print(
        "\nConsider the size of the increase and variation in the "
        "previous measurements."
    )
    print(
        "These labels reflect human judgment about latency, "
        "not proof of a network fault."
    )
    print("\nAlgorithm predictions are intentionally hidden.")

    saved = 0

    for current in batch:
        history = format_history(successful, current)

        print("\n" + "=" * 60)
        print(f"Record ID: {current['id']}")
        print(f"URL: {current['url']}")
        print(f"Checked at: {current['checked_at']}")
        print(f"Response time: {current['response_time_ms']} ms")
        print(f"HTTP status: {current['status_code']}")
        print(f"Previous successful measurements ({len(history)}):")

        for row in history:
            print(
                f"  ID {row['id']}: "
                f"{row['response_time_ms']} ms at {row['checked_at']}"
            )

        while True:
            choice = input("Your label [0/1/s/q]: ").strip().lower()

            if choice == "q":
                print(f"Stopped. Labels saved in this run: {saved}")
                return

            if choice == "s":
                break

            if choice not in {"0", "1"}:
                print("Enter 0, 1, s, or q.")
                continue

            notes = input("Optional note (press Enter to skip): ").strip()

            update_labels(
                current["id"],
                db_path=db_path,
                latency_anomaly_label=int(choice),
                label_notes=notes or None,
            )

            saved += 1
            print(f"Saved latency label for record {current['id']}.")
            break

    print(f"\nReview complete. Labels saved in this run: {saved}")


def main() -> None:
    """Parse command-line arguments and run the label reviewer."""
    parser = argparse.ArgumentParser(
        description="Review successful HTTP measurements and label latency anomalies."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (default: AYNAI's configured database).",
    )
    args = parser.parse_args()
    review_batch(db_path=args.db)

def main() -> None:
    """Parse command-line arguments and run the label reviewer."""
    parser = argparse.ArgumentParser(
        description="Review successful HTTP measurements and label latency anomalies."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (default: AYNAI's configured database).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of records to review (default: 10).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    review_batch(db_path=args.db, batch_size=args.batch_size)


if __name__ == "__main__":
    main()