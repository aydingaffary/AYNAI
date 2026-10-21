import argparse
from math import isfinite

from aynai.analytics import summarize_anomalies, summarize_results
from aynai.monitor import monitor_services
from aynai.scheduler import run_periodic_monitoring


URLS = [
    "https://example.com",
    "https://www.python.org",
]

INTERVAL_SECONDS = 30



def display_results(results: list[dict]) -> None:
    print("\nAYNAI - Network Monitor")
    print("=" * 40)

    for result in results:
        print(f"URL: {result['url']}")
        print(f"Available: {result['is_available']}")
        print(f"HTTP Status: {result['status_code']}")
        print(f"Response Time: {result['response_time_ms']:.2f} ms")
        print(f"Error: {result['error']}")

        if result["is_available"]:
            anomaly = result.get("anomaly")

            if anomaly is None:
                print("Latency anomaly: Not evaluated")
            else:
                print(f"Latency baseline: {anomaly['baseline_ms']} ms")
                print(f"Latency MAD: {anomaly['mad_ms']} ms")

                if anomaly["reason"] == "Insufficient historical data":
                    print(
                        "Latency anomaly: "
                        "Not evaluated (insufficient history)"
                    )
                elif anomaly["is_anomaly"]:
                    print("Latency anomaly: DETECTED")
                else:
                    print("Latency anomaly: No anomaly detected")

        print("-" * 40)


def display_analytics(since_hours: float | None = None) -> None:
    print("\nAYNAI - Monitoring Analytics")
    print("=" * 40)

    for url in URLS:
        summary = summarize_results(url=url, since_hours=since_hours)
        anomaly_summary = summarize_anomalies(
            url=url,
            since_hours=since_hours,
        )

        print(f"URL: {url}")
        print(f"Total checks: {summary['total_checks']}")
        print(f"Successful: {summary['successful_checks']}")
        print(f"Failed: {summary['failed_checks']}")
        print(f"Availability: {summary['availability_rate_percent']}%")
        print(
            "Average response time: "
            f"{summary['average_response_time_ms']} ms"
        )
        print(f"P95 response time: {summary['p95_response_time_ms']} ms")

        print("\nLatency Anomaly Analytics")
        print(f"Evaluated checks: {anomaly_summary['evaluated_checks']}")
        print(f"Anomalies detected: {anomaly_summary['anomaly_count']}")
        print(f"Normal checks: {anomaly_summary['normal_count']}")
        print(
            "Unevaluated checks: "
            f"{anomaly_summary['unevaluated_checks']}"
        )
        print(
            "Anomaly rate: "
            f"{anomaly_summary['anomaly_rate_percent']}%"
        )
        print("-" * 40)


def monitor_and_display(urls: list[str], db_path=None) -> None:
    if db_path is None:
        results = monitor_services(urls)
    else:
        results = monitor_services(urls, db_path=db_path)

    display_results(results)
    display_analytics()


def main() -> None:
    parser = argparse.ArgumentParser(description="AYNAI network monitoring")
    parser.add_argument(
        "--analytics-hours",
        type=float,
        help="Display analytics for the last N hours without starting monitoring",
    )
    args = parser.parse_args()

    if args.analytics_hours is not None:
        if not isfinite(args.analytics_hours) or args.analytics_hours <= 0:
            parser.error("--analytics-hours must be a finite number greater than zero")

        display_analytics(since_hours=args.analytics_hours)
        return

    print("Starting AYNAI periodic monitoring...")
    print(f"Check interval: {INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop.")

    run_periodic_monitoring(
        URLS,
        interval_seconds=INTERVAL_SECONDS,
        monitor_fn=monitor_and_display,
    )


if __name__ == "__main__":
    main()
