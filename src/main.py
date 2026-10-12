
from aynai.monitor import monitor_services


def main() -> None:
    urls = [
        "https://example.com",
        "https://www.python.org",
    ]

    results = monitor_services(urls)

    print("AYNAI - Network Monitor")
    print("=" * 40)

    for result in results:
        print(f"URL: {result['url']}")
        print(f"Available: {result['is_available']}")
        print(f"HTTP Status: {result['status_code']}")
        print(f"Response Time: {result['response_time_ms']} ms")
        print(f"Error: {result['error']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
