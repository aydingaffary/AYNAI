
import time
import urllib.error
import urllib.request


def check_http(url: str, timeout: float = 5.0) -> dict:
    """
    Check whether an HTTP service is reachable.

    Args:
        url: The URL to check.
        timeout: Maximum time to wait for a response.

    Returns:
        A dictionary containing the result of the check.
    """
    start_time = time.perf_counter()

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AYNAI-Monitor/0.1"},
            method="GET",
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "url": url,
            "is_available": 200 <= status_code < 400,
            "status_code": status_code,
            "response_time_ms": round(elapsed_ms, 2),
            "error": None,
        }

    except urllib.error.HTTPError as error:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "url": url,
            "is_available": False,
            "status_code": error.code,
            "response_time_ms": round(elapsed_ms, 2),
            "error": str(error),
        }

    except (urllib.error.URLError, TimeoutError, OSError) as error:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "url": url,
            "is_available": False,
            "status_code": None,
            "response_time_ms": round(elapsed_ms, 2),
            "error": str(error),
        }
