import pytest

from aynai.quality import analyze_data_quality


def make_record(
    *,
    is_available=True,
    response_time_ms=100.0,
    is_anomaly=None,
    latency_anomaly_label=None,
    availability_issue_label=None,
):
    return {
        "is_available": int(is_available),
        "response_time_ms": response_time_ms,
        "is_anomaly": is_anomaly,
        "latency_anomaly_label": latency_anomaly_label,
        "availability_issue_label": availability_issue_label,
    }


def test_empty_database_returns_zero_counts(monkeypatch):
    monkeypatch.setattr("aynai.quality.get_results", lambda **kwargs: [])

    report = analyze_data_quality("https://example.com")

    assert report["url"] == "https://example.com"
    assert report["total_records"] == 0
    assert report["successful_records"] == 0
    assert report["failed_records"] == 0
    assert report["valid_latency_records"] == 0
    assert report["invalid_latency_records"] == 0
    assert report["analyzed_records"] == 0
    assert report["labeled_records"] == 0
    assert report["history_sufficient"] is False


def test_quality_report_counts_records_correctly(monkeypatch):
    records = [
        make_record(
            is_available=True,
            response_time_ms=120.5,
            is_anomaly=0,
            latency_anomaly_label=0,
        ),
        make_record(
            is_available=False,
            response_time_ms=50.0,
            availability_issue_label=1,
        ),
        make_record(
            is_available=True,
            response_time_ms=200.0,
            is_anomaly=1,
        ),
    ]

    monkeypatch.setattr(
        "aynai.quality.get_results",
        lambda **kwargs: records,
    )

    report = analyze_data_quality("https://example.com")

    assert report["total_records"] == 3
    assert report["successful_records"] == 2
    assert report["failed_records"] == 1
    assert report["valid_latency_records"] == 2
    assert report["invalid_latency_records"] == 0
    assert report["analyzed_records"] == 2
    assert report["labeled_records"] == 2
    assert report["history_sufficient"] is False


@pytest.mark.parametrize(
    "invalid_latency",
    [
        None,
        "invalid",
        float("nan"),
        float("inf"),
        -10.0,
    ],
)
def test_invalid_latency_values_are_counted(monkeypatch, invalid_latency):
    records = [
        make_record(response_time_ms=100.0),
        make_record(response_time_ms=invalid_latency),
    ]

    monkeypatch.setattr(
        "aynai.quality.get_results",
        lambda **kwargs: records,
    )

    report = analyze_data_quality("https://example.com")

    assert report["valid_latency_records"] == 1
    assert report["invalid_latency_records"] == 1


def test_history_is_sufficient_with_five_valid_records(monkeypatch):
    records = [
        make_record(response_time_ms=100.0)
        for _ in range(5)
    ]

    monkeypatch.setattr(
        "aynai.quality.get_results",
        lambda **kwargs: records,
    )

    report = analyze_data_quality("https://example.com")

    assert report["valid_latency_records"] == 5
    assert report["history_sufficient"] is True


def test_history_is_insufficient_with_four_valid_records(monkeypatch):
    records = [
        make_record(response_time_ms=100.0)
        for _ in range(4)
    ]

    monkeypatch.setattr(
        "aynai.quality.get_results",
        lambda **kwargs: records,
    )

    report = analyze_data_quality("https://example.com")

    assert report["valid_latency_records"] == 4
    assert report["history_sufficient"] is False