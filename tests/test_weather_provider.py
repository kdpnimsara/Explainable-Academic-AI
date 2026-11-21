
from datetime import datetime, timezone

import pytest

from src.weather.models import WeatherAlert
from src.weather.cap_provider import (
    CAPWeatherProvider,
    WeatherProviderError,
    DEFAULT_FEED_URL,
)


SENT = datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc)
START = datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc)
END = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)


CAP_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
    <identifier>test-alert-001</identifier>
    <sender>info@meteo.gov.lk</sender>
    <sent>2026-11-05T10:00:00Z</sent>
    <status>Actual</status>
    <msgType>Alert</msgType>
    <scope>Public</scope>
    <info>
        <language>en-US</language>
        <event>Lightning</event>
        <urgency>Immediate</urgency>
        <severity>Severe</severity>
        <certainty>Possible</certainty>
        <headline>Lightning Warning</headline>
        <description>Severe lightning expected.</description>
        <instruction>Stay indoors.</instruction>
        <effective>2026-11-05T10:00:00Z</effective>
        <expires>2026-11-05T14:00:00Z</expires>
        <area>
            <areaDesc>Colombo District</areaDesc>
        </area>
        <area>
            <areaDesc>Colombo District</areaDesc>
        </area>
        <area>
            <areaDesc>Gampaha District</areaDesc>
        </area>
    </info>
</alert>
"""


RSS_XML = b"""<?xml version="1.0"?>
<rss version="2.0">
    <channel>
        <item>
            <title>Lightning Warning</title>
            <link>https://was.meteo.gov.lk/cap/en/test.xml</link>
        </item>
    </channel>
</rss>
"""


def make_alert(**overrides):
    """Create a test weather alert."""

    data = {
        "identifier": "test-001",
        "sender": "info@meteo.gov.lk",
        "sent_at": SENT,
        "status": "Actual",
        "message_type": "Alert",
        "scope": "Public",
        "event": "Lightning",
        "urgency": "Immediate",
        "severity": "Severe",
        "certainty": "Possible",
        "headline": "Lightning Warning",
        "description": "Test warning",
        "source_url": "https://was.meteo.gov.lk/test.xml",
        "retrieved_at": SENT,
        "effective_at": START,
        "expires_at": END,
    }

    data.update(overrides)
    return WeatherAlert(**data)


def test_current_alert_is_valid():
    alert = make_alert()

    now = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)

    assert alert.is_current(now) is True


def test_expired_alert_is_not_current():
    alert = make_alert()

    now = datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc)

    assert alert.is_current(now) is False


def test_future_alert_is_not_current():
    alert = make_alert(
        effective_at=datetime(
            2026, 11, 5, 13, 0, tzinfo=timezone.utc
        )
    )

    now = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)

    assert alert.is_current(now) is False


def test_inactive_status_is_not_current():
    alert = make_alert(status="Test")

    now = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)

    assert alert.is_current(now) is False


def test_cancel_message_is_not_current():
    alert = make_alert(message_type="Cancel")

    now = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)

    assert alert.is_current(now) is False


def test_timezone_naive_now_is_rejected():
    alert = make_alert()

    with pytest.raises(ValueError):
        alert.is_current(datetime(2026, 11, 5, 12, 0))


def test_cap_parser_extracts_alert_details():
    provider = CAPWeatherProvider()

    alert = provider._parse_cap(
        CAP_XML,
        "https://was.meteo.gov.lk/cap/en/test.xml",
    )

    assert alert.event == "Lightning"
    assert alert.severity == "Severe"
    assert alert.urgency == "Immediate"
    assert alert.sender == "info@meteo.gov.lk"
    assert alert.headline == "Lightning Warning"
    assert alert.instruction == "Stay indoors."

    assert alert.area_descriptions == (
        "Colombo District",
        "Gampaha District",
    )


def test_cap_parser_rejects_invalid_xml():
    provider = CAPWeatherProvider()

    with pytest.raises(WeatherProviderError):
        provider._parse_cap(
            b"<invalid",
            "https://was.meteo.gov.lk/test.xml",
        )


def test_rss_parser_extracts_cap_links():
    provider = CAPWeatherProvider()

    links = provider._get_cap_links(RSS_XML)

    assert links == [
        "https://was.meteo.gov.lk/cap/en/test.xml"
    ]


def test_rss_parser_rejects_external_links():
    provider = CAPWeatherProvider()

    malicious_rss = b"""
    <rss>
        <channel>
            <item>
                <link>https://example.com/fake.xml</link>
            </item>
        </channel>
    </rss>
    """

    with pytest.raises(WeatherProviderError):
        provider._get_cap_links(malicious_rss)


def test_fetch_alerts_without_live_internet(monkeypatch):
    provider = CAPWeatherProvider()

    def fake_download(url):
        if url == DEFAULT_FEED_URL:
            return RSS_XML

        if url == "https://was.meteo.gov.lk/cap/en/test.xml":
            return CAP_XML

        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(provider, "_download", fake_download)

    alerts = provider.fetch_alerts()

    assert len(alerts) == 1
    assert alerts[0].event == "Lightning"
    assert alerts[0].severity == "Severe"
