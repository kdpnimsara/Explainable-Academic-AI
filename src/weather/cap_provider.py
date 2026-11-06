
"""Provider for official Sri Lanka CAP weather alerts."""

from datetime import datetime, timezone
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from .models import WeatherAlert


DEFAULT_FEED_URL = "https://was.meteo.gov.lk/cap/en/rss.xml"


class WeatherProviderError(RuntimeError):
    """Raised when weather alerts cannot be retrieved or parsed."""


def _local_name(tag: str) -> str:
    """Remove an XML namespace from a tag."""
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    """Get text from a direct XML child."""
    for child in element:
        if _local_name(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _parse_datetime(value: str) -> datetime | None:
    """Parse a timezone-aware CAP timestamp."""
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WeatherProviderError(
            f"Invalid CAP timestamp: {value}"
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WeatherProviderError(
            f"CAP timestamp has no timezone: {value}"
        )

    return parsed.astimezone(timezone.utc)


class CAPWeatherProvider:
    """Retrieve official CAP alerts from the Sri Lankan RSS feed."""

    def __init__(
        self,
        feed_url: str = DEFAULT_FEED_URL,
        timeout: int = 20,
        max_alerts: int = 50,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        if max_alerts <= 0:
            raise ValueError("max_alerts must be greater than zero.")

        self.feed_url = feed_url
        self.timeout = timeout
        self.max_alerts = max_alerts

    def _download(self, url: str) -> bytes:
        """Download XML content from the official source."""
        request = Request(
            url,
            headers={"User-Agent": "Explainable-Academic-AI/1.0"},
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()

        except (URLError, TimeoutError, OSError) as exc:
            raise WeatherProviderError(
                f"Unable to download weather data: {url}"
            ) from exc

    def _get_cap_links(self, rss_data: bytes) -> list[str]:
        """Extract CAP XML links from the RSS feed."""
        try:
            root = ET.fromstring(rss_data)
        except ET.ParseError as exc:
            raise WeatherProviderError(
                "The weather RSS feed contains invalid XML."
            ) from exc

        links = []

        for item in root.iter():
            if _local_name(item.tag) != "item":
                continue

            link = _child_text(item, "link")

            if link:
                resolved_url = urljoin(self.feed_url, link)

                feed_parts = urlparse(self.feed_url)
                alert_parts = urlparse(resolved_url)

                # Only follow links from the configured official host.
                if (
                    alert_parts.scheme != feed_parts.scheme
                    or alert_parts.hostname != feed_parts.hostname
                ):
                    raise WeatherProviderError(
                        "RSS contains a link outside the configured host."
                    )

                links.append(resolved_url)

            if len(links) >= self.max_alerts:
                break

        return links

    def _parse_cap(
        self,
        xml_data: bytes,
        source_url: str,
    ) -> WeatherAlert:
        """Convert a CAP XML document into a WeatherAlert."""
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as exc:
            raise WeatherProviderError(
                f"Invalid CAP XML: {source_url}"
            ) from exc

        info_elements = [
            element
            for element in root
            if _local_name(element.tag) == "info"
        ]

        if not info_elements:
            raise WeatherProviderError(
                "CAP document does not contain an info section."
            )

        # Prefer English information when multiple languages exist.
        info = next(
            (
                element
                for element in info_elements
                if _child_text(element, "language").lower().startswith("en")
            ),
            info_elements[0],
        )

        identifier = _child_text(root, "identifier")
        sender = _child_text(root, "sender")
        sent_at = _parse_datetime(_child_text(root, "sent"))

        if not identifier or not sender or sent_at is None:
            raise WeatherProviderError(
                "CAP document is missing required alert metadata."
            )

        area_descriptions = []

        for area in info:
            if _local_name(area.tag) != "area":
                continue

            area_description = _child_text(area, "areaDesc")

            if area_description and area_description not in area_descriptions:
                area_descriptions.append(area_description)

        retrieved_at = datetime.now(timezone.utc)

        return WeatherAlert(
            identifier=identifier,
            sender=sender,
            sent_at=sent_at,
            status=_child_text(root, "status") or "Unknown",
            message_type=_child_text(root, "msgType") or "Unknown",
            scope=_child_text(root, "scope") or "Unknown",
            event=_child_text(info, "event"),
            urgency=_child_text(info, "urgency"),
            severity=_child_text(info, "severity"),
            certainty=_child_text(info, "certainty"),
            headline=_child_text(info, "headline"),
            description=_child_text(info, "description"),
            instruction=_child_text(info, "instruction") or None,
            effective_at=_parse_datetime(
                _child_text(info, "effective")
            ),
            onset_at=_parse_datetime(
                _child_text(info, "onset")
            ),
            expires_at=_parse_datetime(
                _child_text(info, "expires")
            ),
            area_descriptions=tuple(area_descriptions),
            source_url=source_url,
            retrieved_at=retrieved_at,
        )

    def fetch_alerts(self) -> list[WeatherAlert]:
        """Retrieve all alerts currently listed in the RSS feed."""
        rss_data = self._download(self.feed_url)
        cap_links = self._get_cap_links(rss_data)

        alerts = []

        for link in cap_links:
            cap_data = self._download(link)
            alert = self._parse_cap(cap_data, link)
            alerts.append(alert)

        return alerts

    def fetch_current_alerts(
        self,
        now: datetime | None = None,
    ) -> list[WeatherAlert]:
        """Retrieve only alerts valid at the specified time."""
        current_time = now or datetime.now(timezone.utc)

        return [
            alert
            for alert in self.fetch_alerts()
            if alert.is_current(current_time)
        ]
