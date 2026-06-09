import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BLOCKED_URLS = [
    "http://169.254.169.254/latest/meta-data/",          # AWS metadata endpoint (exact exploit)
    "http://192.168.1.1/admin",                           # Private IP range boundary
    "http://localhost/internal",                          # Loopback internal service
]

ALLOWED_URLS = [
    "https://ultralytics.com/images/zidane.jpg",          # Valid public URL
]

def is_ssrf_safe(url: str) -> bool:
    """Check that the URL does not target private/internal infrastructure."""
    import ipaddress
    import re
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    blocked_patterns = [
        r"^169\.254\.",          # Link-local / cloud metadata
        r"^10\.",                # RFC1918
        r"^172\.(1[6-9]|2\d|3[01])\.",  # RFC1918
        r"^192\.168\.",          # RFC1918
        r"^127\.",               # Loopback
        r"^::1$",                # IPv6 loopback
        r"^localhost$",          # Localhost hostname
    ]

    for pattern in blocked_patterns:
        if re.match(pattern, hostname, re.IGNORECASE):
            return False

    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return False
    except ValueError:
        pass

    return True


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_ssrf_blocked_urls(url):
    """Invariant: URLs targeting private/internal infrastructure must be blocked before fetching."""
    assert not is_ssrf_safe(url), (
        f"SSRF vulnerability: URL '{url}' targets internal infrastructure and must be blocked"
    )


@pytest.mark.parametrize("url", ALLOWED_URLS)
def test_ssrf_allowed_urls(url):
    """Invariant: Valid public URLs must pass the SSRF safety check."""
    assert is_ssrf_safe(url), (
        f"False positive: legitimate URL '{url}' was incorrectly blocked"
    )