from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml

DEFAULT_WHITELIST: dict[str, list[str]] = {
    "tier1_official": [
        "fifa.com",
        "the-afc.com",
        "uefa.com",
        "conmebol.com",
        "cafonline.com",
        "concacaf.com",
        "thecfa.cn",
        "sporttery.cn",
        "lottery.gov.cn",
    ],
    "tier2_data": [
        "fotmob.com",
        "sofascore.com",
        "transfermarkt.com",
        "worldfootball.net",
    ],
    "weather": ["open-meteo.com"],
    "odds": ["the-odds-api.com"],
}


def _default_whitelist_path() -> Path:
    return Path(__file__).resolve().parents[3] / "examples" / "trusted_sources.yaml"


def get_domain(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc or parsed.path
    host = host.split("@")[-1].split(":")[0].strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def load_whitelist(path: str | None = None) -> dict[str, list[str]]:
    whitelist_path = Path(path) if path else _default_whitelist_path()
    if not whitelist_path.exists():
        return {key: list(value) for key, value in DEFAULT_WHITELIST.items()}

    with whitelist_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Whitelist YAML must contain a mapping/object.")

    loaded: dict[str, list[str]] = {}
    for tier, domains in data.items():
        if domains is None:
            loaded[str(tier)] = []
            continue
        if not isinstance(domains, list):
            raise ValueError(f"Whitelist tier {tier!r} must be a list.")
        loaded[str(tier)] = [str(domain).lower().strip() for domain in domains]
    return loaded


def _domain_matches(domain: str, allowed_domain: str) -> bool:
    normalized = allowed_domain.lower().strip()
    if normalized.startswith("www."):
        normalized = normalized[4:]
    return domain == normalized or domain.endswith(f".{normalized}")


def classify_trust_tier(url: str, whitelist: dict) -> str:
    domain = get_domain(url)
    for tier, domains in whitelist.items():
        if any(_domain_matches(domain, str(allowed)) for allowed in domains or []):
            return str(tier)
    return "untrusted"


def is_domain_allowed(url: str, whitelist: dict) -> bool:
    return classify_trust_tier(url, whitelist) != "untrusted"

