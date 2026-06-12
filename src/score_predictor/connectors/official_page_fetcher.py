from __future__ import annotations

import re
from html.parser import HTMLParser

from .http_client import content_has_banned_keywords, fetch_url


class _ReadableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "nav", "header", "footer"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "nav", "header", "footer"}:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def text(self) -> str:
        return " ".join(self._parts)


def extract_factual_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(" ")
    except ModuleNotFoundError:
        parser = _ReadableTextParser()
        parser.feed(html)
        text = parser.text()
    return re.sub(r"\s+", " ", text).strip()


def fetch_official_fact_page(url: str, whitelist_path: str | None = None) -> dict:
    fetched = fetch_url(url, whitelist_path=whitelist_path)
    if not fetched.get("allowed"):
        return {
            **fetched,
            "factual_text": None,
            "metadata": {
                "source_policy": "facts_only_no_prediction_or_betting_content",
            },
        }

    text = extract_factual_text(fetched.get("text") or "")
    warnings = list(fetched.get("warnings") or [])
    allowed = bool(text)
    reason = fetched.get("reason")
    if not text:
        warnings.append("official_page_empty_after_extraction")
        reason = "official_page_empty_after_extraction"
    elif content_has_banned_keywords(text):
        allowed = False
        warnings.append("prediction_or_betting_content_detected")
        reason = "prediction_or_betting_content_detected"

    return {
        **fetched,
        "factual_text": text if allowed else None,
        "allowed": allowed,
        "warnings": list(dict.fromkeys(warnings)),
        "reason": reason,
        "metadata": {
            "source_policy": "facts_only_no_prediction_or_betting_content",
            "text_length": len(text),
        },
    }

