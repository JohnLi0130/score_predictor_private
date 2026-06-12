from __future__ import annotations

import time
from urllib.request import Request, urlopen

from .base import utc_now_iso
from .source_whitelist import classify_trust_tier, load_whitelist

try:
    from score_predictor.intelligence.source_policy import validate_source
except Exception:  # pragma: no cover - only used if the V1 module is unavailable.
    validate_source = None  # type: ignore[assignment]


BANNED_CONTENT_KEYWORDS = [
    "prediction",
    "predictions",
    "betting",
    "betting tips",
    "tips",
    "tipster",
    "recommended bet",
    "pick",
    "picks",
    "odds analysis",
    "score prediction",
    "\u9884\u6d4b",
    "\u63a8\u8350",
    "\u7ade\u5f69\u63a8\u8350",
    "\u7ea2\u5355",
    "\u7206\u51b7",
    "\u7a33\u80c6",
    "\u76d8\u53e3\u5206\u6790",
    "\u6bd4\u5206\u9884\u6d4b",
    "\u6295\u6ce8",
    "\u5b9e\u5355",
    "\u547d\u4e2d",
    "\u4e32\u5173",
    "\u770b\u597d",
    "\u4e34\u573a\u63a8\u8350",
]


class FetchError(RuntimeError):
    pass


def content_has_banned_keywords(text: str) -> bool:
    normalized = text.lower()
    return any(keyword.lower() in normalized for keyword in BANNED_CONTENT_KEYWORDS)


def _request_text(url: str, timeout_seconds: int) -> tuple[int, str]:
    try:
        import httpx

        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "score-predictor-v2 factual-fetcher"},
        ) as client:
            response = client.get(url)
            return response.status_code, response.text
    except ModuleNotFoundError:
        request = Request(
            url,
            headers={"User-Agent": "score-predictor-v2 factual-fetcher"},
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), body


def _request_text_with_retry(url: str, timeout_seconds: int) -> tuple[int, str]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return _request_text(url, timeout_seconds)
        except Exception as exc:  # pragma: no cover - exercised by integration use.
            last_error = exc
            if attempt < 2:
                time.sleep(0.25 * (2**attempt))
    raise FetchError(str(last_error))


def fetch_url(
    url: str,
    *,
    whitelist_path: str | None = None,
    allow_untrusted: bool = False,
    timeout_seconds: int = 15,
) -> dict:
    retrieved_at = utc_now_iso()
    warnings: list[str] = []
    whitelist = load_whitelist(whitelist_path)
    trust_tier = classify_trust_tier(url, whitelist)
    domain_allowed = trust_tier != "untrusted"

    if not domain_allowed and not allow_untrusted:
        return {
            "url": url,
            "status_code": None,
            "text": None,
            "retrieved_at": retrieved_at,
            "trust_tier": trust_tier,
            "allowed": False,
            "warnings": ["domain_not_in_whitelist"],
            "reason": "domain_not_in_whitelist",
        }

    if not domain_allowed and allow_untrusted:
        warnings.append("untrusted_manual_reference_only")

    try:
        status_code, text = _request_text_with_retry(url, timeout_seconds)
    except Exception as exc:
        return {
            "url": url,
            "status_code": None,
            "text": None,
            "retrieved_at": retrieved_at,
            "trust_tier": trust_tier,
            "allowed": False,
            "warnings": warnings + ["fetch_failed"],
            "reason": f"fetch_failed: {exc}",
        }

    allowed = domain_allowed
    reason: str | None = None
    if content_has_banned_keywords(text):
        allowed = False
        reason = "prediction_or_betting_content_detected"
        warnings.append(reason)
    elif validate_source is not None:
        policy = validate_source("", text)
        if policy.get("reason") == "prediction_or_betting_content_detected":
            allowed = False
            reason = "prediction_or_betting_content_detected"
            warnings.append(reason)
        elif not policy.get("allowed"):
            warnings.append(str(policy.get("reason", "source_not_clearly_factual")))

    if not domain_allowed and allow_untrusted:
        allowed = False
        reason = reason or "untrusted_manual_reference_only"

    return {
        "url": url,
        "status_code": status_code,
        "text": text,
        "retrieved_at": retrieved_at,
        "trust_tier": trust_tier,
        "allowed": allowed,
        "warnings": list(dict.fromkeys(warnings)),
        "reason": reason,
    }

