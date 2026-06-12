from __future__ import annotations


def format_research_report(bundle: dict) -> str:
    match = bundle.get("facts", {}).get("match", {})
    match_name = match.get("home_team") and match.get("away_team")
    title = (
        f"{match.get('home_team')} vs {match.get('away_team')}"
        if match_name
        else str(match.get("name", "match"))
    )
    lines = [
        "\u4ec5\u5a31\u4e50\u9884\u6d4b\uff0c\u4e0d\u6784\u6210\u6295\u6ce8\u5efa\u8bae\u3002",
        f"Research bundle: {title}",
        f"Sources used: {len(bundle.get('sources', []))}",
        f"Excluded sources: {len(bundle.get('excluded_sources', []))}",
        f"Requires manual confirmation: {bundle.get('requires_manual_confirmation')}",
    ]
    warnings = bundle.get("warnings", [])
    if warnings:
        lines.append("Warnings: " + ", ".join(warnings))
    return "\n".join(lines)

