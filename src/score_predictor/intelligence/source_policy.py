from __future__ import annotations

BANNED_PREDICTION_KEYWORDS = [
    "预测",
    "推荐",
    "竞彩",
    "红单",
    "爆冷",
    "稳胆",
    "盘口分析",
    "比分预测",
    "实单",
    "投注",
    "赔率解读",
    "看好",
    "命中",
    "串关",
]

ALLOWED_FACT_KEYWORDS = [
    "squad",
    "lineup",
    "starting xi",
    "roster",
    "injury",
    "suspension",
    "venue",
    "kickoff",
    "weather",
    "official",
    "大名单",
    "首发",
    "阵容",
    "伤停",
    "停赛",
    "比赛地点",
    "开球时间",
]


def is_prediction_source(title: str, text: str) -> bool:
    combined = f"{title}\n{text}".lower()
    return any(keyword.lower() in combined for keyword in BANNED_PREDICTION_KEYWORDS)


def is_fact_source(title: str, text: str) -> bool:
    combined = f"{title}\n{text}".lower()
    return any(keyword.lower() in combined for keyword in ALLOWED_FACT_KEYWORDS)


def validate_source(title: str, text: str) -> dict:
    if is_prediction_source(title, text):
        return {
            "allowed": False,
            "reason": "prediction_or_betting_content_detected",
        }
    if is_fact_source(title, text):
        return {
            "allowed": True,
            "reason": "fact_source_detected",
        }
    return {
        "allowed": False,
        "reason": "source_not_clearly_factual",
    }

