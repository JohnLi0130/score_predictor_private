from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, validator

from .intelligence.schemas import IntelligenceInput


def _validate_decimal_odds(value: float) -> float:
    if value <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.")
    return float(value)


def _validate_positive_lambda(value: float) -> float:
    if value <= 0:
        raise ValueError("Lambda values must be positive.")
    return float(value)


class Odds1X2(BaseModel):
    home: float
    draw: float
    away: float

    _odds_are_valid = validator("home", "draw", "away", allow_reuse=True)(
        _validate_decimal_odds
    )


class OverUnderOdds(BaseModel):
    line: float
    over_odds: float
    under_odds: float

    _odds_are_valid = validator("over_odds", "under_odds", allow_reuse=True)(
        _validate_decimal_odds
    )

    @validator("line")
    def line_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Over/under line must be positive.")
        return float(value)


class AsianHandicapOdds(BaseModel):
    line: float
    home_odds: float
    away_odds: float

    _odds_are_valid = validator("home_odds", "away_odds", allow_reuse=True)(
        _validate_decimal_odds
    )


class SportteryRqspfOdds(BaseModel):
    handicap: float
    home: float
    draw: float
    away: float

    _odds_are_valid = validator("home", "draw", "away", allow_reuse=True)(
        _validate_decimal_odds
    )


class OddsChannelConfig(BaseModel):
    role: str
    source: str
    provider: Any = None
    weight: float = 1.0

    @validator("role", "source")
    def required_channel_text_must_not_be_blank(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Odds channel role/source must not be blank.")
        return normalized

    @validator("weight")
    def channel_weight_must_be_bounded(cls, value: float) -> float:
        if value < 0 or value > 5:
            raise ValueError("Odds channel weight must stay between 0 and 5.")
        return float(value)


class OddsChannels(BaseModel):
    international: OddsChannelConfig = Field(
        default_factory=lambda: OddsChannelConfig(
            role="primary_calibration",
            source="the_odds_api",
            provider="pinnacle",
            weight=1.0,
        )
    )
    sporttery: OddsChannelConfig = Field(
        default_factory=lambda: OddsChannelConfig(
            role="supplemental_calibration",
            source="yaml",
            provider="sporttery",
            weight=0.35,
        )
    )


class BttsOdds(BaseModel):
    yes: float
    no: float

    _odds_are_valid = validator("yes", "no", allow_reuse=True)(_validate_decimal_odds)


class InternalModel(BaseModel):
    home_lambda: float
    away_lambda: float

    _lambdas_are_valid = validator("home_lambda", "away_lambda", allow_reuse=True)(
        _validate_positive_lambda
    )


class PredictionSettings(BaseModel):
    market_weight: float = 0.65
    max_goals: int = 7
    score_model: str = "poisson"
    dc_enabled: bool = False
    market_only_mode: bool = False
    h2h_weight: float = 1.0
    x1x2_weight: float = 1.0
    totals_weight: float = 1.0
    ou_weight: float = 1.0
    alternate_totals_weight: float = 0.8
    btts_weight: float = 0.6
    spreads_weight: float = 0.5
    correct_score_weight: float = 0.35
    sporttery_1x2_weight: float = 0.15
    sporttery_handicap_3way_weight: float = 0.15
    sporttery_total_goals_weight: float = 0.30
    sporttery_correct_score_weight: float = 0.20
    sporttery_half_full_weight: float = 0.0
    team_adjustment_strength: float = 1.0

    @validator("market_weight")
    def market_weight_must_be_probability(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("market_weight must be between 0 and 1.")
        return float(value)

    @validator("max_goals")
    def max_goals_must_be_reasonable(cls, value: int) -> int:
        if value < 3 or value > 20:
            raise ValueError("max_goals must be between 3 and 20.")
        return int(value)

    @validator("score_model")
    def score_model_must_be_supported(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized != "poisson":
            raise ValueError("V0 supports only the poisson score_model.")
        return normalized

    @validator(
        "x1x2_weight",
        "h2h_weight",
        "totals_weight",
        "ou_weight",
        "alternate_totals_weight",
        "btts_weight",
        "spreads_weight",
        "correct_score_weight",
        "sporttery_1x2_weight",
        "sporttery_handicap_3way_weight",
        "sporttery_total_goals_weight",
        "sporttery_correct_score_weight",
        "sporttery_half_full_weight",
        "team_adjustment_strength",
    )
    def calibration_weights_must_be_bounded(cls, value: float) -> float:
        if value < 0 or value > 5:
            raise ValueError("Calibration weights must stay between 0 and 5.")
        return float(value)

    @validator("sporttery_half_full_weight")
    def half_full_weight_must_remain_zero(cls, value: float) -> float:
        if float(value) != 0.0:
            raise ValueError("sporttery_half_full_weight must remain 0 in phase one.")
        return 0.0


class MarketRoles(BaseModel):
    calibration_sources: list[str] = Field(default_factory=list)
    value_comparison_sources: list[str] = Field(default_factory=lambda: ["sporttery"])
    roles_configured: bool = False

    @validator("calibration_sources", "value_comparison_sources", pre=True)
    def sources_must_be_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @validator("calibration_sources", "value_comparison_sources", each_item=True)
    def source_names_must_not_be_blank(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("Market role source names must not be blank.")
        return normalized


class LambdaAdjustments(BaseModel):
    home_factors: list[float] = Field(default_factory=list)
    away_factors: list[float] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @validator("home_factors", "away_factors", each_item=True)
    def factors_must_be_bounded(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Adjustment factors must be positive.")
        if value < 0.5 or value > 1.5:
            raise ValueError("Adjustment factors must stay between 0.5 and 1.5 in V0.")
        return float(value)


class MatchInput(BaseModel):
    match: str
    kickoff_time: str
    timezone: str = "Asia/Shanghai"
    target: str = "90min_score"
    venue_type: str
    prediction_time: str
    odds_1x2: Odds1X2
    internal_model: InternalModel
    over_under: OverUnderOdds | None = None
    over_under_markets: list[OverUnderOdds] = Field(default_factory=list)
    btts: BttsOdds | None = None
    correct_score_odds: dict[str, float] = Field(default_factory=dict)
    correct_score_other_odds: dict[str, float] = Field(default_factory=dict)
    sporttery_1x2: Odds1X2 | None = None
    sporttery_handicap_3way: SportteryRqspfOdds | None = None
    sporttery_correct_score_odds: dict[str, float] = Field(default_factory=dict)
    sporttery_total_goals_odds: dict[str, float] = Field(default_factory=dict)
    half_full_time_odds: dict[str, float] = Field(default_factory=dict)
    asian_handicap: AsianHandicapOdds | None = None
    rqspf: SportteryRqspfOdds | None = None
    calibration_market: dict[str, Any] = Field(default_factory=dict)
    value_comparison_market: dict[str, Any] = Field(default_factory=dict)
    international_market: dict[str, Any] = Field(default_factory=dict)
    sporttery_market: dict[str, Any] = Field(default_factory=dict)
    market_roles: MarketRoles = Field(default_factory=MarketRoles)
    odds_channels: OddsChannels = Field(default_factory=OddsChannels)
    settings: PredictionSettings = Field(default_factory=PredictionSettings)
    adjustments: LambdaAdjustments = Field(default_factory=LambdaAdjustments)
    injuries: dict[str, Any] = Field(default_factory=dict)
    expected_lineups: dict[str, Any] = Field(default_factory=dict)
    tactical_notes: dict[str, Any] = Field(default_factory=dict)
    motivation: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    intelligence: IntelligenceInput | None = None

    @validator("match", "kickoff_time", "venue_type", "prediction_time")
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Required text fields must not be blank.")
        return value.strip()

    @validator("target")
    def target_must_be_90min_score(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != "90min_score":
            raise ValueError("V0 supports only target='90min_score'.")
        return normalized

    @validator("over_under_markets", always=True)
    def include_primary_over_under(
        cls, value: list[OverUnderOdds], values: dict[str, Any]
    ) -> list[OverUnderOdds]:
        markets = list(value or [])
        primary = values.get("over_under")
        if primary is not None and all(market.line != primary.line for market in markets):
            markets.append(primary)
        return markets

    @validator("correct_score_odds")
    def correct_score_odds_must_be_valid(cls, value: dict[str, float]) -> dict[str, float]:
        for score, odds in value.items():
            if not isinstance(score, str) or "-" not in score:
                raise ValueError("Correct score keys must look like '1-0'.")
            _validate_decimal_odds(float(odds))
        return {str(score): float(odds) for score, odds in value.items()}

    @validator("sporttery_correct_score_odds")
    def sporttery_correct_score_odds_must_be_valid(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        for score, odds in value.items():
            if not isinstance(score, str) or "-" not in score:
                raise ValueError("Sporttery correct score keys must look like '1-0'.")
            _validate_decimal_odds(float(odds))
        return {str(score): float(odds) for score, odds in value.items()}

    @validator("sporttery_total_goals_odds", "half_full_time_odds", "correct_score_other_odds")
    def sporttery_odds_must_be_valid(cls, value: dict[str, float]) -> dict[str, float]:
        for outcome, odds in value.items():
            if not isinstance(outcome, str) or not outcome.strip():
                raise ValueError("Market outcome keys must not be blank.")
            _validate_decimal_odds(float(odds))
        return {str(outcome): float(odds) for outcome, odds in value.items()}
