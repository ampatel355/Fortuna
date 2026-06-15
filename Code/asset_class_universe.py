"""Asset-class-aware peer-universe selection for relative-strength rotation."""

from __future__ import annotations

import os


def _env_list(name: str, default: str) -> list[str]:
    """Read a comma-separated ticker list from the environment."""
    raw_text = os.environ.get(name, default).strip()
    return [ticker.strip().upper() for ticker in raw_text.split(",") if ticker.strip()]


ASSET_CLASS_DISPLAY_NAMES = {
    "us_equity_stock": "US equity stock",
    "us_equity_etf": "US equity ETF",
    "commodity_etf": "Commodity ETF",
    "crypto": "Crypto",
    "forex": "Forex",
    "equity_index_future": "Equity index future",
    "commodity_future": "Commodity future",
    "rate_future": "Rate future",
    "generic_future": "Generic future",
    "manual_override": "Manual override",
}
SUPPORTED_ASSET_CLASSES = list(ASSET_CLASS_DISPLAY_NAMES.keys())

# These universes are intentionally narrower than the original mixed basket.
# The goal is to compare each asset only to realistic peers in the same asset
# class, which makes cross-sectional momentum much harder to overstate.
ASSET_CLASS_UNIVERSES = {
    "us_equity_stock": _env_list(
        "RELATIVE_STRENGTH_US_EQUITY_STOCK_UNIVERSE",
        "AAPL,MSFT,INTC,CSCO,JPM,XOM,JNJ,PG,WMT,KO,HD",
    ),
    "us_equity_etf": _env_list(
        "RELATIVE_STRENGTH_US_EQUITY_ETF_UNIVERSE",
        "SPY,QQQ,DIA,IWM,MDY,XLK,XLF,XLV,XLY,XLP",
    ),
    "commodity_etf": _env_list(
        "RELATIVE_STRENGTH_COMMODITY_ETF_UNIVERSE",
        "GLD,SLV,USO,UNG,DBA,DBC",
    ),
    "crypto": _env_list(
        "RELATIVE_STRENGTH_CRYPTO_UNIVERSE",
        "BTC-USD,ETH-USD,XRP-USD,LTC-USD,BCH-USD,ADA-USD,DOGE-USD",
    ),
    "forex": _env_list(
        "RELATIVE_STRENGTH_FOREX_UNIVERSE",
        "EURUSD=X,GBPUSD=X,JPYUSD=X,AUDUSD=X,CADUSD=X,CHFUSD=X",
    ),
    "equity_index_future": _env_list(
        "RELATIVE_STRENGTH_EQUITY_INDEX_FUTURE_UNIVERSE",
        "ES=F,NQ=F,YM=F,RTY=F",
    ),
    "commodity_future": _env_list(
        "RELATIVE_STRENGTH_COMMODITY_FUTURE_UNIVERSE",
        "CL=F,NG=F,GC=F,SI=F,HG=F,ZC=F",
    ),
    "rate_future": _env_list(
        "RELATIVE_STRENGTH_RATE_FUTURE_UNIVERSE",
        "ZN=F,ZB=F,ZF=F",
    ),
    "generic_future": _env_list(
        "RELATIVE_STRENGTH_GENERIC_FUTURE_UNIVERSE",
        "ES=F,NQ=F,YM=F,RTY=F",
    ),
}

US_EQUITY_ETF_HINTS = set(
    ASSET_CLASS_UNIVERSES["us_equity_etf"]
    + ["VOO", "IVV", "VTI", "SCHX", "EFA", "VEA", "EEM", "VT"]
)
COMMODITY_ETF_HINTS = set(
    ASSET_CLASS_UNIVERSES["commodity_etf"]
    + ["CPER", "PDBC", "GSG", "PPLT", "DBB", "DBE"]
)
EQUITY_INDEX_FUTURE_HINTS = set(ASSET_CLASS_UNIVERSES["equity_index_future"])
COMMODITY_FUTURE_HINTS = set(ASSET_CLASS_UNIVERSES["commodity_future"])
RATE_FUTURE_HINTS = set(ASSET_CLASS_UNIVERSES["rate_future"])

FALLBACK_RULE_DESCRIPTION = (
    "If a ticker does not match crypto, forex, futures, or known ETF/commodity lists, "
    "treat it as a US equity stock by default."
)


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker symbol for classification and display."""
    return str(ticker).strip().upper()


def is_crypto_ticker(ticker: str) -> bool:
    """Return whether the symbol looks like a Yahoo crypto pair."""
    return normalize_ticker(ticker).endswith("-USD")


def is_forex_ticker(ticker: str) -> bool:
    """Return whether the symbol looks like a Yahoo forex pair."""
    return normalize_ticker(ticker).endswith("=X")


def is_futures_ticker(ticker: str) -> bool:
    """Return whether the symbol looks like a Yahoo futures contract."""
    return normalize_ticker(ticker).endswith("=F")


def classify_ticker_asset_class(ticker: str) -> str:
    """Map one ticker into the peer universe used for relative-strength ranking."""
    normalized_ticker = normalize_ticker(ticker)
    forced_asset_class = os.environ.get("RELATIVE_STRENGTH_FORCE_ASSET_CLASS", "").strip().lower()
    if forced_asset_class:
        return forced_asset_class

    if is_crypto_ticker(normalized_ticker):
        return "crypto"
    if is_forex_ticker(normalized_ticker):
        return "forex"
    if normalized_ticker in COMMODITY_ETF_HINTS:
        return "commodity_etf"
    if is_futures_ticker(normalized_ticker):
        if normalized_ticker in EQUITY_INDEX_FUTURE_HINTS:
            return "equity_index_future"
        if normalized_ticker in COMMODITY_FUTURE_HINTS:
            return "commodity_future"
        if normalized_ticker in RATE_FUTURE_HINTS:
            return "rate_future"
        return "generic_future"
    if normalized_ticker in US_EQUITY_ETF_HINTS or (normalized_ticker.startswith("XL") and len(normalized_ticker) <= 4):
        return "us_equity_etf"

    return "us_equity_stock"


def deduplicate_tickers(tickers: list[str]) -> list[str]:
    """Preserve order while removing duplicates and blank values."""
    unique_tickers: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        normalized_ticker = normalize_ticker(ticker)
        if not normalized_ticker or normalized_ticker in seen:
            continue
        seen.add(normalized_ticker)
        unique_tickers.append(normalized_ticker)
    return unique_tickers


def resolve_relative_strength_setup(anchor_ticker: str) -> dict[str, object]:
    """Choose a realistic peer universe for the anchor ticker.

    If RELATIVE_STRENGTH_UNIVERSE is populated, it acts as a manual override.
    Otherwise the function selects a peer set from the ticker's asset class.
    """
    normalized_anchor = normalize_ticker(anchor_ticker)
    manual_override = deduplicate_tickers(
        _env_list("RELATIVE_STRENGTH_UNIVERSE", "")
    )

    if manual_override:
        universe = manual_override.copy()
        if normalized_anchor not in universe:
            universe.append(normalized_anchor)
        return {
            "anchor_ticker": normalized_anchor,
            "asset_class": "manual_override",
            "asset_class_label": ASSET_CLASS_DISPLAY_NAMES["manual_override"],
            "universe": deduplicate_tickers(universe),
            "selection_source": "manual_override",
            "selection_reason": (
                "Using the explicit RELATIVE_STRENGTH_UNIVERSE override supplied by the user."
            ),
            "fallback_rule": FALLBACK_RULE_DESCRIPTION,
        }

    asset_class = classify_ticker_asset_class(normalized_anchor)
    base_universe = ASSET_CLASS_UNIVERSES.get(asset_class, ASSET_CLASS_UNIVERSES["us_equity_stock"]).copy()
    include_anchor = os.environ.get("RELATIVE_STRENGTH_INCLUDE_ANCHOR", "0") == "1"
    anchor_removed_for_independence = False
    if normalized_anchor not in base_universe:
        base_universe.append(normalized_anchor)
    if not include_anchor and normalized_anchor in base_universe and len(base_universe) > 2:
        base_universe = [ticker for ticker in base_universe if ticker != normalized_anchor]
        anchor_removed_for_independence = True

    return {
        "anchor_ticker": normalized_anchor,
        "asset_class": asset_class,
        "asset_class_label": ASSET_CLASS_DISPLAY_NAMES.get(asset_class, asset_class.replace("_", " ").title()),
        "universe": deduplicate_tickers(base_universe),
        "selection_source": "asset_class_default",
        "selection_reason": (
            f"Using the {ASSET_CLASS_DISPLAY_NAMES.get(asset_class, asset_class)} peer universe so "
            "relative-strength ranking stays within a sensible asset class."
            + (
                " The anchor ticker is excluded by default to avoid self-referential ranking "
                "and to keep cross-anchor validations more independent."
                if anchor_removed_for_independence
                else ""
            )
        ),
        "fallback_rule": FALLBACK_RULE_DESCRIPTION,
    }
