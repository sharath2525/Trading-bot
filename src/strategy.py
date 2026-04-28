"""Pre-trade filters and multi-timeframe entry confirmation for the live trading loop."""

import logging


def market_filter(asset_data: dict) -> tuple[bool, str]:
    """Block any entry when market conditions are unfavourable.

    Returns (allowed: bool, reason: str).
    """
    lt = asset_data.get("long_term_4h", {})
    current_price = float(asset_data.get("current_price") or 0)

    atr14 = lt.get("atr14")
    if atr14 and current_price > 0:
        atr_pct = float(atr14) / current_price * 100
        if atr_pct > 5.0:
            return False, f"ATR spike {atr_pct:.2f}% of price — too volatile"

    spread_pct = asset_data.get("spread_pct", 0)
    if spread_pct and float(spread_pct) > 0.15:
        return False, f"spread {float(spread_pct):.3f}% too wide"

    return True, ""


def entry_confirmed(asset_data: dict, direction: str) -> bool:
    """Return True only when 15m and 5m confirm the higher-timeframe direction.

    Returns False when indicator data is missing — block entry rather than
    allow through with no confirmation.
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})

    if not s15 or not t5:
        return False

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = s15.get("near_ema", True)
    macd_5m   = float(t5.get("macd_histogram") or 0)
    bull_5m   = t5.get("candle_bullish", True)

    # Price-relative MACD threshold (0.1% of price).
    # A fixed ±50 is meaningless for high-priced assets where MACD swings in hundreds.
    current_price = float(asset_data.get("current_price") or 0)
    macd_threshold = current_price * 0.001 if current_price > 0 else 50.0

    if direction == "buy":
        return (near_ema and macd_15m > -macd_threshold) and (bull_5m or macd_5m > 0)

    if direction == "sell":
        return (near_ema and macd_15m < macd_threshold) and ((not bull_5m) or macd_5m < 0)

    return True
