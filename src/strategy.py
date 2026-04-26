"""Rule-based scoring engine for trade signal generation.

Produces directional signals from multi-timeframe indicator alignment.
Claude is called only for borderline scores (5-6) as commentary.
"""

import logging


def market_filter(asset_data: dict) -> tuple[bool, str]:
    """
    Pre-trade filters that block any signal regardless of score.
    Returns (pass: bool, reason: str).
    """
    lt = asset_data.get("long_term_4h", {})
    current_price = float(asset_data.get("current_price") or 0)

    # ATR spike filter — skip if current ATR is much higher than typical
    atr14 = lt.get("atr14")
    if atr14 and current_price > 0:
        atr_pct = float(atr14) / current_price * 100
        if atr_pct > 5.0:
            return False, f"ATR spike {atr_pct:.2f}% of price — too volatile"

    # Spread filter — skip if bid/ask spread is too wide
    spread_pct = asset_data.get("spread_pct", 0)
    if spread_pct and float(spread_pct) > 0.15:
        return False, f"spread {float(spread_pct):.3f}% too wide"

    return True, ""


def score_setup(asset_data: dict) -> tuple[int, str]:
    """
    Score the setup 0-10 and return (score, direction).
    direction is 'buy', 'sell', or 'hold'.

    Weights:
      4h EMA trend: +3  |  1h EMA trend: +2  |  4h MACD: +2
      1h RSI:       +2  |  Funding tilt: +1
    """
    buy_votes = 0
    sell_votes = 0

    trend_4h = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h = asset_data.get("trend_1h", "UNKNOWN")
    momentum_4h = asset_data.get("momentum_4h", "NEUTRAL")
    intraday = asset_data.get("intraday_1h", {})

    # 4h EMA trend — highest weight (3 pts)
    if trend_4h == "BULLISH":
        buy_votes += 3
    elif trend_4h == "BEARISH":
        sell_votes += 3

    # 1h EMA trend (2 pts)
    if trend_1h == "BULLISH":
        buy_votes += 2
    elif trend_1h == "BEARISH":
        sell_votes += 2

    # 4h MACD histogram (2 pts)
    if momentum_4h == "BULLISH":
        buy_votes += 2
    elif momentum_4h == "BEARISH":
        sell_votes += 2

    # 1h RSI direction (2 pts)
    rsi14_1h = intraday.get("rsi14")
    if rsi14_1h is not None:
        rsi_val = float(rsi14_1h)
        if rsi_val > 55:
            buy_votes += 2
        elif rsi_val < 45:
            sell_votes += 2
        elif rsi_val > 50:
            buy_votes += 1
        elif rsi_val < 50:
            sell_votes += 1

    # Funding rate tilt (1 pt)
    funding = asset_data.get("funding_rate")
    if funding is not None:
        funding_val = float(funding)
        if funding_val < -0.0001:
            buy_votes += 1
        elif funding_val > 0.0001:
            sell_votes += 1

    if buy_votes > sell_votes:
        return min(buy_votes, 10), "buy"
    if sell_votes > buy_votes:
        return min(sell_votes, 10), "sell"
    return 0, "hold"


def entry_confirmed(asset_data: dict, direction: str) -> bool:
    """
    Returns True if 15m + 5m confirm the 1h direction.
    If data is missing, returns True (do not block on missing data).
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})

    if not s15 or not t5:
        return True  # missing data → do not block

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = s15.get("near_ema", True)

    macd_5m  = float(t5.get("macd_histogram") or 0)
    bull_5m  = t5.get("candle_bullish", True)

    if direction == "buy":
        # 15m: price near EMA20 (pullback) AND MACD recovering
        # 5m: bullish candle OR positive histogram (trigger)
        setup_ok   = near_ema and macd_15m > -50
        trigger_ok = bull_5m or macd_5m > 0
        return setup_ok and trigger_ok

    if direction == "sell":
        # Mirror logic for shorts
        setup_ok   = near_ema and macd_15m < 50
        trigger_ok = (not bull_5m) or macd_5m < 0
        return setup_ok and trigger_ok

    return True


def make_decision(asset: str, asset_data: dict) -> dict:
    """
    Primary rule-based decision function.
    Returns dict with: direction, score, filter_reason.
    """
    from src.config_loader import CONFIG

    filter_pass, filter_reason = market_filter(asset_data)
    if not filter_pass:
        logging.info("[FILTER] %s blocked — %s", asset, filter_reason)
        return {"direction": "hold", "score": 0, "filter_reason": filter_reason}

    score, direction = score_setup(asset_data)
    logging.info("[SCORE] %s score=%d direction=%s", asset, score, direction)

    min_score = int(CONFIG.get("min_trade_score") or 7)
    if score < min_score:
        logging.info("[SCORE] %s score=%d below min=%d — hold", asset, score, min_score)
        return {"direction": "hold", "score": score, "filter_reason": f"score {score} < {min_score}"}

    if direction != "hold":
        if not entry_confirmed(asset_data, direction):
            logging.info(
                "[ENTRY] %s direction=%s blocked — "
                "15m/5m not confirmed, waiting for pullback",
                asset, direction
            )
            direction = "hold"

    return {"direction": direction, "score": score, "filter_reason": ""}
