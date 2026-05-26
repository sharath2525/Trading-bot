"""Pre-trade filters and multi-timeframe entry confirmation for the live trading loop."""

import logging
from datetime import datetime, timezone
from src.config_loader import CONFIG


def market_filter(asset_data: dict, btc_trend_1h: str = "UNKNOWN", btc_candles_5m: list = None, direction: str = "") -> tuple[bool, str]:
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

    # ── Time-of-day gate ──────────────────────────────────────────────────────
    # Block 00:00–05:59 UTC (low liquidity, bad fills, fake moves)
    # Also block weekend midnight-to-6am (Friday 22:00 UTC → Monday 06:00 UTC)
    _now_utc = datetime.now(timezone.utc)
    _utc_hour = _now_utc.hour
    _utc_weekday = _now_utc.weekday()  # 0=Mon, 5=Sat, 6=Sun
    if 0 <= _utc_hour <= 5:
        return False, f"time gate — UTC {_utc_hour:02d}:xx blocked (00:00–06:00 UTC)"
    # Outside prime session hours (08:00–20:00 UTC): require score ≥7.5 to trade.
    # Extended session window to 20:00 UTC — captures NY close session (high volume).
    # Threshold lowered from 8.5 → 7.5: 8.5 required near-perfect alignment and blocked
    # ~15 of 24 trading hours almost completely. 7.5 still requires 4h+1h trend + MACD OR
    # near_ema, which is a genuine setup. Low-liquidity filter still active via volume gate.
    _in_session = 8 <= _utc_hour < 20
    _score_context = asset_data.get("_current_score")  # injected by main.py when available
    if not _in_session and _score_context is not None and float(_score_context) < 7.5:
        return False, (f"session gate — UTC {_utc_hour:02d}:xx is outside 08:00–20:00 "
                       f"window; score {float(_score_context):.1f} < 7.5 required off-session")
    # Block Friday 20:00 UTC through Sunday 08:00 UTC
    # Friday=4, Saturday=5, Sunday=6
    _is_weekend_block = (
        (_utc_weekday == 4 and _utc_hour >= 20) or   # Friday from 20:00 UTC
        (_utc_weekday == 5) or                         # All of Saturday
        (_utc_weekday == 6 and _utc_hour < 8)          # Sunday until 08:00 UTC
    )
    if _is_weekend_block:
        return False, f"weekend gate — Fri 20:00→Sun 08:00 UTC (low liquidity window)"

    # ── BTC correlation filter ─────────────────────────────────────────────────
    # Block ETH/SOL/AVAX BUY when BTC 1h trend is BEARISH
    # Also check BTC 5m momentum: if last 3 candles all red on BTC, block altcoin BUY
    _asset_name = asset_data.get("asset", "")
    _correlated = _asset_name in ("ETH", "SOL", "AVAX")
    if _correlated and direction == "buy":
        if btc_trend_1h == "BEARISH":
            return False, f"{_asset_name} BUY blocked — BTC 1h is BEARISH (correlation filter)"
        # BTC 5m momentum check — if last 3 BTC candles are bearish, block altcoin BUY
        _btc_5m = btc_candles_5m or []
        if len(_btc_5m) >= 3:
            _last3 = _btc_5m[-3:]
            _all_red = all(
                float(c.get("close", 0)) < float(c.get("open", 1))
                for c in _last3
            )
            if _all_red:
                return False, f"{_asset_name} BUY blocked — BTC 5m momentum bearish (3 red candles)"

    # ── S&R zone check ────────────────────────────────────────────────────────
    # Block entries near known support/resistance levels:
    # 1. Round numbers ($100k/$50k/$10k for BTC-class, $5k/$1k for ETH, $100/$50 for altcoins)
    # 2. Previous day high/low (PDH/PDL) from 1h candle data
    # 3. Swing highs/lows from last 50 candles on 1h timeframe
    if current_price > 0:
        _sr_blocked = False
        _sr_reason = ""

        # Round number check
        _round_levels = []
        if current_price > 10000:
            _round_levels = [round(current_price / 5000) * 5000,
                             round(current_price / 10000) * 10000,
                             round(current_price / 50000) * 50000]
        elif current_price > 1000:
            _round_levels = [round(current_price / 500) * 500,
                             round(current_price / 1000) * 1000]
        elif current_price > 50:
            _round_levels = [round(current_price / 50) * 50,
                             round(current_price / 100) * 100]

        for _level in _round_levels:
            if _level > 0:
                _dist_pct = abs(current_price - _level) / current_price * 100
                if _dist_pct < 0.5:
                    _sr_blocked = True
                    _sr_reason = (f"price ${current_price:.2f} within 0.5% of round S&R ${_level:.0f} "
                                  f"(dist={_dist_pct:.3f}%)")
                    break

        # Swing high/low from 1h candles (last 50 candles)
        if not _sr_blocked:
            _candles_1h = asset_data.get("candles_1h", [])
            if len(_candles_1h) >= 10:
                _recent_1h = _candles_1h[-50:]
                _highs = [float(c.get("high", 0)) for c in _recent_1h if c.get("high")]
                _lows  = [float(c.get("low", 0)) for c in _recent_1h if c.get("low")]
                if _highs and _lows:
                    _swing_high = max(_highs)
                    _swing_low  = min(_lows)
                    _dist_high = abs(current_price - _swing_high) / current_price * 100
                    _dist_low  = abs(current_price - _swing_low)  / current_price * 100
                    # Only block BUY approaching swing high from BELOW (not confirmed breakout)
                    if direction == "buy" and _dist_high < 0.3 and current_price <= _swing_high:
                        _sr_blocked = True
                        _sr_reason = f"BUY within 0.3% of 1h 50-candle swing high ${_swing_high:.2f}"
                    # Only block SELL when AT the swing low (very tight, not approaching from far above).
                    # 0.15% threshold — avoids blocking every continuation sell in a downtrend where
                    # the 50-candle low is always near the current price.
                    elif direction == "sell" and _dist_low < 0.15 and current_price >= _swing_low:
                        _sr_blocked = True
                        _sr_reason = f"SELL within 0.3% of 1h 50-candle swing low ${_swing_low:.2f}"

                    # PDH/PDL: last 24 1h candles approximate yesterday's range
                    if not _sr_blocked and len(_recent_1h) >= 24:
                        _pdh = max(float(c.get("high", 0)) for c in _recent_1h[-24:])
                        _pdl = min(float(c.get("low", 0))  for c in _recent_1h[-24:])
                        # Only block BUY near PDH when price hasn't cleared it yet
                        if _pdh > 0 and abs(current_price - _pdh) / current_price * 100 < 0.2 and current_price <= _pdh:
                            _sr_blocked = True
                            _sr_reason = f"price within 0.2% of PDH ${_pdh:.2f}"
                        # Only block SELL when exactly at PDL (0.1% — avoids blocking every downtrend continuation)
                        elif _pdl > 0 and abs(current_price - _pdl) / current_price * 100 < 0.1 and current_price >= _pdl:
                            _sr_blocked = True
                            _sr_reason = f"price within 0.2% of PDL ${_pdl:.2f}"

        # Swing high/low from 4h candles (last 50 candles) — catches larger structural S&R
        if not _sr_blocked:
            _candles_4h = asset_data.get("candles_4h", [])
            if len(_candles_4h) >= 10:
                _recent_4h = _candles_4h[-50:]
                _highs_4h = [float(c.get("high", 0)) for c in _recent_4h if c.get("high")]
                _lows_4h  = [float(c.get("low", 0)) for c in _recent_4h if c.get("low")]
                if _highs_4h and _lows_4h:
                    _swing_high_4h = max(_highs_4h)
                    _swing_low_4h  = min(_lows_4h)
                    # Only block when approaching resistance from below / support from above
                    if direction == "buy" and abs(current_price - _swing_high_4h) / current_price * 100 < 0.3 and current_price <= _swing_high_4h:
                        _sr_blocked = True
                        _sr_reason = f"BUY within 0.3% of 4h 50-candle swing high ${_swing_high_4h:.2f}"
                    elif direction == "sell" and abs(current_price - _swing_low_4h) / current_price * 100 < 0.15 and current_price >= _swing_low_4h:
                        _sr_blocked = True
                        _sr_reason = f"SELL within 0.3% of 4h 50-candle swing low ${_swing_low_4h:.2f}"

        if _sr_blocked:
            return False, f"S&R gate — {_sr_reason}"

    # ── Funding rate hard gate ─────────────────────────────────────────────────
    # Extreme funding = crowded positioning = reversal risk
    # This makes funding a CODE gate, not just Claude context
    _funding = float(asset_data.get("funding_rate") or 0)
    if direction == "buy" and _funding > 0.0005:
        return False, f"funding gate — {_asset_name} rate {_funding:.5f}/8h too high for BUY (crowded longs)"
    if direction == "sell" and _funding < -0.0005:
        return False, f"funding gate — {_asset_name} rate {_funding:.5f}/8h too negative for SELL (crowded shorts)"

    return True, ""


def _compute_signal_score(asset_data: dict, direction: str) -> int:
    """Return an integer score 0–5 counting how many entry conditions are met.

    Each of five conditions contributes 1 point. MIN_TRADE_SCORE sets the
    minimum number of conditions that must pass before entry is allowed.
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})
    current_price = float(asset_data.get("current_price") or 0)
    macd_threshold = current_price * 0.001 if current_price > 0 else 0.0

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = bool(s15.get("near_ema", False))
    macd_5m   = float(t5.get("macd_histogram") or 0)
    bull_5m   = bool(t5.get("candle_bullish", False))
    trend_4h  = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h  = asset_data.get("trend_1h", "UNKNOWN")

    score = 0
    if direction == "buy":
        if trend_4h == "BULLISH":           score += 1
        if trend_1h == "BULLISH":           score += 1
        if macd_15m > macd_threshold:       score += 1
        if near_ema:                        score += 1
        if bull_5m or macd_5m > 0:         score += 1
    elif direction == "sell":
        if trend_4h == "BEARISH":           score += 1
        if trend_1h == "BEARISH":           score += 1
        if macd_15m < -macd_threshold:      score += 1
        if near_ema:                        score += 1
        if (not bull_5m) or macd_5m < 0:   score += 1
    return score


def compute_signal_score(asset_data: dict, direction: str) -> float:
    """Return a weighted float score 0–11 for the pre-gate in main.py.

    Weights: trend_4h=3, trend_1h=2, MACD_15m=2, near_ema=1.5, trigger_5m=1.5.
    Base reachable values: 0, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 10.
    Score 9 base is mathematically unreachable. Bonuses can push score above 10 (cap 11.0).
    MIN_SIGNAL_SCORE (default 6) is the execution threshold in main.py.
    Do NOT call this from entry_confirmed() — that uses _compute_signal_score() (0–5 system).
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})
    current_price = float(asset_data.get("current_price") or 0)
    macd_threshold = current_price * 0.001 if current_price > 0 else 0.0

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = bool(s15.get("near_ema", False))
    macd_5m   = float(t5.get("macd_histogram") or 0)
    bull_5m   = bool(t5.get("candle_bullish", False))
    trend_4h  = asset_data.get("trend_4h", "UNKNOWN")
    trend_1h  = asset_data.get("trend_1h", "UNKNOWN")

    score = 0.0
    if direction == "buy":
        if trend_4h == "BULLISH":           score += 3.0
        if trend_1h == "BULLISH":           score += 2.0
        if macd_15m > macd_threshold:       score += 2.0
        if near_ema:                        score += 1.5
        if bull_5m or macd_5m > 0:         score += 1.5
    elif direction == "sell":
        if trend_4h == "BEARISH":           score += 3.0
        if trend_1h == "BEARISH":           score += 2.0
        if macd_15m < -macd_threshold:      score += 2.0
        if near_ema:                        score += 1.5
        if (not bull_5m) or macd_5m < 0:   score += 1.5

    # ── Volume bonus (+1.0 when vol ≥1.5× 5-period avg) ──────────────────────
    # BUG-v8-L3 FIX: Use last CLOSED candle (_c5m[-2]) not the live forming candle (_c5m[-1]).
    # The forming candle accumulates volume from 0; near candle-open it is always below average,
    # and near candle-close it can spike for unrelated reasons. Using [-2] (the last completed
    # candle) gives a stable, reliable volume reading.
    _c5m = asset_data.get("candles_5m", [])
    if len(_c5m) >= 7:  # need at least 7: [-7:-2] = 5 avg candles, [-2] = last closed
        _vols = [c.get("volume", 0) for c in _c5m[-7:-2]]
        _avg_vol = sum(_vols) / len(_vols) if _vols else 0
        _cur_vol = _c5m[-2].get("volume", 0)  # last closed candle
        if _avg_vol > 0 and _cur_vol >= _avg_vol * 1.5:
            score += 1.0

    # ── Candle pattern bonus (+0.5 for engulfing or hammer/pin bar) ───────────
    if len(_c5m) >= 2:
        _c = _c5m[-1]
        _cp = _c5m[-2]
        try:
            _o = float(_c.get("open", 0))
            _h = float(_c.get("high", 0))
            _l = float(_c.get("low", 0))
            _cl = float(_c.get("close", 0))
            _body = abs(_cl - _o)
            _wick = _h - _l if _h > _l else 0
            _prev_cl = float(_cp.get("close", 0))
            if direction == "buy" and _o > 0 and _wick > 0:
                # Bullish engulfing: green candle, body ≥60% of range, close > prev close
                if _cl > _o and _body >= _wick * 0.6 and _cl > _prev_cl:
                    score += 0.5
                # Hammer/pin bar: green candle, upper wick small, lower wick > body
                elif _cl > _o and (_h - _cl) < _body * 0.3 and (_o - _l) > _body:
                    score += 0.5
            elif direction == "sell" and _o > 0 and _wick > 0:
                # Bearish engulfing: red candle, body ≥60% of range
                if _cl < _o and _body >= _wick * 0.6:
                    score += 0.5
                # Bearish pin bar: red candle, lower wick small, upper wick > body
                elif _cl < _o and (_cl - _l) < _body * 0.3 and (_h - _o) > _body:
                    score += 0.5
        except (TypeError, ValueError):
            pass

    return min(11.0, score)


def entry_confirmed(asset_data: dict, direction: str) -> bool:
    """Return True only when 15m and 5m confirm the higher-timeframe direction.

    Returns False when indicator data is missing — block entry rather than
    allow through with no confirmation.
    """
    s15 = asset_data.get("setup_15m", {})
    t5  = asset_data.get("trigger_5m", {})

    if not s15 or not t5:
        return False

    # Signal score gate — requires minimum aligned conditions before entry.
    # Default fallback is 2 (lowered from 3): MIN_TRADE_SCORE=2 in .env means
    # only 2 of the 5 inner conditions need to pass (e.g. trend_4h + trend_1h).
    _score = _compute_signal_score(asset_data, direction)
    _min_score = int(CONFIG.get("min_trade_score") or 2)
    if _score < _min_score:
        logging.info(
            "[SCORE] %s %s blocked — score %d < min %d",
            asset_data.get("asset", "?"), direction, _score, _min_score,
        )
        return False

    # RSI gate — block extreme overbought/oversold only.
    # Threshold widened from 70/30 → 78/22:
    # In a genuine strong trend, 15m RSI routinely stays 65-80 (BUY) or 20-35 (SELL).
    # The old 70/30 threshold was systematically blocking trend-continuation entries
    # during the exact conditions this strategy targets. 78/22 only blocks truly extreme
    # levels where reversal risk is highest (parabolic moves, exhaustion spikes).
    rsi_15m = s15.get("rsi14")
    if rsi_15m is not None:
        if direction == "buy" and float(rsi_15m) > 78:
            logging.info("buy blocked — 15m RSI %.1f extreme overbought (>78)", float(rsi_15m))
            return False
        if direction == "sell" and float(rsi_15m) < 22:
            logging.info("sell blocked — 15m RSI %.1f extreme oversold (<22)", float(rsi_15m))
            return False

    macd_15m = float(s15.get("macd_histogram") or 0)
    near_ema  = bool(s15.get("near_ema", False))  # E-1 FIX: default False — missing data blocks entry
    macd_5m   = float(t5.get("macd_histogram") or 0)
    bull_5m   = t5.get("candle_bullish", False)  # V3-MEDIUM-1 FIX: default False — missing data blocks entry

    # Price-relative MACD threshold (0.1% of price).
    # A fixed ±50 is meaningless for high-priced assets where MACD swings in hundreds.
    current_price = float(asset_data.get("current_price") or 0)
    macd_threshold = current_price * 0.001 if current_price > 0 else 50.0

    # Volume confirmation: trigger candle must have at least 70% of the recent
    # average volume. This filters dead/low-liquidity candles that produce fake
    # MACD crossovers without real buying/selling pressure behind them.
    # Candle dicts use the key "volume" (mapped from raw Hyperliquid "v" field).
    candles_5m = asset_data.get("candles_5m", [])
    if len(candles_5m) >= 5:
        recent_vols = [c.get("volume", 0) for c in candles_5m[-21:-1]]
        avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 0
        trigger_vol = candles_5m[-1].get("volume", 0)
        # Volume threshold lowered from 1.2× → 0.7×.
        # 1.2× required a SURGE to confirm — this killed pullback entries (which are the
        # highest R:R setups) because volume on pullback candles is by definition lower.
        # 0.7× means at least 70% of average volume — filters dead/zero-volume candles
        # while allowing normal consolidation candles to pass.
        # Zero avg_vol still fails closed (data feed issue).
        vol_ok = trigger_vol >= avg_vol * 0.7 if avg_vol > 0 else False
        if not vol_ok:
            logging.info(
                "Entry rejected: very low volume on 5m trigger (%.0f vs avg %.0f, need 0.7×)",
                trigger_vol, avg_vol,
            )
    else:
        vol_ok = False  # insufficient candle history — block entry rather than allow with no volume data

    # NOTE: Stale setup check (>0.5×ATR from EMA) removed.
    # In a genuine trend, price IS extended from the 15m EMA — that is the definition of
    # a trending move. The stale check was systematically killing trend-continuation entries,
    # which are the core of this strategy. The 15m MACD and near_ema signal in the score
    # already penalise setups where price is too far from structure.

    # BUY: use OR between near_ema and macd_15m.
    # In a strong trend, price extends away from EMA (near_ema=False) but MACD is positive.
    # Old AND logic required BOTH — made every extended-trend entry impossible.
    if direction == "buy":
        return vol_ok and (near_ema or macd_15m > macd_threshold) and (bull_5m or macd_5m > 0)

    # SELL: use OR between near_ema and macd_15m — in a downtrend price is below EMA so near_ema=False,
    # but MACD is strongly negative. Requiring AND made this mutually exclusive with actual downtrends.
    if direction == "sell":
        return vol_ok and (near_ema or macd_15m < -macd_threshold) and ((not bull_5m) or macd_5m < 0)

    return True


def is_trending_regime(asset_data: dict) -> bool:
    """[INACTIVE — NOT CALLED ANYWHERE] BB width regime detection.

    Status: Dead code retained for potential Tier 2 use.
    The BB width gate was removed from _code_decide_direction() (CHANGE 3, 2026-05-21)
    because ADX ≥ 15 already filters ranging markets, making this redundant.

    To reactivate: call this from _code_decide_direction() after the ADX gate.
    Do NOT reactivate without removing the ADX gate first — they are redundant.
    """
    # DEAD_CODE_MARKER: is_trending_regime — search for this tag to find all inactive functions
    lt = asset_data.get("long_term_4h", {})
    bb_width = lt.get("bb_width_pct")
    bb_width_series = lt.get("bb_width_series", [])

    if bb_width is None or len(bb_width_series) < 10:
        return True  # insufficient data — don't block

    try:
        _values = sorted([float(v) for v in bb_width_series if v is not None])
        _median = _values[len(_values) // 2]
        _current = float(bb_width)
        _is_trending = _current >= _median
        if not _is_trending:
            logging.info(
                "[BB] ranging market detected — BB width %.3f%% below 20-period median %.3f%%",
                _current, _median
            )
            print(f"[BB] {asset_data.get('asset', '?')} ranging — BB width {_current:.3f}% below median {_median:.3f}%")
        return _is_trending
    except (TypeError, ValueError):
        return True  # data error — don't block


def oi_confirmed(asset_data: dict, direction: str) -> tuple[bool, str]:
    """Return (True, "") if OI confirms the trade direction, (False, reason) if not.

    Rules:
    - OI must be increasing over last 2 periods (oi_series last 2 values increasing)
    - OI spike >5% in single period → HOLD (manipulation/liquidation cascade risk)
    - If OI data unavailable → return True (don't block when data missing)
    """
    oi = asset_data.get("open_interest")
    oi_series = asset_data.get("oi_series", [])

    if not oi or len(oi_series) < 2:
        return True, ""  # no data — don't block

    try:
        prev_oi = float(oi_series[-2])
        curr_oi = float(oi_series[-1])

        if prev_oi > 0:
            oi_change_pct = (curr_oi - prev_oi) / prev_oi * 100

            # OI spike >5% in one period — abnormal, block entry
            if abs(oi_change_pct) > 5.0:
                return False, f"OI spike {oi_change_pct:.1f}% in last period — manipulation risk"

            # OI must be increasing to confirm real money entering
            if curr_oi < prev_oi * 0.995:
                return False, f"OI not increasing ({curr_oi:.0f} <= {prev_oi:.0f}) — no new money confirming move"
    except (TypeError, ValueError):
        return True, ""

    return True, ""
