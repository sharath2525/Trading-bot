"""Trading agent — confirm_trade() is the only live method.

CODE-FIRST HYBRID: all direction/TP/SL/size decisions are made by code in main.py.
Claude's role is strictly APPROVE or REJECT on score-10 setups via confirm_trade().
MASTER RULE 2: code decides direction. MASTER RULE 3: Claude role is fixed.

DEAD CODE NOTE: decide_trade() and _decide() were removed (2026-05-10).
They implemented the pre-2026-04-30 Claude-as-decision-engine architecture and
violated MASTER RULE 2. confirm_trade() below is the only active method.
"""

import re
import anthropic
from src.config_loader import CONFIG
import logging
from datetime import datetime, timezone


class TradingAgent:
    """Wrapper for the Claude APPROVE/REJECT gate. confirm_trade() is the only live method."""

    def __init__(self, hyperliquid=None):
        self.client = anthropic.Anthropic(api_key=CONFIG["anthropic_api_key"])
        self.hyperliquid = hyperliquid
        logging.info("TradingAgent initialized — confirm_trade() gate only (code-first hybrid)")

    def confirm_trade(self, asset: str, direction: str, entry_price: float,
                      tp_price: float, sl_price: float,
                      score: float, indicators: dict,
                      macro_context: dict | None = None,
                      asset_data: dict | None = None) -> str:
        """Call Claude for deep market analysis on a confluence-confirmed setup.

        Called when score >= MIN_AI_SCORE and all timeframes are aligned.
        Returns 'APPROVE' or 'REJECT'. Fails closed on any error.
        max_tokens from AI_MAX_TOKENS config (default 4000).
        Parses for 'VERDICT: APPROVE' anywhere in response — anything else → REJECT.
        """
        _model = "claude-sonnet-4-6"
        _max_tok = int(CONFIG.get("ai_max_tokens") or 4000)
        _now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # BUG-v7-S5 FIX: was datetime.now() (local time)

        # Last 5 completed trades for this asset — gives Claude recent performance context
        _recent_trades_ctx = "No recent trade history available."
        try:
            import json as _rj
            import os as _os
            # BUG-P9-8 FIX: Use absolute path so diary.jsonl is found regardless of CWD.
            # Relative path fails silently when launched via systemd or from a different dir.
            _diary_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "diary.jsonl")
            if _os.path.exists(_diary_path):
                with open(_diary_path, "r") as _rf:
                    _all_lines = [_rj.loads(l) for l in _rf if l.strip()]
                _asset_trades = [t for t in _all_lines if t.get("asset") == asset][-5:]
                if _asset_trades:
                    _recent_trades_ctx = "\n".join(
                        f"  {t.get('timestamp','?')[:16]} {t.get('action','?').upper()} "
                        f"entry={t.get('entry_price','?')} result={t.get('pnl_pct','?')}%"
                        for t in _asset_trades
                    )
        except Exception:
            pass

        # ── Hard pre-checks — run BEFORE calling Claude to avoid soft-rule inconsistency ──
        # These mirror two of Claude's auto-reject conditions as hard Python guards.
        # market_filter() already checks these but confirm_trade() receives pre-filtered data;
        # adding them here prevents Claude from applying stricter/inconsistent thresholds.

        # 1. Round S&R check — block if price within 0.5% of a major round level
        _pre_price = float((asset_data or {}).get("current_price") or entry_price or 0)
        if _pre_price > 0:
            _round_levels_pre = []
            if _pre_price > 10000:
                _round_levels_pre = [
                    round(_pre_price / 5000) * 5000,
                    round(_pre_price / 10000) * 10000,
                ]
            elif _pre_price > 1000:
                _round_levels_pre = [round(_pre_price / 500) * 500, round(_pre_price / 1000) * 1000]
            elif _pre_price > 50:
                _round_levels_pre = [round(_pre_price / 50) * 50, round(_pre_price / 100) * 100]
            for _rl in _round_levels_pre:
                if _rl > 0 and abs(_pre_price - _rl) / _pre_price < 0.003:  # 0.3% (was 0.5% — too tight, blocked too many setups)
                    logging.info(
                        "[PRE-REJECT] %s %s — price $%.2f within 0.5%% of round level $%.0f — skipping Claude call",
                        asset, direction, _pre_price, _rl
                    )
                    return "REJECT"

        # 2. Extreme funding pre-check — matches market_filter() threshold exactly
        _pre_funding = float((asset_data or {}).get("funding_rate") or 0)
        if direction == "buy" and _pre_funding > 0.0005:
            logging.info("[PRE-REJECT] %s BUY — funding %.5f > 0.0005 threshold — skipping Claude call", asset, _pre_funding)
            return "REJECT"
        if direction == "sell" and _pre_funding < -0.0005:
            logging.info("[PRE-REJECT] %s SELL — funding %.5f < -0.0005 threshold — skipping Claude call", asset, _pre_funding)
            return "REJECT"
        # ── End hard pre-checks ──────────────────────────────────────────────────

        ad = asset_data or {}
        mc = macro_context or {}

        intra_1h = ad.get("intraday_1h", {})
        lt_4h    = ad.get("long_term_4h", {})
        s30m     = ad.get("setup_30m", {})
        s15m     = ad.get("setup_15m", {})
        t5m      = ad.get("trigger_5m", {})

        _atr14   = lt_4h.get("atr14")
        _atr_pct = round(_atr14 / entry_price * 100, 3) if _atr14 and entry_price > 0 else None
        _funding = ad.get("funding_rate") or 0
        _fund_status = "EXTREME" if abs(_funding) > 0.0005 else "NORMAL"

        _setup = (
            f"TRADE SETUP\n{'='*60}\n"
            f"Asset:         {asset}\n"
            f"Direction:     {direction.upper()}  <- set by code, you cannot change this\n"
            f"Entry price:   {entry_price:,.4f}\n"
            f"Take-profit:   {tp_price:,.4f}\n"
            f"Stop-loss:     {sl_price:,.4f}\n"
            f"Signal score:  {score:.1f} / 10\n"
            f"UTC time:      {_now_utc}"
        )

        _tf = (
            f"TIMEFRAME ALIGNMENT\n{'='*60}\n"
            f"4h  trend:   {ad.get('trend_4h','UNKNOWN')}  "
            f"(EMA20={lt_4h.get('ema20')} vs EMA50={lt_4h.get('ema50')})\n"
            f"    MACD hist (last 3): {lt_4h.get('macd_histogram_series',[])}\n"
            f"    ADX: {lt_4h.get('adx')}  RSI14: {lt_4h.get('rsi14')}\n\n"
            f"1h  trend:   {ad.get('trend_1h','UNKNOWN')}  "
            f"(EMA20={intra_1h.get('ema20')} vs EMA50={intra_1h.get('ema50')})\n"
            f"    MACD hist (last 3): {intra_1h.get('series',{}).get('macd_histogram',[])}\n"
            f"    ADX: {intra_1h.get('adx')}  RSI14: {intra_1h.get('rsi14')}\n\n"
            f"30m EMA20={s30m.get('ema20')} EMA50={s30m.get('ema50')}\n"
            f"    MACD hist: {s30m.get('macd_histogram')}  RSI14: {s30m.get('rsi14')}\n\n"
            f"15m setup:   (near EMA: {s15m.get('near_ema')})\n"
            f"    EMA20={s15m.get('ema20')}  MACD hist={s15m.get('macd_histogram')}  RSI14={s15m.get('rsi14')}\n\n"
            f"5m  trigger: candle_bullish={t5m.get('candle_bullish')}  "
            f"MACD hist={t5m.get('macd_histogram')}  RSI14={t5m.get('rsi14')}\n\n"
            f"NOTE: All timeframes confirmed confluence before this AI call."
        )

        _vol = (
            f"VOLATILITY\n{'='*60}\n"
            f"ATR14 (4h):           {_atr14}  ({_atr_pct}% of price)\n"
            f"Bollinger Band width: {lt_4h.get('bb_width_pct')}%  (normal: 2-5%, high vol: >6%)\n"
            f"Spread:               {ad.get('spread_pct')}%"
        )

        _pos = (
            f"POSITIONING\n{'='*60}\n"
            f"Funding rate:       {_funding} per 8h  ({_fund_status})\n"
            f"Funding annualized: {ad.get('funding_annualized_pct')}%\n"
            f"Open interest:      {ad.get('open_interest')}"
        )

        _events   = mc.get("events", [])
        _headlines = mc.get("headlines", [])
        if _events or _headlines:
            _macro = (
                f"MACRO CONTEXT  [fetched {mc.get('fetched_at','unknown')}]\n{'='*60}\n"
                f"UPCOMING EVENTS:\n"
                + ("\n".join(f"  - {e}" for e in _events) if _events else "  None detected")
                + f"\n\nRECENT HEADLINES:\n"
                + ("\n".join(f"  {h}" for h in _headlines[:10]) if _headlines else "  None fetched")
            )
        else:
            _macro = (
                f"MACRO CONTEXT\n{'='*60}\n"
                "MACRO DATA: UNAVAILABLE — analyze on technicals and timestamp only.\n"
                "Default to conservative — if near a typical high-impact time window, lean REJECT."
            )

        _instr = (
            f"INSTRUCTIONS\n{'='*60}\n"
            "You are a professional trading risk analyst validating a code-confirmed setup.\n"
            "Code has set direction, entry, TP, SL — you CANNOT change any of these.\n\n"
            "═══ STEP 1: AUTO-REJECT CHECK ═══\n"
            "NOTE: Round S&R proximity and extreme funding rate have already been verified by pre-checks.\n"
            "Focus your auto-reject evaluation on: RSI divergence, high-impact news events, and weak candle body only.\n\n"
            "If ANY of the following are true, write VERDICT: REJECT immediately:\n"
            "  - RSI divergence on 4h or 1h (price at new high/low but RSI is NOT)\n"
            "  - High-impact event within 2 hours: FOMC, CPI, NFP, ECB, PCE, GDP, earnings\n"
            "  - Candle body < 30% of total candle range on 15m trigger (indecision/wick-dominated)\n\n"
            "═══ STEP 2: SCORE EACH FACTOR 1–5 ═══\n"
            "(Proceed only if no auto-reject triggered)\n\n"
            "FACTOR 1 — Trend strength (quality of multi-timeframe alignment)\n"
            "  1=TFs conflicting  3=most aligned  5=all TFs aligned + MACD accelerating\n"
            "  Score: __/5\n\n"
            "FACTOR 2 — Entry quality (price vs EMA, S&R, market structure)\n"
            "  1=chasing far from EMA  3=acceptable  5=at key level with clean structure\n"
            "  Score: __/5\n\n"
            "FACTOR 3 — Risk/reward validity (is TP reachable without major resistance?)\n"
            "  1=TP blocked by resistance  3=reasonable path  5=TP in open air + volume\n"
            "  Score: __/5\n\n"
            "FACTOR 4 — Macro/news environment (safety of this trading window)\n"
            "  1=event imminent  3=no major events  5=post-event calm + strong trend\n"
            "  Score: __/5\n\n"
            "FACTOR 5 — Volume/OI confirmation (real money behind this move?)\n"
            "  1=low volume fake-out  3=average volume  5=volume spike + OI expanding\n"
            "  Score: __/5\n\n"
            "═══ STEP 3: RESPOND IN THIS EXACT FORMAT ═══\n"
            "Factor 1: [score]/5 — [one sentence reason]\n"
            "Factor 2: [score]/5 — [one sentence reason]\n"
            "Factor 3: [score]/5 — [one sentence reason]\n"
            "Factor 4: [score]/5 — [one sentence reason]\n"
            "Factor 5: [score]/5 — [one sentence reason]\n"
            "TOTAL: [sum]/25\n"
            "CONFIDENCE: [1-10]\n\n"
            "VERDICT: APPROVE  ← only if TOTAL >= 15 and no auto-reject\n"
            "VERDICT: REJECT   ← if TOTAL < 15 or any auto-reject triggered\n\n"
            "Lean toward REJECT when uncertain. A missed trade beats a trapped position."
        )

        _history = (
            f"RECENT TRADE HISTORY — {asset}\n{'='*60}\n"
            f"{_recent_trades_ctx}"
        )

        _user = "\n\n".join([_setup, _tf, _vol, _pos, _macro, _history, _instr])
        _system = (
            "You are a professional trading risk analyst. "
            "Analyze market conditions thoroughly, then end with VERDICT: APPROVE or VERDICT: REJECT."
        )

        try:
            resp = self.client.messages.create(
                model=_model,
                max_tokens=_max_tok,
                system=_system,
                messages=[{"role": "user", "content": _user}],
                timeout=30.0,
            )
            answer = resp.content[0].text.strip() if resp.content else ""
            verdict = "APPROVE" if "VERDICT: APPROVE" in answer.upper() else "REJECT"

            # P5: Extract CONFIDENCE score from response (format: "CONFIDENCE: X/10" or "CONFIDENCE: X")
            _conf_match = re.search(r"CONFIDENCE:\s*(\d+)(?:/10)?", answer, re.IGNORECASE)
            confidence = int(_conf_match.group(1)) if _conf_match else None

            input_tokens  = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
            cost_usd = (input_tokens * 0.000003) + (output_tokens * 0.000015)

            logging.info("[CONFIRM] %s score=%.1f direction=%s verdict=%s confidence=%s cost=$%.5f",
                         asset, score, direction, verdict, confidence, cost_usd)

            with open("llm_requests.log", "a", encoding="utf-8") as _lf:
                _lf.write(
                    f"\n=== MARKET ANALYSIS {asset} score={score:.1f} {_now_utc} ===\n"
                    f"direction={direction} verdict={verdict} confidence={confidence}/10\n"
                    f"input_tokens={input_tokens} output_tokens={output_tokens} cost=${cost_usd:.5f}\n"
                    f"--- ANALYSIS ---\n{answer}\n"
                    f"{'='*60}\n"
                )
            with open("prompts.log", "a", encoding="utf-8") as _pl:
                _pl.write(
                    f"\n=== CONFIRM PROMPT {asset} {_now_utc} ===\n{_user}\n"
                    f"=== RESPONSE ===\n{answer}\n{'='*60}\n"
                )

            return verdict
        except Exception as _e:
            logging.warning("[CONFIRM] %s error — failing closed: %s", asset, _e)
            return "REJECT"
