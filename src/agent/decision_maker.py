"""Trading agent — confirm_trade() is the only live method.

CODE-FIRST HYBRID: all direction/TP/SL/size decisions are made by code in main.py.
Claude's role is strictly APPROVE or REJECT on score-10 setups via confirm_trade().
MASTER RULE 2: code decides direction. MASTER RULE 3: Claude role is fixed.

DEAD CODE NOTE: decide_trade() and _decide() were removed (2026-05-10).
They implemented the pre-2026-04-30 Claude-as-decision-engine architecture and
violated MASTER RULE 2. confirm_trade() below is the only active method.
"""

import anthropic
from src.config_loader import CONFIG
import logging
from datetime import datetime


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
        _haiku = "claude-haiku-4-5-20251001"
        _max_tok = int(CONFIG.get("ai_max_tokens") or 4000)
        _now_utc = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

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
            "You are a professional trading risk analyst and market environment validator.\n\n"
            "The code has already confirmed:\n"
            f"  Multi-timeframe alignment across all timeframes (above)\n"
            f"  Direction, entry, TP, SL — all set by code. You CANNOT change these.\n\n"
            "YOUR ONLY JOB: assess whether this is a REAL, HIGH-QUALITY setup or FALSE/RISKY.\n\n"
            "Analyze each category:\n\n"
            "BREAKOUT VALIDITY:\n"
            "- Are all timeframes genuinely aligned or is one borderline?\n"
            "- Is MACD acceleration consistent across timeframes (not just one TF)?\n"
            "- Is RSI at a level that supports further move (not already overbought)?\n"
            "- Does price structure suggest genuine breakout or potential fake-out?\n\n"
            "MACRO & NEWS RISK:\n"
            "- Any high-impact scheduled events in next 4-12 hours?\n"
            "  (FOMC, CPI, NFP, ECB, PCE, PPI, GDP, PMI, earnings, options expiry)\n"
            "- Any active geopolitical shocks in headlines?\n"
            "- Does current UTC time put entry near a high-risk window?\n"
            "  (US open 13:30 UTC, US close 20:00 UTC, Asia open 23:00 UTC)\n\n"
            "VOLATILITY RISK:\n"
            "- Is ATR in normal range or spike regime?\n"
            "- Is Bollinger Band width healthy for this entry type?\n"
            "- Is spread normal?\n\n"
            "POSITIONING RISK:\n"
            "- Is funding rate at extreme suggesting crowded positioning (>0.05% per 8h)?\n"
            "- Is open interest confirming trend or warning of reversal?\n\n"
            "End your response with exactly one of:\n"
            "VERDICT: APPROVE\n"
            "VERDICT: REJECT\n\n"
            "Lean toward REJECT when uncertain. A missed trade is better than a trapped position."
        )

        _user = "\n\n".join([_setup, _tf, _vol, _pos, _macro, _instr])
        _system = (
            "You are a professional trading risk analyst. "
            "Analyze market conditions thoroughly, then end with VERDICT: APPROVE or VERDICT: REJECT."
        )

        try:
            resp = self.client.messages.create(
                model=_haiku,
                max_tokens=_max_tok,
                system=_system,
                messages=[{"role": "user", "content": _user}],
                timeout=30.0,
            )
            answer = resp.content[0].text.strip() if resp.content else ""
            verdict = "APPROVE" if "VERDICT: APPROVE" in answer.upper() else "REJECT"

            input_tokens  = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
            cost_usd = (input_tokens * 0.0000008) + (output_tokens * 0.000004)

            logging.info("[CONFIRM] %s score=%.1f direction=%s verdict=%s cost=$%.5f",
                         asset, score, direction, verdict, cost_usd)

            with open("llm_requests.log", "a", encoding="utf-8") as _lf:
                _lf.write(
                    f"\n=== MARKET ANALYSIS {asset} score={score:.1f} {_now_utc} ===\n"
                    f"direction={direction} verdict={verdict}\n"
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
