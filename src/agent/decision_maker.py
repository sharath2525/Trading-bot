"""Decision-making agent that orchestrates LLM prompts and indicator lookups.

Uses the Anthropic Claude API directly for trade decisions.
"""

import asyncio
import anthropic
from src.config_loader import CONFIG
from src.indicators.local_indicators import compute_all, last_n, latest
import json
import logging
from datetime import datetime


class TradingAgent:
    """High-level trading agent that delegates reasoning to Claude.

    ════════════════════════════════════════════════════════════════════════════════
    TRADING STRATEGY OVERVIEW
    ════════════════════════════════════════════════════════════════════════════════

    This agent implements a QUANTITATIVE, MULTI-ASSET PERPETUAL FUTURES TRADING STRATEGY
    optimized for Hyperliquid. The strategy uses Claude as the decision engine combined
    with local technical analysis to identify high-probability setups.

    CORE APPROACH:
    ─────────────────────────────────────────────────────────────────────────────────
    1. DUAL-TIMEFRAME ANALYSIS
       • 1h (intraday): Entry signals and trend confirmation
       • 4h (swing): Establishes directional bias and structural trend
       • Higher timeframe structures take precedence (avoids counter-trend trades)

    2. TECHNICAL INDICATOR FOUNDATION
       Indicators computed locally from Hyperliquid OHLCV candles:
       • TREND: EMA(20) vs EMA(50) crossover — pre-computed as trend_4h / trend_1h labels
       • MOMENTUM: MACD histogram (not just MACD line), RSI(14)
       • VOLATILITY: ATR(14) for TP/SL sizing (SL = 1×ATR, TP = 3×ATR)

    3. DECISION LOGIC (Claude-driven)
       Claude analyzes multi-dimensional context:
       ✓ Structural alignment (trend, EMA crosses, HH/HL vs LH/LL)
       ✓ Momentum confirmation (MACD histogram, RSI slope, Stochastic RSI)
       ✓ Liquidity & volatility (ATR, volume profile, funding rates)
       ✓ Position management (existing trades, exit plans, cooldowns)
       ✓ Risk-reward validation (expected move > 3× round-trip fees)

       Claude chooses ONE action per asset: BUY | SELL | HOLD
       Each decision includes: allocation, order type, TP/SL targets, exit conditions

    4. POSITION MANAGEMENT (Hysteresis + Cooldowns)
       • HYSTERESIS: Stronger evidence required to CHANGE direction than to KEEP it
         - Requires BOTH higher-timeframe support AND intraday confirmation
         - Avoids whipsaws and reduces slippage/fees
       • COOLDOWN: After any position action, enforce 3+ bars (e.g., 3×1h = 3h)
         before another direction change (unless hard invalidation occurs)
       • EXIT PLANS: Every trade includes explicit invalidation triggers
         (e.g., "close if 4h close above EMA50" or "flip if 1h EMA cross reverses")

    5. FUNDING RATE TILT (Not a trigger, context only)
       • Positive funding (shorts paying longs): Favors longs but isn't a trade signal
       • Negative funding (longs paying shorts): Favors shorts but watch position duration
       • Only triggers action if expected funding accrual > expected edge over horizon

    6. FEE-AWARE TRADING
       • Hyperliquid taker fee: 0.045% per side → ~0.09% round-trip
       • NEVER enter if expected move < 3× fee (~0.27%)
       • Prefer limit orders (0% maker fee) → saves 0.045% per entry
       • Factor fee costs into TP targets (min TP ~0.3% from entry)

    ════════════════════════════════════════════════════════════════════════════════
    RISK MANAGEMENT (Hard guardrails — non-bypassable)
    ════════════════════════════════════════════════════════════════════════════════
    1. MAX_POSITION_PCT: Individual position size capped as % of account
    2. MAX_LEVERAGE: Effective leverage (notional / equity) cannot exceed limit
    3. MAX_TOTAL_EXPOSURE_PCT: Sum of all open positions cannot exceed % of account
    4. DAILY_LOSS_CIRCUIT_BREAKER: No new trades once daily drawdown exceeds threshold
    5. CONCURRENT_POSITION_LIMIT: Max number of simultaneous open trades
    6. MIN_BALANCE_RESERVE: Minimum cash buffer to avoid liquidation
    7. MANDATORY_STOP_LOSS: Auto-enforced if Claude omits it (risk control)
    8. FORCE_CLOSE_AT_LOSS: Auto-close any position at/beyond MAX_LOSS_PER_POSITION_PCT

    All risk checks run BEFORE execution. Claude's recommendations are capped or
    blocked if they violate any guardrail.
    ════════════════════════════════════════════════════════════════════════════════
    """

    def __init__(self, hyperliquid=None):
        # Override via LLM_MODEL in .env: haiku (cheapest) / sonnet / opus
        self.model = CONFIG.get("llm_model") or "claude-haiku-4-5-20251001"
        self.client = anthropic.Anthropic(api_key=CONFIG["anthropic_api_key"])
        self.hyperliquid = hyperliquid
        # Sanitize model: cheap Haiku used only to fix malformed JSON output
        self.sanitize_model = CONFIG.get("sanitize_model") or "claude-haiku-4-5-20251001"
        self.max_tokens = int(CONFIG.get("max_tokens") or 1200)

        logging.info("TradingAgent initialized — main model: %s | sanitize model: %s",
                     self.model, self.sanitize_model)

    def decide_trade(self, assets, context):
        """Decide for multiple assets in one call."""
        return self._decide(context, assets=assets)

    def _decide(self, context, assets):
        """Dispatch decision request to Claude and enforce output contract.

        ════════════════════════════════════════════════════════════════════════════════
        CLAUDE'S DECISION FRAMEWORK
        ════════════════════════════════════════════════════════════════════════════════

        Claude is instructed to act as a QUANTITATIVE TRADER with the following mindset:
        1. Respect existing exit plans (don't close early unless invalidation occurred)
        2. Use hysteresis: require stronger evidence to change direction than to hold
        3. Impose self-cooldowns: 3+ bars before another direction change
        4. Treat funding as context, not a trigger (unless huge outlier)
        5. Don't rely on RSI extremes alone; pair with structure + momentum
        6. Prefer adjustments (tighter SL, trail TP) over exits when thesis weakens
        7. Close winners at high-quality opportunities; respect risk/reward

        DECISION INPUTS (passed in context JSON):
        ─────────────────────────────────────────────────────────────────────────────
        • Current timestamp (for validating cooldowns, exit triggers)
        • Account state: balance, free collateral, margin ratio
        • Open positions: asset, direction, entry price, quantity, TP/SL orders
        • Market data per asset:
          ├─ 1h candles (last 50): intraday_1h indicators with 3-bar series
          ├─ 4h candles (last 30): long_term_4h indicators with 3-bar series
          ├─ Pre-computed labels: trend_4h, trend_1h, momentum_4h
          ├─ Funding rate: current rate
          ├─ Open interest: Long/short ratio for positioning tilt
          └─ Mid price: Current spot price for context

        DECISION OUTPUTS (structured JSON):
        ─────────────────────────────────────────────────────────────────────────────
        Per asset, Claude decides:
        • action: "buy" | "sell" | "hold"
        • allocation_usd: Position size in USD (0 if hold)
        • order_type: "market" (immediate) | "limit" (resting order with better entry)
        • limit_price: Required if order_type="limit", ignored otherwise
        • tp_price: Take-profit target (null if holding or exit plan handles it)
        • sl_price: Stop-loss target (mandatory; system auto-applies if omitted)
        • exit_plan: Text description of closure conditions + any cooldown guidance
        • rationale: Explanation of the decision (structure, indicators, risk-reward)

        DECISION VALIDATION LOOP:
        ─────────────────────────────────────────────────────────────────────────────
        1. Claude is given up to 6 iterations to finalize a decision
        2. If Claude calls fetch_indicator tool, gather fresh data and loop
        3. On first successful JSON parse, extract trade decisions
        4. Fallback: if JSON is malformed, use a cheap Haiku model to normalize it
        5. Final output must be valid JSON with reasoning + trade_decisions array
        ════════════════════════════════════════════════════════════════════════════════
        """
        system_prompt = (
            "You are a rigorous QUANTITATIVE TRADER and interdisciplinary MATHEMATICIAN-ENGINEER optimizing risk-adjusted returns for perpetual futures under real execution, margin, and funding constraints.\n"
            "You will receive market + account context for SEVERAL assets, including:\n"
            f"- assets = {json.dumps(list(assets))}\n"
            "- per-asset 1h and higher-timeframe (4h) metrics\n"
            "- Active Trades with Exit Plans\n"
            "- Recent Trading History\n"
            "- Risk management limits (hard-enforced by the system, not just guidelines)\n\n"
            "Always use the 'current time' provided in the user message to evaluate any time-based conditions, such as cooldown expirations or timed exit plans.\n\n"
            "Your goal: make decisive, first-principles decisions per asset that minimize churn while capturing edge.\n\n"
            "Aggressively pursue setups where calculated risk is outweighed by expected edge; size positions so downside is controlled while upside remains meaningful.\n\n"
            "Core policy (low-churn, position-aware)\n"
            "1) Respect prior plans: If an active trade has an exit_plan with explicit invalidation (e.g., \"close if 4h close above EMA50\"), DO NOT close or flip early unless that invalidation (or a stronger one) has occurred.\n"
            "2) Hysteresis: Require stronger evidence to CHANGE a decision than to keep it. Only flip direction if BOTH:\n"
            "   a) Higher-timeframe structure supports the new direction (e.g., 4h EMA20 vs EMA50 and/or MACD regime), AND\n"
            "   b) Intraday structure confirms with a decisive break beyond ~0.5×ATR (recent) and momentum alignment (MACD or RSI slope).\n"
            "   Otherwise, prefer HOLD or adjust TP/SL.\n"
            "3) Cooldown: After opening, adding, reducing, or flipping, impose a self-cooldown of at least 3 bars of the decision timeframe (e.g., 3×1h = 3h) before another direction change, unless a hard invalidation occurs. Encode this in exit_plan (e.g., \"cooldown_bars:3 until 2025-10-19T15:00Z\"). You must honor your own cooldowns on future cycles.\n"
            "4) Funding is a tilt, not a trigger: Do NOT open/close/flip solely due to funding unless expected funding over your intended holding horizon meaningfully exceeds expected edge (e.g., > ~0.25×ATR). Consider that funding accrues discretely and slowly relative to 1h bars.\n"
            "5) Overbought/oversold ≠ reversal by itself: Treat RSI extremes as risk-of-pullback. You need structure + momentum confirmation to bet against trend. Prefer tightening stops or taking partial profits over instant flips.\n"
            "6) Prefer adjustments over exits: If the thesis weakens but is not invalidated, first consider: tighten stop (e.g., to a recent swing or ATR multiple), trail TP, or reduce size. Flip only on hard invalidation + fresh confluence.\n\n"
            "Decision discipline (per asset)\n"
            "- Choose one: buy / sell / hold.\n"
            "- Proactively harvest profits when price action presents a clear, high-quality opportunity that aligns with your thesis.\n"
            "- You control allocation_usd (but the system will cap it — see risk limits below).\n"
            "- Order type: set order_type to \"market\" for immediate execution, or \"limit\" for resting orders.\n"
            "  • For limit orders, you MUST set limit_price. Use limit orders when you want better entry prices (e.g., buying a dip, selling a bounce).\n"
            "  • For market orders, limit_price should be null.\n"
            "  • Default is \"market\" if omitted.\n"
            "- TP/SL sanity:\n"
            "  • BUY: tp_price > current_price, sl_price < current_price\n"
            "  • SELL: tp_price < current_price, sl_price > current_price\n"
            "  If sensible TP/SL cannot be set, use null and explain the logic. A mandatory SL will be auto-applied if you don't set one.\n"
            "- exit_plan must include at least ONE explicit invalidation trigger and may include cooldown guidance you will follow later.\n\n"
            "Leverage policy (perpetual futures)\n"
            "- You can use leverage, but the system enforces a hard cap. Stay within the limits.\n"
            "- In high volatility (elevated ATR) or during funding spikes, reduce or avoid leverage.\n"
            "- Treat allocation_usd as notional exposure; keep it consistent with safe leverage and available margin.\n\n"
            "Fee policy (CRITICAL — directly affects profitability)\n"
            "- Hyperliquid charges a 0.045% taker fee on market orders. A full round-trip (open + close) costs ~0.09%.\n"
            "- The risk limits in context show taker_fee_pct and min_tp_pct_from_entry (system-enforced minimum TP distance).\n"
            "- NEVER open a trade if the expected price move is less than 3× the round-trip fee (~0.27%).\n"
            "- BUY: set tp_price at least 0.3% above entry. SELL: set tp_price at least 0.3% below entry.\n"
            "- Prefer limit orders (order_type: \"limit\") when practical — they pay 0% maker fee, saving the full taker cost.\n"
            "- Factor funding costs into your holding horizon: if funding is strongly negative for your direction, either reduce size or target a larger TP to compensate.\n\n"
            "Tool usage\n"
            "- Use the fetch_indicator tool whenever an additional datapoint could sharpen your thesis; parameters: indicator (ema/sma/rsi/macd/bbands/atr/adx/obv/vwap/stoch_rsi/all), asset (e.g. \"BTC\", \"OIL\", \"GOLD\"), interval (\"1h\"/\"4h\"), optional period.\n"
            "- Indicators are computed locally from Hyperliquid candle data — works for ALL perp markets (crypto, commodities, indices).\n"
            "- Incorporate tool findings into your reasoning, but NEVER paste raw tool responses into the final JSON — summarize the insight instead.\n"
            "- Use tools to upgrade your analysis; lack of confidence is a cue to query them before deciding.\n\n"
            "Indicator interpretation — CANONICAL RULES (apply exactly, never invert):\n"
            "- EMA structure (use the pre-computed trend_4h / trend_1h fields in context):\n"
            "  • EMA20 > EMA50 = BULLISH structural bias → favor BUY\n"
            "  • EMA20 < EMA50 = BEARISH structural bias → favor SELL\n"
            "  • EMA20 > EMA50 > EMA200 = strong BULLISH; EMA20 < EMA50 < EMA200 = strong BEARISH\n"
            "  • trend_4h='BULLISH' in context means 4h EMA20 is ABOVE 4h EMA50 — this is an uptrend, lean BUY\n"
            "  • trend_4h='BEARISH' in context means 4h EMA20 is BELOW 4h EMA50 — this is a downtrend, lean SELL\n"
            "- RSI: RSI14 > 50 = bullish momentum → lean toward BUY; RSI14 < 50 = bearish momentum → lean toward SELL\n"
            "- MACD histogram (macd_histogram field = MACD line minus signal line):\n"
            "  • histogram > 0 = MACD line ABOVE signal = bullish acceleration → favor BUY\n"
            "  • histogram < 0 = MACD line BELOW signal = bearish deceleration → favor SELL\n"
            "  • momentum_4h='BULLISH' in context means histogram > 0 — confirms upward momentum\n"
            "- Price vs EMA20: price > EMA20 = short-term bullish; price < EMA20 = short-term bearish\n"
            "- Primary decision rule: BULLISH trend + BULLISH momentum → BUY; BEARISH trend + BEARISH momentum → SELL\n"
            "- Counter-trend trades (e.g., SELL when trend_4h=BULLISH) require exceptionally strong contrary signals\n"
            "  with explicit justification and tighter stops. Default to trend direction.\n\n"
            "Reasoning recipe (first principles)\n"
            "- Structure (trend_4h/trend_1h labels, EMAs slope/cross, HH/HL vs LH/LL), Momentum (MACD histogram regime, RSI slope), Liquidity/volatility (ATR, volume), Positioning tilt (funding, OI).\n"
            "- Favor alignment across 4h and 1h. Counter-trend trades require stronger 1h confirmation and tighter risk.\n\n"
            "Output contract\n"
            "- Output raw JSON only — no ```json fences, no markdown .\n"
            "- Output ONLY a strict JSON object with exactly two properties:\n"
            "  • \"reasoning\": max 2 sentences. Be concise.\n"
            "  • \"trade_decisions\": array ordered to match the provided assets list.\n"
            "- Each item inside trade_decisions must contain the keys: asset, action, allocation_usd, order_type, limit_price, tp_price, sl_price, exit_plan, rationale.\n"
            "  • order_type: \"market\" (default) or \"limit\"\n"
            "  • limit_price: required if order_type is \"limit\", null otherwise\n"
        )

        tools = [{
            "name": "fetch_indicator",
            "description": (
                "Fetch technical indicators computed locally from Hyperliquid candle data. "
                "Works for ALL Hyperliquid perp markets including crypto (BTC, ETH, SOL), "
                "commodities (OIL, GOLD, SILVER), indices (SPX), and more. "
                "Available indicators: ema, sma, rsi, macd, bbands, atr, adx, obv, vwap, stoch_rsi, all. "
                "Returns the latest values and recent series."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "enum": ["ema", "sma", "rsi", "macd", "bbands", "atr", "adx", "obv", "vwap", "stoch_rsi", "all"],
                    },
                    "asset": {
                        "type": "string",
                        "description": "Hyperliquid asset symbol, e.g. BTC, ETH, OIL, GOLD, SPX",
                    },
                    "interval": {
                        "type": "string",
                        "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                    },
                    "period": {
                        "type": "integer",
                        "description": "Indicator period (default varies by indicator)",
                    },
                },
                "required": ["indicator", "asset", "interval"],
            },
        }]

        messages = [{"role": "user", "content": context}]

        def _log_request(model, messages_to_log):
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {model}\n")
                f.write(f"Messages count: {len(messages_to_log)}\n")
                # Log last message content (truncated)
                last = messages_to_log[-1]
                content_str = str(last.get("content", ""))[:500]
                f.write(f"Last message role: {last.get('role')}\n")
                f.write(f"Last message content (truncated): {content_str}\n")

        enable_tools = CONFIG.get("enable_tool_calling", False)

        def _call_claude(msgs, use_tools=True):
            """Make a Claude API call with optional tool use."""
            _log_request(self.model, msgs)
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                # cache_control marks the system prompt for Anthropic prompt caching.
                # Cached tokens cost ~10% of normal price; effective from the 2nd call onward.
                "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                "messages": msgs,
            }
            if use_tools and enable_tools:
                kwargs["tools"] = tools
            if CONFIG.get("thinking_enabled"):
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(CONFIG.get("thinking_budget_tokens") or 10000),
                }
                if self.max_tokens < 16000:
                    logging.warning("THINKING_ENABLED forces max_tokens from %d to 16000 — this increases API cost significantly", self.max_tokens)
                kwargs["max_tokens"] = max(self.max_tokens, 16000)

            response = self.client.messages.create(**kwargs, timeout=45.0)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
            cost_usd = (input_tokens * 0.0000008) + (output_tokens * 0.000004)
            logging.info(
                "[API] input=%d output=%d cache_read=%d cost=$%.5f",
                input_tokens, output_tokens, cache_read, cost_usd
            )
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"Response stop_reason: {response.stop_reason}\n")
                f.write(f"Usage: input={input_tokens}, output={output_tokens}, cache_read={cache_read}\n")
                f.write(f"Cost: ${cost_usd:.5f}\n")
            return response

        def _handle_tool_call(tool_name, tool_input):
            """Execute a tool call and return the result string."""
            if tool_name != "fetch_indicator":
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            try:
                asset = tool_input["asset"]
                interval = tool_input["interval"]
                indicator = tool_input["indicator"]

                # Fetch candles — must run in a fresh event loop on a worker thread
                # because this method is called from inside the running asyncio loop.
                import concurrent.futures as _cf
                def _fetch():
                    import asyncio as _a
                    loop = _a.new_event_loop()
                    try:
                        return loop.run_until_complete(
                            self.hyperliquid.get_candles(asset, interval, 100)
                        )
                    finally:
                        loop.close()
                with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                    candles = pool.submit(_fetch).result(timeout=30)

                all_indicators = compute_all(candles)

                if indicator == "all":
                    result = {k: {"latest": latest(v) if isinstance(v, list) else v,
                                  "series": last_n(v, 10) if isinstance(v, list) else v}
                              for k, v in all_indicators.items()}
                elif indicator == "macd":
                    result = {
                        "macd": {"latest": latest(all_indicators.get("macd", [])), "series": last_n(all_indicators.get("macd", []), 10)},
                        "signal": {"latest": latest(all_indicators.get("macd_signal", [])), "series": last_n(all_indicators.get("macd_signal", []), 10)},
                        "histogram": {"latest": latest(all_indicators.get("macd_histogram", [])), "series": last_n(all_indicators.get("macd_histogram", []), 10)},
                    }
                elif indicator == "bbands":
                    result = {
                        "upper": {"latest": latest(all_indicators.get("bbands_upper", [])), "series": last_n(all_indicators.get("bbands_upper", []), 10)},
                        "middle": {"latest": latest(all_indicators.get("bbands_middle", [])), "series": last_n(all_indicators.get("bbands_middle", []), 10)},
                        "lower": {"latest": latest(all_indicators.get("bbands_lower", [])), "series": last_n(all_indicators.get("bbands_lower", []), 10)},
                    }
                elif indicator in ("ema", "sma"):
                    try:
                        period = max(2, min(200, int(tool_input.get("period", 20))))
                    except (ValueError, TypeError):
                        period = 20
                    from src.indicators.local_indicators import ema as _ema, sma as _sma
                    closes = [c["close"] for c in candles]
                    series = _ema(closes, period) if indicator == "ema" else _sma(closes, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "rsi":
                    try:
                        period = max(2, min(200, int(tool_input.get("period", 14))))
                    except (ValueError, TypeError):
                        period = 14
                    from src.indicators.local_indicators import rsi as _rsi
                    series = _rsi(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "atr":
                    try:
                        period = max(2, min(200, int(tool_input.get("period", 14))))
                    except (ValueError, TypeError):
                        period = 14
                    from src.indicators.local_indicators import atr as _atr
                    series = _atr(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "stoch_rsi":
                    result = {
                        "k": {"latest": latest(all_indicators.get("stoch_rsi_k", [])), "series": last_n(all_indicators.get("stoch_rsi_k", []), 10)},
                        "d": {"latest": latest(all_indicators.get("stoch_rsi_d", [])), "series": last_n(all_indicators.get("stoch_rsi_d", []), 10)},
                    }
                else:
                    key_map = {"adx": "adx", "obv": "obv", "vwap": "vwap"}
                    mapped = key_map.get(indicator, indicator)
                    series = all_indicators.get(mapped, [])
                    result = {"latest": latest(series) if isinstance(series, list) else series,
                              "series": last_n(series, 10) if isinstance(series, list) else series}

                return json.dumps(result, default=str)
            except Exception as ex:
                logging.error("Tool call error: %s", ex)
                return json.dumps({"error": str(ex)})

        def _sanitize_output(raw_content: str, assets_list):
            """Use a cheap Claude model to normalize malformed output."""
            try:
                response = self.client.messages.create(
                    model=self.sanitize_model,
                    max_tokens=700,
                    system=(
                        "You are a strict JSON normalizer. Return ONLY a JSON object with two keys: "
                        "\"reasoning\" (string) and \"trade_decisions\" (array). "
                        "Each trade_decisions item must have: asset, action (buy/sell/hold), "
                        "allocation_usd (number), order_type (\"market\" or \"limit\"), "
                        "limit_price (number or null), tp_price (number or null), sl_price (number or null), "
                        "exit_plan (string), rationale (string). "
                        f"Valid assets: {json.dumps(list(assets_list))}. "
                        "If input is wrapped in markdown or has prose, extract just the JSON. Do not add fields. Output raw JSON only. No ```json fences, no markdown of any kind."
                    ),
                    messages=[{"role": "user", "content": raw_content}],
                )
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "trade_decisions" in parsed:
                    return parsed
                return {"reasoning": "", "trade_decisions": []}
            except Exception as se:
                logging.error("Sanitize failed: %s", se)
                return {"reasoning": "", "trade_decisions": []}

        # Main loop: up to 6 iterations to handle tool calls
        def _hold_all(reason: str):
            logging.warning("[CLAUDE] %s — returning hold for all assets", reason)
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"Fallback hold: {reason}\n")
            return {
                "reasoning": reason,
                "trade_decisions": [{
                    "asset": a, "action": "hold", "allocation_usd": 0.0,
                    "order_type": "market", "limit_price": None,
                    "tp_price": None, "sl_price": None,
                    "exit_plan": "", "rationale": reason,
                } for a in assets]
            }

        for iteration in range(6):
            try:
                response = _call_claude(messages)
            except asyncio.TimeoutError:
                return _hold_all("Claude API asyncio timeout (45s)")
            except anthropic.APITimeoutError as e:
                return _hold_all(f"Claude API timeout (45s): {e}")
            except anthropic.APIError as e:
                logging.error("Claude API error: %s", e)
                with open("llm_requests.log", "a", encoding="utf-8") as f:
                    f.write(f"API Error: {e}\n")
                return _hold_all(f"Claude API error: {e}")

            # Check if the response contains tool use
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if tool_use_blocks and response.stop_reason == "tool_use":
                # Build assistant message with all content blocks
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    elif block.type == "thinking":
                        assistant_content.append({
                            "type": "thinking",
                            "thinking": block.thinking,
                        })
                messages.append({"role": "assistant", "content": assistant_content})

                # Process each tool call
                tool_results = []
                for block in tool_use_blocks:
                    result_str = _handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            # No tool calls — parse the text response as JSON
            raw_text = ""
            for block in text_blocks:
                raw_text += block.text

            if not raw_text.strip():
                logging.error("Empty response from Claude")
                break

            # Strip markdown code fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()

            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    logging.error("Expected dict, got: %s; attempting sanitize", type(parsed))
                    return _sanitize_output(raw_text, assets)

                reasoning_text = parsed.get("reasoning", "") or ""
                decisions = parsed.get("trade_decisions")

                if isinstance(decisions, list):
                    normalized = []
                    for item in decisions:
                        if isinstance(item, dict):
                            item.setdefault("allocation_usd", 0.0)
                            item.setdefault("order_type", "market")
                            item.setdefault("limit_price", None)
                            item.setdefault("tp_price", None)
                            item.setdefault("sl_price", None)
                            item.setdefault("exit_plan", "")
                            item.setdefault("rationale", "")
                            normalized.append(item)
                    return {"reasoning": reasoning_text, "trade_decisions": normalized}

                logging.error("trade_decisions missing or invalid; attempting sanitize")
                sanitized = _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {"reasoning": reasoning_text, "trade_decisions": []}

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logging.error("JSON parse error: %s, content: %s", e, raw_text[:200])
                sanitized = _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {
                    "reasoning": "Parse error",
                    "trade_decisions": [{
                        "asset": a,
                        "action": "hold",
                        "allocation_usd": 0.0,
                        "tp_price": None,
                        "sl_price": None,
                        "exit_plan": "",
                        "rationale": "Parse error"
                    } for a in assets]
                }

        # Exhausted tool loop
        return {
            "reasoning": "tool loop cap",
            "trade_decisions": [{
                "asset": a,
                "action": "hold",
                "allocation_usd": 0.0,
                "tp_price": None,
                "sl_price": None,
                "exit_plan": "",
                "rationale": "tool loop cap"
            } for a in assets]
        }