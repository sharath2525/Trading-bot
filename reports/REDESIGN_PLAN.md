# AI Role Redesign Plan
### From: One-word rubber stamp at score 10
### To: Deep market analyst triggered by multi-timeframe confluence

*Status: PLAN ONLY — no code changed yet*
*Date: 2026-05-01*
*Updated: 2026-05-01 — max_tokens 4000, full macro taxonomy, multi-timeframe gate, timing flexibility*

---

## What You Asked For

> "When the score reaches good, AI should analyze market conditions, check if volatile or not,
> check war conditions / CPI data / macro risk — decide if the trade is necessary.
> Code still runs on code logic. AI is only called some times, not every tick."

> "Bot cycle should not be locked to 1pm / 2pm. If conditions align at 1:45, the bot should
> catch it. 5m, 30m, 1h, 4h all approved — THEN call AI. AI should decide: real breakout
> or false breakout?"

This plan turns Claude from a one-word rubber stamp into a real market analyst. Code still
decides direction, TP, SL, and position size. Claude decides whether the market environment
is safe enough AND whether the breakout is genuine.

---

## What Changes vs What Stays the Same

### STAYS THE SAME
- `_code_decide_direction()` — code decides buy/sell. Claude never touches this.
- `_code_compute_tpsl()` — code sets TP/SL. Claude never touches this.
- `atr_position_size()` — code sets position size. Claude never touches this.
- Score below 7 → HOLD. No change.
- All 8 risk checks in `risk_manager.py` — recommended to keep (see Section 6).
- `market_filter()` — ATR spike + spread check stays in code.
- `entry_confirmed()` — 15m/5m technical confirmation stays in code.

### CHANGES
| What | Before | After |
|------|--------|-------|
| AI trigger | `score == 10.0` only | `score >= 7` **AND** multi-timeframe confluence confirmed |
| AI call timing | Only on outer 1h cycle | Any 5m tick — whenever confluence fires, regardless of cycle position |
| AI call frequency | ~5-20×/month | ~10-40×/month — confluence gate keeps it rare |
| Timeframes used | 4h, 1h, 15m, 5m | **Added 30m** — 4h, 1h, 30m, 15m, 5m |
| Context to Claude | ~150 tokens: asset, direction, entry, TP, SL, score | ~1200 tokens: + all timeframe signals, volatility, macro news, breakout context |
| Claude model | `claude-haiku-4-5-20251001` | Same model — `max_tokens` raised from 10 to **4000** |
| Claude task | "Approve or reject this score-10 setup" | "Analyze confluence across 4 timeframes. Real breakout or false? Safe macro environment?" |
| Claude response | `APPROVE` or `REJECT` (10 tokens) | Full chain-of-thought analysis, ends with `VERDICT: APPROVE` or `VERDICT: REJECT` |
| Cache invalidation | Per outer cycle only | Per confluence event — fresh AI call each time a new confluence is detected |
| Macro/news data | None | Web search for financial headlines fetched once per outer cycle |
| Cost estimate | ~$0.003/month | ~$1–5/month depending on how often score ≥ 7 fires |

---

## 1. New AI Trigger Rule

**Current:** Claude called only when `score == 10.0`, only on the outer 1h cycle.

**Problem with that design:** If the market aligns perfectly at 1:45 PM — all signals green
across 5m, 30m, 1h, 4h — the bot would either miss it (waiting for 2:00 PM outer cycle) or
execute without any AI check because the inner tick doesn't call Claude. You need the AI
check to fire whenever the market is genuinely ready, not on a fixed clock schedule.

**New rule:** Claude is called when **both conditions are true simultaneously:**
1. `compute_signal_score() >= MIN_AI_SCORE` (default 7) — the weighted score gate
2. `multi_timeframe_confluence()` returns True — ALL active timeframes agree (new gate, see Section 1b)

This can trigger on any 5-minute tick — outer loop or inner loop — because market alignment
doesn't respect hourly clock boundaries.

---

### 1a. New Score Pipeline Order (same logic, any tick)

```
EVERY TICK (outer loop OR inner loop tick):

1. Daily cap check (code)
2. SL cooldown check (code)
3. _code_decide_direction() — 4h + 1h EMA gate (code)
   └─ None → HOLD
4. compute_signal_score() — weighted 0-10 float (code)
   └─ score < MIN_SIGNAL_SCORE (7) → HOLD
5. multi_timeframe_confluence() — NEW GATE (code)
   └─ Not all timeframes aligned → HOLD
6. _code_compute_tpsl() — TP = entry + 2×ATR, SL = entry - 1×ATR (code)
7. atr_position_size() scaled by score/10 (code)
8. Check _ai_verdict_cache — is there a FRESH verdict for this confluence? (code)
   └─ Fresh cache hit → use cached verdict, skip API call
   └─ No cache or stale → proceed to step 9
9. confirm_trade() — Claude deep-analyzes market + macro + breakout (AI CALL)
   └─ Cache the verdict tagged with confluence_fingerprint
10. Execution phase — state gate, macro filter, entry_confirmed, risk checks (code)
```

---

### 1b. Multi-Timeframe Confluence Gate (New Code Function)

This is the new gate between score and AI call. It answers: **"Do ALL timeframes agree
right now, not just the dominant ones?"**

The function `multi_timeframe_confluence(asset_data, direction)` returns `True` only when:

| Timeframe | Required for BUY | Required for SELL |
|-----------|-----------------|-------------------|
| 4h | EMA20 > EMA50 (trend_4h = BULLISH) | EMA20 < EMA50 (BEARISH) |
| 1h | EMA20 > EMA50 (trend_1h = BULLISH) | EMA20 < EMA50 (BEARISH) |
| 30m *(new)* | EMA20 > EMA50 OR MACD histogram > 0 | EMA20 < EMA50 OR MACD histogram < 0 |
| 15m | MACD histogram > 0 AND price near EMA | MACD histogram < 0 AND price near EMA |
| 5m | Bullish candle OR positive MACD | Bearish candle OR negative MACD |

**All 5 rows must be True.** If any one timeframe conflicts → HOLD. This is what makes
the AI call meaningful — by the time Claude is called, the entire timeframe stack has
already confirmed the direction. Claude's job is then purely market environment + breakout validity.

**Why this keeps AI calls rare:**
- Getting all 5 timeframes simultaneously aligned is uncommon — maybe 20-30% of hours
- The score gate (≥ 7) already requires 4h + 1h + 15m + 5m to mostly agree
- The confluence gate adds 30m as an intermediate check and requires STRICT alignment
- Combined: genuine full-stack alignment happens perhaps 5-15 times per week per asset

---

### 1c. AI Call Timing — Fires on Any Tick, Not Just Outer Loop

**The old problem:** Inner ticks could NOT call Claude. If confluence happened at minute 35
of the hour, the bot would wait 25 more minutes for the outer loop.

**The new design:** Claude can fire on any tick — outer or inner — when confluence is
detected. The difference is what gets passed to Claude:

- **Outer loop tick:** Has fresh 4h/1h/30m/15m data (just fetched). Full context available.
- **Inner loop tick:** Has fresh 5m data only. 4h/1h/30m/15m data is from the last outer cycle.
  - If the outer cycle ran less than 55 minutes ago → use cached higher-TF data (acceptable)
  - If the outer cycle data is stale (> 55 min) → do NOT call AI, HOLD until next outer cycle

This gives you genuine timing flexibility without sacrificing data quality.

---

### 1d. AI Verdict Cache — Now Keyed on Confluence, Not Just Cycle

**Old cache key:** `invocation_count` (the outer cycle number)

**New cache key:** `confluence_fingerprint` — a hash of:
```
asset + direction + trend_4h + trend_1h + score_bucket (rounded to 0.5) + date_hour
```

**Why this matters:**
- If the same confluence pattern fires on tick 3 and tick 7 of the same hour → use
  the cached verdict from tick 3. Same setup, no need to call Claude twice.
- If the direction or a key trend label flips between ticks → fingerprint changes →
  new AI call (the market genuinely changed).
- Cache expires after 60 minutes regardless — market conditions drift.

**Cache invalidation rules:**
1. Fingerprint changes (direction or trend flips) → invalidate immediately, new AI call
2. Age > 60 minutes → invalidate, new AI call next confluence
3. New outer cycle starts → invalidate all asset caches (fresh data)
4. Claude returned REJECT → cache the REJECT for 30 minutes
   (if the bot re-scores ≥ 7 on the next tick, don't hammer Claude — the REJECT stands
   for 30 min unless the market meaningfully changes, i.e., fingerprint changes)

---

### 1e. Hard Minimum Gap Between AI Calls (New Rule — Closes the Fingerprint Loophole)

**The problem the cache doesn't fully close:**

The fingerprint includes `score_bucket` (rounded to 0.5). If the score was 8.5 on tick 3
(fingerprint: `BTC_buy_BULLISH_BULLISH_8.5_2026050113`) and rises to 10.0 on tick 7
(fingerprint: `BTC_buy_BULLISH_BULLISH_10.0_2026050113`), the fingerprints differ —
the cache considers this a new setup and fires a second Claude call. But the market
context hasn't changed meaningfully; only the score ticked up by 1.5 points. This is
exactly the "1:30, 1:35, 1:40 repeated approval" problem applied to score drift.

**The fix — a hard per-asset minimum gap:**

Add `_last_ai_call_time: dict[str, float]` (per asset, unix timestamp).

**Rule:** No Claude call for an asset is allowed within `MIN_AI_CALL_GAP_MINUTES` of the
previous call, **regardless of fingerprint**. Default: 30 minutes.

```
Before firing any AI call:
  gap_elapsed = now - _last_ai_call_time.get(asset, 0)
  if gap_elapsed < MIN_AI_CALL_GAP_MINUTES * 60:
      use previous verdict if still valid
      if no valid previous verdict → HOLD (skip entry this tick)
      log: [AI GAP] BTC — last call 14 min ago, minimum gap is 30 min — holding
```

**How this interacts with the cache:**

| Situation | Behavior |
|-----------|----------|
| Cache hit (fingerprint match, not expired) | Use cached verdict — no gap check needed |
| Cache miss + gap not elapsed | Use previous verdict if available; otherwise HOLD |
| Cache miss + gap elapsed + fresh higher-TF data | Fire new AI call, update `_last_ai_call_time` |
| Cache miss + gap elapsed + stale higher-TF data | HOLD, wait for outer cycle |

The gap rule is an **additional layer beneath the cache**, not a replacement for it.
Cache handles the "same fingerprint" case. Gap handles the "similar but slightly different
fingerprint" case where the market hasn't genuinely changed but score drift caused a new hash.

**Why 30 minutes:**

- 30 min is the REJECT cache duration — so a REJECT verdict's validity naturally aligns with the gap
- Within a single outer cycle (1h), a maximum of 2 AI calls per asset can fire (at minute 0 and minute 31)
- In practice, with confluence required, this is almost always 0 or 1 per hour
- Does NOT prevent genuine direction changes from triggering a new call — if direction flips from
  buy → sell, the gap check uses the **previous-direction** timestamp, not cross-direction

**New config key:**

| Key | Default | Purpose |
|-----|---------|---------|
| `MIN_AI_CALL_GAP_MINUTES` | `30` | Hard minimum gap between Claude calls for the same asset, regardless of fingerprint changes. |

---

## 2. What Claude Receives (New Context Payload ~1200 tokens)

The context is structured in clearly labelled sections so Claude can work through each
category systematically before reaching a verdict.

---

### Section A — Trade Setup (code-computed, read-only for Claude)
```
TRADE SETUP
═══════════════════════════════════════════════════
Asset:          BTC
Direction:      BUY  ← set by code, you cannot change this
Entry price:    $94,250
Take-profit:    $96,400  (entry + 2×ATR14 — set by code)
Stop-loss:      $93,100  (entry - 1×ATR14 — set by code)
Signal score:   8.5 / 10
UTC timestamp:  2026-05-01T13:45:00Z
```

### Section B — Multi-Timeframe Confluence (NEW — all timeframes passed)
```
TIMEFRAME ALIGNMENT
═══════════════════════════════════════════════════
4h  trend:   BULLISH  (EMA20 $93,800 > EMA50 $91,200)
    MACD histogram last 3 bars: [+0.18, +0.22, +0.29]  ← accelerating
    ADX: 28  (trending)
    RSI14: 54

1h  trend:   BULLISH  (EMA20 $94,100 > EMA50 $93,400)
    MACD histogram last 3 bars: [+0.08, +0.11, +0.14]  ← rising
    ADX: 31  (strong trend)
    RSI14: 61

30m trend:   BULLISH  (EMA20 $94,180 > EMA50 $93,900)    ← NEW timeframe
    MACD histogram: +0.09
    RSI14: 63

15m setup:   BULLISH  (price $94,250 near EMA20 $94,190 — 0.06% distance)
    MACD histogram: +0.12
    RSI14: 58  (not overbought)

5m  trigger: BULLISH  (last candle bullish, close $94,250 > open $94,180)
    MACD histogram: +0.04

NOTE: All 5 timeframes are aligned. This confluence was detected at 13:45 UTC.
```

### Section C — Volatility & Market Structure (NEW)
```
VOLATILITY
═══════════════════════════════════════════════════
ATR14 (4h):         $1,150  (1.22% of price — NORMAL range for BTC)
ATR14 as % of SL:   SL is 1.22% below entry — ATR-matched
Bollinger Band (4h) width: 3.4%  (normal: 2-5%, high vol: >6%)
Recent 4h price range (last 10 bars): $92,800 – $95,100  ($2,300 range)
Spread:             0.04%  (well within 0.15% block threshold)
```

### Section D — Market Positioning (NEW)
```
POSITIONING
═══════════════════════════════════════════════════
Funding rate:           +0.012% per 8h  (longs paying)
Funding annualized:     +13.1%
Funding status:         NORMAL  (extreme = >0.05% per 8h)
Open interest:          $2.1B
Recent OI change:       +2.3% in last 4h  (growing — new longs entering)
```

### Section E — Macro / News Context (fetched via RSS before this call)
```
MACRO CONTEXT  [fetched 3 min ago — 2026-05-01T13:42Z]
═══════════════════════════════════════════════════
UPCOMING HIGH-IMPACT EVENTS (next 12 hours):
  - None scheduled

RECENT HEADLINES (last 4 hours):
  [CoinDesk] "Bitcoin breaks above $94k as ETF inflows resume" (13:20 UTC)
  [Reuters]  "Fed officials signal no rate change expected before June" (11:45 UTC)
  [Reuters]  "US equity futures flat ahead of earnings season" (12:30 UTC)

GEOPOLITICAL: No active escalation events detected in headlines
```
*If web fetch failed: "MACRO DATA: UNAVAILABLE — analyze on technicals and timestamp only"*

### Section F — Claude's Instructions
```
INSTRUCTIONS
═══════════════════════════════════════════════════
You are a professional trading risk analyst and market environment validator.

The code has already confirmed:
✓ Multi-timeframe alignment across 5 timeframes (above)
✓ Signal score 8.5/10
✓ Direction, entry, TP, SL — all set by code. You CANNOT change any of these.

YOUR ONLY JOB is to assess whether this is a REAL, HIGH-QUALITY setup or a FALSE/RISKY one.

Work through EACH of these questions in your analysis:

BREAKOUT VALIDITY:
- Are all 5 timeframes genuinely aligned or is one timeframe borderline?
- Is the MACD acceleration consistent across timeframes (not just one TF)?
- Is the RSI at a level that supports further move (not already overbought)?
- Does the price structure suggest a genuine breakout or a potential fake-out?
- Is volume/OI confirming the move or is this a low-conviction candle?

MACRO & NEWS RISK:
- Are there any high-impact scheduled events in the next 4-12 hours?
  (FOMC, CPI, NFP, ECB, PCE, PPI, GDP, PMI, earnings, options expiry)
- Are there any active geopolitical shocks in the headlines?
- Does the current UTC time put this entry near a high-risk window?
  (US market open 13:30 UTC, US market close 20:00 UTC, Asia open 23:00 UTC)

VOLATILITY RISK:
- Is ATR suggesting normal conditions or a spike regime?
- Is the Bollinger Band width in a healthy range for this type of entry?
- Is the spread normal?

POSITIONING RISK:
- Is the funding rate at an extreme that suggests crowded positioning?
- Is open interest growing (confirms trend) or shrinking (warns of reversal)?

FINAL DECISION:
After working through all of the above, give your verdict.

End your response with exactly one of these two lines:
VERDICT: APPROVE
VERDICT: REJECT

Lean toward REJECT when uncertain. A missed trade is better than a trapped position.
```

---

## 3. Complete Taxonomy of Market-Moving Conditions Claude Must Analyze

This is the full list of conditions that affect trading. Claude needs to be aware of ALL of
these, not just CPI and wars. The prompt will explicitly instruct Claude to check each category.

---

### CATEGORY 1 — Central Bank & Monetary Policy
Events that move ALL markets (crypto, equities, commodities, forex):

| Event | Impact | Frequency |
|-------|--------|-----------|
| FOMC interest rate decision | EXTREME — 3-8% moves typical | 8×/year |
| FOMC minutes release | HIGH — clarifies Fed intent | 8×/year |
| Fed Chair speech / press conference | HIGH | Variable |
| ECB rate decision | HIGH (affects USD/EUR, gold) | 8×/year |
| Bank of England, BOJ, RBA decisions | MEDIUM | Monthly |
| Fed officials speaking (hawkish/dovish) | MEDIUM | Weekly |

**What Claude checks:** Is there a central bank event in the next 4 hours? Next 12 hours?
Even a Fed speech can cause 2-3% swings. Entering before = high gap risk.

---

### CATEGORY 2 — Economic Data Releases
Scheduled data that surprises markets:

| Data | Impact | Frequency |
|------|--------|-----------|
| US CPI (Consumer Price Index) | EXTREME — inflation data | Monthly |
| US PCE (Fed's preferred inflation gauge) | EXTREME | Monthly |
| US NFP (Non-Farm Payrolls) | EXTREME — jobs data, first Friday of month | Monthly |
| US GDP | HIGH | Quarterly |
| US PPI (Producer Price Index) | HIGH | Monthly |
| US Retail Sales | HIGH | Monthly |
| US Initial Jobless Claims | MEDIUM | Weekly (Thursday) |
| ISM Manufacturing/Services PMI | MEDIUM | Monthly |
| JOLTS Job Openings | MEDIUM | Monthly |
| University of Michigan Consumer Sentiment | MEDIUM | Monthly |
| China CPI, PMI | MEDIUM (affects commodities) | Monthly |
| EU CPI, GDP | MEDIUM | Monthly |

**What Claude checks:** Any of these in the next 4-12 hours? Surprise = violent move.
"Expected" releases still move markets; unexpected prints are catastrophic for levered positions.

---

### CATEGORY 3 — Crypto-Specific Events
Events unique to crypto markets:

| Event | Impact | Notes |
|-------|--------|-------|
| Bitcoin ETF approvals/rejections | EXTREME | SEC/CFTC announcements |
| Exchange hack or insolvency | EXTREME | e.g., FTX-style events |
| Regulatory action (ban, lawsuit) | EXTREME | SEC vs exchanges |
| Bitcoin halving | HIGH | Predictable — every 4 years |
| Large exchange maintenance/downtime | HIGH | Liquidity drops |
| Major protocol upgrade/fork | HIGH | Ethereum upgrades etc. |
| Whale on-chain movements ($500M+) | MEDIUM | CryptoQuant data |
| Large liquidation cascades | HIGH | Check funding/OI extremes |
| Stablecoin depeg | EXTREME | USDT/USDC issues |
| Crypto options expiry (monthly/quarterly) | MEDIUM | CME, Deribit |

**What Claude checks:** Any of these in the news? Is funding rate showing extreme
positioning that suggests a cascade is imminent?

---

### CATEGORY 4 — Geopolitical Events
Events that trigger risk-off/risk-on globally:

| Event | Impact | Notes |
|-------|--------|-------|
| War outbreak or major escalation | EXTREME | BTC can spike OR crash |
| Military strikes (airstrikes, missile launches) | HIGH | Risk-off initially |
| Sanctions announcements | HIGH | Especially US/EU sanctions |
| Nuclear threat rhetoric | EXTREME | Extreme risk-off |
| Coup or government collapse | HIGH | Emerging market shock |
| US-China tensions escalating | HIGH | Trade war, Taiwan |
| Energy infrastructure attack (pipelines, etc.) | HIGH | Oil price spike |
| Major natural disaster | MEDIUM | Supply chain + sentiment |

**What Claude checks:** Any active conflict escalation in headlines? Is there a credible
near-term geopolitical shock? These are hard to predict but easy to identify when they
are happening in real-time news.

---

### CATEGORY 5 — Equity Market Conditions
Crypto and equities are increasingly correlated (70-80% in recent years):

| Condition | Impact | Notes |
|-----------|--------|-------|
| US stock market circuit breaker triggered | EXTREME | -7%, -13%, -20% halts |
| S&P 500 / Nasdaq crashing (>2% down) | HIGH | BTC typically follows |
| VIX spike above 30 | HIGH | Fear index — risk-off |
| Major tech earnings miss (AAPL, MSFT, NVDA) | MEDIUM | Sentiment contagion |
| US stock market futures limit down | HIGH | Pre-market risk signal |
| Major index ETF (SPY/QQQ) unusual options activity | MEDIUM | Potential volatility |

**What Claude checks:** Is there an equity market stress signal in the data? Is VIX elevated?

---

### CATEGORY 6 — Commodity & Energy Markets
Relevant mainly for GOLD, OIL, and macro sentiment:

| Condition | Impact | Notes |
|-----------|--------|-------|
| Oil price spike or crash (>5%) | HIGH | Affects inflation, rates |
| OPEC/OPEC+ surprise decision | HIGH | Supply shock |
| Gold flash crash or spike | MEDIUM | Safe-haven flows |
| Natural gas price spike | MEDIUM | Energy crisis indicator |
| Agricultural commodity shock | LOW-MEDIUM | Food inflation |

**What Claude checks:** For GOLD/OIL assets directly. For crypto — oil spike = inflation = Fed hawkish = crypto negative.

---

### CATEGORY 7 — Market Microstructure / Liquidity
Conditions the bot can partially measure itself:

| Condition | Impact | Notes |
|-----------|--------|-------|
| Abnormally wide spread (>0.15%) | HIGH | Low liquidity — already blocked by code |
| Funding rate extreme (>0.05% per 8h) | HIGH | Squeeze risk |
| Open interest dropping sharply | MEDIUM | Position unwinding |
| Volume significantly below average | MEDIUM | Low conviction move |
| Price far from EMA (>2%) | MEDIUM | Chasing signal |
| Weekend/holiday thin liquidity | MEDIUM | Amplified moves |

**What Claude checks:** The funding rate and OI data passed in context. Also: is this a
weekend? Are markets in a holiday period (Thanksgiving, Christmas, Chinese New Year)?

---

### CATEGORY 8 — Scheduled Time-Based Risks

| Condition | Impact | Notes |
|-----------|--------|-------|
| Monthly crypto options expiry (last Friday) | HIGH | Large gamma exposure |
| CME BTC futures expiry | HIGH | Usually last Friday of month |
| End of quarter (March 31, June 30, Sept 30, Dec 31) | HIGH | Rebalancing flows |
| US market open (9:30 ET) | MEDIUM | Volatility spike |
| US market close (4:00 PM ET) | MEDIUM | Position squaring |
| Asia open (varies by time zone) | LOW-MEDIUM | Overnight gaps |

**What Claude checks:** Given the current UTC timestamp passed in context, are we near
a time-based risk window?

---

### How This Gets to Claude

All of the above falls into two buckets:

**Bucket 1 — Pre-fetched data (web search before calling Claude)**
- Economic calendar for the next 24 hours (upcoming CPI, FOMC, NFP)
- Last 5-10 news headlines for the asset and macro terms
- Fetched once per outer cycle, cached for all inner ticks

**Bucket 2 — Real-time data already in the bot**
- Funding rate (already fetched from Hyperliquid every cycle)
- Open interest (already fetched)
- Spread from impactPxs (already calculated)
- ATR, Bollinger Bands (already computed)
- Current UTC timestamp (available in every cycle)
- ADX, MACD series (already computed)

Claude receives both buckets in a structured prompt. Bucket 1 provides the "what is happening in the world" context. Bucket 2 provides the "what is the market doing right now" context. With 4000 tokens of output budget, Claude can reason through every category before giving a verdict.

---

### Macro / News Fetch Implementation

**Option A — Web Search + Economic Calendar RSS (Recommended)**

Before calling `confirm_trade()` in the outer loop:

Step 1 — Economic Calendar (structured, reliable):
- Source: Forex Factory RSS feed (free, no API key) — `https://rss.forexfactory.com`
- Parse events with `impact: HIGH` scheduled in next 12 hours
- Pass as a clean list: `[{"time": "16:00 UTC", "event": "US CPI", "impact": "HIGH"}]`

Step 2 — News Headlines (breaking events):
- Source: CoinDesk RSS (crypto) + Reuters RSS (macro) — both free, no API key
- Fetch last 5 headlines from each, published within the last 4 hours
- Pass as a simple list of strings

Step 3 — Timeout protection:
- Both fetches run with a 3-second timeout
- On timeout or failure → pass `"UNAVAILABLE"` for that section
- Never block the trading cycle waiting for news

Step 4 — Cache:
- Cache both fetches in `_macro_context` dict at the start of each outer cycle
- Inner ticks use cached context — do NOT re-fetch every 5 minutes
- Cache is invalidated at the start of each new outer loop cycle

**Option B — Fallback (no internet / API down)**

If all fetches fail → pass to Claude:
```
MACRO DATA: UNAVAILABLE (fetch failed or not configured)
Analyze based on technical data and current time only.
Default to conservative — if near a typical high-impact time window, lean REJECT.
```

Claude will still analyze volatility, funding, ADX, MACD, and timestamp-based risks
(e.g., "it's 14:55 UTC — US market open in 35 minutes, elevated volatility expected").

---

### Cost Estimate with 4000 max_tokens

### Updated Cost Estimate — With 4000 max_tokens + Confluence Gate

The confluence gate (all 5 timeframes must agree) is much stricter than score-only.
This keeps AI calls rare despite the lower score threshold.

| Scenario | AI calls/day | Input tokens | Output tokens (avg, of 4000 max) | Cost/call | Cost/month |
|----------|-------------|-------------|--------------------------------|----------|-----------|
| Low — confluence rare (20% alignment rate) | ~10 | ~1,200 | ~800 | ~$0.0033 | ~$1.00 |
| Medium — confluence moderate (40%) | ~19 | ~1,200 | ~800 | ~$0.0033 | ~$1.90 |
| High — confluence frequent (60%) | ~29 | ~1,200 | ~800 | ~$0.0033 | ~$2.90 |

Claude Haiku pricing: input $0.0000008/token, output $0.000004/token.
800 output tokens = full analysis paragraph + verdict. Most Claude responses will be
400-800 tokens — well within the 4000 ceiling. Claude won't pad output unnecessarily.

**Cost comparison vs old design:**
- Old (score==10 only): ~$0.003/month
- New (confluence + full analysis): ~$1-3/month
- Difference: ~$1-3/month for genuine market intelligence on every trade entry
- This is acceptable. If cost exceeds $10/month, confluence is too loose — raise `MIN_AI_SCORE` to 8.

**How to monitor cost:** Every call logs to `llm_requests.log` with `input_tokens`,
`output_tokens`, and `cost_usd`. Check this file weekly. The pattern `grep "CONFIRM\|ANALYZE" llm_requests.log | tail -50` gives you the last 50 AI decisions at a glance.

---

## 4. Claude's Response Format

### Current: `max_tokens=10`, returns `APPROVE` or `REJECT` (no reasoning)

### New: `max_tokens=4000`, Claude reasons fully then ends with a verdict line

**Why 4000 tokens:**

With only 10 tokens, Claude is forced to guess instantly with no reasoning. 4000 tokens
lets Claude actually think through the full market picture — volatility, macro calendar,
news risk, funding conditions — before deciding. This is what makes the analysis genuinely
useful rather than a coin flip. Claude will use most of its token budget on internal
chain-of-thought reasoning, then end with a clear verdict. You pay for the reasoning but
get a real decision.

**Response structure Claude is instructed to follow:**

```
MARKET ANALYSIS — [ASSET] [DIRECTION] [DATE/TIME]

VOLATILITY ASSESSMENT:
[Claude's analysis of ATR, Bollinger Bands, recent price range]
[Is volatility in a safe range? Too high? Too compressed?]

TREND STRENGTH:
[ADX, MACD series analysis — is momentum real or fading?]
[Is the entry timing good or chasing a move?]

MACRO CALENDAR RISK:
[Any scheduled high-impact events in the next 2-12 hours?]
[FOMC, CPI, NFP, ECB, earnings, options expiry, etc.]
[Risk: HIGH / MEDIUM / LOW]

GEOPOLITICAL / NEWS RISK:
[Any breaking events that could cause irrational market moves?]
[Wars, sanctions, exchange outages, regulatory announcements]
[Risk: HIGH / MEDIUM / LOW]

MARKET POSITIONING:
[Funding rate analysis — are longs/shorts over-extended?]
[Open interest — is there squeeze risk?]
[Spread — is liquidity normal?]

OVERALL RISK ASSESSMENT:
[Summary: is this a clean setup in a safe environment?]
[Key risk factors if any]

VERDICT: APPROVE
```
or
```
VERDICT: REJECT
```

**Parsing rule — extract the last `VERDICT:` line:**
```python
# Find "VERDICT: APPROVE" or "VERDICT: REJECT" anywhere in the response
# Fail closed — if no VERDICT line found, treat as REJECT
if "VERDICT: APPROVE" in answer.upper():
    return "APPROVE"
else:
    return "REJECT"
```

This is robust: Claude could write `VERDICT: APPROVE` anywhere and it will be found.
If Claude's output is malformed, truncated, or missing the verdict line → REJECT (fail closed).

**Full reasoning is written to `llm_requests.log` and `prompts.log`** so you can review
every decision Claude made and tune the prompt over time.

**Example approved response:**
```
MARKET ANALYSIS — BTC BUY 2026-05-01T14:00Z

VOLATILITY ASSESSMENT:
ATR14 is $1,150 on 4h, representing 1.22% of current price. This is within normal range
for BTC (typical 1-2%). Bollinger Band width at 3.4% indicates moderate compression —
consistent with a breakout setup rather than excessive volatility. No spike conditions.
Volatility: ACCEPTABLE

TREND STRENGTH:
ADX at 28 on 4h confirms a genuine trend (above 25 threshold). MACD histogram rising
across last 3 bars [+0.12, +0.18, +0.22] shows accelerating bullish momentum, not fading.
RSI at 58 — not overbought, room to run. Entry timing looks clean.
Trend: CONFIRMED BULLISH

MACRO CALENDAR RISK:
No FOMC meetings scheduled today. CPI data was released yesterday — no repeat event
in the next 48 hours. No NFP this week. No major earnings from BTC-correlated equities.
Economic calendar is clear for the next 12 hours.
Macro Risk: LOW

GEOPOLITICAL / NEWS RISK:
No breaking geopolitical events detected in fetched headlines. No exchange outage reports.
No regulatory announcements from SEC/CFTC. Market sentiment appears neutral-to-positive.
News Risk: LOW

MARKET POSITIONING:
Funding rate +0.012% per 8h is mildly positive — longs slightly dominant but nowhere near
extreme squeeze territory (extreme would be >0.05%). Open interest stable. Spread at 0.04%
is well within the 0.15% block threshold. No liquidity concerns.
Positioning Risk: LOW

OVERALL RISK ASSESSMENT:
Clean technical setup in a calm macro environment. No scheduled catalysts, volatility normal,
momentum confirmed. All risk factors LOW. This is the type of environment the strategy is
designed to trade.

VERDICT: APPROVE
```

**Example rejected response:**
```
MARKET ANALYSIS — ETH BUY 2026-05-01T20:00Z

VOLATILITY ASSESSMENT:
ATR14 is elevated at 2.8% of price — approaching the 3% level where SL distance becomes
unreliable. Bollinger Bands have expanded to 7.2% width, indicating a high-volatility regime.
Recent candles show a 4.1% range in the last 8 hours. Conditions are choppy.
Volatility: ELEVATED — caution

MACRO CALENDAR RISK:
FOMC minutes release is scheduled in 2 hours and 15 minutes (22:15 UTC). These releases
routinely cause 2-5% moves across risk assets including ETH. Entering a leveraged position
90 minutes before a high-impact Fed event creates significant gap risk — the SL at 1×ATR
may not protect adequately against a 3% sudden move.
Macro Risk: HIGH — FOMC in 135 minutes

VERDICT: REJECT
```

---

## 5. Verdict Cache Logic (Revised)

### Cache structure

```python
_ai_verdict_cache: dict[str, dict] = {}
# Per asset:
# {
#   "verdict":      "APPROVE" or "REJECT",
#   "fingerprint":  "BTC_buy_BULLISH_BULLISH_8.5_2026050113",   # see 1d above
#   "score":        8.5,
#   "timestamp":    "2026-05-01T13:45:00Z",
#   "expires_at":   "2026-05-01T14:45:00Z",    # 60 min for APPROVE
#                   "2026-05-01T14:15:00Z",    # 30 min for REJECT
#   "analysis":     "<Claude's full reasoning text>"
# }

_macro_context_cache: dict = {}
# {
#   "events":     [...],          # economic calendar next 12h
#   "headlines":  [...],          # last 5 news headlines
#   "fetched_at": "13:42 UTC",
#   "expires_at": "14:42 UTC"    # 60 min cache
# }
```

### Decision tree — any tick, any time

```
Score >= 7 AND confluence confirmed?
    │
    ├── YES → check _ai_verdict_cache[asset]
    │             │
    │             ├── Cache exists AND fingerprint matches AND not expired?
    │             │       → use cached verdict (no API call)
    │             │       → log: [AI CACHE HIT] BTC APPROVE (expires in 23min)
    │             │
    │             └── Cache miss OR fingerprint changed OR expired?
    │                     → check if higher-TF data is fresh (< 55 min old)
    │                         ├── Fresh → fetch macro context (if cache expired)
    │                         │          → call confirm_trade() → cache result
    │                         └── Stale → HOLD, log: [AI HOLD] stale 4h data, wait for outer cycle
    │
    └── NO → HOLD (score or confluence not met — no AI call)
```

### What "fresh higher-TF data" means

The inner loop only refreshes 5m candles. The 4h, 1h, 30m, 15m indicators come from the
last outer cycle. The rule:
- If the outer cycle ran **less than 55 minutes ago** → higher-TF data is acceptable,
  Claude can be called with a note: "4h/1h data from XX:XX UTC (NN min ago)"
- If the outer cycle ran **55+ minutes ago** (clock drift, slow API, or missed cycle) →
  HOLD and wait — calling Claude with severely stale 4h data is worse than waiting

### Why REJECT caches for only 30 minutes (not 60)

If Claude rejects at 1:45pm because of a nearby FOMC event at 2:30pm, and the event passes
cleanly with no shock, by 3:00pm the REJECT is no longer valid. A 30-minute REJECT window
means the bot can re-evaluate in the next outer cycle (2:00pm or 3:00pm) and potentially
APPROVE a fresh confluence if the macro risk cleared.

A 60-minute REJECT would lock out a valid 3:00pm setup that emerged after the event resolved.

---

## 6. Risk Checks After Claude — Recommendation

**User said: "not sure"**

**Recommendation: KEEP all 8 hard risk checks running AFTER Claude approval.**

Here is why this is important:

Claude analyzing "market conditions are safe" does NOT mean the specific trade parameters
are safe. These are two different questions:

| Question | Who answers it |
|----------|---------------|
| "Is the macro environment safe to trade right now?" | Claude |
| "Is this specific allocation within position size limits?" | Risk manager |
| "Does my total exposure exceed the safety cap?" | Risk manager |
| "Is the account balance above the reserve floor?" | Risk manager |
| "Does the SL meet the minimum required distance?" | Risk manager |

Claude APPROVE + Risk manager REJECT is a valid and expected outcome. Example:
- Score 8.5 on BTC — Claude approves (market looks fine)
- But the user already has 2 positions open, and `MAX_CONCURRENT_POSITIONS = 2`
- Risk manager blocks the trade — correct behavior
- Claude had no way to know about concurrent position limits

**If risk checks are removed:** a Claude APPROVE could result in a trade that exceeds
leverage limits, wipes the reserve floor, or fires on a day the circuit breaker should
have blocked. The risk manager is the last line of defense.

**Updated pipeline with both layers:**
```
Claude APPROVE  →  Risk manager validates  →  Execute   ← safest path
Claude APPROVE  →  Risk manager blocks     →  HOLD      ← correct, not a bug
Claude REJECT   →  (no trade)              →  HOLD
```

---

## 7. Files That Need Changing

| File | What Changes |
|------|-------------|
| `src/config_loader.py` | Add: `MIN_AI_SCORE`, `NEWS_FETCH_ENABLED`, `AI_MAX_TOKENS`, `AI_REJECT_CACHE_MINUTES`, `AI_APPROVE_CACHE_MINUTES` |
| `src/agent/decision_maker.py` | Rewrite `confirm_trade()`: new multi-section prompt, `max_tokens=4000`, parse for `VERDICT: APPROVE/REJECT`, log full analysis |
| `src/main.py` — new helpers | Add `multi_timeframe_confluence()` function, `_build_confluence_fingerprint()`, `_fetch_macro_context()` (RSS fetch with timeout) |
| `src/main.py` — outer loop | Add 30m candle fetch to the `asyncio.gather` block. Build 30m indicators. Pass into market_sections. |
| `src/main.py` — pipeline | Replace `score >= 10.0` AI gate with: `score >= MIN_AI_SCORE AND confluence_ok`. Add cache check. Cache the verdict with fingerprint. |
| `src/main.py` — inner loop | Allow AI call on inner ticks IF: confluence fires AND higher-TF data is fresh AND no cache hit. |
| `src/main.py` — state vars | Add `_ai_verdict_cache: dict`, `_macro_context_cache: dict`, `_outer_cycle_timestamp`, `_last_ai_call_time: dict` |
| `MASTER_RULES.md` | Update Rule 3: Claude is a market analyst + breakout validator at score ≥ 7, not a score-10 rubber stamp. |
| `.env.example` | Add all new config keys with documented defaults and comments. |

### Files that must NOT change

| File | Why |
|------|-----|
| `src/strategy.py` `compute_signal_score()` | Score system is sacred — Rule 1 |
| `src/strategy.py` `entry_confirmed()` | Still runs as a code gate after AI APPROVE |
| `src/risk_manager.py` | All 8 checks sacred — Rule 4 |
| `src/main.py` lines 58-79 | `_code_decide_direction` and `_code_compute_tpsl` — Rule 2 |
| `src/indicators/local_indicators.py` | Indicator math does not change. 30m uses the same `compute_all()` function. |
| `src/trade_state.py` | State machine does not change |

---

## 8. New Config Keys

| Key | Default | Purpose |
|-----|---------|---------|
| `MIN_AI_SCORE` | `7` | Score threshold that triggers Claude analysis. Must be ≥ `MIN_SIGNAL_SCORE`. Separate key — never merge. |
| `NEWS_FETCH_ENABLED` | `True` | Toggle macro news fetching. Set `False` for testnet or isolated environments. |
| `AI_MAX_TOKENS` | `4000` | Max output tokens for Claude's full market analysis. Allows complete chain-of-thought reasoning. |
| `AI_APPROVE_CACHE_MINUTES` | `60` | How long an APPROVE verdict stays cached before a fresh AI call is required. |
| `AI_REJECT_CACHE_MINUTES` | `30` | How long a REJECT verdict is cached. Shorter so the bot can re-evaluate after the risk event passes. |
| `CONFLUENCE_REQUIRE_30M` | `True` | Whether 30m timeframe is required in the confluence gate. Set `False` to revert to 4-timeframe check. |
| `AI_STALE_TF_MINUTES` | `55` | Max age of higher-TF data before AI call is blocked on inner ticks (waits for outer cycle refresh). |
| `MIN_AI_CALL_GAP_MINUTES` | `30` | Hard minimum gap between Claude calls per asset, regardless of fingerprint. Closes the score-drift loophole where a tick-up from 8.5→10 would bypass the fingerprint cache. |

> **Three separate score keys — NEVER merge any of them:**
> - `MIN_TRADE_SCORE` (0–5 int) — `entry_confirmed()` internal gate in `strategy.py`
> - `MIN_SIGNAL_SCORE` (0–10 float) — main loop pre-gate in `main.py`
> - `MIN_AI_SCORE` (0–10 float) — Claude analysis trigger in `main.py`
>
> Default all three to 7 initially. `MIN_AI_SCORE` can be raised to 8 to reduce AI call frequency.

---

## 9. Impact on MASTER_RULES.md

**Rule 1 — Score System Is Sacred:** No change. Score math is untouched.

**Rule 2 — Code Is Primary Decision Maker:** No change. Direction, TP, SL, size all still code.

**Rule 3 — Claude Role Is Fixed:** UPDATED.

Old Rule 3:
> Claude called ONLY when score == 10.0. Returns APPROVE or REJECT only. max_tokens=10.

New Rule 3:
> Claude called when score >= MIN_AI_SCORE (default 7), at most once per asset per outer
> cycle (1h). Claude analyzes macro conditions, volatility, and market environment. Returns
> APPROVE or REJECT (with optional brief reason, max_tokens=80). Claude never sets direction,
> TP, SL, or position size. Any output not starting with APPROVE = REJECT (fail closed).

**Rule 4 — Risk Management Is Fixed:** No change. All 8 checks remain mandatory.

---

## 10. Cost Estimate

| Scenario | Calls/day | Input tokens/call | Output tokens/call | Cost/day | Cost/month |
|----------|-----------|-------------------|-------------------|----------|-----------|
| Conservative (score ≥ 7 fires 30% of cycles, 2 assets) | ~14 | ~800 | ~40 | ~$0.012 | ~$0.36 |
| Moderate (fires 60% of cycles) | ~29 | ~800 | ~40 | ~$0.024 | ~$0.72 |
| Heavy (fires 90% of cycles) | ~43 | ~800 | ~40 | ~$0.037 | ~$1.10 |

Haiku pricing: input $0.0000008/token, output $0.000004/token.
All scenarios are under $2/month — very acceptable.

If cost exceeds $5/month, investigate — score ≥ 7 is firing too often, OR inner ticks are
calling Claude when they should be using cached verdicts.

---

## 11. What This Does NOT Change

- **Trade frequency**: Claude rejecting more setups means fewer trades — this is the goal.
  You want quality over quantity.
- **Direction logic**: Code still owns buy/sell entirely. Claude never suggests direction.
- **TP/SL math**: Code still owns ATR-based targets. Claude cannot override them.
- **Force-close logic**: If a position loses 8%, it is closed regardless of Claude's opinion.
- **TP/SL guardian**: Re-places missing orders regardless of Claude.
- **Time-based exit**: Closes stale trades regardless of Claude.

---

## 12. Open Questions — Decided

| Question | Decision |
|----------|---------|
| News API source | **Free RSS** — CoinDesk (crypto) + Forex Factory (economic calendar). No API key, no rate limit, 3-second timeout, graceful fallback to UNAVAILABLE. |
| Which model for analysis | **Claude Haiku** — same as before. Haiku is fast enough and cheap enough. Sonnet would give better reasoning but costs 5× more. Start with Haiku, upgrade if reasoning quality is poor. |
| Keep risk checks after Claude? | **YES** — Claude checks market environment, risk manager checks trade parameters. Two independent layers, neither replaces the other. |
| Score range that skips Claude? | **None** — all scores ≥ 7 go through Claude. Even score 10 goes through the market analysis. A perfect technical setup in a pre-CPI window is still a bad trade. |
| Max tokens | **4000** — lets Claude reason fully. Expected actual usage: 400-800 tokens per call. |
| 30m timeframe required? | **Yes** — adds meaningful intermediate confirmation. Uses existing `compute_all()`, just needs 30m candle fetch added to the gather block. |
| Inner ticks can call AI? | **Yes, if:** confluence fires AND higher-TF data < 55 min old AND no cache hit. This catches setups like the 1:45pm example. |
| REJECT cache duration | **30 minutes** — short enough that a REJECT before FOMC doesn't block the 3pm setup after FOMC resolves cleanly. |
| APPROVE cache duration | **60 minutes** — market conditions don't change that fast. One AI call per hour per asset maximum under normal conditions. |

---

## 13. Complete New Pipeline — Visual Summary

```
Every 5-minute tick (outer OR inner loop):
┌─────────────────────────────────────────────────────────────────────┐
│  CODE GATES (fast, no API calls)                                    │
│                                                                     │
│  1. Daily cap check            → skip if hit                        │
│  2. SL cooldown check          → skip if cooling                    │
│  3. _code_decide_direction()   → HOLD if trends conflict            │
│  4. compute_signal_score()     → HOLD if score < 7                  │
│  5. multi_timeframe_confluence()→ HOLD if any TF disagrees          │
│  6. _code_compute_tpsl()       → TP/SL set by code                  │
│  7. atr_position_size()        → size set by code                   │
│                                                                     │
│  8. Cache check                → if fresh APPROVE exists → use it   │
│                                   if fresh REJECT exists → HOLD     │
│                                   if stale or no cache → AI CALL ↓  │
└─────────────────────────────────────────────────────────────────────┘
                              │ AI CALL (rare)
┌─────────────────────────────▼───────────────────────────────────────┐
│  CLAUDE ANALYSIS (confirm_trade — max_tokens=4000)                  │
│                                                                     │
│  Receives: trade setup + 5-TF confluence data + volatility          │
│            + funding/OI + macro calendar + news headlines           │
│                                                                     │
│  Analyzes: breakout validity · macro risk · volatility regime       │
│            geopolitical risk · positioning extremes · timing risk   │
│                                                                     │
│  Returns:  full chain-of-thought analysis                           │
│            ending with VERDICT: APPROVE or VERDICT: REJECT          │
│                                                                     │
│  Cache result with fingerprint for 60 min (APPROVE) / 30 min (REJ) │
└─────────────────────────────────────────────────────────────────────┘
                              │ APPROVE only
┌─────────────────────────────▼───────────────────────────────────────┐
│  POST-APPROVE CODE GATES (non-bypassable)                           │
│                                                                     │
│  9.  State machine gate (IDLE only)                                 │
│  10. Trend guard (UNKNOWN blocks)                                   │
│  11. Inversion assert                                               │
│  12. Daily macro filter (1d trend + ADX)                            │
│  13. market_filter() — ATR spike + spread                           │
│  14. entry_confirmed() — 15m/5m + volume gate                       │
│  15. risk_mgr.validate_trade() — 8 hard checks                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │ All pass
┌─────────────────────────────▼───────────────────────────────────────┐
│  EXECUTE TRADE                                                      │
│  Market/limit order → fill confirmation → TP/SL orders → diary log  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 14. Fee-Adjusted TP/SL Formula

**Status: Confirmed — worth adding. Pure code change in `_code_compute_tpsl()`.**

### The current formula

```
TP = entry + 2 × ATR14
SL = entry − 1 × ATR14
```

The risk manager already enforces a *minimum* TP that covers fees (8th guard: TP ≥ 0.27% from
entry), but this is a post-hoc clamp. If ATR happens to produce a TP that barely passes the
0.27% minimum, the effective reward after fees is near zero. The formula itself is fee-blind.

### The problem this creates

Taker fee is 0.045% per side → 0.09% round-trip. For a 2×ATR TP to be genuinely profitable
after fees, the TP must exceed entry by *at least* ATR + fees. On low-volatility assets where
ATR is small relative to price, a 2×ATR TP might net almost nothing after fees.

Example: BTC at $94,000, ATR = $150 (0.16% of price).
- Current: TP = $94,300 → reward = 0.32% → after fees = 0.23%
- SL = $93,850 → risk = 0.16%
- Risk/reward after fees = 0.16% risk / 0.23% net reward = 0.70 — barely above 1:1 after fees

### The fix — bake fees into TP/SL computation

```
fee_buffer = entry × TAKER_FEE_PCT × 2   # round-trip (entry + exit)

TP = entry + (2 × ATR14) + fee_buffer    # TP must clear fees before profit starts
SL = entry − (1 × ATR14) − fee_buffer    # SL must account for exit fee on loss side
```

This guarantees that every trade, regardless of ATR magnitude, has a genuine reward *net of fees*
at the TP level. The risk manager's 0.27% minimum TP guard remains as a hard floor backstop.

**What this changes in practice:**
- On large-ATR assets (BTC, ETH with normal volatility): fee buffer is ~$85 on BTC → immaterial
- On low-ATR or low-price assets: fee buffer becomes more significant → correctly widens TP
- SL also widens slightly → fewer stop-hunts on tight SLs
- The risk manager's fee-coverage guard may now trigger less often since the formula pre-corrects

**File:** `src/main.py` → `_code_compute_tpsl()` only. No other file changes.

**No MASTER_RULES.md change required.** Rule 2 says code owns TP/SL — this stays true.
The formula is just more accurate. The `TAKER_FEE_PCT` config key already exists.

---

## 15. ADX-Based Position Size Reduction (Ranging Market Guard)

**Status: Confirmed — worth adding. Pure code change in sizing logic after `atr_position_size()`.**

### The current behaviour

`atr_position_size()` computes size based on the 1% ATR rule and scales by `score / 10`.
It does not reduce size for ranging (low-trend) markets. ADX is used only in `entry_confirmed()`
as a hard block below 20. Between ADX 20 and 25 (technically "emerging trend"), trades still
fire at full size.

### The problem this creates

A score of 8.5 can fire with ADX at 21 — technically above the block threshold, but the market
is barely trending. A full-size position in a ranging market risks getting chopped by a reversal
before TP is reached. The entry is allowed, but full risk allocation is excessive.

### The fix — half-size when ADX is weak AND score is not at maximum

```
Rule: if ADX_1h < 20 AND score < 9.0 → apply 50% size multiplier

# In the sizing step (after atr_position_size()):
adx_1h = asset_data.get("intraday_1h", {}).get("adx", 25)
if float(adx_1h) < 20 and score < 9.0:
    position_size *= 0.5
    logging.info(
        "[SIZE] %s — ADX %.1f < 20 + score %.1f < 9 → half-size applied",
        asset, adx_1h, score
    )
```

**Why `score < 9.0` is the exception boundary:**

Score 9 is mathematically unreachable (the scoring system has no path to 9 — see Section on
achievable scores). In practice this means the rule fires for ALL valid scores (7, 8, 8.5)
and is effectively "always apply in ranging markets." The `< 9.0` phrasing is intentional —
it future-proofs the rule if the scoring system ever changes, and communicates the design intent:
only an exceptionally high-confidence setup should escape the ranging-market size reduction.

**What this changes in practice:**
- ADX 20–25 zone: position size is halved. Trade can still fire.
- ADX < 20: `entry_confirmed()` already blocks entry entirely. This rule adds redundancy there.
- ADX ≥ 25: no change. Full size applies.
- No interaction with risk_manager guards — the half-size output is still passed through all
  8 risk checks. If the halved size is still too large for the account, the risk manager blocks it.

**Why NOT remove the ADX < 20 block in `entry_confirmed()`:**

The block in `entry_confirmed()` is a hard gate — it prevents entry in clearly ranging markets.
This sizing rule is a softer guard for the borderline zone (ADX 20–25). Both serve different
purposes and should coexist:

| ADX Range | `entry_confirmed()` | Sizing Rule | Net Effect |
|-----------|--------------------|-----------| -----------|
| < 20 | BLOCK entry | Half-size (redundant) | No trade |
| 20–25 | Allow entry | Half-size | Trade fires at 50% allocation |
| ≥ 25 | Allow entry | Full size | Trade fires at full allocation |

**File:** `src/main.py` — sizing step immediately after `atr_position_size()` call.
No changes to `risk_manager.py`, `strategy.py`, or any other file.

**New config key (optional):**

| Key | Default | Purpose |
|-----|---------|---------|
| `ADX_HALF_SIZE_THRESHOLD` | `20` | ADX level below which position size is halved (when score < 9). Matches the existing `entry_confirmed()` block threshold. |

---

## 16. Updated Files-That-Need-Changing (Final)

| File | What Changes |
|------|-------------|
| `src/config_loader.py` | Add: `MIN_AI_SCORE`, `NEWS_FETCH_ENABLED`, `AI_MAX_TOKENS`, `AI_REJECT_CACHE_MINUTES`, `AI_APPROVE_CACHE_MINUTES`, `CONFLUENCE_REQUIRE_30M`, `AI_STALE_TF_MINUTES`, `MIN_AI_CALL_GAP_MINUTES`, `ADX_HALF_SIZE_THRESHOLD` |
| `src/agent/decision_maker.py` | Rewrite `confirm_trade()`: new multi-section prompt, `max_tokens=4000`, parse for `VERDICT: APPROVE/REJECT`, log full analysis |
| `src/main.py` — new helpers | Add `multi_timeframe_confluence()`, `_build_confluence_fingerprint()`, `_fetch_macro_context()` |
| `src/main.py` — outer loop | Add 30m candle fetch to `asyncio.gather`. Build 30m indicators. Pass into market_sections. |
| `src/main.py` — pipeline | Replace `score >= 10.0` gate with `score >= MIN_AI_SCORE AND confluence_ok`. Add cache + gap checks. |
| `src/main.py` — inner loop | Allow AI call on inner ticks IF: confluence fires AND higher-TF data fresh AND no cache or gap block. |
| `src/main.py` — `_code_compute_tpsl()` | Add fee buffer: `TP += entry × TAKER_FEE_PCT × 2`, `SL -= entry × TAKER_FEE_PCT × 2` |
| `src/main.py` — sizing step | After `atr_position_size()`: if ADX_1h < `ADX_HALF_SIZE_THRESHOLD` AND score < 9.0 → multiply size by 0.5 |
| `src/main.py` — state vars | Add `_ai_verdict_cache: dict`, `_macro_context_cache: dict`, `_outer_cycle_timestamp`, `_last_ai_call_time: dict` |
| `MASTER_RULES.md` | Update Rule 3: Claude is market analyst + breakout validator at score ≥ 7, not a score-10 rubber stamp |
| `.env.example` | Add all new config keys with documented defaults |

### Files that must NOT change

| File | Why |
|------|-----|
| `src/strategy.py` `compute_signal_score()` | Score system sacred — Rule 1 |
| `src/strategy.py` `entry_confirmed()` | Still runs as code gate after AI APPROVE |
| `src/risk_manager.py` | All 8 checks sacred — Rule 4. Fee-coverage guard remains as backstop even after fee-adjusted TP formula. |
| `src/main.py` lines for `_code_decide_direction()` | Rule 2 — code owns direction |
| `src/indicators/local_indicators.py` | Indicator math unchanged. 30m uses existing `compute_all()`. |
| `src/trade_state.py` | State machine unchanged |

---

## 17. Complete New Pipeline — Final Visual Summary

```
Every 5-minute tick (outer OR inner loop):
┌─────────────────────────────────────────────────────────────────────┐
│  CODE GATES (fast, no API calls)                                    │
│                                                                     │
│  1. Daily cap check             → skip if hit                       │
│  2. SL cooldown check           → skip if cooling                   │
│  3. _code_decide_direction()    → HOLD if trends conflict           │
│  4. compute_signal_score()      → HOLD if score < MIN_SIGNAL_SCORE  │
│  5. multi_timeframe_confluence()→ HOLD if any TF disagrees          │
│  6. _code_compute_tpsl()        → TP/SL set by code                 │
│        └─ fee buffer baked in:  TP += 2×fee, SL -= fee              │
│  7. atr_position_size()         → base size set by code             │
│        └─ ADX guard: if ADX < 20 AND score < 9 → size × 0.5        │
│                                                                     │
│  8. Gap check  → if last AI call < MIN_AI_CALL_GAP_MINUTES → HOLD  │
│  9. Cache check→ if fresh APPROVE exists → use it                   │
│                  if fresh REJECT exists  → HOLD                     │
│                  if stale or no cache   → AI CALL ↓                 │
└─────────────────────────────────────────────────────────────────────┘
                              │ AI CALL (rare)
┌─────────────────────────────▼───────────────────────────────────────┐
│  CLAUDE ANALYSIS (confirm_trade — max_tokens=4000)                  │
│                                                                     │
│  Receives: trade setup + 5-TF confluence data + volatility          │
│            + funding/OI + macro calendar + news headlines           │
│                                                                     │
│  Analyzes: breakout validity · macro risk · volatility regime       │
│            geopolitical risk · positioning extremes · timing risk   │
│                                                                     │
│  Returns:  full chain-of-thought analysis                           │
│            ending with VERDICT: APPROVE or VERDICT: REJECT          │
│                                                                     │
│  Cache with fingerprint: 60 min (APPROVE) / 30 min (REJECT)        │
│  Update _last_ai_call_time[asset]                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │ APPROVE only
┌─────────────────────────────▼───────────────────────────────────────┐
│  POST-APPROVE CODE GATES (non-bypassable)                           │
│                                                                     │
│  10. State machine gate (IDLE only)                                 │
│  11. Trend guard (UNKNOWN blocks)                                   │
│  12. Inversion assert                                               │
│  13. Daily macro filter (1d trend + ADX)                            │
│  14. market_filter() — ATR spike + spread                           │
│  15. entry_confirmed() — 15m/5m RSI/ADX/volume/near-EMA            │
│  16. risk_mgr.validate_trade() — 8 hard checks                      │
│        (fee-coverage guard remains as backstop even after           │
│         fee-adjusted TP — belt-and-suspenders safety)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │ All pass
┌─────────────────────────────▼───────────────────────────────────────┐
│  EXECUTE TRADE                                                      │
│  Market/limit order → fill confirmation → TP/SL orders → diary log  │
└─────────────────────────────────────────────────────────────────────┘
```

---

*This document is FINAL — all design decisions confirmed.*
*No code has been modified. This is analysis and planning only.*
*Next step: implementation begins file-by-file, starting with `src/config_loader.py`.*
