# 🏛️ TRADING BOT BUG COUNCIL — V5 FINAL REPORT

**Scan date:** 2026-05-10
**Scope:** Full re-scan of every file after VS Code applied all v4 fixes. Deep double-check of `dashboard.html`.
**Files read (line-by-line):** `dashboard.html` (459 lines) · `src/main.py` (2165+ lines) · `src/agent/decision_maker.py` · `src/strategy.py` · `src/risk_manager.py` · `src/trade_state.py` · `src/trading/hyperliquid_api.py` · `src/indicators/local_indicators.py` · `src/config_loader.py` · `src/utils/prompt_utils.py` · `Dockerfile` · `MASTER_RULES.md` · `CLAUDE.md`
**Bot:** Hyperliquid perpetual futures · Code-first hybrid · Score-gated · Claude as APPROVE/REJECT gate

---

## ✅ V4 FIX VERIFICATION — ALL 10 CONFIRMED

Every fix applied since the v4 report was verified against live source code.

| ID | Description | Status | Evidence |
|----|-------------|--------|----------|
| V4-HIGH-1 | No API auth on port 3000 | ✅ FIXED | `auth_middleware` added; `API_HOST` defaults to `127.0.0.1` (localhost-only bind) |
| V4-HIGH-2 | `save_active_trades()` non-atomic write | ✅ FIXED | Now uses `.tmp` + `os.replace()` — matches `_save()` pattern |
| V4-HIGH-3 | Hardcoded IP + CORS mismatch | ✅ FIXED | `const API = ''`; CORS reflects localhost origins; hardcoded IP removed |
| V4-MEDIUM-1 | `togR()` onclick breaks on `"` in reasoning | ✅ FIXED | `data-full` + `encodeURIComponent`; `togR(btn)` reads `dataset` |
| V4-MEDIUM-2 | Diary fetches 5000 entries every 60s | ✅ FIXED | `limit=200`; server default also set to 200 |
| V4-MEDIUM-3 | P&L includes unrealized, no disclosure | ✅ FIXED | `*` suffix added; footnote note appended (minor rendering issue — see V5-MEDIUM-2) |
| V4-LOW-1 | Dockerfile missing `web3`, `rich` | ✅ FIXED | Both packages added to Dockerfile |
| V4-LOW-2 | No HEALTHCHECK in Dockerfile | ✅ FIXED | `HEALTHCHECK` polling `http://localhost:3000/` added |
| V4-LOW-3 | "Active Since" resets on diary rotation | ✅ FIXED | `bot_started.json` written at first boot; `/meta` endpoint exposes it; `checkAuth()` reads it |
| V4-LOW-4 | "Invalid Date" rendered for malformed timestamps | ✅ FIXED | `safeDate()` helper added; used in `renderFills()` and `renderAI()` |

---

## 🔴 NEW BUGS — COUNCIL FINDINGS

### 🔨 AGENT 1 — LOGIC BREAKER

**V5-MEDIUM-1 · Daily macro filter activates when daily ADX is `None` — logic inverted from comment**
**File:** `src/main.py` · Lines 1469, 1861 (outer and inner loop)

```python
_macro_trending = _adx_1d is None or float(_adx_1d) > 20
```

The intent per the inline comment is: *"Only applied when the daily trend has actual momentum (ADX > 20). A near-cross with low ADX is a ranging market — blocking all longs or shorts would unnecessarily suppress valid intraday setups."*

The implementation is the opposite for the `None` case: when `_adx_1d` is `None` (daily ADX not yet computable — startup, data gap), the condition evaluates `True` and the macro filter **fires**. The comment says the filter should only fire when ADX is confirmed above 20.

**Practical scenario:** At first bot startup, daily EMA is computed from the first available 1d candles. EMA needs 50 candles. ADX needs additional computation on top of that. There is a brief startup window where `trend_1d` is `"BEARISH"` or `"BULLISH"` but `_adx_1d` is `None`. During this window, buys against a BEARISH daily trend, or sells against a BULLISH daily trend, are blocked — which is actually conservative and arguably correct. However, it directly contradicts the stated intent ("only apply when ADX > 20").

**Financial consequence:** Missed entries at startup. No fund loss. Low practical impact since EMA and ADX share the same candle batch and both are usually available together. However the code does not match its own documentation, creating a future maintenance hazard.

**Fix:** Change both instances to:
```python
_macro_trending = _adx_1d is not None and float(_adx_1d) > 20
```

---

### 👤 AGENT 5 — OPERATOR SAFETY

**No new critical or high operator-safety issues found.** The auth middleware, localhost-only API bind, and DASHBOARD_TOKEN env var together provide a solid baseline. The KILLSWITCH (file + signal), circuit breaker, and drawdown halt all remain intact from prior scans.

---

### 🌪️ AGENT 3 — CHAOS TESTER

**No new crash-risk or state-corruption paths found.** The `save_active_trades()` atomic fix eliminates the OOM double-position risk. The inner loop's `_daily_trade_count` is correctly incremented (line 1931). The state gate prevents double-entry within a cycle. The fill deduplication `_seen_fill_tids` set is correctly scoped per-order. All 3 fill-poll attempts complete regardless of early fill detection.

---

### ⚡ AGENT 4 — PERFORMANCE ENGINEER

**No new blocking or memory-growth issues.** The diary limit=200 fix eliminates the 25MB per-minute payload. `handle_diary()` reads `decisions.jsonl` synchronously inside the async handler — this was pre-existing and low risk at 200 entries.

---

### 🔐 AGENT 2 — SECURITY ANALYST

**V5-LOW-1 · CORS origin check matches substrings — `localhost.attacker.com` would pass**
**File:** `src/main.py` · Lines 292-293

```python
if "127.0.0.1" in _origin or "localhost" in _origin:
    _allowed = _origin
```

This reflects ANY origin that contains the substring `"localhost"` — including crafted origins like `http://localhost.attacker.com`. If an attacker serves a page at such a domain and the operator visits it, that page can send credentialed requests to the bot API from within the browser.

**Mitigated by:** `API_HOST=127.0.0.1` default — the API only accepts connections from the local machine. An external page at `localhost.attacker.com` cannot reach `http://localhost:3000` unless the browser itself is running on the same machine as the bot (which is unusual for server deployments). For desktop deployments where operator and bot share the same machine, this is a real but low-probability vector.

**Fix:** Use exact origin matching:
```python
_allowed_origins = {f"http://localhost:{_port}", f"http://127.0.0.1:{_port}"}
if _origin in _allowed_origins:
    _allowed = _origin
```

---

## 🟡 MEDIUM DASHBOARD BUGS

### V5-MEDIUM-2 · P&L footnote appended to wrong DOM node — renders inside value cell
**File:** `dashboard.html` · Lines 316-323

```javascript
const pnlSection = document.getElementById('p0')?.closest('section,div[id]');
if(pnlSection && !pnlSection.querySelector('.pnl-note')) {
    ...
    pnlSection.appendChild(note);
}
```

`document.getElementById('p0')` returns the `<div class="pnl-usd" id="p0">` element. `closest('section,div[id]')` searches upward for a `<section>` or `<div>` with ANY id attribute — and since `#p0` itself is a `div` with `id="p0"`, `.closest()` returns `#p0` itself immediately (the element is always its own closest ancestor matching the selector).

The footnote div is therefore appended **inside** the `#p0` value cell. On the very next `renderPnL()` call, `u.textContent = ...` (line 300) clears all children of `#p0`, removing the footnote. The `pnl-note` check then finds no existing note and appends it again, only for it to be wiped next call.

**Consequence:** The `* Includes unrealized mark-to-market P&L` footnote never actually renders for the operator. The `*` suffix on the value IS visible (it's in `textContent`), but it has no explanation since the footnote div is never stably present. Operators see an unexplained `*` with no context.

**Fix:** Use an id-less parent or a direct container reference:
```javascript
const pnlSection = document.querySelector('.pnl-row');
// or: const pnlSection = document.getElementById('p0')?.parentElement?.parentElement;
```
And add `id="pnl-row"` to the `.pnl-row` div for a cleaner selector.

---

## 🔵 LOW / INFORMATIONAL

### V5-LOW-2 · Seven dead config keys in `config_loader.py` confuse operator tuning
**File:** `src/config_loader.py` · Lines 81-89

These keys are defined in `CONFIG` but never read by any active code path (`grep -r "CONFIG.get" src/` confirms zero usage):

| Key | Env Var | Comment |
|-----|---------|---------|
| `llm_model` | `LLM_MODEL` | Model hardcoded to `claude-haiku-4-5-20251001` in `confirm_trade()` |
| `sanitize_model` | `SANITIZE_MODEL` | Legacy from removed `_decide()` |
| `max_tokens` | `MAX_TOKENS` | `confirm_trade()` uses `ai_max_tokens` |
| `enable_tool_calling` | `ENABLE_TOOL_CALLING` | Feature never implemented |
| `max_tool_iterations` | `MAX_TOOL_ITERATIONS` | Feature never implemented |
| `thinking_enabled` | `THINKING_ENABLED` | Feature never implemented |
| `thinking_budget_tokens` | `THINKING_BUDGET_TOKENS` | Feature never implemented |

An operator reading `.env.example` or `config_loader.py` and setting `ENABLE_TOOL_CALLING=true` or `LLM_MODEL=claude-opus-4-6` would have zero effect on bot behavior — no error, no warning.

**Fix:** Remove these 7 keys from `CONFIG`. If tool-calling or extended thinking features are planned, add them when implemented.

---

### V5-LOW-3 · `bot_started.json` written with non-atomic `write_text()` — inconsistent with codebase pattern
**File:** `src/main.py` · Lines 2141-2144

```python
_started_path.write_text(
    json.dumps({"started_at": datetime.now(timezone.utc).isoformat()}),
    encoding="utf-8",
)
```

All other persistent writes in the codebase use `.tmp` + `os.replace()`. This write is tiny (~60 bytes) and only happens once (guarded by `if not _started_path.exists()`), making mid-write corruption extremely unlikely. However it's stylistically inconsistent and worth aligning for completeness.

**Financial consequence:** None — only affects the "Active Since" dashboard display.

---

## 📊 CONFIRMED CLEAN — DEEP RE-SCAN NO ISSUES FOUND

**`dashboard.html` (full re-read):**
- Auth flow: `checkAuth()` → `apiFetch('/meta')` → 401 → token prompt → localStorage — correct
- `apiFetch()` wrapper correctly injects `Authorization: Bearer` on all requests
- `safeDate()` helper correctly returns `'—'` for null/NaN timestamps — used in both `renderFills()` and `renderAI()`
- `togR(btn)` uses `decodeURIComponent(btn.dataset.full)` — immune to double-quote attribute breakage
- Chart downsampling: `pts.length > 250 → filter by stride` — correct
- `renderPositions()`, `renderOrders()`, `renderFills()` — all use live data, no XSS via `textContent`
- Badge injection (`${d.asset||'?'}`) — values come from bot config (ASSETS env), not user input; acceptable

**`src/main.py` trading loop (full re-read):**
- `_code_decide_direction()` — correct 4h EMA gate, returns `None` for conflict
- `_code_compute_tpsl()` — `TP = entry + 2×ATR, SL = entry − 1×ATR (buy)` with fee buffer — correct
- `multi_timeframe_confluence()` — optional 30m gate via `confluence_require_30m` config — correct
- ADX half-size guard (lines 1287-1292): `adx_1h < threshold AND score < 9 → 0.5×` — correct
- Daily trade cap checked in both outer (line 1247) and inner (line 1728) loops — correct
- `_daily_trade_count` incremented in both outer (1638) and inner (1931) — correct
- SL cooldown map (`_sl_cooldown_map`) checked in both loops — correct
- State gate (`ENTERED`/`COOLDOWN`) checked before every entry in both loops — correct
- KILLSWITCH: file-based (line 617) and signal-based (lines 2156-2157) — both wired correctly
- `global _shutdown` declaration in `run_loop()` at line 373 — confirmed, V3-CRITICAL-1 still fixed
- `_rotate_if_needed()` called for diary, decisions, llm_requests.log, prompts.log — all 4 covered
- Fill deduplication via `_seen_fill_tids` set — correct, no double-counting
- `_can_place_tpsl = filled_qty > 0 or order_type != "limit"` — correct semantics for market vs limit
- Inner loop market_filter + entry_confirmed + risk_mgr.validate_trade all wired — correct
- C-3/C-4 account/price refresh before inner execution — confirmed present

**`src/risk_manager.py`:** All 8 checks intact. Circuit breaker atomic write. Dangerous-defaults startup checker. ✅

**`src/trade_state.py`:** Both `_save()` and `save_active_trades()` now atomic. State machine transitions correct. ✅

**`src/strategy.py`:** Score system 0-10 unchanged. MASTER RULE 1 intact. ✅

**`src/agent/decision_maker.py`:** `confirm_trade()` correct. Dead code (`_decide()`) removed. ✅

**`src/indicators/local_indicators.py`:** All indicators correct. ✅

**`Dockerfile`:** `web3`, `rich` added. HEALTHCHECK present. `API_HOST=127.0.0.1` not needed in Dockerfile (set in .env or config_loader). ✅

---

## 📊 V5 FINDINGS TABLE

| ID | Sev | File | Lines | Issue | Financial Impact |
|----|-----|------|-------|-------|-----------------|
| V5-MEDIUM-1 | 🟡 MEDIUM | main.py | 1469, 1861 | Daily ADX=None → macro filter fires (code contradicts comment) | Missed entries at startup only; no fund loss |
| V5-MEDIUM-2 | 🟡 MEDIUM | dashboard.html | 316-323 | P&L footnote appended inside value cell — never renders stably | Operator sees `*` with no explanation; medium monitoring confusion |
| V5-LOW-1 | 🔵 LOW | main.py | 292-293 | CORS reflects any origin containing "localhost" substring | Theoretical: crafted origin passes; mitigated by 127.0.0.1 bind |
| V5-LOW-2 | 🔵 LOW | config_loader.py | 81-89 | 7 dead config keys (LLM_MODEL, ENABLE_TOOL_CALLING, etc.) | Operator confusion only; setting these has zero effect |
| V5-LOW-3 | 🔵 LOW | main.py | 2141-2144 | `bot_started.json` written non-atomically | Affects only "Active Since" display; no trading impact |

---

## 🏛️ COUNCIL VERDICT

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     🏛️  TRADING BOT BUG COUNCIL — V5 FINAL REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 BOT SUMMARY
   Strategy:      Code-first hybrid perpetual futures
   Exchange:      Hyperliquid (mainnet)
   Leverage:      3× configured (10× permissive default without .env)
   Order types:   Market entry + reduce-only TP/SL trigger orders
   Risk controls: ATR 1% rule · 8-check risk manager · KILLSWITCH ·
                  Daily cap · Cooldown · Concurrent position limit ·
                  Drawdown circuit breaker · Daily macro trend filter ·
                  ADX half-size guard
   Claude role:   APPROVE/REJECT only on score ≥ MIN_AI_SCORE setups
   Dashboard:     Auth-gated · 127.0.0.1 only · 200-entry diary · Fixed

✅ ALL V4 FIXES VERIFIED: 10 for 10 confirmed in live source code.

📊 V5 FINDINGS SUMMARY:
   CRITICAL:  0
   HIGH:      0
   MEDIUM:    2  (one logic/comment mismatch · one dashboard DOM bug)
   LOW:       3  (CORS substring · dead config keys · non-atomic small write)

🔴 TOP 3 TO FIX:
  1. V5-MEDIUM-1 — Fix daily macro filter ADX=None logic
     Change: `_adx_1d is None or float(_adx_1d) > 20`
       → `_adx_1d is not None and float(_adx_1d) > 20`
     (Appears at lines 1469 AND 1861 — fix both)

  2. V5-MEDIUM-2 — Fix P&L footnote DOM insertion
     `closest('section,div[id]')` selects #p0 itself → appends inside value cell.
     Change to `document.querySelector('.pnl-row')` as the container.

  3. V5-LOW-2 — Remove 7 dead config keys from config_loader.py
     Prevents operators from thinking LLM_MODEL or ENABLE_TOOL_CALLING do anything.

CAPITAL RECOMMENDATION: Bot is in the best shape it has ever been.
  Trading loop: ✅ no exploitable bugs
  Dashboard:    ✅ auth, localhost-only, correct data — 2 cosmetic/minor fixes remain
  Docker:       ✅ packages, health check, all present

GO LIVE STATUS: ✅ READY
  No blocking bugs remain. The 2 medium findings are comment-code mismatches
  and a dashboard cosmetic issue — neither affects trade execution or fund safety.
  Apply the 3 fixes above at next maintenance window.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
