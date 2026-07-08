# Strategy Tournament — Design Document

**Status:** implemented alongside this document; see "Implementation plan" at the bottom
for the module map. Research/educational tool only — backtesting only, no live trading,
no broker integration, no order placement.

**Prime directive:** do not optimize for the highest backtest return. Optimize for
robustness, risk-adjusted performance, drawdown control, repeatability, and honest
evidence quality. Every "great" result is treated as suspicious until proven otherwise.

---

## 1. Current architecture summary (audit)

The repo is a per-sleeve, day-by-day event-driven backtester:

- **`src/engine.py`** — walks one strategy through a canonical trading calendar one day
  at a time. Fills scheduled `TargetEvent`s at that day's close, marks to market, then
  asks the strategy for new events. Sizing (`requested_notional = target_weight x
  sleeve_equity`) is frozen at signal time; only the share count is resolved at the
  next day's fill. Average-cost accounting, cash-constrained proportional buy scaling,
  gross + net-of-cost realized P&L, duplicate/fill-date/calendar event validation.
- **`src/market_view.py`** — `MarketDataView`, the restricted accessor handed to
  strategies. Any read dated after the walk date raises `LookaheadError`. This makes
  lookahead structurally impossible for strategy code, not just discouraged.
- **`src/strategies/base.py`** — `Strategy` ABC: `reset()`, `prepare()` (vectorized
  indicator precompute), `initial_events()` (pre-start warmup decisions),
  `on_day()` (daily decisions).
- **Existing strategies:**
  - `mean_reversion` — RSI < 35 oversold entries on ~25 hardcoded large caps, exits on
    stop/timeout/SMA/RSI, max 5 slots at 20% each.
  - `sector_rotation` — monthly top-K (K=2) of the 11 SPDR sector ETFs by trailing
    3-completed-calendar-month return, equal weight.
- **`src/data.py`** — yfinance fetch + metadata-keyed CSV cache + canonical calendar
  (intersection-based, no forward-fill; gapped tickers dropped or hard-fail by policy).
- **`src/metrics.py`** — total/annual/monthly returns, CAGR, max DD + duration, Sharpe,
  Sortino, Calmar, net & gross win rate, turnover, cost drag, exposure
  (days-with-any-position, average-capital-invested), benchmark (SPY) comparison.
- **`src/universe.py`** — optional `us_50b` mean-reversion universe (current US-listed
  common stock >= $50B, Nasdaq-Trader-gated, window-aware price-data validation).
  Explicitly documented as survivorship-biased and current-snapshot.
- **`src/robustness.py` + `--strategy robustness`** — 5 fixed windows x 5 allocation
  mixes (sector/mean-reversion blends) vs SPY, with average-rank and beats-SPY-fraction
  aggregation.
- **Reporting** — timestamped run dirs with target_events/transactions/trades/positions
  CSVs, metrics.json, report.txt, equity PNG; comparison and robustness report writers.

### Current strategy limitations (audit findings)

1. **Only two strategy families**, hardwired into the CLI (`mean_reversion`,
   `sector_rotation`, plus blends of exactly those two). No shared registry; adding a
   strategy means editing several call sites.
2. **In-sample-tuned defaults.** `config.py` comments record that RSI entry was moved
   35 <- 30 "for more aggressive entry in bull markets", SMA 30 <- 50 "for faster exits",
   max holding 10 <- 20, sector top-K 2 <- 3 "for more concentrated momentum bet", and
   git history contains "Improve strategy parameters to beat SPY performance". The README
   discloses this, but it means the shipped defaults are, to an unknown degree, fitted to
   one historical stretch. The tournament must therefore judge strategies across multiple
   windows, and parameter-sensitivity reporting must show how fragile these choices are.
3. **No trend/momentum family on individual stocks**, no regime awareness anywhere: mean
   reversion happily buys falling knives in a 2008/2022-style downtrend; sector rotation
   stays 100% invested in the "least bad" sectors during a broad bear market.
4. **No cross-strategy fairness harness**: `compare` mode compares exactly
   mean_reversion/sector_rotation/both; costs, windows, and universes are aligned there,
   but nothing generalizes to N strategies.
5. **Survivorship bias in every stock universe** (default hardcoded list and `us_50b`
   both current-snapshot). Documented loudly, but no point-in-time universe exists — and
   none is feasible with free data (see infeasible list).
6. **Execution realism limits**: close-to-close fills, one-day lag, no intraday stops, no
   spreads/impact. Fine for research ranking; not a claim of live performance.

---

## 2. Strategy families

### Feasible with current free data (daily OHLCV from Yahoo)

| Family | Implemented as | Notes |
|---|---|---|
| Mean reversion (baseline) | `mean_reversion` (existing, untouched) | Anchor/control for the tournament. |
| Improved mean reversion | `mean_reversion_filtered` (new) | Same rules + regime gate + falling-knife guard. Differs by FILTERS, not by re-tuned thresholds. |
| Momentum / trend following | `momentum` (new) | Classic 6-month cross-sectional momentum with a 200-day trend + absolute-momentum filter, monthly rebalance. |
| Sector rotation | `sector_rotation` (existing, untouched) | Second control. |
| Regime-aware hybrid | `regime_switch` (new) | Risk-on: sector rotation's exact ranking. Risk-off (SPY < 200-day SMA): 100% cash. |

- **Breakout** (e.g. 52-week-high breakout with ATR stops) is feasible with daily OHLCV
  but overlaps heavily with the momentum family (both are trend-continuation bets, and
  52-week-high proximity is itself a momentum proxy). To keep the tournament small,
  honest, and interpretable, it is **deliberately deferred** — a candidate for a future
  batch if momentum earns further research.

### Infeasible without paid / fundamental / point-in-time data (not implemented)

- **Earnings drift (PEAD)** — needs reliable historical earnings dates AND surprise
  magnitudes. Yahoo's free earnings calendar is spotty/unversioned historically;
  building this on it would produce silently wrong event dates. Rejected as fragile.
- **Quality / low-volatility factor portfolios** — quality needs fundamentals
  (ROE, accruals, leverage) with point-in-time correctness (as-reported, not restated).
  Free sources restate history. Low-vol alone is feasible on prices but on a 25-stock
  universe it degenerates into "hold the 5 least volatile mega-caps" — a portfolio too
  small/correlated to say anything; deferred until a broader universe with survivorship
  handling exists.
- **Any point-in-time index-constituent universe** — historical S&P 500 membership
  lists are licensed data. Without them, every stock universe here remains
  survivorship-biased, and every report says so.

---

## 3. New strategy specifications

Parameter-choice principle: every new parameter is either (a) one of the most-cited,
decades-old conventions in the literature (chosen for canonical status, NOT tuned on
this repo's data), or (b) a round, severe threshold documented as a heuristic. The
parameter-sensitivity report (section 6) shows how results move when each is nudged —
the point is disclosure of fragility, never selection of the best variant.

### 3.1 `momentum` — cross-sectional trend/momentum on large caps

- **Universe:** the same mean-reversion universe (default hardcoded 25; `--universe
  us_50b` supported). Same survivorship warning applies and is printed.
- **Signal (month-end only, same month-end mechanism as sector rotation):**
  trailing 126-trading-day (~6-month) total return, per ticker.
  6-month momentum is the canonical Jegadeesh–Titman horizon; 200-day SMA is the
  canonical long-term trend line (Faber's timing work) — both predate this repo by
  decades.
- **Eligibility filter at signal time:** close > 200-day SMA (trend filter) AND
  trailing 126-day return > 0 (absolute momentum). Both must hold.
- **Entry/rebalance rule:** hold the top `K=5` eligible tickers by trailing return,
  equal-weighted at 1/K each (matches mean reversion's 5x20% slots for comparability).
  Full rebalance each month-end; unchanged holdings re-emit targets so price drift is
  corrected (same convention as sector rotation).
- **Exit rule:** a holding that drops out of the top-K or fails either filter gets
  target weight 0 at the next monthly rebalance. No intramonth exits — this is
  deliberately the simplest possible implementation; adding daily stops would be a
  second (confounded) change.
- **Defensive cash fallback:** if only m < K tickers are eligible, the remaining
  (K-m)/K of the sleeve stays in cash. In a broad downtrend the strategy de-risks by
  construction rather than being forced fully invested.
- **Position sizing:** fixed equal weight 1/K of sleeve equity, sized at signal time
  (engine convention). No leverage, no averaging down.
- **Risk controls:** trend + absolute-momentum filters (the primary control), cash
  fallback, max 5 concurrent names, monthly cadence caps turnover.
- **Warmup:** 200-day SMA + 126-day lookback needs ~330 trading days; fetch warmup is
  500 calendar days. Tickers with insufficient history by the first walk day are
  dropped with a printed warning (same policy as mean reversion).

### 3.2 `mean_reversion_filtered` — baseline mean reversion + two filters

Subclasses the baseline strategy; entry/exit thresholds, sizing, slots, stop, timeout
are **inherited unchanged**. The difference is exactly two new entry-eligibility
filters (exits are never filtered — risk-reducing exits must not be blocked):

1. **Market-regime gate:** no NEW entries while SPY closes below its 200-day SMA.
   Financial intuition: buying oversold stocks works when dips are noise around an
   uptrend; in a downtrend "oversold" keeps getting more oversold (2008, 2022).
2. **Falling-knife guard:** skip a candidate whose trailing 5-trading-day return is
   <= -15%. RSI < 35 after an orderly pullback is a dip; -15% in a week is an event
   (earnings blowup, fraud, guidance cut), and mean reversion has no edge there.
   -15%/5d is a round, severe, disclosed heuristic — not fitted.

Implementation detail: the baseline class gains one overridable hook,
`_entry_allowed(ticker, day, market) -> bool`, which returns `True` in the baseline
(regression-tested to produce byte-identical events). The subclass overrides only that
hook. SPY is declared as a **signal-only ticker** (`signal_tickers = ["SPY"]`):
fetched for data, structurally never tradable (it is not in `self.universe`, which is
the only set the strategy iterates for entries/exits).

### 3.3 `regime_switch` — regime-aware hybrid (sector rotation <-> cash)

- **Risk-on** (SPY close > 200-day SMA at the month-end signal date): behave exactly
  like sector rotation — same trailing-3-month ranking code path (subclass, not copy),
  same top-K=2, same equal weights.
- **Risk-off** (SPY <= 200-day SMA): target weight 0 on all 11 ETFs — 100% cash. Cash
  earns nothing (conservative; no synthetic T-bill yield is credited, disclosed).
- **Cadence:** month-end only, same as sector rotation — the regime is NOT re-checked
  intramonth (simple, auditable; avoids whipsaw trading and hidden extra turnover).
- **No new tuned parameters:** reuses sector rotation's lookback/top-K unchanged and
  the same shared 200-day regime SMA used by `mean_reversion_filtered`
  (`config.REGIME_SMA_PERIOD`, one shared constant so there is exactly one regime
  definition in the codebase).
- **Assumption stated in every report:** the 200-day regime filter is a blunt
  instrument — it sacrifices upside around V-bottoms (2020) to avoid deep sustained
  bears (2008/2022). The tournament windows are chosen to show both cases honestly.

---

## 4. Tournament infrastructure

- **Registry** (`src/tournament.py`): `STRATEGY_REGISTRY` mapping name -> factory +
  data plan (warmup days, gap policy, whether the stock universe applies, signal
  tickers). Adding a future strategy = one registry entry + its class + tests.
- **Fairness contract:** every strategy in one tournament run gets the same capital,
  the same `--cost-bps`, the same requested window, the same canonical-calendar
  construction, the same benchmark (SPY over each strategy's own effective range,
  plus a SPY row over the common intersection), and writes the same artifact set.
  Stock strategies share the same resolved universe (default or `us_50b`).
- **`mean_reversion` and `sector_rotation` keep their existing sleeve runners** —
  tournament mode calls them unchanged, so tournament results for the two incumbents
  are identical to what `--strategy mean_reversion`/`sector_rotation` produce.
- **CLI:** `--strategy tournament`, with `--tournament-strategies` (default: all
  registered), `--tournament-windows` (default: the single `--start`/`--end` window;
  `regimes` expands to the named regime windows below; or explicit
  `start:end,start:end,...`), `--tournament-cost-bps-list` (cost sensitivity),
  `--tournament-param-sensitivity` (opt-in small variant sweep). Existing CLI modes
  and defaults are untouched.
- **Report** (`write_tournament_report`): per-window comparison table (all required
  metrics), assumptions/warnings block per strategy (from `Strategy.describe()`),
  cross-window robustness section, cost-sensitivity table, parameter-sensitivity
  table, and the standard disclaimer/execution-model/pre-tax/survivorship warnings.

### Required metrics (per strategy, per window)

total return, CAGR, max drawdown (+ duration), Sharpe, Sortino, Calmar, net win rate
(+ gross), round-trip trade count, transactions, turnover ($ and cost drag %), time in
market (days-with-any-position % and average-capital-invested %), best/worst month,
best/worst calendar year (when the window spans >= 2 years), SPY comparison (total
return, CAGR, max DD, excess return, correlation), and — across windows — the
robustness components below.

### Robustness score (disclosed formula, reported with its components)

For each strategy across the tested windows:

- `pct_windows_beats_spy_return` — fraction of windows with total return > SPY's.
- `pct_windows_positive_return` — fraction of windows with total return > 0.
- `worst_window_max_drawdown` — the worst max-drawdown across windows.
- `return_dispersion` — std-dev of window total returns (consistency).
- **`robustness_score` = mean(pct_windows_beats_spy_return,
  pct_windows_positive_return)** — deliberately simple and stated everywhere it
  appears; the components are always printed next to it so nobody has to trust the
  composite.

---

## 5. Bias risks and mitigations

| Risk | Mitigation |
|---|---|
| Lookahead | `MarketDataView` raises on any future read (existing, structural). Every new strategy gets mutate-all-future-dates decision-invariance tests. |
| Survivorship | Unfixable with free data; every stock-universe report prints the warning; the tournament never claims validated alpha, only relative mechanics. |
| In-sample tuning | New params are canonical-convention or disclosed heuristics; incumbent defaults' tuning history is called out in reports; multi-window + sensitivity reporting shows fragility instead of hiding it. |
| Cherry-picked window | Default full window PLUS named regime windows (bull/bear/choppy); robustness score aggregates across them. |
| Cost blindness | Cost-sensitivity mode reruns at 0/5/10/20 bps; excess-return sign flips are flagged. |
| Overfit parameter sweeps | Sensitivity variants are few (<= 4 per strategy), fixed in code with justification comments, and NEVER auto-selected. |
| Execution realism | Close-to-close + 1-day lag disclosed in every report (existing line); no intraday stops claimed. |
| Tax drag | Pre-tax warning in every report (existing); mean-reversion-family strategies are short-term-gains heavy — stated in describe(). |

### Named regime windows (for `--tournament-windows regimes`)

- `2019_bull` 2019-01-01..2019-12-31 — steady uptrend.
- `2020_covid` 2020-01-01..2020-12-31 — crash + V-recovery (regime filters look worst here).
- `2021_bull` 2021-01-01..2021-12-31 — low-vol melt-up.
- `2022_bear` 2022-01-01..2022-12-31 — sustained bear (regime filters should earn their keep).
- `2023_2024_bull` 2023-01-01..2024-12-31 — recovery bull with chop.
- `2019_2024_full` 2019-01-01..2024-12-31 — the whole span (in-sample for incumbents' tuned defaults; treated as such).

These labels describe consensus market character, chosen before looking at any
tournament result. ETF-history constraints (XLC inception 2018 + 3-month lookback)
make earlier windows infeasible for the sector strategies.

---

## 6. Data limitations

- Yahoo daily OHLCV, auto-adjusted; synthetic adjusted prices, not tradable prints.
- No point-in-time universes; both stock universes are current snapshots.
- No intraday data — stops are delayed close-to-close exit rules.
- Cash earns 0% — mildly punishes cash-heavy regimes (regime_switch, momentum
  fallback); conservative direction, disclosed.
- This development sandbox has no market-data egress; all live-data validation runs
  must be executed locally with the commands in the README/PR.

## 7. Success criteria

A strategy is tournament-"strong" only if ALL hold across the tested windows:
1. Beats SPY total return in a majority of windows (the account's goal is to beat
   buy-and-hold SPY — otherwise the $15k belongs in an index fund),
2. with max drawdown no worse than SPY's in most windows,
3. robustness_score >= 0.5 with return_dispersion not dominated by one lucky window,
4. survives 4x the default cost assumption (20 bps) with positive excess return,
5. and its parameter-sensitivity spread does not change the conclusion (a variant
   flip from beat-SPY to lose-to-SPY = fragile = fail).

Anything less is "paper-trade only" or "abandon". Final verdicts live in
`docs/RED_TEAM.md` and must be re-validated on live local runs.

## 8. Implementation plan (module map)

1. `src/config.py` — new constants only (momentum/regime/knife params), nothing changed.
2. `src/strategies/base.py` — additive: `signal_tickers`, `family`, `describe()`.
3. `src/strategies/mean_reversion.py` — additive `_entry_allowed` hook (baseline returns True; regression-tested identical).
4. `src/strategies/momentum.py`, `src/strategies/mean_reversion_filtered.py`, `src/strategies/regime_switch.py` — new.
5. `src/tournament.py` — registry, generic sleeve runner, multi-window/cost/param-sensitivity orchestration, robustness scoring.
6. `src/reporting.py` — `write_tournament_report` (+ helpers), best/worst-year support.
7. `src/cli.py` — `tournament` mode + flags; existing modes untouched.
8. `tests/` — test_momentum.py, test_mean_reversion_filtered.py, test_regime_switch.py, test_tournament.py, CLI coverage.
9. `README.md` — tournament section; `docs/RED_TEAM.md` — bias attack + final recommendation framework.
