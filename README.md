# Wallst — Strategy Backtesting & Tournament Research Platform

**Research / educational tool only. Not financial advice. This does not place live-money trades against Robinhood or any brokerage.** The strategy platform is a historical backtester. The isolated execution subsystem includes an in-process fake broker and a hard-pinned Alpaca paper workflow. Its terminal can submit only an immutable, exact-hash-approved batch to Alpaca paper after separate network and paper-submit confirmations; it contains no live-money endpoint or arbitrary order-entry command. Past performance in these reports does not indicate future results. Always do your own research and consider consulting a licensed financial advisor before risking real capital.

## What this is

A backtesting platform for a hypothetical $15,000 account, built around two original strategy ideas — each with its own $7,500 sleeve when run together:

1. **Mean reversion** — buy oversold large-cap stocks, exit on a bounce, a stop-loss, or a timeout.
2. **Sector rotation** — hold the strongest-momentum SPDR sector ETFs, rebalanced monthly.

— plus a **strategy tournament** (`--strategy tournament`) that pits those two against three newer strategy families (trend/momentum, filtered mean reversion, a regime-switch hybrid) under identical conditions, across multiple market-regime windows, with cost- and parameter-sensitivity probes and a disclosed robustness score, a **portfolio-combination mode** (`--strategy portfolio`) that backtests one account split across several of those strategies at fixed weights (e.g. 60/35/5), and a **walk-forward / out-of-sample mode** (`--strategy walk_forward`) that evaluates that portfolio only on periods that did not influence its parameters. See "`--strategy tournament`", "`--strategy portfolio`", and "`--strategy walk_forward`" below and `docs/TOURNAMENT_DESIGN.md`; `docs/RED_TEAM.md` collects everything known to be wrong or biased about these backtests.

A shared backtesting engine simulates every strategy against historical data pulled from `yfinance`, with less-biased execution timing than a naive same-close backtest (signals lag fills by one trading day), transaction costs, and a full audit trail of every decision and trade. This is a simplified, close-to-close simulation, not a claim of realistic live-market execution — see "Design decisions" and "Known limitations" below.

The order-management, risk, ledger, fake-broker, reconciliation, and paper-only
operating boundaries are documented in
[`docs/LIVE_TRADING_PHASE1.md`](docs/LIVE_TRADING_PHASE1.md) and
[`docs/LIVE_TRADING_PHASE2.md`](docs/LIVE_TRADING_PHASE2.md). Reviewed paper
aggregation and execution are in
[`docs/LIVE_TRADING_PHASE3.md`](docs/LIVE_TRADING_PHASE3.md). The adversarial
review and residual-risk register are in
[`docs/LIVE_TRADING_RED_TEAM.md`](docs/LIVE_TRADING_RED_TEAM.md). Supervised
paper automation is documented in
[`docs/LIVE_TRADING_PHASE4.md`](docs/LIVE_TRADING_PHASE4.md). Phase 4 runs the
registered fixed 60% momentum / 35% sector rotation / 5% regime-switch mix,
publishes signed immutable targets, schedules against the exchange calendar,
recovers paper order streams through the same OMS/ledger, alerts, backs up, and
produces soak evidence. It remains Alpaca paper only. Observe is the default;
shadow plans are durably non-submitting; paper submission still requires the
reviewed approval and risk/reconciliation controls. Phase 5/live money is not
implemented.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python backtest.py --strategy mean_reversion
python backtest.py --strategy sector_rotation
python backtest.py --strategy both

# with options
python backtest.py --strategy both \
  --start 2018-01-01 --end 2025-01-01 \
  --capital 15000 --cost-bps 5 \
  --output-dir output --refresh-cache
```

Flags: `--strategy {mean_reversion,sector_rotation,both,compare,robustness}` (required), `--start`, `--end` (default: last 7 years through today), `--capital` (default 15000), `--cost-bps` (default 5), `--output-dir` (default `output`), `--refresh-cache` (force a full re-download instead of using the local cache), `--no-fractional-shares` (round position sizes down to whole shares), `--compare-years` (comma-separated years for `--strategy compare`'s annual-returns table, e.g. `2022,2023,2024`; default: every calendar year spanned by `--start`/`--end`; ignored for other `--strategy` values), `--robustness-windows` (comma-separated `start:end` windows for `--strategy robustness`, e.g. `2019-01-01:2021-12-31,2020-01-01:2022-12-31`; default: the 5 standard windows below; ignored for other `--strategy` values), `--universe {default,us_50b}` (mean-reversion universe; default `default` — see below; ignored by `sector_rotation`), `--universe-csv PATH` (use a custom ticker CSV instead, overrides `--universe`), `--refresh-universe` (rebuild the `us_50b` universe from live data instead of using its cache), `--max-universe-candidates N` (debug/safety cap on screener results considered when rebuilding `us_50b`; not applied by default), `--universe-allow-screener-only` (debug escape hatch: proceed without the Nasdaq Trader eligibility gate if its directories are unreachable; not the default).

Each run writes a timestamped directory under `output/` (e.g. `output/20250101T120000_mean_reversion_2018-01-01_to_2025-01-01/`) containing:

| File | Contents |
|---|---|
| `report.txt` | Human-readable summary: returns, risk, costs, exposure, benchmark comparison, sample transactions, warnings |
| `metrics.json` | Full run config + computed metrics, for reproducibility |
| `equity_curve.csv` / `.png` | Daily portfolio value |
| `target_events.csv` | **Every** signal the strategy generated, including ones that resulted in no trade |
| `transactions.csv` | Every executed (nonzero) buy/sell, with requested vs. actual weight/notional |
| `trades.csv` | Realized P&L events (`partial_sell` and `full_exit`) |
| `positions.csv` | Daily snapshot of shares/value/weight per ticker |

### `--strategy compare`: diagnostics, not optimization

```bash
python backtest.py --strategy compare --start 2022-01-01 --end 2024-12-31 --capital 15000
```

Runs mean reversion, sector rotation, and combined exactly as `--strategy both` does (writing all of their normal individual artifacts, unchanged), **plus** a side-by-side comparison against each other and against a plain SPY buy-and-hold. This exists to help answer *"is underperformance coming from mean reversion, sector rotation, transaction costs, or the combined allocation?"* before touching any strategy parameter — it changes no RSI/SMA/holding-day/top-K/stop/allocation/cost setting, it only reports on runs made with whatever those settings already are.

Writes an extra `output/{ts}_comparison_{start}_to_{end}/` directory:

| File | Contents |
|---|---|
| `comparison.csv` / `.txt` | One row per metric, one column per {mean_reversion, sector_rotation, both, SPY}: total return, CAGR, max drawdown (+ duration), Sharpe, Sortino, Calmar, best/worst month, turnover, transaction costs, cost drag, transaction/trade counts, average capital invested, correlation to SPY, excess return vs. SPY |
| `annual_returns.csv` | Calendar-year return per strategy/SPY for each requested year (a year with no data for a given row is blank, not 0%) |
| `monthly_returns.csv` | Month-by-month return per strategy/SPY, for finer-grained comparison than the annual table |
| `comparison.json` | Same data as the CSVs/txt, machine-readable |

`comparison.txt` also includes a **strategy contribution** section for the combined sleeve: each sleeve's starting/final equity over the combined (intersection) window, each sleeve's dollar contribution to the combined gain, and each sleeve's share of combined transaction costs — computed directly from each sleeve's own equity curve and transactions (not the naive concatenation `both` uses internally), so a sleeve that started trading before the intersection window began doesn't get misattributed gains/costs from before that window.

Each row's underlying date range can differ (mean reversion needs indicator warmup; sector rotation clips to `latest ETF inception + lookback`; `both`/SPY use the intersection) — `comparison.txt` prints the effective range used per row so this isn't hidden.

### `--strategy robustness`: does the allocation choice hold up across time and mix?

```bash
python backtest.py --strategy robustness --start 2019-01-01 --end 2024-12-31 --capital 15000 --refresh-cache
```

`--strategy compare` answers "how did mean reversion vs. sector rotation vs. 50/50 do in ONE window?" `--strategy robustness` asks the next question: does the current 50/50 split hold up across **different capital allocations** and **different historical windows**, or does it only look reasonable in the one window already examined? Still diagnostics only — no RSI/SMA/holding-day/top-K/stop-loss/universe/transaction-cost parameter is touched, and `--strategy both`'s allocation stays a fixed 50/50 exactly as before; this mode's allocation logic is isolated to itself.

**Default windows** (5 overlapping 3-year-ish calendar-year windows, override with `--robustness-windows`):
`2019-2021`, `2020-2022`, `2021-2023`, `2022-2024`, `2019-2024`.

**Allocation mixes tested in each window:**
100% sector / 0% mean-reversion, 75/25, 50/50, 25/75, 0% sector / 100% mean-reversion, plus a SPY row.

**How allocation mixes are computed:** mean_reversion and sector_rotation are each run **once per window** (reusing `run_mean_reversion_sleeve`/`run_sector_rotation_sleeve` completely unmodified — the same strategy logic and config.py defaults as every other mode, both sleeves run at the full requested `--capital`), not once per allocation. The 5 allocation mixes are then built by capital-weighting those two already-computed **raw** equity curves — `w_sr * sector_equity + w_mr * mean_reversion_equity` — and their transaction totals. This preserves each sleeve's raw dollar economics, including any first-day fill/transaction-cost drag baked into its equity curve, rather than normalizing each curve to a clean, cost-free starting point before scaling. Because every dollar decision the engine makes is sized as a pure fraction of that sleeve's own equity (`target_weight * sleeve_equity`) and `cost_bps` is a fixed rate — both scale linearly with capital — this blend is equivalent to actually re-running each sleeve at its allocated capital split, provided both sleeves were run at the same capital being blended (true for every caller here). See `src/robustness.py`'s module docstring for the full reasoning and its edge cases (fixed-dollar dust thresholds and whole-share rounding, neither of which binds at realistic position sizes). This turns 5 windows × 6 allocations × 2 sleeves (60 backtests) into 5 windows × 2 sleeves (10 backtests) plus arithmetic.

Writes `output/{ts}_robustness_{start}_to_{end}/`:

| File | Contents |
|---|---|
| `robustness_summary.csv` / `.txt` | One row per (window, allocation): the same 17 metrics as `--strategy compare`'s table |
| `robustness_rankings.csv` | Per allocation: average rank across windows by total return/Sharpe/max drawdown/Calmar, and how often it beats SPY on return and on max drawdown |
| `robustness_heatmap_data.csv` | Wide matrix (allocation × window) of total return, ready to feed into a heatmap |
| `robustness_summary.json` | Same data, machine-readable |

`robustness_summary.txt` directly answers: best allocation per window (by return/Sharpe/drawdown/Calmar), average rank across windows, how often each allocation beats SPY (on return, and on drawdown), and a "does mean reversion's drawdown protection justify its cost drag" section comparing every non-100%-sector allocation against the 100%-sector baseline (a simple heuristic — drawdown improved by more percentage points than cost drag increased — not a rigorous risk-adjusted verdict).

Since each window reuses the individual-sleeve functions unmodified, every window also writes its own normal `mean_reversion`/`sector_rotation` report directories as a side effect (10 extra directories for the 5 default windows) — useful for auditing any one window's underlying trades, not something you need to look at to read the robustness report itself.

### `--universe`: default vs. a larger current-snapshot universe

Mean reversion's universe is pluggable. This does not change any strategy threshold (RSI/SMA/holding-days/stop-loss/position-count) — only which tickers the same rules run against. Sector rotation is unaffected either way; it always trades the 11 fixed sector ETFs.

```bash
# Default: unchanged, reproducible ~25-stock universe (nothing new to opt into).
python backtest.py --strategy mean_reversion

# A much larger universe: every US-listed common stock currently >= $50B market cap.
python backtest.py --strategy mean_reversion --universe us_50b --refresh-universe

# Debug/dev: cap how many screener results to consider, for a fast bounded run.
python backtest.py --strategy mean_reversion --universe us_50b --refresh-universe --max-universe-candidates 20

# Your own ticker list (CSV needs at minimum a `ticker` column).
python backtest.py --strategy mean_reversion --universe-csv my_tickers.csv
```

- **`--universe default`** (the default) is the existing hardcoded `config.MEAN_REVERSION_UNIVERSE` list — old results stay exactly reproducible; nothing about this mode's behavior changed.
- **`--universe us_50b`** builds (or reuses a cached build of) a universe of US-listed common stock with a *current* market cap >= $50B, combining a **bulk** screener query (fast) with the Nasdaq Trader symbol directories as a **required eligibility gate** (correct):
  1. Queries Yahoo Finance's screener (`yfinance`'s `yf.screen`) for US equities with `intradaymarketcap >= $50B`, paginated (Yahoo caps each request at 250 results — realistically only a few hundred US companies clear that bar, so this is typically 1-3 requests total, not thousands). Progress is printed per page: candidates found, running qualifying count, failures.
  2. Fetches and parses Nasdaq Trader's public symbol directories (`nasdaqlisted.txt`, `otherlisted.txt`) into a candidate set, excluding test issues, ETFs (via each directory's own ETF flag), and — by name pattern — warrants, rights, units, preferred shares, notes, and funds. **Every screener result is cross-checked against this candidate set; a screener ticker NOT present in it is excluded.** This is a real correctness requirement, not just extra metadata: Yahoo's screener alone is not reliable enough to define "US-listed publicly traded common stock" — it can return symbols that aren't tradable/normal listings at all. Because of this, **the Nasdaq Trader fetch is required by default**: if it fails, the build hard-fails with a clear error rather than silently proceeding screener-only. Pass `--universe-allow-screener-only` to explicitly opt into that reduced-guarantee fallback if you understand the risk.
  3. **Presence in the candidate set alone still isn't proof the screener's quote is about that same listing.** A live smoke test found the screener returning a ~$2T "Space Exploration Technologies Corp." for a ticker that, it turned out, *was* present in the Nasdaq Trader directory — just as an unrelated, different company. Two more checks close this gap: (a) once a ticker clears the candidate-set gate, its **Nasdaq Trader security name is authoritative** for non-common-stock filtering and saved metadata — not the screener's own `longName` — though the screener's name is still checked too, as a defense-in-depth backstop (the name-pattern check also now covers "ETF"/"ETN" by name, not just each directory's ETF flag, in case that flag is ever wrong); and (b) an **identity-mismatch guard** compares the screener's name against the Nasdaq Trader name for the same ticker and excludes the result if they share no meaningful word in common. Deliberately does **not** exclude on the word "Trust" alone, since legitimate common-stock companies (REITs in particular) routinely have "Trust" in their listed name.
  4. Normalizes tickers to Yahoo Finance's convention (`BRK.B` → `BRK-B`).
  5. Keeps tickers with market cap >= $50B (hard-fails rather than running on a near-empty universe if fewer than 5 qualify; warns if more than half of the screener's results couldn't be parsed).
  6. **Deduplicates by company**: some companies list more than one share class as separate tickers that can each independently clear $50B (e.g. `GOOGL`/`GOOG`, `BRK-A`/`BRK-B`). Since the goal is "companies >= $50B," not "securities >= $50B," these are collapsed to one ticker per company (grouped by a normalized company name with corporate-suffix/share-class wording stripped — a heuristic, not a perfect company-identity resolution; the higher-market-cap ticker of the group is kept, alphabetical tie-break). Real Nasdaq Trader names for the same company's classes don't always agree on the generic word after the class letter (e.g. `Alphabet Inc. - Class A Common Stock` vs. `Alphabet Inc. - Class C Capital Stock`, or `Berkshire Hathaway Inc. Common Stock` vs. `Berkshire Hathaway Inc. New Common Stock`), so "Capital Stock"/"New Common Stock" wording is stripped alongside "Common Stock", not just the latter.
  7. **Validates price-data availability as a last line of defense**, on only the small final selected/deduped list (not the full candidate pool): each remaining ticker gets one lightweight price-history lookup, and any ticker with no usable price data is excluded. When a specific backtest window is known (i.e. a real `--start`/`--end` run, not a bare `--refresh-universe`), this checks that window specifically (warmup-adjusted), not just a fixed recent lookback — a ticker can have a perfectly normal *recent* quote while having no price history at all over an older window a backtest actually needs, which is exactly how a bad symbol can pass a "recent data" check, get cached, and then still get dropped by the backtest itself.
  8. Caches the result to `data_cache/universe_us_50b.csv` (ticker, name, market cap, exchange, snapshot timestamp, and the price-data-validated window if one was checked). Subsequent runs reuse this cache; pass `--refresh-universe` to rebuild it from live data.
- **A cached universe's price-data validation is window-specific, not permanent.** Each cache records the `[start, end]` window its tickers were actually checked against. A later run requesting a window not fully covered by that record (or a cache from before this field existed) triggers a lightweight re-validation of the cached tickers only against the new window (no Nasdaq Trader/screener re-fetch) — dropping anything that now fails and re-saving the cache — instead of silently trusting a cache that was only ever proven good for a different historical period. Robustness mode resolves the universe once and shares it across every tested window, so it validates against the union of all those windows' start/end.
- An earlier version of this looked up market cap per-ticker across the entire ~8,000+-ticker Nasdaq Trader directory via `yfinance`'s `Ticker.fast_info`/`.info`. In practice this was impractically slow (each call can trigger several additional sub-requests for share count/last price/history metadata) and printed no progress, so a real run looked hung. That per-ticker path (`fetch_market_caps`/`_get_market_cap`) is still in `src/universe.py` as a tested utility but is no longer used by the default build.
- **`--max-universe-candidates N`** caps how many screener results are considered — a debug/safety knob for fast, bounded dev/test runs, not applied by default (a real run considers every qualifying result).
- **`--universe-allow-screener-only`** is a debug escape hatch: if the Nasdaq Trader directories can't be fetched, proceed with screener-only results (reduced eligibility guarantees, including no identity-mismatch check) instead of hard-failing. Not the default.
- **`--universe-csv PATH`** uses your own CSV instead (a `ticker` column is required; `name`/`market_cap`/`exchange`/`snapshot_date` columns are used if present, same schema as the `us_50b` cache file). This always overrides `--universe`.
- Every report (`report.txt`, `metrics.json`) for a non-default universe records: universe mode, tickers selected, screener results considered, tickers excluded for not being in the Nasdaq Trader candidate set, tickers excluded for a screener/Nasdaq Trader name mismatch, tickers excluded as non-common-stock by name, tickers excluded for missing price data, duplicate-company tickers collapsed, unparseable screener results, the min/max market cap actually in the universe, the cache file used, the universe snapshot's timestamp, and the price-data-validated window (if any).

**Why this is still survivorship-biased, and NOT point-in-time historical membership:** `us_50b` membership is decided once, using *today's* market cap, then applied uniformly across every historical date a backtest touches. A ticker in this universe was not necessarily >= $50B (or even public) for the entire window you're backtesting — a company that IPO'd in 2023 and is >= $50B today appears in a 2019-2024 backtest as if it had been tradeable the whole time; a company that WAS >= $50B in 2019 but has since shrunk or been delisted does not appear at all. Market caps also drift with the market, so re-running `--refresh-universe` on a different day can change which tickers qualify — this is a live/current value, not a fixed historical fact. Treat `us_50b` results exactly like the default universe's results: they validate strategy mechanics on a large, liquid, currently-large-cap set of names, not a general historical edge.

### `--strategy tournament`: compare strategy families under identical conditions

Runs every registered strategy — the two originals plus three newer ones — under the **same capital, cost assumption, requested windows, canonical-calendar construction, and benchmark**, and writes one side-by-side report. Each strategy runs at the FULL `--capital` (the tournament compares *alternatives* for one account; `--strategy both`/`compare` are what split capital into simultaneous sleeves). Nothing about the pre-existing modes or defaults changes; `tournament` is purely additive.

```bash
# One window, all five strategies:
python backtest.py --strategy tournament --start 2019-01-01 --end 2024-12-31 --capital 15000

# Across named bull/bear/choppy regime windows, with cross-window robustness scoring:
python backtest.py --strategy tournament --start 2019-01-01 --end 2024-12-31 --tournament-windows regimes

# Cost sensitivity (re-runs every strategy at each cost level) + parameter sensitivity:
python backtest.py --strategy tournament --start 2019-01-01 --end 2024-12-31 \
    --tournament-cost-bps-list 0,5,10,20 --tournament-param-sensitivity

# A subset of strategies, custom windows:
python backtest.py --strategy tournament --tournament-strategies momentum,sector_rotation \
    --tournament-windows 2020-01-01:2020-12-31,2022-01-01:2022-12-31
```

The five registered strategies (see `docs/TOURNAMENT_DESIGN.md` for full specs and parameter provenance):

- **`mean_reversion`** — the existing baseline, unchanged, run through its existing sleeve runner.
- **`mean_reversion_filtered`** — the baseline plus exactly two **entry-only** filters (every inherited threshold unchanged, so differences isolate the filters): no new entries while SPY closes below its 200-day SMA (regime gate), and no entry into a name whose trailing 5-trading-day return is ≤ -15% (falling-knife guard; it rejects entries into an *established* crash — day 1 of a crash is indistinguishable from a dip without foresight and is then managed by the inherited stop). Exits are never gated.
- **`momentum`** — monthly top-5 by trailing 126-trading-day (~6-month) return, eligible only if close > 200-day SMA AND that return is positive; equal 1/5 weights; if fewer than 5 qualify the remainder stays in **cash** (defensive by construction). 6-month momentum and the 200-day SMA are decades-old literature conventions chosen for canonical status, not tuned on this repo's data.
- **`sector_rotation`** — the existing baseline, unchanged.
- **`regime_switch`** — sector rotation's exact (inherited) ranking while SPY > its 200-day SMA; **100% cash otherwise** (cash earns 0%, conservative). No new tuned parameters.

SPY is a *signal-only* ticker for the two regime-aware strategies: its data is required (they hard-fail without it) but it is structurally untradable — it is never in the tradable universe the strategies iterate.

What the report (`tournament_report.txt` + `tournament_summary.csv` + `tournament.json`) contains:

- Per-window table of every required metric: total return, CAGR, max drawdown (+duration), Sharpe, Sortino, Calmar, net win rate, trade/transaction counts, turnover and cost drag, **time in market** (days-with-any-position and average-capital-invested), best/worst month, best/worst calendar year, excess return and correlation vs SPY — plus a SPY buy-and-hold row over the strategies' common date range.
- With multiple windows, a **cross-window robustness section**: fraction of windows beating SPY, fraction positive, worst-window drawdown, return dispersion, and a composite `robustness_score = mean(beats-SPY fraction, positive fraction)` — the formula is printed next to every use and its components are always shown alongside it. **The fractions are computed over the number of windows a strategy was *expected* to run (every window that ran for at least one strategy), not just the windows it survived** — so a strategy that failed or disappeared in a hard window is penalized for it exactly like a lost window, and cannot outrank a full-coverage peer by only clearing its easy windows. Each strategy's `ran/exp` coverage is shown, and any strategy missing windows is flagged as INCOMPLETE.
- **Per-window listing-history filtering (important for `--universe us_50b`):** a current-snapshot universe contains names that did not exist for older windows (ARM, GEV, HOOD, SNOW, ABNB, APP, RKLB, …). For each strategy and each window, any universe ticker that lacks enough listing history to warm up that strategy's indicators for that window is **excluded, with an explicit reason** (never silently run with missing early history) — reported per strategy/window in the tournament report's "Per-strategy/window universe exclusions" section and in each run's own `report.txt`. This is a no-op for the default hardcoded universe (whose members all predate any window), so default-universe behavior is unchanged. yfinance's per-ticker "possibly delisted / no price data" log noise (expected for a young ticker in an old window) is quieted so the structured exclusion report is readable.
- With `--tournament-cost-bps-list`, a **cost-sensitivity table** (`cost_sensitivity.csv`), with an explicit warning when a strategy's excess return vs SPY flips sign as costs rise — such a strategy has no margin for real-world frictions and should be treated as NOT beating SPY.
- With `--tournament-param-sensitivity`, a **small, disclosed variant sweep** (`param_sensitivity.csv`): a handful of fixed variants per strategy, each with a written rationale (see `PARAM_SENSITIVITY_VARIANTS` in `src/tournament.py`), **never auto-selected** — the report shows dispersion so fragility is visible, and warns explicitly when the beat-SPY conclusion flips across variants.
- Every strategy's self-declared parameters and plain-language assumptions (via `Strategy.describe()`), and any runs that failed (listed, not hidden).

To compare the default universe against `us_50b` for the stock strategies, run the tournament twice — once per `--universe` value — and compare the two reports; the resolved universe applies uniformly to every stock strategy within a run, so each run is internally fair.

Fairness/consistency guarantees, tested: the incumbents' tournament rows are produced by the same sleeve-runner functions their standalone modes use, and the generic runner used for the newer strategies and sensitivity re-runs is regression-tested to reproduce those runners' equity curves exactly.

### `--strategy portfolio`: run ONE account allocated across weighted sleeves

Where `--strategy tournament` runs each strategy at the *full* capital to compare *alternatives*, `--strategy portfolio` backtests a **single account split across several strategies at once** — each sleeve gets its share of the starting capital and they run side by side as one portfolio.

```bash
# The 60% momentum / 35% sector_rotation / 5% regime_switch portfolio on a $15k account:
python backtest.py --strategy portfolio --start 2019-01-01 --end 2024-12-31 --capital 15000 \
    --portfolio-weights momentum=0.60,sector_rotation=0.35,regime_switch=0.05
```

`--portfolio-weights` defaults to exactly that 60/35/5 mix, so the command above is equivalent to omitting the flag. On a $15,000 account it allocates **$9,000 to momentum, $5,250 to sector_rotation, and $750 to regime_switch**. Any registered tournament strategy can be used; weights must be non-negative and sum to 1.0 (a small floating-point tolerance is allowed), and a strategy may not be listed twice.

**Static allocation (v1), stated plainly:** capital is allocated **once** at the start. Each sleeve then runs as a **fully independent** engine simulation with its own cash/shares/lots — exactly like `--strategy both`'s sleeves — so **no cash is ever shared or transferred between sleeves and no capital is double-counted**. Sleeve weights are allowed to **drift** with performance (a winning sleeve becomes a larger share of the portfolio over time); there is **no periodic rebalancing** back to the target weights. The portfolio equity curve is the sum of the sleeves' independent daily equity curves over the **common date intersection** of their valid ranges. This is disclosed prominently at the top of the report.

The report (`portfolio_report.txt` + `portfolio_summary.csv` + `portfolio_equity.csv` + `portfolio_sleeves.csv` + `portfolio.json`) reports, for the whole portfolio: total return, CAGR, calendar-year returns, max drawdown (+duration), Sharpe, Sortino, Calmar, turnover, transaction costs; each sleeve's **final value, ending weight, and dollar contribution to total P&L** (which reconcile exactly — sleeve final values sum to the portfolio's, ending weights sum to 100%, P&L contributions sum to the portfolio's total gain); and the **SPY benchmark return and excess return** over the same common window. It honors the same `--start/--end`, `--capital`, `--cost-bps`, `--universe`/`--universe-csv` (for stock sleeves), and `--no-fractional-shares` options as every other mode.

### `--strategy walk_forward`: out-of-sample validation of the portfolio

Every backtest above is in-sample by construction. `--strategy walk_forward` answers the harder question: does the portfolio stay credible when it is only ever *evaluated* on periods that did not influence its parameters? It splits `--start/--end` into a sequence of **(train, test) folds**, evaluates the portfolio on each fold's **test** period with capital carried forward, and **stitches the test-period equity curves into one continuous out-of-sample curve**.

```bash
# Walk-forward the 60/35/5 portfolio: 3-year training, 1-year test, step 1 year (all defaults):
python backtest.py --strategy walk_forward --start 2015-01-01 --end 2024-12-31 --capital 15000 \
    --portfolio-weights momentum=0.60,sector_rotation=0.35,regime_switch=0.05

# Rolling (fixed-width) training window instead of expanding:
python backtest.py --strategy walk_forward --start 2015-01-01 --end 2024-12-31 --walk-forward-window rolling

# Optional optimize mode: rank each sleeve's predefined variants on the training window and freeze:
python backtest.py --strategy walk_forward --start 2015-01-01 --end 2024-12-31 --walk-forward-optimize
```

Fold geometry is set by `--walk-forward-train-years` (default 3), `--walk-forward-test-years` (default 1), `--walk-forward-step-years` (default 1), and `--walk-forward-window` (`expanding`, the default, anchors every fold's training window at `--start`; `rolling` slides a fixed train-years-wide window). Each fold's training window **ends the day before its test window begins** (`train_end < test_start`), so no test-period data can influence the parameters used; a trailing fold whose test year would run past `--end` is dropped.

- **Fixed mode (v1, default):** the portfolio's shipped, fixed parameters are evaluated on each test period. **No parameters are selected or tuned**, so there is nothing to overfit — this is the clean baseline (and a test asserts the selection routine is never even called). The training window is still reported and its dates enforced, but it does not influence results.
- **Optimize mode (`--walk-forward-optimize`, optional):** for each fold, each sleeve's small, **predefined** sensitivity variants (from `PARAM_SENSITIVITY_VARIANTS` in `src/tournament.py` — a handful per strategy, each with a written rationale, **never a free sweep**) are ranked on the **training window only** (by training Sharpe, tie-broken by training return), the best is **frozen**, and only then is the test period run with it. The selected variant per fold is reported.

Capital **compounds across folds** — each test period starts with the prior fold's ending equity — and each fold re-establishes the portfolio from cash, so fold boundaries incur re-entry costs (conservative). The report (`walk_forward_report.txt` + `walk_forward_folds.csv` + `walk_forward_equity.csv` + `walk_forward.json`) gives, per fold: train dates, test dates, portfolio return, SPY return, excess return, max drawdown, Sharpe, transaction costs, and the selected variant; and in aggregate: the stitched out-of-sample total return, CAGR, and max drawdown, the **percentage of test folds beating SPY** and the **percentage of profitable test folds**.

**Walk-forward does not fix survivorship bias.** It validates that the *parameters* are not overfit; it does nothing about the stock universe being a **current snapshot of today's survivors** (not a point-in-time constituent list) in *every* fold. A clean out-of-sample result here is still not evidence of a general, tradable edge — see `docs/RED_TEAM.md` (§A1) and "Known Limitations." Research / paper-trading only; no live orders.

## Strategy rules

### Mean reversion ($7,500 sleeve)
- Universe: ~25 liquid, sector-diverse large caps by default (`src/config.py: MEAN_REVERSION_UNIVERSE`), or a larger current US-listed >= $50B universe via `--universe us_50b` (see "`--universe`" above) — either way, the entry/exit rules below are unchanged.
- **Entry**: RSI(14) drops below `RSI_ENTRY_THRESHOLD` (`src/config.py`, default **35**), if a slot is free (`MAX_CONCURRENT_POSITIONS`, default 5, sized at 1/5 of the sleeve each). If more candidates than free slots signal the same day, the lowest RSI (most oversold) wins.
- **Exit** (priority order — first match wins): (1) delayed exit rule ("stop-loss"), close ≤ 92% of the entry fill price; (2) timeout, `MAX_HOLDING_DAYS` trading days held (default **10**); (3) close crosses back above `SMA_PERIOD` (default **30**); (4) RSI rises back to `RSI_EXIT_THRESHOLD` (default 50). See "Design decisions" for why (1) is not a real intraday stop.
- All signals fill at the **next** trading day's close, never the signal day's close (see "Execution timing" below).

### Sector rotation ($7,500 sleeve)
- Universe: the 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLU, XLI, XLB, XLRE, XLC).
- Every month-end, rank all 11 by trailing *completed-calendar-month* return over `SECTOR_LOOKBACK_MONTHS` (default 3 — not a fixed trading-day lag). The top `SECTOR_TOP_K` (`src/config.py`, default **2**) get equal weight; the rest get 0.
- **Every** ETF gets a rebalance decision each month — including ones that stay in the top-K, since price drift since the last rebalance means their actual dollar weight has moved and a monthly rebalance corrects that. This is why `target_events.csv` always has 11 rows per rebalance while `transactions.csv` may have fewer (a ticker whose weight didn't drift enough to cross a minimum trade threshold generates no transaction).
- XLC (inception 2018) and XLRE (2015) have shorter histories than the rest. The effective backtest start date is clipped to `(latest ETF inception + 3 months)` if the requested start predates that — this is reported explicitly, not silently applied.

### A note on these particular default values

The defaults above (RSI 35 / SMA-30 / 10-day timeout / top-2 sectors) are **not the strategy's original design defaults** (RSI 30 / SMA-50 / 20-day timeout / top-3 sectors). They were changed mid-development, in response to a single 2024 backtest run that underperformed SPY (~11% vs. SPY's ~25%), specifically to trade more often and hold more concentrated positions and try to close that gap.

This is disclosed here deliberately because tuning parameters to fit one specific historical window is a real overfitting risk, not just a caveat to note in passing — see "No walk-forward or out-of-sample validation" below. Nothing about the more aggressive defaults has been validated out-of-sample; they were chosen to see if a more aggressive posture would help in the one window already looked at, which is close to the textbook definition of curve-fitting. Before trusting any comparison between these settings and the original ones, re-run both across multiple non-overlapping periods and compare, rather than taking either single run at face value.

## Design decisions worth knowing about

- **Execution timing**: a strategy's decision on day *t* uses only price/indicator data through day *t*'s close. The trade then fills at day *t+1*'s close. This is enforced structurally — strategies only ever see a `MarketDataView` that raises an error if asked for data dated after the day it's currently processing — rather than relying on careful coding to avoid look-ahead bias.
- **Signal-time sizing**: the dollar amount of a trade (`requested_notional`) is fixed using the portfolio's equity *at the moment the signal fires*, not the equity on the later fill day. The number of shares that dollar amount buys is only resolved at the fill price (that's just normal execution, not hindsight). `actual_weight_after` in `transactions.csv` shows what was actually achieved, which can differ slightly from what was requested.
- **Average-cost accounting**: positions track shares and an average cost basis. A partial sell (e.g. sector rotation trimming a still-held ETF) realizes P&L on the trimmed portion (`trades.csv` type `partial_sell`) without closing the position; a full exit (`full_exit`) closes it. `trades.csv` is not limited to complete round trips.
- **No shared cash pool**: in `--strategy both` mode, the two strategies run as fully independent simulations, each with its own $7,500 and its own share/cash bookkeeping. They're only combined afterward by summing their two independently-computed equity curves, over the *intersection* of the two curves' valid dates (never an outer join, which would risk implying one sleeve was invested before it actually started).
- **Cash-constrained buy sizing**: if a batch of buy orders would need more cash than is available (including transaction costs), all the buy orders in that batch are scaled down proportionally. Cash never goes negative.
- **Fractional shares**: allowed by default (avoids leftover-cash drift from rounding). `--no-fractional-shares` rounds every position down to a whole share count.
- **Adjusted prices**: all data uses `yfinance`'s `auto_adjust=True`, so prices/trades are **synthetic adjusted-price units** (dividends and splits already factored in) — not literal historical execution prices you could have actually transacted at. This is called out in every report.
- **The mean-reversion "stop-loss" is a delayed exit rule, not a real stop-loss**: if a stock gaps down 15% overnight, this backtester still only "sees" and reacts to that day's closing price, with the exit executing the following day's close. In a live crash it could not protect you the way a broker-side stop order might.
- **Cache correctness**: adjusted historical prices can change retroactively (a correction, a newly-declared dividend, etc.), so the local cache in `data_cache/` does a full re-download if it's more than 7 days old, rather than trusting a naive append-only cache indefinitely. If a cache-extension fetch fails (e.g. a network hiccup), the run falls back to the last good cached data rather than failing outright — but this fallback is never silent: a warning naming the affected ticker and the shortfall is recorded in the run's warnings and printed in `report.txt`.
- **Delta-fetch window is deliberately widened, not exact**: when extending a cache toward a later requested end, the fetch starts several days before the cached end rather than requesting just the exact missing tail. yfinance was observed to handle very narrow (e.g. single-day) date ranges unreliably in practice; the overlap is deduped on merge. If the benchmark (SPY) still ends up short of a requested weekday end after this, the run hard-fails rather than silently using a shortened effective end — the benchmark's date range drives the canonical trading calendar for the whole run, so a silent shortfall there would silently shorten everything downstream of it.
- **Gross vs. net P&L**: `trades.csv` reports both gross realized P&L (`realized_pnl`, ignoring transaction costs) and net-of-costs P&L (`realized_pnl_net`, after both the buy and sell transaction cost on that lot). `win_rate` in the report and `metrics.json` is net-of-costs by default (`win_rate_gross` is also reported) — a trade that's barely profitable before costs but a loser after costs is correctly counted as a loss.
- **Holding days**: `trades.csv`'s `holding_days` counts **trading days** (matching `MAX_HOLDING_DAYS`'s units in `src/config.py`); `holding_calendar_days` is also reported for the calendar-day count, since these differ (weekends/holidays).
- **Minimum usable universe**: mean reversion hard-fails rather than silently running on a degraded universe if data issues (fetch failures, gaps) drop more than 20% of the configured universe (`src/config.py: MIN_MEAN_REVERSION_UNIVERSE_FRACTION`).

## Known limitations

- **Survivorship bias**: the mean-reversion universe — whether the default ~25-stock list or the larger `--universe us_50b` current-snapshot universe (see "`--universe`" above) — is today's surviving large caps, not a point-in-time constituent list from whenever the backtest starts. A stock that existed in 2018 but has since been delisted, acquired, or gone bankrupt is not in either universe, which tends to overstate historical performance. Every mean-reversion report prints an explicit warning about this. Treat mean-reversion results as validation of the strategy's *mechanics*, not proof of a durable historical edge.
- No intraday price simulation — everything executes at a daily close.
- No bid/ask spread or market-impact modeling; only a flat per-trade cost assumption (`--cost-bps`, default 5 bps).
- No dividend/borrow cost modeling beyond what's already baked into `auto_adjust=True` adjusted closes.
- **Pre-tax only**: every metric and report figure is pre-tax. Mean reversion in particular can generate short-term round trips that would typically be taxed at ordinary income rates in the US; realistic after-tax results can be meaningfully worse than what's shown.
- **No walk-forward or out-of-sample validation**: strategy parameters (RSI thresholds, SMA period, holding-day limits, sector lookback/top-K) are fixed defaults in `src/config.py`, not fit or validated out-of-sample. A backtest that looks good on one historical window is not proof the parameters generalize — treat any single run as one data point, not a validated edge, and be skeptical of hand-tuning these thresholds to make one specific historical window look better.
- **Sector universe is today's 11 SPDR sectors**: two of them (XLC, XLRE) only exist from 2018/2015 onward, which clips how far back sector rotation can be tested. There is currently no "legacy universe" mode to test older regimes (dot-com, 2008) with a smaller ETF set.
- **Benchmark is SPY only**: no equal-weight-universe or alternative-lookback comparisons yet, so a strategy can look better than it is if SPY happens to be an easy bar in the tested window.

## Tests

```bash
pytest tests/
```

The suite covers indicator/strategy/backtest behavior, no-lookahead invariants,
data/universe integrity, portfolio and walk-forward equivalence, and the Phase
1–4 execution path. Phase 4 coverage includes signed snapshots, strategy and
universe parity, calendar scheduling, mode isolation, quote/risk gates,
idempotent stream events and partial fills, REST recovery, alerts, backup and
restore integrity, soak evidence, endpoint pinning, and preservation of every
existing non-live strategy mode. CI runs the full suite on every push.
