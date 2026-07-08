# Wallst — Mean Reversion & Sector Rotation Backtester

**Research / educational tool only. Not financial advice. This does not place live trades against your Robinhood account or any brokerage — it is a historical backtester, nothing more.** Past performance in these reports does not indicate future results. Always do your own research and consider consulting a licensed financial advisor before risking real capital.

## What this is

Two trading strategy ideas, each with its own $7,500 sleeve of a hypothetical $15,000 account, backtested against historical price data:

1. **Mean reversion** — buy oversold large-cap stocks, exit on a bounce, a stop-loss, or a timeout.
2. **Sector rotation** — hold the strongest-momentum SPDR sector ETFs, rebalanced monthly.

A shared backtesting engine simulates both against historical data pulled from `yfinance`, with less-biased execution timing than a naive same-close backtest (signals lag fills by one trading day), transaction costs, and a full audit trail of every decision and trade. This is a simplified, close-to-close simulation, not a claim of realistic live-market execution — see "Design decisions" and "Known limitations" below.

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

Flags: `--strategy {mean_reversion,sector_rotation,both,compare,robustness}` (required), `--start`, `--end` (default: last 7 years through today), `--capital` (default 15000), `--cost-bps` (default 5), `--output-dir` (default `output`), `--refresh-cache` (force a full re-download instead of using the local cache), `--no-fractional-shares` (round position sizes down to whole shares), `--compare-years` (comma-separated years for `--strategy compare`'s annual-returns table, e.g. `2022,2023,2024`; default: every calendar year spanned by `--start`/`--end`; ignored for other `--strategy` values), `--robustness-windows` (comma-separated `start:end` windows for `--strategy robustness`, e.g. `2019-01-01:2021-12-31,2020-01-01:2022-12-31`; default: the 5 standard windows below; ignored for other `--strategy` values).

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

## Strategy rules

### Mean reversion ($7,500 sleeve)
- Universe: ~25 liquid, sector-diverse large caps (`src/config.py: MEAN_REVERSION_UNIVERSE`).
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

- **Survivorship bias**: the mean-reversion universe is today's surviving large caps, not a point-in-time constituent list from whenever the backtest starts. A stock that existed in 2018 but has since been delisted, acquired, or gone bankrupt is not in this universe, which tends to overstate historical performance. Every mean-reversion report prints an explicit warning about this. Treat mean-reversion results as validation of the strategy's *mechanics*, not proof of a durable historical edge.
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

111 tests across indicator math, the look-ahead-prevention accessor, both strategies' decision logic (including explicit "does the decision change if I mutate any future date" checks), the engine's cash/lot accounting (including net-of-cost P&L and trading-day holding periods), cost/turnover metrics, the data/caching layer (including the yfinance end-date-inclusivity fix, the widened-delta-fetch fix, and stale-cache-fallback warnings), the CLI's minimum-usable-universe guard, the `--strategy compare` diagnostics report (annual returns, sleeve contribution, and a full end-to-end CLI run), and the `--strategy robustness` allocation-blending math (raw-equity-curve blending that preserves first-day cost drag rather than normalizing it away, cost/turnover weighting, ranking/beats-SPY analysis, and a full end-to-end CLI run). CI runs this suite on every push via GitHub Actions.
