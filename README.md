# Wallst — Mean Reversion & Sector Rotation Backtester

**Research / educational tool only. Not financial advice. This does not place live trades against your Robinhood account or any brokerage — it is a historical backtester, nothing more.** Past performance in these reports does not indicate future results. Always do your own research and consider consulting a licensed financial advisor before risking real capital.

## What this is

Two trading strategy ideas, each with its own $7,500 sleeve of a hypothetical $15,000 account, backtested against historical price data:

1. **Mean reversion** — buy oversold large-cap stocks, exit on a bounce, a stop-loss, or a timeout.
2. **Sector rotation** — hold the strongest-momentum SPDR sector ETFs, rebalanced monthly.

A shared backtesting engine simulates both against historical data pulled from `yfinance`, with realistic execution timing (signals lag fills by one trading day), transaction costs, and a full audit trail of every decision and trade.

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

Flags: `--strategy {mean_reversion,sector_rotation,both}` (required), `--start`, `--end` (default: last 7 years through today), `--capital` (default 15000), `--cost-bps` (default 5), `--output-dir` (default `output`), `--refresh-cache` (force a full re-download instead of using the local cache), `--no-fractional-shares` (round position sizes down to whole shares).

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

## Strategy rules

### Mean reversion ($7,500 sleeve)
- Universe: ~25 liquid, sector-diverse large caps (`src/config.py: MEAN_REVERSION_UNIVERSE`).
- **Entry**: RSI(14) drops below 30, if a slot is free (max 5 concurrent positions, sized at 1/5 of the sleeve each). If more candidates than free slots signal the same day, the lowest RSI (most oversold) wins.
- **Exit** (priority order — first match wins): (1) stop-loss, close ≤ 92% of the entry fill price; (2) timeout, 20 trading days held; (3) close crosses back above the 50-day SMA; (4) RSI rises back to 50.
- All signals fill at the **next** trading day's close, never the signal day's close (see "Execution timing" below).

### Sector rotation ($7,500 sleeve)
- Universe: the 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLU, XLI, XLB, XLRE, XLC).
- Every month-end, rank all 11 by trailing 3-*completed-calendar-month* return (not a fixed 63-trading-day lag). The top 3 get equal weight (1/3 each); the rest get 0.
- **Every** ETF gets a rebalance decision each month — including ones that stay in the top 3, since price drift since the last rebalance means their actual dollar weight has moved and a monthly rebalance corrects that. This is why `target_events.csv` always has 11 rows per rebalance while `transactions.csv` may have fewer (a ticker whose weight didn't drift enough to cross a minimum trade threshold generates no transaction).
- XLC (inception 2018) and XLRE (2015) have shorter histories than the rest. The effective backtest start date is clipped to `(latest ETF inception + 3 months)` if the requested start predates that — this is reported explicitly, not silently applied.

## Design decisions worth knowing about

- **Execution timing**: a strategy's decision on day *t* uses only price/indicator data through day *t*'s close. The trade then fills at day *t+1*'s close. This is enforced structurally — strategies only ever see a `MarketDataView` that raises an error if asked for data dated after the day it's currently processing — rather than relying on careful coding to avoid look-ahead bias.
- **Signal-time sizing**: the dollar amount of a trade (`requested_notional`) is fixed using the portfolio's equity *at the moment the signal fires*, not the equity on the later fill day. The number of shares that dollar amount buys is only resolved at the fill price (that's just normal execution, not hindsight). `actual_weight_after` in `transactions.csv` shows what was actually achieved, which can differ slightly from what was requested.
- **Average-cost accounting**: positions track shares and an average cost basis. A partial sell (e.g. sector rotation trimming a still-held ETF) realizes P&L on the trimmed portion (`trades.csv` type `partial_sell`) without closing the position; a full exit (`full_exit`) closes it. `trades.csv` is not limited to complete round trips.
- **No shared cash pool**: in `--strategy both` mode, the two strategies run as fully independent simulations, each with its own $7,500 and its own share/cash bookkeeping. They're only combined afterward by summing their two independently-computed equity curves, over the *intersection* of the two curves' valid dates (never an outer join, which would risk implying one sleeve was invested before it actually started).
- **Cash-constrained buy sizing**: if a batch of buy orders would need more cash than is available (including transaction costs), all the buy orders in that batch are scaled down proportionally. Cash never goes negative.
- **Fractional shares**: allowed by default (avoids leftover-cash drift from rounding). `--no-fractional-shares` rounds every position down to a whole share count.
- **Adjusted prices**: all data uses `yfinance`'s `auto_adjust=True`, so prices/trades are adjusted-price units (dividends and splits already factored in), not literal historical execution prices. This is called out in every report.
- **Stop-loss is a delayed close-to-close stop**, not an intraday one: if a stock gaps down 15% overnight, this backtester still only "sees" and reacts to that day's closing price, with the exit executing the following day's close.
- **Cache correctness**: adjusted historical prices can change retroactively (a correction, a newly-declared dividend, etc.), so the local cache in `data_cache/` does a full re-download if it's more than 7 days old, rather than trusting a naive append-only cache indefinitely.

## Known limitations

- **Survivorship bias**: the mean-reversion universe is today's surviving large caps, not a point-in-time constituent list from whenever the backtest starts. A stock that existed in 2018 but has since been delisted, acquired, or gone bankrupt is not in this universe, which tends to overstate historical performance. Every mean-reversion report prints an explicit warning about this. Treat mean-reversion results as validation of the strategy's *mechanics*, not proof of a durable historical edge.
- No intraday price simulation — everything executes at a daily close.
- No bid/ask spread or market-impact modeling; only a flat per-trade cost assumption (`--cost-bps`, default 5 bps).
- No dividend/borrow cost modeling beyond what's already baked into `auto_adjust=True` adjusted closes.

## Tests

```bash
pytest tests/
```

66 tests across indicator math, the look-ahead-prevention accessor, both strategies' decision logic (including explicit "does the decision change if I mutate any future date" checks), the engine's cash/lot accounting, cost/turnover metrics, and the data/caching layer.
