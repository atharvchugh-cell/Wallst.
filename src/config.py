"""Default parameters for the backtesting system. Edit freely to try different
universes or strategy thresholds."""

# --- Portfolio-level defaults ---
DEFAULT_CAPITAL = 15000.0
DEFAULT_COST_BPS = 5.0
DEFAULT_START_YEARS_BACK = 7
BENCHMARK_TICKER = "SPY"

# --- Mean reversion ---
# ~25 liquid, sector-diverse large caps. This is a SURVIVORSHIP-BIASED research
# universe (today's large-cap survivors), not a point-in-time constituent list.
# See README "Known Limitations".
MEAN_REVERSION_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "JPM", "BAC", "V", "MA",
    "JNJ", "UNH", "PFE",
    "XOM", "CVX",
    "HD", "MCD", "DIS", "KO", "PEP", "WMT", "COST",
    "CAT", "BA", "NFLX",
]

# An alternate hand-picked watchlist, used only by the universe-sensitivity test
# to sanity-check that results aren't wildly fragile to the specific tickers chosen.
MEAN_REVERSION_UNIVERSE_ALT = [
    "ORCL", "CRM", "ADBE", "CSCO", "IBM",
    "GS", "MS", "AXP", "SCHW",
    "ABBV", "LLY", "MRK",
    "COP", "SLB",
    "LOW", "SBUX", "NKE", "PG", "CL", "TGT", "T",
    "DE", "GE", "UPS",
]

RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 35.0     # increased from 30 for more aggressive entry in bull markets
RSI_EXIT_THRESHOLD = 50.0
SMA_PERIOD = 30                # reduced from 50 for faster exits
STOP_LOSS_PCT = -0.08          # delayed close-to-close stop, see README
MAX_HOLDING_DAYS = 10          # reduced from 20 for faster mean reversion turnover
MAX_CONCURRENT_POSITIONS = 5

# Warmup buffer (calendar days) fetched before the requested start date so that
# SMA-50/RSI-14 are valid by the time the requested start date arrives.
MEAN_REVERSION_WARMUP_CALENDAR_DAYS = 200

# Minimum fraction of the configured mean-reversion universe that must survive
# data fetch/gap checks. Below this, the run hard-fails instead of silently
# producing a thin (or all-cash, "no trade") backtest on a degraded universe.
MIN_MEAN_REVERSION_UNIVERSE_FRACTION = 0.8

# --- Sector rotation ---
SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLY",
    "XLP", "XLU", "XLI", "XLB", "XLRE", "XLC",
]
SECTOR_LOOKBACK_MONTHS = 3
SECTOR_TOP_K = 2                # reduced from 3 for more concentrated momentum bet
# Extra calendar-day buffer fetched beyond the "latest ETF inception + lookback
# months" effective-start calculation, to comfortably cover month-end alignment.
SECTOR_WARMUP_BUFFER_CALENDAR_DAYS = 40

# --- Data layer ---
CACHE_DIR = "data_cache"
CACHE_REFRESH_THRESHOLD_DAYS = 7
YFINANCE_AUTO_ADJUST = True
YFINANCE_INTERVAL = "1d"

# --- Reporting ---
OUTPUT_DIR = "output"
SHORT_PERIOD_WARNING_CALENDAR_DAYS = 90
