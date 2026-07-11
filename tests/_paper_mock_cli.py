"""Standalone helper invoked via `subprocess.run([sys.executable, __file__, ...])`
by the true cross-process restart test in test_paper.py. NOT a pytest test
module itself (no test_ prefix) -- it installs the same deterministic mocked
price data as the `env` fixture (regenerated independently per process from a
SHA-256-derived seed, so two separate processes agree without any IPC) and
then runs `src.cli.main()` with the CLI args passed on argv, exiting with its
return code. This proves paper-trading state genuinely survives a process
exit/restart, not just object reuse within one pytest process.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cli, paper, tournament  # noqa: E402

FULL_DATES = pd.bdate_range("1990-01-01", "2024-12-31")


def _seed(ticker: str) -> int:
    return int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16) % (2**31)


def _make_frame(seed: int, drift: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(drift, 0.5, size=len(FULL_DATES)))
    base = np.clip(base, 10.0, None)
    return pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=FULL_DATES
    )


_frames: dict[str, pd.DataFrame] = {}


def _frame(ticker: str) -> pd.DataFrame:
    if ticker not in _frames:
        _frames[ticker] = _make_frame(_seed(ticker))
    return _frames[ticker]


def _fake_price(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kw):
    fs = pd.Timestamp(start) - pd.Timedelta(days=warmup_calendar_days)
    out = {}
    for t in tickers:
        df = _frame(t)
        out[t] = df.loc[(df.index >= fs) & (df.index <= pd.Timestamp(end))].copy()
    return out, []


def _fake_bench(start, end, **kw):
    df = _frame("SPY")
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].copy()


def install_mocks() -> None:
    paper.data.get_price_data = _fake_price
    paper.data.get_benchmark_data = _fake_bench
    tournament.data.get_price_data = _fake_price
    tournament.data.get_benchmark_data = _fake_bench


if __name__ == "__main__":
    install_mocks()
    sys.exit(cli.main(sys.argv[1:]))
