# Frozen baseline reference

The immutable control for all strategy-lab research: the fixed
**60% momentum / 35% sector_rotation / 5% regime_switch** portfolio,
$15,000 starting capital, 5 bps per-trade costs, fractional shares,
2019-01-01 through 2024-12-31, default universe.

- `baseline_snapshot.json` — the frozen numeric reference (final equity
  $52,495.09, CAGR 23.23%, Sharpe 1.07, max drawdown -27.61%), produced by
  the lab engine with **every enhancement disabled**, which
  `tests/test_lab_equivalence.py` proves identical to `--strategy portfolio`.
- `run_manifest.json` — the full reproducibility manifest for that run
  (git SHA, dependency versions, strategy parameters, universe, data-cache
  fingerprint, artifact hashes).

Reproduce with:

```bash
python3 backtest.py --strategy strategy_lab --experiment baseline \
    --start 2019-01-01 --end 2024-12-31 --capital 15000
```

Caveats (see docs/RED_TEAM.md): results depend on the local Yahoo-adjusted
price cache identified by the manifest's `data_cache` fingerprint; the stock
universe is survivorship-biased; all figures are pre-tax, close-to-close,
research-only. These numbers are a reference check for the lab's experiment
comparisons, not hardcoded test expectations.
