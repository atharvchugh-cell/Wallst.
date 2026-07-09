# Red Team — everything known to be wrong or biased about these backtests

This document attacks the tournament's own results before anyone else can.
Findings are split into (A) **established from the code and data model alone**
— true regardless of what any backtest run prints — and (B) **what the live
runs must additionally establish**, with the verdict framework to apply.
This development environment has no market-data egress, so every empirical
number must come from the local runs listed at the bottom; nothing in this
repo should be read as claiming validated live-data results until then.

---

## A. Standing findings (true independent of any run's numbers)

### A1. Survivorship bias — worst for momentum, bad for everything on stocks
Both stock universes (the hardcoded 25 and `us_50b`) are **today's
survivors**. This inflates every stock strategy, but it inflates
**momentum most**: momentum buys past winners, and a today-selected universe
is literally a list of long-horizon past winners — the losers momentum would
have ridden down (Enron-, Lucent-, GE-2017-shaped paths) are absent by
construction. Mean reversion is somewhat less exposed (it bets on bounces of
temporarily-down survivors) but still benefits: names that dipped and never
recovered aren't in the list. The sector strategies are far cleaner — the 11
SPDR ETFs are a fixed, complete family with no membership selection.
**Consequence:** cross-family comparisons (momentum-on-stocks vs
sector-rotation-on-ETFs) are biased IN FAVOR of the stock strategies.
Momentum beating sector rotation in this tournament is weak evidence;
sector rotation beating momentum despite momentum's tailwind is strong.

**Partial mitigation — the look-ahead half of the problem is now fixed, the
selection half is not.** A current-$50B `us_50b` snapshot contains names that
did not exist for older windows (ARM, GEV, HOOD, SNOW, ABNB, APP, RKLB, …).
Running them in, say, a 2019 window would be a *look-ahead* error (trading a
company before it was public). The tournament now filters, per strategy and
per window, to tickers with enough listing history to warm up that strategy's
indicators for that window, excluding the rest with an explicit reason
(reported per strategy/window). That removes the look-ahead artifact and the
noisy degraded runs it caused. It does NOT remove survivorship bias: the
universe is still *today's* survivors, so within any window the members are
the ones that went on to reach $50B and stay listed. Treat the filter as
"no longer also wrong about listing dates," not as "survivorship-corrected."

### A2. The incumbents' defaults are in-sample tuned; the challengers' are not
`config.py` records that mean reversion's RSI 35 / SMA-30 / 10-day timeout
and sector rotation's top-2 were re-tuned mid-development against one
historical stretch specifically to beat SPY ("Improve strategy parameters to
beat SPY performance" is in the git history). The three new strategies
deliberately use canonical literature constants (126-day momentum, 200-day
SMA) or round disclosed heuristics (-15%/5-day). **Consequence:** on any
window overlapping the tuning period, the incumbents carry an unfair,
unquantified advantage. The `--tournament-param-sensitivity` variants
deliberately include the incumbents' ORIGINAL pre-tuning values
(RSI 30, 20-day timeout, top-3) — if the shipped defaults beat those
variants only on the full 2019-2024 window but not on sub-windows, that gap
is the tuning showing, not edge.

### A3. Execution model flatters high-turnover strategies
Close-to-close fills with 5 bps and no spread/impact/slippage is optimistic
for a small account trading US large caps (spreads are tight, but not zero,
and fills are not closing prints). The bias scales with turnover:
mean reversion (10-day timeout, 5 slots) trades far more than the monthly
strategies. The "stop-loss" is a delayed close-to-close exit — a real
overnight gap through the stop fills worse than modeled, and mean-reversion
entries are exactly the names most likely to gap. **Consequence:** treat the
20 bps cost-sensitivity row as closer to "honest" for mean reversion than
the 5 bps headline; a mean-reversion edge that dies at 20 bps was never real.

### A4. Tax drag is ignored and hits the families unequally
Everything is pre-tax. Mean reversion's gains are ~100% short-term (10-day
max hold) — taxed as ordinary income. Momentum/sector rotation hold 1+
months (still mostly short-term but with lower turnover); regime_switch can
sit in cash. **Consequence:** after-tax, the ranking between a
mean-reversion family and a monthly family tightens or flips even when
pre-tax returns look similar. Any final decision made on pre-tax numbers
alone overstates the high-turnover strategies.

### A5. Regime filters are exposed to a known, structural failure mode
A 200-day-SMA regime filter exits after crashes start and re-enters after
recoveries start. In a V-shaped crash-recovery (2020) it takes the loss AND
misses the rebound — the worst of both. The named `2020_covid` window exists
precisely to price this; the `2022_bear` window prices the payoff case.
**Consequence:** judging `regime_switch`/`mean_reversion_filtered` on either
window alone is cherry-picking, in either direction. Only the cross-window
robustness section is a fair read.

### A6. Small sample sizes for monthly strategies
A 1-year window gives a monthly strategy ~12 decisions; even 2019-2024 gives
~72. Sector rotation's top-2 concentration makes single lucky/unlucky months
material. Sharpe ratios computed on ~250 daily points of a monthly-rebalanced
sleeve are statistically fragile, and CAGR over one-year windows is mostly
noise. **Consequence:** differences in robustness_score of one window's worth
(e.g. 0.67 vs 0.83 over 6 windows) are not meaningful separations. Demand
consistency, not decimal places.

### A7. Cash earns 0%
`regime_switch` (risk-off) and `momentum` (defensive fallback) sit in cash
that earns nothing, while 2022-2024 T-bills paid ~4-5%. **Consequence:**
these strategies' risk-off windows are UNDERSTATED by roughly the bill yield
times time-in-cash — the one bias in this repo that runs against the
defensive strategies. Noted rather than fixed: crediting a yield would
require a rate series and open its own accuracy questions.

### A8. SPY row uses each window's common intersection range
Strategies' effective ranges differ (warmup, ETF-inception clipping), so the
SPY comparison row covers the intersection. For windows clipped by XLC's
2018 inception + lookback this is honest but means "the tournament window"
isn't always the full labeled window. The report prints the actual range —
read it before quoting any excess return.

### A9. Data-quality single points of failure
All prices are Yahoo's adjusted series: retroactive adjustment revisions,
occasional bad prints, and the synthetic nature of adjusted prices are
inherited silently. The canonical calendar is SPY's own trading dates — a
hole in Yahoo's SPY series silently shrinks everyone's calendar (guarded
for the known cases: unfinalized today-bar excluded, benchmark short-end
hard-fails; not guarded against interior holes, which have not been
observed). Interior gaps in any non-benchmark signal ticker hard-fail.

### A10. What "no lookahead" does and does not cover
`MarketDataView` structurally prevents strategies from reading prices or
indicators dated after the walk day, and mutate-all-future-dates tests cover
every strategy family. It does NOT protect against *specification-level*
leakage: choosing the regime windows, the -15% knife threshold, or the
robustness-score formula while knowing (from general market knowledge) what
2020/2022 looked like is a mild, unavoidable form of hindsight shared by all
research on named historical windows. Disclosed rather than pretended away.

---

## B. Verdict framework for the live runs

Run the commands at the bottom locally, then apply — mechanically, no
special pleading:

1. **Beats-SPY gate.** A strategy must beat SPY's total return in a majority
   of the regime windows (robustness section). The account's alternative is
   an index fund; a strategy that loses to SPY most windows is dead on
   arrival regardless of how clever it is. *(A1 means momentum needs a wider
   margin: if momentum-on-stocks only narrowly beats SPY, its survivorship
   tailwind likely accounts for it.)*
2. **Drawdown gate.** Max drawdown no worse than SPY's in most windows, and
   worst-window drawdown must not be catastrophic for a $15k account (a
   -40% worst window on $15k is $6k gone — behaviorally fatal even if the
   backtest recovers).
3. **Cost gate.** Excess return vs SPY must stay positive at 20 bps
   (`--tournament-cost-bps-list 0,5,10,20`). Sign flip = fail (per A3 this
   is the honest cost level for the high-turnover family).
4. **Fragility gate.** No beat-SPY sign flip across the disclosed parameter
   variants (`--tournament-param-sensitivity`). If nudging one knob turns a
   winner into a loser, the "edge" is the knob, not the strategy.
5. **Sample-size discipline** (A6): treat any two strategies within one
   window's worth of robustness_score of each other as tied.

Verdict vocabulary — every strategy lands in exactly one bucket:
- **paper-trade candidate**: passes all four gates across the regime
  windows. (Even then: paper trading only. Nothing here validates live
  execution, and A1 cannot be fixed with free data.)
- **research further**: fails exactly one gate, or passes but within tie
  range of SPY. Worth one more targeted investigation, not capital.
- **abandon**: fails two or more gates, or loses to SPY in most windows.

### What can already be said without the runs (from A-findings + construction)

- **`mean_reversion` (baseline)** starts with the weakest hand: highest
  turnover (A3), worst tax profile (A4), in-sample-tuned defaults (A2), and
  a known failure mode in sustained downtrends (it keeps catching knives —
  the exact motivation for the filtered variant). Expect the cost gate to be
  its likeliest failure. Its job in this tournament is to be the control.
- **`mean_reversion_filtered`** is the cleanest *experiment* in the repo:
  identical to the control except two entry filters, so (filtered −
  baseline) per window directly prices the filters. If it does NOT improve
  drawdown in `2022_bear`, the filters are not doing their one job — abandon
  the variant regardless of raw return.
- **`momentum`** must clear the widest bar (A1). Judge it primarily on
  drawdown control and cost robustness, which survivorship inflates least.
- **`sector_rotation`** is the least-biased stock-free strategy but is
  always fully invested (top-2 of 11 even in a bear) — expect `2022_bear`
  drawdown near or worse than SPY's; top_k=2 is also its tuned, fragile knob
  (watch the top_k_3 variant).
- **`regime_switch`** is the purest robustness bet: it will lose the
  `2020_covid` window by design (A5) and should earn its keep in
  `2022_bear` + score on worst-window drawdown. If it fails the drawdown
  gate too, it has nothing.

**No strategy in this repo is eligible for real money on the basis of these
backtests.** The ceiling for any of them is paper trading, because: the
universes are survivorship-biased (unfixable with free data, A1), execution
is simulated close-to-close (A3), results are pre-tax (A4), and none of this
has been validated on data the strategies' parameters could not have seen.
What paper trading must then prove before that conversation changes:
live-quote fills within the modeled cost envelope, real spread/slippage
measurement, and out-of-sample persistence over a period chosen in advance.

---

## C. Exact local commands (this sandbox has no market-data egress)

```bash
# 1. The headline tournament: all five strategies, regime windows,
#    cost + parameter sensitivity. One command, everything in section B.
python3 backtest.py --strategy tournament --start 2019-01-01 --end 2024-12-31 \
    --capital 15000 --tournament-windows regimes \
    --tournament-cost-bps-list 0,5,10,20 --tournament-param-sensitivity

# 2. Repeat on the larger current-snapshot universe (stock strategies only
#    are affected) and compare the two tournament_report.txt files:
python3 backtest.py --strategy tournament --start 2019-01-01 --end 2024-12-31 \
    --capital 15000 --tournament-windows regimes --universe us_50b --refresh-universe

# 3. Existing modes, unchanged, for continuity with earlier results:
python3 backtest.py --strategy compare --start 2019-01-01 --end 2024-12-31 --capital 15000
python3 backtest.py --strategy robustness --start 2019-01-01 --end 2024-12-31 --capital 15000
```

Read `tournament_report.txt` bottom-up: assumptions block first, then
robustness, then the per-window tables. If a number looks great, re-read
section A before believing it.
