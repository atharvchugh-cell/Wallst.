# Reviewed Paper Execution — Phase Three

## Completion boundary

Phase three implements the complete code path from reviewed strategy-sleeve
weights to broker-neutral account targets and explicitly confirmed Alpaca paper
orders. It does **not** add a live-money URL, generic order-entry command,
scheduler, or unattended strategy publisher. No paper or real order was sent
while building or testing it; all automated execution tests use `FakeBroker`.

The only order-capable terminal flow is:

```text
strict deployment + full sleeve snapshot
  -> fresh paper account/reconciliation/clock/IEX snapshots
  -> immutable quantity preview + SHA-256 hash
  -> offline approval of that exact full hash
  -> fresh revalidation
  -> approved Alpaca paper orders only
  -> disarm + reconciliation
```

There is no terminal `arm`, arbitrary-symbol `buy`, arbitrary-symbol `sell`,
raw-order, live-host, or endpoint-override command.

## What was added

- Strict JSON deployment and signal schemas reject duplicate keys, unknown
  fields, missing sleeves, symbols outside the managed universe, unsafe risk
  booleans, non-finite numbers, excessive weights, and symlinked artifacts.
- Deterministic `Decimal` aggregation combines independent sleeve weights into
  one account-level target per managed symbol. A missing target means zero only
  inside the explicitly reviewed managed universe.
- A dedicated paper account may not contain unmanaged, short, or fractional
  positions. Preview also requires zero open orders.
- Whole-share target quantities are rounded down using the current IEX ask.
  Preview enforces aggregate gross/symbol limits, order notional, cash reserve,
  maximum batch size, and the complete batch's projected daily turnover before
  approval; the OMS independently repeats risk checks immediately before each
  order.
- Read-only market data is hard-pinned to
  `https://data.alpaca.markets`, feed `iex`, with redirects and environment
  proxies disabled for the real credentialed session. A snapshot must contain
  an exact symbol set plus positive bid, ask, and last trade. The conservative
  timestamp is the older of latest quote/latest trade, and either component
  being future-dated fails closed.
- The paper trading clock must be fresh, open, internally consistent, and in
  regular America/New_York hours. The authenticated exchange calendar must
  prove that the signal follows the official close of the immediately prior
  trading session, including holidays and early closes. Approved batches
  expire at the end of their preview trading date, and signal staleness may
  never be configured beyond seven calendar days.
- Ledger schema v3 persists previous-close day-start equity and a monotonic
  high-water mark. A new trading date cannot silently reset either control;
  initialization/roll requires `--confirm-new-equity-session` and uses the
  broker's `last_equity` field.
- Preview contents are immutable JSON with a SHA-256 plan hash. Approval is
  offline, requires the full 64-character hash, operator, reason, and explicit
  acknowledgement. Another approved/executing/submitted batch for the account
  blocks approval. Approval reparses the complete plan; execution cross-checks
  every denormalized batch field against the hashed plan before changing
  control state.
- A separate file lock serializes batch orchestrators across processes. The
  underlying OMS retains its own submission/control fence and deterministic
  client IDs.
- Sells run before buys. Any risk rejection, broker rejection, or cancellation
  stops later batch items. A partially submitted batch is never claimed to be
  atomic; its durable status and every completed item remain visible.
- At most one order may remain unresolved. If a paper order is asynchronous,
  the batch stops at `submitted`; settle/reconcile it, then explicitly resume
  the same approved batch before the next item. This prevents later risk checks
  from reusing cash or exposure reserved by an accepted-but-unfilled order.
- A crash after broker acceptance leaves the batch resumable and disarmed.
  Recovery looks up the deterministic client ID and never blindly resubmits a
  broker-missing order.
- Execution always disarms in a `finally` path without requiring another
  broker call, then reports `complete`, `submitted`, `failed`, or an auditable
  interrupted/resumable state.

## Runtime prerequisite

Use an OpenSSL-backed Python runtime as described in
`docs/LIVE_TRADING_PHASE2.md`. The macOS system Python currently available in
this workspace uses LibreSSL 2.8.3, so the real network adapters deliberately
fail closed. Offline approval, batch inspection, and mocked tests remain usable.

## Artifact contract

Start from:

- `examples/live/paper_deployment.example.json`
- `examples/live/paper_targets.example.json`

Copy them to operator-controlled files. Do not edit the examples into a hidden
production configuration. Replace the account ID and signal timestamp/version.
The three-symbol target file is a **schema illustration, not output from the
research strategies and not a recommended portfolio**.

The target artifact is a **full current sleeve snapshot**, not the raw
`target_events.csv` event stream. Research `TargetEvent` rows are state changes
inside independent simulated sleeves and cannot safely be sent to an account.
For every configured sleeve, list its complete desired weights at one fresh,
timezone-aware `signal_at`. Omitted managed symbols are explicitly aggregated
to zero. Every configured sleeve must be present, even when its object is empty
because it is entirely in cash.

## Which strategy would trade?

Phase 3 does not choose a strategy. It trades exactly the manually supplied,
exact-hash-approved full target snapshot. There is no automatic bridge from a
backtest run to this order path yet. Calling `execute` without a target artifact
that an operator created and approved cannot make momentum, sector-rotation,
or regime decisions.

The repository's intended default research portfolio is:

- **60% momentum:** at each month-end, rank the 25 configured large-cap stocks
  by 126-trading-day return; a stock is eligible only when that return is
  positive and its close is above its 200-day SMA. Hold up to five at 20% of
  the momentum sleeve each, so each selected stock contributes up to 12% of
  the account. Empty slots remain cash.
- **35% sector rotation:** at each month-end, rank the 11 SPDR sector ETFs by
  return over the prior three completed calendar months and hold the top two
  at 50% of that sleeve each, or 17.5% of the account each.
- **5% regime switch:** when SPY is above its 200-day SMA at month-end, use the
  same top-two sector rotation at 2.5% of the account per ETF; otherwise keep
  this sleeve in cash. If it selects the same ETF as sector rotation, the
  account-level weights add.

That 60/35/5 mix is now published by the separate Phase 4 workflow; it is not
silently assumed by Phase 3. The illustrative Phase 3 target file aggregates
to AAPL 15%, XLF 17.5%, and SPY 0.5%—33% invested and 67% cash—because it
contains illustrative sleeve weights of 25%, 50%, and 10%. Those are not the
latest strategy selections. See `LIVE_TRADING_PHASE4.md` for the authentic
full-universe signed publisher; the older file remains only a Phase 3 schema
example.

The deployment must explicitly state
`"allocation_policy": "rebalance_to_deployment_weights"`. Under that policy,
each preview reapplies the configured 60/35/5-style capital weights before
combining the sleeve targets. This is intentionally explicit because it is not
the same as `src/portfolio.py`'s historical static-start allocation, where
sleeve weights drift and cash is never transferred between sleeves. Do not use
paper results as a parity claim for that backtest. A future drift-preserving
virtual-sleeve ledger and parity study remain outside the current Phase 4.

It must also state
`"execution_policy": "next_session_regular_hours_market"`. The existing
research strategies decide from daily-close data, so Phase 3 queries the
authenticated exchange calendar and rejects a signal unless it is timestamped
at or after the official close of the immediately prior session. This handles
weekends, holidays, and early closes without a hard-coded 16:00 assumption.
The example is timestamped just after the prior close. This prevents a same-day/intraday
look-ahead bridge, but a regular-hours market order still does **not** reproduce
the backtest's next-day-close synthetic fill. Market-on-close staging and a
measured research/paper timing policy remain Phase-4 work.

The account ID and managed symbol set are hard ownership boundaries. Phase
three refuses a dedicated account that holds a nonzero position outside that
set rather than guessing whether it is safe to leave or liquidate it.

## First preview

Complete Phase 2 bootstrap first and verify the dedicated paper dashboard.
Preview contacts Alpaca paper trading plus read-only IEX market data, but cannot
submit an order:

```bash
python3 -m src.live.paper_cli preview \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --deployment /absolute/path/to/paper-deployment.json \
  --targets /absolute/path/to/fresh-paper-targets.json \
  --confirm-paper-network \
  --confirm-new-equity-session
```

The equity-session flag authorizes initialization or a new-date roll only. It
does not reset an already-created day-start value on repeated same-day preview.

Review all of the output, especially:

- authenticated account ID and preview trading date;
- `market_data_feed` exactly `iex`;
- each sleeve allocation and aggregated target weight in the source files;
- every symbol's current quantity, target quantity, delta, quote timestamp,
  and reference price;
- total number of nonzero deltas;
- deployment risk limits; and
- the full `batch_id` and 64-character `plan_hash`.

If any field is wrong, do not approve it. Create a corrected target version and
run a new preview.

A target source/version is single-use. If market/account state changes enough
to create a different preview, publish a fresh `target_version`; the ledger
will not silently reprice the old version. A never-started preview or approval
can be retired offline:

```bash
python3 -m src.live.paper_cli void-batch \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --batch-id batch-REVIEWED_ID \
  --operator 'operator-name' \
  --reason 'targets withdrawn before execution' \
  --confirm-void-paper-batch
```

Voiding cannot roll back or hide a batch that already started.

## Offline exact-hash approval

Approval makes no network connection and constructs no broker/data adapter:

```bash
python3 -m src.live.paper_cli approve \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --batch-id batch-REVIEWED_ID \
  --plan-hash REVIEWED_FULL_64_CHARACTER_HASH \
  --operator 'operator-name' \
  --reason 'reviewed quantities, prices, date, account, and limits' \
  --confirm-approve-paper-batch
```

Inspect it again without credentials or network access:

```bash
python3 -m src.live.paper_cli batch-status \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --batch-id batch-REVIEWED_ID
```

## Explicit paper execution

This is the only Phase-3 terminal command that may submit orders. It is hard
pinned to Alpaca paper, but it changes the paper account:

```bash
python3 -m src.live.paper_cli execute \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --batch-id batch-REVIEWED_ID \
  --operator 'operator-name' \
  --reason 'execute reviewed paper batch' \
  --confirm-paper-network \
  --confirm-submit-paper-orders
```

Immediately inspect the output, paper dashboard, and ledger. `complete` means
all target intents are filled/no-op in the local synchronized lifecycle.
`submitted` means one or more broker orders remain active and require recovery
and observation. `failed` can still follow earlier successful items; inspect
positions rather than treating the batch as rolled back.

To synchronize an already-started batch without granting permission to submit
any new item, use:

```bash
python3 -m src.live.paper_cli settle-batch \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --batch-id batch-REVIEWED_ID \
  --operator 'operator-name' \
  --reason 'check terminal paper outcomes' \
  --confirm-paper-network
```

This is the required next-day path for an overnight `submitted`/interrupted
batch. It can synchronize fills/cancellations and close the durable batch, but
cannot submit a missing item. An unfinished item from an expired preview date
settles to failure rather than being sent on a later day.

For a restart or ambiguous acknowledgement:

```bash
python3 -m src.live.paper_cli recover \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --confirm-paper-network

python3 -m src.live.paper_cli reconcile \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --confirm-paper-network

python3 -m src.live.paper_cli execute \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --batch-id batch-REVIEWED_ID \
  --operator 'operator-name' \
  --reason 'resume after clean recovery' \
  --confirm-paper-network \
  --confirm-submit-paper-orders
```

The last command synchronizes the approved deterministic targets. It does not
give recovery permission to invent a replacement for a broker-missing order.
Use Phase 2's kill/cancel/abandon procedures when reconciliation is not clean.

## Emergency rule

At any uncertainty, run `local-kill` immediately if the network is unreliable,
or the normal networked `kill` if the paper broker can be reached. A local kill
does not cancel broker orders. Never clear the kill switch until dashboard,
ledger, open orders, fills, positions, and cash have been reviewed and a fresh
reconciliation completes after the reset.

## Phase-3 verification performed

Automated tests cover strict parsing and duplicate-key attacks, sleeve
aggregation, whole-share sizing, concentration/cash/order caps, unmanaged
positions, stale/future quotes, false/stale exchange clocks, wrong-day batches,
previous-close equity, high-water persistence, migration without invented
state, hash and denormalized-metadata tampering, approval bypass, holiday and
early-close calendars, quote ageing during preview, whole-batch turnover,
open-order preview, price movement, partial-batch stop, crash-after-acceptance
recovery, replay, terminal settlement verification, concurrent execute
processes, CLI confirmation separation, TLS/redirect/proxy guards, and all
Phase 1/2 regressions. Tests use stub HTTP sessions or `FakeBroker`; none
contacts Alpaca.

## Residual risk and Phase Four

Phase three is suitable for a manually supervised paper pilot, not unattended
operation and not real money. Important residual risks:

- IEX is one venue, while the paper simulator may use a different best-price
  view. Market orders have no guaranteed execution price and can move after
  the last risk check.
- A multi-order batch is not atomic. An outage or rejection can leave a valid
  partial portfolio requiring operator handling.
- The strategy snapshot is deliberately operator-provided; there is not yet a
  signed, automated publisher proving parity with a fresh research run.
- The checked-in three-symbol artifacts are schema examples and cannot
  represent the full possible 60/35/5 strategy universe. Their safety limits
  are illustrative; an operator must prove that a proposed full rebalance is
  compatible with gross exposure, turnover, cash, order-count, and managed-
  symbol limits. Phase 3 rejects rather than scales an incompatible target.
- The implemented allocation policy rebalances sleeves to reviewed deployment
  weights. It does not reproduce the backtest portfolio's drifting independent
  sleeve capital; that requires a virtual-sleeve ledger and parity evidence.
- The implemented execution policy uses a next-session regular-hours market
  order. It is not equivalent to the research engine's next-day-close fill.
- There is no streaming trade-update consumer, scheduler, alert delivery,
  process supervisor, off-host signed audit export, secrets manager, or tested
  backup/restore service.
- Paper simulation omits or simplifies market impact, queue position, latency
  slippage, some fees, dividends, and liquidity constraints.

Phase Four is a credentialed paper soak/shadow phase: run this flow repeatedly
with zero unexplained reconciliation issues, compare intended/research/paper
positions, inject crash/network/partial-fill drills, add alerts/supervision and
backup restore, and collect an operating record. Phase Five remains a separate,
explicitly authorized micro-live rollout; no Phase-3 code or instruction
authorizes it.

## Primary references

- Alpaca latest multi-symbol stock snapshots:
  https://docs.alpaca.markets/us/reference/stocksnapshots-1
- Alpaca latest stock quotes and feed choices:
  https://docs.alpaca.markets/us/reference/stocklatestquotes-1
- Alpaca US market clock used by the paper adapter:
  https://docs.alpaca.markets/us/reference/legacyclock
- Alpaca market calendar, including trading days and early closes:
  https://docs.alpaca.markets/us/v1.1/reference/getcalendar-1
- Alpaca paper environment and simulation limits:
  https://docs.alpaca.markets/us/docs/paper-trading
- Alpaca trading account `last_equity` definition:
  https://docs.alpaca.markets/us/v1.1/docs/account-plans
- Alpaca paper credential/domain separation:
  https://docs.alpaca.markets/us/v1.1/docs/authentication-1
