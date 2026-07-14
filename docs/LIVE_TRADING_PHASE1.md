# Live-Trading Foundation — Phase One

## Status and safety boundary

Phase one is an **offline execution foundation**, not a live-trading feature.
Its adapter is `FakeBroker`, an in-process, long-only cash-account simulator
used by tests and the demo. Phase two adds a separately isolated Alpaca
paper-only operations adapter; see `docs/LIVE_TRADING_PHASE2.md`. Neither phase
contains a live-money endpoint or a terminal order-submission command.

The historical strategies are unchanged. Their `TargetEvent` objects are not
orders and are not wired directly into this subsystem.

## Architecture

```text
research strategies
        |
        | future, reviewed portfolio aggregation (not in phase one)
        v
immutable account-level TargetPositionIntent
        |
        v
durable SQLite intent/order plan -> pre-trade risk -> broker contract
        ^                                             |
        |                                             v
reconciliation <- expected positions + cash <- durable fills/events
```

The account-level intent boundary is deliberate. If momentum and sector
rotation both want the same ETF, an allocator must combine their desired
shares before the OMS sees the target. Two sleeves must never independently
trade one brokerage position.

## Components

- `src/live/models.py` — immutable intents, orders, fills, quotes, positions,
  account snapshots, and lifecycle enums. Decimal quantities and prices avoid
  binary-float drift. All market timestamps must be timezone-aware.
- `src/live/broker.py` — the broker-neutral execution contract.
- `src/live/fake_broker.py` — deterministic, long-only fake broker with
  idempotent client order IDs, optional partial fills, and an acknowledgement-
  loss fault used to prove restart safety.
- `src/live/ledger.py` — SQLite schema for safety state, intents, orders, fills,
  fill-derived expected positions and cash, reconciliation runs, and append-only
  audit events. SQLite uses full synchronization and WAL mode for file-backed ledgers.
- `src/live/risk.py` — deterministic pre-trade controls.
- `src/live/oms.py` — target-to-delta conversion, durable-before-submit order
  planning, idempotent submission/recovery, event ingestion, arming, disarming,
  and kill-switch cancellation.
- `src/live/reconcile.py` — broker-to-ledger order/fill synchronization and
  strict position/open-order/fill comparison.
- `src/live/cli.py` — fake-broker demo and read-only ledger inspection.

## Safety invariants

1. Execution starts disarmed. Arming requires an explicit opening-position
   baseline and a recent clean reconciliation. Arming sessions expire rather
   than remaining valid indefinitely.
2. A kill switch is persistent. Clearing it leaves execution disarmed; arming
   is a separate action.
3. An order is written durably before broker submission.
   Only the SQLite transaction that actually inserts that order receives
   submission authority; competing processes can synchronize but cannot POST.
4. The client order ID is deterministically derived from the immutable intent.
   Recovery can synchronize that ID but never automatically resurrects a
   broker-missing order after a restart, disarm, or kill.
5. If broker submission succeeds but its acknowledgement is lost, the local
   order remains pending. Recovery queries the broker by client ID before any
   resubmission.
6. Broker fills are deduplicated by fill ID and atomically applied to expected
   positions and expected cash, including commissions.
7. New order construction stops and disarms execution when broker and ledger
   position quantities or cash differ.
8. Reconciliation issues disarm execution. Reconciliation never silently
   adopts current broker positions as a new baseline.
9. Phase one is long-only. Negative targets and sell quantities beyond the
   current long position are rejected.
10. File-backed ledgers are bound to one broker account, owner-readable only,
    integrity-checked on open, and protected by a cross-process execution fence.
    Audit rows have database triggers preventing update or deletion through the
    normal schema.

## Pre-trade checks

The default `RiskLimits` enforce:

- explicit arming and a disengaged kill switch;
- confirmed market-open state;
- quote symbol, recency, and future-timestamp validation;
- broker account-snapshot recency and future-timestamp validation;
- signal recency and future-timestamp validation;
- a price collar around the signal reference price;
- whole shares only;
- maximum order notional, open-order count, and daily turnover;
- broker-reported buying power and a configurable cash buffer;
- projected gross and per-symbol exposure;
- daily-loss and high-water drawdown stops; and
- no accidental short position.

Daily-loss and drawdown violations engage the persistent kill switch, not just
reject the current order. A unique partial SQLite index also enforces at most
one active order per account and symbol, including across competing processes.

Defaults are intentionally conservative examples, not account-specific
recommendations. A future deployment configuration must set and review every
limit for the actual account.

## Offline demonstration

The demo is opt-in and names its adapter in the output:

```bash
python -m src.live.cli demo --confirm-fake
```

To inspect persistence across processes, choose a temporary SQLite path:

```bash
python -m src.live.cli demo --confirm-fake \
  --db /tmp/wallst_phase1.sqlite3 --account-id FAKE-DEMO

python -m src.live.cli status \
  --db /tmp/wallst_phase1.sqlite3 --account-id FAKE-DEMO
```

Use a fresh database for each standalone demo. `FakeBroker` itself is
in-memory and intentionally does not pretend to survive between processes;
in production the external paper broker would provide that durable truth.

## Operating sequence for a future paper adapter

This is the required sequence followed by the separately reviewed Phase 2
paper-only adapter:

1. Open the durable ledger and broker session while disarmed.
2. On the account's first run only, explicitly record its opening positions and cash.
3. Recover pending orders by synchronizing their existing client IDs; never
   automatically resubmit a broker-missing order.
4. Reconcile orders, fills, positions, and cash; stop on every issue.
5. Record day-start and high-water equity from a durable account-state service.
6. Arm with an operator reason only after a recent clean reconciliation.
7. Convert an aggregated account target into an immutable target intent.
8. Run pre-trade risk and submit through the OMS.
9. Consume broker order/fill events and reconcile again.
10. Disarm at the end of the execution window. Engage the kill switch and
    cancel open orders on any unexplained state change.

## Failure recovery

- **Process dies before submit:** the ledger contains `pending_submit`;
  recovery records broker absence and leaves it disarmed for explicit operator
  resolution. The operator can auditably abandon the missing local order, then
  create a fresh target version after reconciliation.
- **Acknowledgement is lost:** recovery finds the broker order by client ID,
  imports its fills, and does not create another order.
- **Fill event is replayed:** the fill ID already exists, so positions do not
  change twice.
- **Manual or unknown broker activity appears:** reconciliation reports the
  external order/fill or position mismatch and disarms execution.
- **Kill switch is engaged:** all broker-reported open orders are canceled by
  default, the local lifecycle follows the broker's terminal truth (including
  a fill/cancel race), and execution remains persistently blocked. A failed
  cancellation is surfaced as an error while leaving the kill engaged.

## Phase-two paper connectivity

Phase two adds a hard-pinned Alpaca paper adapter plus guarded terminal
diagnostics, baseline, reconciliation, recovery, cancellation, and emergency
controls. Its complete operating procedure is in
`docs/LIVE_TRADING_PHASE2.md`. Phase 3 subsequently added the only terminal
paper-order path: immutable account-level aggregation, reviewed market
data/calendar checks, exact-hash offline approval, and a second explicit
paper-submit acknowledgement. See `docs/LIVE_TRADING_PHASE3.md`. There is still
no raw-order command or live endpoint.

## Later-phase prerequisites and exclusions

Phase 3 implemented timestamped IEX snapshots/trading-clock checks,
strategy-sleeve aggregation, durable equity guardrails, deployment policy, and
manual exact-hash-approved paper orchestration. Phase 4 still needs:

- streaming event ingestion, alert delivery, process supervision, and backups;
- credentialed paper soak testing and a shadow-vs-research parity report; and
- operator-reviewed limits, runbooks, and emergency drills.

Real-money endpoints, credentials, tax-lot policy, margin, shorts, options,
fractional orders, market-on-close staging, and automatic scheduling are all
explicitly outside phase one.
