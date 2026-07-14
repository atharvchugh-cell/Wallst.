# Phase 4 — Supervised Alpaca Paper Trading

## Status and boundary

Phase 4 is implemented as a **paper-only** layer around the Phase 1–3 OMS,
ledger, risk engine, approval workflow, reconciliation, kill switch, and
arming controls. It adds authentic strategy publication, signed immutable
snapshots, a restart-safe scheduler, paper order-stream recovery, durable
alerts, automatic backups, health reporting, and paper-soak evidence.

It does not implement or authorize Alpaca live endpoints, live credentials,
margin, leverage, shorts, options, crypto, extended hours, or unattended
real-money execution. Phase 5 is separate and unimplemented.

No finite implementation is “bulletproof.” Host compromise, stolen signing
keys, upstream data errors, broker defects, exchange anomalies, and operator
mistakes remain possible. The system therefore fails closed and requires
supervised paper evidence rather than claiming certainty.

## Strategy published

The publisher runs the registered repository strategy classes; it does not
copy their formulas.

| Sleeve | Account allocation | Registered decision |
|---|---:|---|
| Momentum | 60% | Month-end top five by 126-session return, positive return and above SMA-200; 12% of account per selected stock; unused slots cash |
| Sector rotation | 35% | Top two of the configured 11 SPDR sectors by three completed calendar months; 17.5% of account each |
| Regime switch | 5% | If SPY is above SMA-200, use the same top two sectors at 2.5% each; otherwise cash |

Mean reversion is excluded. Overlapping sector and regime targets aggregate
before the OMS. A selected sector can therefore reach 20%. Sleeve attribution
remains in the signed snapshot.

The configured momentum universe is a current, hand-picked large-cap list. It
is **survivorship-biased**, not point-in-time. Every snapshot repeats that
disclosure and freezes the exact universe and broker asset identities.

## Publication and authenticity

`StrategyTargetPublisher` uses each registered class's `prepare()` and
`initial_events()` methods with a `MarketDataView` bounded at the decision
session. It requires:

- the authenticated official exchange calendar and completed official close,
  including early closes;
- the last session of the month and the next regular execution session;
- exact SPY calendar completion and exact required-symbol coverage;
- a forced post-close refresh (same-day caches are never trusted for publication),
  at least 200 complete finalized sessions, finite/positive adjusted OHLCV across
  the full indicator window, valid OHLC geometry, no carried-forward prior OHLC
  row even if volume changes, and no strategy-dropped registered ticker;
- current active, tradable US-equity asset metadata, supported exchange,
  stable asset ID, and no unresolved duplicate share class;
- a dedicated paper account with no short, fractional, duplicate, or unmanaged
  position; and
- exact fixed 60/35/5 configuration.

The snapshot records git SHA and dirty state, exact hashes/sizes of every
modified/staged/untracked file when labelled dirty state is permitted,
policy/deployment hashes, source
file hashes, asset metadata, data provenance and retrieval time, full sleeve
and aggregated targets, current positions, estimated deltas, pricing
semantics, publisher identity, account fingerprint, and expiration.

Canonical JSON is hashed with SHA-256. Authenticity is separate: an optional
signer interface is implemented by an operator-owned local HMAC-SHA256 key.
The key must be 32–4096 bytes, owned by the current operator, owner-only
(`0600` or stricter), a regular non-symlink file, and outside the repository.
It is never accepted as a CLI value or stored in the ledger.
When policy requires signing, unsigned or unverifiable snapshots cannot create
or execute a plan.

The example allows at most seven calendar days of snapshot/signal age so a
Friday month-end can reach Monday (including a holiday), but execution is still
pinned to the signed immediately-next exchange session. A catch-up is allowed
only with explicit confirmation before that session closes; afterward the run
is durably failed and publication is refused.

Snapshots are inserted once into the execution ledger and protected by SQLite
update/delete triggers. Optional exported snapshot files are created with
exclusive creation and mode `0400`.

The first execution artifact irreversibly binds a ledger to either the legacy
Phase-3 approval model or the Phase-4 signed-snapshot model. Migration refuses
mixed artifacts, the profile is protected by update/delete triggers, and a
Phase-4 plan can be approved only through its immutable snapshot link. Use a
fresh ledger when changing execution phases; do not reuse a Phase-3 ledger.

## Research prices versus execution prices

Adjusted research closes decide ranks and provide only a non-executable delta
estimate. On the expected next regular session, the existing Phase 3 preview
retrieves fresh Alpaca paper quotes and freezes whole-share quantities. Phase 4
adds exact quote coverage, maximum spread, and maximum deviation checks.
Phase 3 independently repeats freshness, price-collar, account, position,
turnover, concentration, cash, order-count, and OMS risk checks.

Phase-4 deployment, minimum-trade, concentration, order-count, session, and
cash-deployment limits are rechecked against current account equity and current
quotes at both preview and execution. Every active order reserves its remaining
quantity at a durable conservative risk price against account-wide cash,
buying-power, exposure, concentration, and turnover limits.

Orders are regular-hours `market` / `day`, whole-share, long-only target
position orders. Sells sort before buys. At most one unresolved order is
allowed; a later order requires fill recovery and clean reconciliation. An
exit never exceeds the reconciled broker quantity.

Immediately before the broker POST, after repeating deterministic client-ID
discovery, OMS invokes the Phase-4 durable authorizer while holding the same
cross-process fence used by disarm/kill, critical-alert, and stream-recovery
writers. Any intervening mode, batch, snapshot, reconciliation, stream, alert,
or control-state change blocks the POST and disarms the account before the
fence is released.

A market order does **not** reproduce the backtest's synthetic next-session
adjusted close. Paper fills also do not reproduce live queue position, market
impact, availability, latency, or slippage.

## Operating modes

- `observe`: publishes and audits targets; creates no execution plan.
- `shadow`: creates and exact-hash approves a simulated plan, then finalizes it
  as voided; an immutable ledger link independently prevents submission,
  including via the older Phase 3 command, without blocking the next month.
- `paper_manual`: prepares a plan but requires explicit offline exact-hash
  operator approval and an explicit paper-submit confirmation. It also requires
  the recovered stream, recent clean reconciliation, clear kill switch, and no
  unresolved critical alert.
- `paper_supervised`: retains explicit approval and additionally requires a
  connected/recovered stream, clean reconciliation, no kill switch, valid
  signature/snapshot, matching code/config/assets, and no unresolved critical
  alerts. It is not the example default.

The example policy defaults to `observe`. Changing mode changes the policy
hash and requires a newly published snapshot.

## Scheduler semantics

The scheduler queries the authenticated exchange calendar in bounded chunks,
selects the month's final actual session, respects early closes and closures,
and stages the first following regular session. A unique ledger constraint
prevents duplicate decisions. Runs are recorded as due, published, skipped,
delayed, or failed. Publication after the expected next open is a catch-up and
requires explicit operator confirmation.

An outstanding due/delayed/failed run remains the oldest next action, including
across month and year boundaries. After a prior terminal run, restart recovery
walks forward to the earliest missing due month with a bounded 24-month scan;
a fresh ledger does not invent older obligations. A missed run whose execution
window closed must be explicitly dispositioned with the offline
`skip-schedule` command, operator, reason, and confirmation. Skipped evidence is
terminal and immutable, and the scheduler then advances to the next missing
month.

Correctness does not depend on a continuously running process. A one-shot
scheduler can be invoked by `launchd`, cron, or another supervisor; after a
restart it reads the durable run/snapshot state and refuses duplicates. Health
computes the next action in the exchange session timezone and surfaces an
outstanding prior-month due/delayed/failed run instead of hiding it behind the
next calendar month.

## Stream and reconciliation semantics

The Alpaca paper `trade_updates` transport is hard-pinned to the paper
websocket. It handles bounded exponential reconnect, local sequence tracking,
duplicate event IDs, out-of-order events, multiple partial fills, fills while
cancel is pending, cancellation, rejection, expiration, and disconnects. The
top-level channel and event allowlist are strict; unsupported correction/bust
events and non-US-equity payloads fail closed into REST recovery rather than
being guessed at.

The durable stream state is not marked ready until websocket authentication,
`trade_updates` subscription, and REST reconciliation succeed. The stream
renews a 45-second ledger lease during quiet periods and performs an all-status
continuous reconciliation at least every 60 seconds; a stale lease or a clean
reconciliation older than 120 seconds blocks submission. A received event is
first persisted as `pending`; only a fully applied and reconciled event becomes
`applied`. A crash in between forces REST recovery before replay. Every applied
event reconciles broker and ledger truth. Order
replacement is intentionally unsupported: an old `replaced` order is treated
as terminal, while any successor not created under the known deterministic
client ID is an external-order mismatch requiring operator resolution.

The ledger remains authoritative locally. Every connect/reconnect first blocks
submission, then fetches account, positions, open orders, overlap-watermarked
recent orders in all statuses (including terminal orders), bounded recent fills
and known client IDs through REST, recovers unresolved orders, and reconciles.
The watermark starts before the previous reconciliation began, so activity that
arrived during that run is not skipped; recovery does not issue lifetime-wide
order or fill scans.
Only a clean result clears the recovery block. Unknown orders, positions, cash
differences, duplicate identifiers, missing local/broker state, or conflicts
stay recorded and generate critical alerts; nothing is silently deleted or
adopted.

## Alerts, health, backups, and soak evidence

Alerts have severity, deduplication, occurrence count, acknowledgement,
explicit resolution, and repeated critical escalation. Repeats update the
durable count without flooding every sink. Critical incidents remain a paper
submission blocker after recovery until an operator resolves them. Structured
console output is available;
an HTTPS webhook sink is optional, sends no broker credentials, disables
redirects and environment proxies, requires an explicit hostname allowlist,
rejects non-public literal/resolved addresses, pins the request to a validated
public address while retaining TLS SNI/hostname verification, and has a bounded
timeout. No URL is committed or persisted.

The health command is read-only unless explicit alert-write flags are used. It
reports mode, armed/kill state, broker connectivity, stream state, recent
reconciliation, latest signature- and provenance-valid snapshot, unresolved/overdue orders,
alerts, backup, database integrity, next action, and blockers.
`armed=false` is the safe boundary state and is not itself a readiness failure;
Phase 3 arms only inside the guarded submission sequence. Observe/shadow and
offline paper inspection are never reported submission-ready. Connected checks
also validate account identity, active status, USD currency, trade/account
blocks, cash, equity, buying power, and available account-age metadata.

Every state-changing trading-workflow command (`publish`, `skip-schedule`,
`prepare-plan`, `approve`, `run-paper`, `reconcile`, and `soak-observe`) creates a SQLite-safe
versioned backup automatically. If the workflow fails after or during a durable
state transition, it still attempts that evidence backup and preserves the
original workflow exception if backup also fails. Stream event ingestion remains in the durable WAL ledger
and is covered by scheduled/manual backups rather than copying on every frame.
Backups include a verified SQLite snapshot plus
credential-scanned deployment/policy copies and SHA-256 manifest. The backup
root must be dedicated, private, and separate from the ledger directory; files
and directories are fsynced before success is recorded. Retention deletes only
strictly valid backup IDs. Restore does not open a damaged active database,
re-verifies the temporary copy hash and SQLite integrity immediately before
atomic replacement, and fsyncs the replacement. Colocated SHA-256 files prove
consistency/corruption detection, not hostile authenticity; signed/off-host
manifests remain a deployment responsibility.

Every file-backed ledger holds a shared lifetime lock. Restore requires a
nonblocking exclusive lock on the destination and therefore refuses replacement
while any cooperating scheduler, stream, CLI, or other ledger process remains
open; the runbook still requires stopping everything before replacement.

Soak reports provide daily and cumulative decisions, snapshots, plans, orders,
fills, rejects, cancellations, partials, reconciliation, reference slippage,
optional next-close slippage, target-weight error, alerts, uptime,
disconnect/recovery counts, database integrity, and uniqueness evidence.
Daily fill slippage joins fills to their originating order even when the order
was created on a prior day. Daily stream disconnect/recovery counts come from
dated state-transition audit events; routine reconciliations are not mislabeled
as recovery events.

## CLI and credentials

Use only:

```bash
export ALPACA_PAPER_API_KEY='...'
export ALPACA_PAPER_API_SECRET='...'
export WSLAB_PHASE4_SIGNING_KEY_FILE="$HOME/.wallst-strategy-lab/phase4-signing.key"
```

The CLI rejects live-style environment names. There is no base-URL option and
the adapters remain pinned to `https://paper-api.alpaca.markets`,
`https://data.alpaca.markets`, and the paper websocket. Secrets are never
printed, persisted, included in snapshots, or passed as normal CLI arguments.

The signing key is required for publish, prepare, approve, and paper execution.
It is deliberately not required merely to reconcile, start REST/websocket
recovery, create/verify a backup, inspect health, or record/report soak evidence;
those recovery paths must remain available after key loss. They do not authenticate
or authorize a target without the key.

Create a key once outside the repository:

```bash
umask 077
mkdir -p "$HOME/.wallst-strategy-lab"
openssl rand -out "$HOME/.wallst-strategy-lab/phase4-signing.key" 32
chmod 600 "$HOME/.wallst-strategy-lab/phase4-signing.key"
```

Keep the ledger, snapshots, and backups outside the git worktree so runtime
files do not change the code-state proof.

## Exact supervised commands

Set operator paths:

```bash
DB="$HOME/.wallst-strategy-lab/paper-ledger.sqlite3"
DEPLOYMENT="$HOME/.wallst-strategy-lab/phase4-deployment.json"
POLICY="$HOME/.wallst-strategy-lab/phase4-policy.json"
SNAPSHOTS="$HOME/.wallst-strategy-lab/snapshots"
```

The shipped example policy is deliberately `observe`. The shadow rehearsal
below requires a copied policy whose JSON `mode` is exactly `shadow`; the CLI
flag is an assertion and will not override the file. Likewise, moving to
`paper_manual` or `paper_supervised` requires changing the policy first and
publishing a fresh signed snapshot—never reuse the shadow snapshot or plan.

After the official final monthly close, publish only:

```bash
python3 -m src.live.phase4_cli publish \
  --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY" \
  --snapshot-dir "$SNAPSHOTS" --mode shadow \
  --confirm-paper-network --confirm-publish
```

If the process missed the expected next open, add
`--confirm-manual-catch-up` only after following the monthly checklist.
If the signed next-session execution window has closed, never catch up that
target. Record the reviewed disposition offline so later months can proceed:

```bash
python3 -m src.live.phase4_cli skip-schedule \
  --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY" \
  --run-id schedule-REVIEWED_ID --operator 'operator-name' \
  --reason 'execution window closed; incident and broker state reviewed' \
  --confirm-skip-schedule
```

This command does not load the signing key or contact the broker. Resolve its
missed-run critical alert only after the alert-response recovery checks.

Start/order-stream recovery in a supervised terminal:

```bash
python3 -m src.live.phase4_cli stream \
  --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY" \
  --confirm-paper-network --confirm-start-paper-stream
```

During the expected next regular session, prepare the plan (shadow will
auto-approve but can never submit):

```bash
python3 -m src.live.phase4_cli prepare-plan \
  --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY" \
  --snapshot-id snapshot-REVIEWED_ID \
  --confirm-paper-network --confirm-prepare-paper-plan \
  --confirm-new-equity-session
```

For a `paper_manual` or `paper_supervised` policy, review the complete plan,
then approve offline:

```bash
python3 -m src.live.phase4_cli approve \
  --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY" \
  --batch-id batch-REVIEWED_ID --plan-hash FULL_64_CHARACTER_HASH \
  --operator 'operator-name' --reason 'reviewed snapshot, quantities, limits, and session' \
  --confirm-approve-paper-plan
```

Submit only to Alpaca paper:

```bash
python3 -m src.live.phase4_cli run-paper \
  --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY" \
  --batch-id batch-REVIEWED_ID --operator 'operator-name' \
  --reason 'supervised paper rebalance' \
  --confirm-paper-network --confirm-submit-paper-orders
```

Inspect operations:

```bash
python3 -m src.live.phase4_cli health --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY"
python3 -m src.live.phase4_cli alerts --db "$DB" --unresolved-only
python3 -m src.live.phase4_cli soak-report --db "$DB"
```

The default health command is read-only. A host monitor that should persist,
escalate, deliver, and back up health-derived alerts must opt into the write:

```bash
python3 -m src.live.phase4_cli health \
  --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY" \
  --record-health-alerts --confirm-health-alert-write
```

Optional measurements that cannot be derived offline from the OMS (next-close
slippage, actual-weight error, and process uptime) are recorded explicitly and
backed up before reporting:

```bash
python3 -m src.live.phase4_cli soak-observe \
  --db "$DB" --deployment "$DEPLOYMENT" --policy "$POLICY" \
  --trading-date 2026-07-01 --metric process_uptime_seconds --value 3600 \
  --operator 'operator-name' --reason 'supervisor uptime reading' \
  --confirm-record-soak-observation
```

## Runbooks

- [Operator startup](PHASE4_OPERATOR_STARTUP.md)
- [Monthly rebalance](PHASE4_MONTHLY_REBALANCE.md)
- [Stream recovery](PHASE4_STREAM_RECOVERY.md)
- [Reconciliation](PHASE4_RECONCILIATION.md)
- [Backup and restore](PHASE4_BACKUP_RESTORE.md)
- [Alert response](PHASE4_ALERT_RESPONSE.md)
- [Paper-soak graduation](PHASE4_SOAK_GRADUATION.md)
