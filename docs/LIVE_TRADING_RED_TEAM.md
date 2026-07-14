# Phase 1–4 Adversarial Review

## Verdict

Reviewed and hardened on 2026-07-14. The result is suitable for offline OMS
testing and a supervised, signed-target, exact-hash-approved Alpaca paper pilot.
No credentialed paper request or order was made during implementation. It is
**not bulletproof**, is not ready for unattended operation, and is not
authorized for live money.

No finite test suite can prove the absence of defects, broker failures, host
compromise, exchange anomalies, or operator error. “Bulletproof” is therefore
not a defensible engineering claim. This report records what was attacked,
what was fixed, and what remains outside the boundary.

## Fixed findings

| Severity | Attack | Resolution |
|---|---|---|
| Critical | Two processes racing one intent could both believe they created the durable order and both become eligible to POST. | `plan_order_with_created` grants submission authority only to the transaction that inserted the row. A two-connection concurrency test proves exactly one grant. |
| Critical | Disarm or kill could race between risk approval and broker submission. | A cross-process file fence serializes control changes and POST. Submission rechecks armed/kill state, arming age, durable order state, and client ID inside the fence. |
| Critical | A ledger could be operated using credentials for a different paper account. | The opening baseline permanently binds the ledger to one account. Every OMS, reconciliation, CLI, and emergency path verifies that binding. |
| High | Reconciliation or broker transport failure could abort while leaving an earlier armed state intact. | Broker errors during intent processing and all reconciliation exceptions persistently disarm and append failure audits. A failed reconciliation invalidates the prior clean authorization sequence. |
| High | Networked kill attempted account discovery before persisting the emergency stop. | A bound ledger is killed locally before the first network call. `local-kill` provides the same persistent stop without claiming broker cancellation. |
| High | A repeated fill ID with changed price, quantity, order, or account fields was treated as an idempotent replay. | Only byte-equivalent semantic fill content is idempotent. Any changed field is a ledger conflict; cumulative activities also cannot exceed broker-reported filled quantity. |
| High | Alpaca `done_for_day` could be misclassified as canceled even though it may receive later updates. | It remains active; if it has filled quantity it remains partially filled. `stopped`, pending-cancel, and pending-replace are also kept active. |
| High | Cancel/fill races and asynchronous cancellation could be reported as safely canceled. | Cancellation rereads broker truth. A fill is imported; an order still open makes cancellation/kill fail while the kill remains engaged. Kill re-queries all open orders after cancellation. |
| High | HTTP redirects or environment proxy variables could move paper credentials outside the pinned request path. | Redirects and `requests` environment inheritance are disabled. The trading base URL remains exactly the Alpaca paper host. |
| Medium | Stale order snapshots, duplicate broker rows, account-ID changes, or malformed/non-finite values could be collapsed or leak raw exceptions. | These now raise sanitized broker errors and invalidate execution authorization. |
| Medium | Limit-order risk used the current quote rather than the maximum buy limit. | Cash, notional, turnover, and collar calculations use the explicit limit as the risk price. |
| Medium | Ledger files inherited permissive filesystem modes and audit rows were only append-only by convention. | Database and lock files are forced to mode `0600`; quick and foreign-key checks run on open; SQLite triggers reject audit update/delete operations. |
| Medium | Fractional or short positions from an older ledger could reconcile cleanly. | New baselines reject them and reconciliation explicitly flags legacy unsupported holdings even when quantities otherwise match. |
| Critical | Independent strategy sleeves could fight over one brokerage position or implicitly liquidate an unrelated holding. | Strict full-snapshot aggregation creates one account-level target per explicitly managed symbol and refuses any nonzero unmanaged account position. |
| Critical | A terminal submission could bypass review or execute a changed preview. | Preview is immutable and SHA-256 hashed; offline approval requires the exact full hash, operator, reason, and acknowledgement. Execute accepts only that stored approved plan. |
| High | Two batch processes could interleave items even though individual order IDs were idempotent. | A separate cross-process batch lock serializes orchestration; the OMS submission fence remains independently active. |
| High | Restarting intraday could reset day-start/high-water controls to a convenient current value. | Schema v3 uses broker previous-close `last_equity`, preserves a monotonic high-water, and requires explicit confirmation to roll a new trading date. |
| High | A broker `is_open` flag or one fresh field could mask an impossible/stale market snapshot. | Regular-hours geometry, clock freshness, next-open/close consistency, exact symbol coverage, and both quote/trade timestamps are validated. |
| High | A later item could continue after an earlier batch member was rejected. | Any risk rejection, broker rejection, or cancellation stops subsequent items; the already-completed subset remains auditable as a non-atomic partial batch. |
| High | Multiple asynchronous accepted orders could each reuse cash/exposure not yet reflected in fills. | Batch execution stops after the first unresolved order. Settlement and another explicit same-day resume are required before the next item. |
| Medium | Raw research target events could be mistaken for executable portfolio state. | Phase 3 accepts only a complete, versioned sleeve snapshot. Raw event streams are explicitly outside the execution contract. |
| High | A holiday, skipped session, or early close could make a weekday/16:00 timestamp look like a valid latest close. | Preview now requires the authenticated exchange calendar to prove the signal follows the official close of the immediately prior session; signal age is capped at seven days. |
| High | SQL tampering with denormalized batch metadata could redirect pre-integrity control-state changes or let approval rely only on a hash column. | Approval reparses the full plan, and load/execute/settle cross-check hash, source, account, deployment, date, and signal metadata before acting. |
| High | A multi-order plan could be guaranteed to exceed the daily-turnover limit only after earlier orders filled. | Preview now rejects the whole batch when existing plus planned bid/ask turnover exceeds the configured cap; per-order risk remains independently active. |
| Medium | Quotes could age past their limit while account and reconciliation calls completed after the initial freshness check. | Preview repeats exact-coverage and freshness validation immediately before quantities are frozen. |
| Medium | Replaying `execute` or settling a terminal batch could claim a fresh clean reconciliation without performing one. | Idempotent execute replay explicitly reports no fresh reconciliation; `settle-batch` validates account state, reconciles even terminal batches, and audits operator/reason. |

The focused adversarial tests cover concurrency, control fencing, lost
acknowledgements, duplicate intents, partial fills, replay collisions, restart
recovery, wrong-account credentials, stale/future data, broker outages,
redirects, TLS/runtime checks, cancellation races, local/network kills, cash and
position drift, and audit immutability. Phase-3 coverage additionally attacks
duplicate JSON keys, hash/database plan tampering, omitted sleeves, unmanaged
holdings, concentration/cash/batch caps, stale/future IEX components, false
exchange clocks, holiday/early-close calendars, wrong-day approval, approval
bypass, denormalized metadata changes, quote-age races, guaranteed batch
turnover failure, open-order preview, price-collar movement, partial-batch stop,
crash-after-acceptance resumption, terminal settlement truthfulness, CLI
confirmation separation, and concurrent batch execution.

## Phase 4 additions — 2026-07-14

Phase 4 was attacked across strategy equivalence, lookahead, signing,
scheduling, quote sizing, mode bypass, restart boundaries, stream replay,
reconciliation, endpoint/credential injection, backups, and operator recovery.

| Severity | Attack | Resolution |
|---|---|---|
| Critical | An automatic publisher could quietly reimplement or drift from the researched formulas. | Publication instantiates the registered strategy classes and calls their existing `prepare`/`initial_events` paths. It verifies the exact 60/35/5 weights, strategy parameters, event session, output shape, and no mean-reversion sleeve. |
| Critical | A normal SHA-256 digest could be presented as authenticity. | Canonical content hashing is separate from an operator-controlled signing interface. The shipped signer uses a mode-0600, non-symlink local HMAC key; required-signature policy blocks unsigned, wrong-key, malformed, changed, or expired snapshots. |
| Critical | A signed target could be executed after code, policy, deployment, input data, universe, account, or broker asset identity changed. | Execution rechecks git SHA/dirty entries, policy/deployment hashes, absolute input hashes, registered parameters/universe, account fingerprint, and current broker asset metadata/IDs before plan creation and again before submission. |
| Critical | An automatically approved shadow plan could be submitted through the older Phase-3 CLI. | The plan-to-snapshot mode link is inserted atomically with the execution batch and is immutable. `PaperExecutionService.execute` refuses every linked non-submitting plan, regardless of caller. |
| Critical | A restart between plan persistence and mode linkage could leave an unclassified executable plan. | Phase-4 link insertion occurs in the same SQLite transaction as batch creation; replay cross-checks the stored policy tuple. |
| High | A duplicate or late scheduler process could publish twice or treat a guessed weekday as month-end. | Bounded authenticated-calendar queries find the final actual session and next regular session. Decision session is unique in the ledger; published runs cannot be rewritten; post-next-open catch-up requires explicit confirmation. |
| High | A current-session/early-close signal could use an unfinished daily bar. | Publication requires current time after the official authenticated close, the data cutoff exactly at that session, complete SPY/reference calendar, exact symbol coverage, finite OHLCV, and no stale final row. Early close comes from the calendar, never a hard-coded time. |
| High | Adjusted research closes could be mistaken for executable quotes. | Snapshot semantics label adjusted closes non-executable. Quantity freeze uses fresh paper bid/ask, then checks age, spread, deviation, collar, turnover, concentration, order count, cash deployment, minimum notional, and Phase-3/OMS limits. |
| High | Stream duplicates, regressions, multiple partials, or a fill while cancel was pending could corrupt local state. | Event IDs and payload hashes are durable; changed replays conflict; sequence/time regressions block submission and require REST recovery; cumulative partials are idempotent fills; active pending-cancel states can still fill. |
| High | Reconnect could resume submission from in-memory stream state. | Every connect/reconnect disarms, marks recovery active, fetches account/positions/open orders/fills/known client IDs, recovers unresolved orders, and reconciles. Only a clean durable result clears the recovery block. |
| High | External broker activity could be silently adopted or deleted. | Unknown orders, fills, positions, cash differences, missing orders, and duplicate identifiers remain reconciliation issues, create critical alerts, and require explicit resolution. |
| High | Backups made by copying a live WAL database could be inconsistent or contain credentials. | The online SQLite backup API creates a standalone copy; quick/foreign-key checks and hashes are verified. Only credential-scanned JSON configs are copied. Restore re-verifies all content and requires explicit replacement confirmation. |
| Medium | Repeated alerts could flood operators or disappear after acknowledgement. | Alerts persist in SQLite with unresolved deduplication, occurrence counts, acknowledgement separate from resolution, and durable critical escalation. Sink failure creates another durable alert. |
| Critical | A snapshot could verify in memory but fail after a cold-process JSON reload because the config loader changed JSON numbers to `Decimal` before hashing. | Signed snapshots use a strict duplicate-key loader that preserves native JSON number types. File round-trip and cold-process tests verify the identical envelope and signature. |
| Critical | A websocket process could mark itself connected before authentication, or leave a timeless healthy flag after crashing. | Readiness is persisted only after paper-stream auth/subscription and REST recovery. A quiet-stream heartbeat renews a 45-second durable lease; stale state blocks execution and health. |
| High | Finalized shadow approvals stayed active and prevented the next month's equity session, encouraging unsafe manual cleanup. | Shadow plans are exact-hash approved, then finalized as `voided`; the immutable non-submit link remains an independent bypass guard. A three-process/two-month replay test proves rollover and zero orders. |
| High | A two-day snapshot/signal age failed legitimate Friday-to-Monday month ends, while an unbounded catch-up could publish after the execution opportunity was gone. | Age is bounded at seven calendar days to cover weekends/holidays, execution is pinned to the signed immediately-next session, and catch-up is refused after that session closes. |
| High | Code/input drift checks could be skipped when replaying an existing plan or immediately before paper execution. | Publisher provenance is revalidated at persistence, plan replay, approval, and submission; current broker asset identity is also rechecked before planning and submission. |
| High | A rewritten backup manifest could use path traversal, or restore could replace a symlink/open SQLite path. | Verification enforces an exact manifest schema, fixed ledger basename, safe config basenames, regular non-symlink files, and hashes. Restore rejects symlink targets and the CLI closes SQLite before atomic replacement. |
| Critical | A Phase-4 manual/supervised plan could be passed directly to the legacy Phase-3 execution service, bypassing current signature, provenance, stream, reconciliation, and alert gates. | Every Phase-4-linked batch now requires a supervisor reauthorization callback inside the cross-process batch guard. Direct Phase-3 execution is rejected both before and after approval; the supervisor rechecks the complete boundary immediately before OMS use. |
| High | A same-day unfinished cache, a short/NaN indicator window, or a silently dropped momentum ticker could publish an artificial cash/risk-off target. | Publication forces a post-close refresh, requires at least 200 complete finite positive OHLCV sessions with valid geometry, rejects repeated final bars, and validates every registered universe member and decision indicator after strategy preparation. |
| High | `allow_labelled` froze only porcelain path/status lines, so edited dirty-file bytes could change after signing without changing git status. | Labelled dirty snapshots now freeze path, state, size, and SHA-256 for every modified, staged, deleted, and untracked file. Persistence, replay, approval, health, and submission re-enumerate and compare exact bytes. |
| High | Reconciliation looked only at open orders and fills; an externally created terminal order could disappear from current broker state. | The broker-neutral contract now fetches recent orders in every status. Reconciliation flags unseen system/external terminal orders and duplicate all-status IDs, and persists each incident as a durable critical latch at the reconciliation boundary. |
| High | A process crash after persisting a stream event but before applying it made its replay look idempotently complete. | Applicable events persist as `pending`, become `applied` only after ledger synchronization and clean reconciliation, and force disarm/REST recovery on pending replay. Clean recovery explicitly marks the event `recovered`. |
| High | A clean reconnect could erase the operational significance of a disconnect/reconciliation incident. | Critical incident alerts are never auto-resolved. Both manual and supervised paper submission block on every unresolved critical alert; CLI resolution requires a later clean reconciliation plus recovered stream or clear kill state where applicable. |
| High | Backup creation could chmod an arbitrary existing root, delete similarly named directories, copy config bytes different from those scanned, or install a changed source after restore verification. | Roots must be dedicated/private/separate, retention accepts only strict IDs, configs are read once via no-follow descriptors and those bytes are copied, restored temporary bytes are rehashed immediately before replacement, and files/directories are fsynced before success. |
| Medium | Quiet websocket periods renewed a lease without rechecking broker truth, and an arbitrarily old clean reconciliation could appear ready. | Heartbeats run all-status reconciliation at least every 60 seconds; health and submission require a clean reconciliation no older than 120 seconds as well as a fresh 45-second stream lease. |
| Medium | Alert repeats flooded sinks, `info` could not upgrade to `warning`, and the webhook accepted private/DNS-rebound targets. | Severity now upgrades across all levels, duplicate occurrences update durable counts without redelivery, and webhook use requires a hostname allowlist plus public literal/resolved IP validation, redirect/proxy disabling, and bounded timeouts. |
| Medium | Backup manifests were described too strongly despite colocated unkeyed hashes. | Runbooks now state that the hashes prove consistency/corruption detection, not hostile authenticity; signed/MACed off-host digests and host hardening remain explicit deployment work. |

Combined mocked lifecycle/adversarial coverage includes publish, immutable
persistence, plan freeze, approval, OMS submission, multiple partial fills,
stream interruption, restart/REST recovery, terminal reconciliation,
backup/restore, and soak reporting. A separate cold-process simulation runs two
authentic monthly shadow rebalances plus a duplicate replay and produces zero
orders. No test opens a real socket or submits a real paper order.

## Residual risks and prohibited assumptions

- The Alpaca contract tests are mocked. A credentialed paper soak test has not
  yet validated real latency, rate limits, pagination volume, account resets,
  or long-running session behavior.
- Phase 3 by itself still polls REST. Phase 4 supervises Alpaca paper
  `trade_updates` and REST recovery, but trade corrections/busts and every
  possible future broker event shape still require soak validation.
- Non-fill cash changes—including dividends, transfers, fees, and paper-account
  resets—cause reconciliation failure and require operator review; they are not
  automatically adopted.
- A local kill cannot cancel orders while the broker is unreachable. It blocks
  this system from submitting, but the operator must inspect the broker
  dashboard and cancel any already-open orders separately.
- Database triggers are not cryptographic attestation. A user or attacker with
  host/database administration access can drop triggers, rewrite history,
  replace code, or steal environment credentials. External signed audit export,
  secrets management, host hardening, and least-privilege deployment remain
  future operational work.
- The execution fence protects cooperating processes using the same ledger and
  lock path. Code that bypasses the OMS and calls a broker adapter directly is
  outside that protection.
- Phase 4's publisher is signed and runs the registered strategy code, but the
  current stock universe remains survivorship-biased and Yahoo adjusted daily
  data is not an institutional point-in-time source.
- The older three-symbol Phase-3 files remain schema illustrations. Phase 4
  adds a separate complete 36-symbol deployment example.
- The required allocation policy explicitly rebalances to deployment weights;
  it does not reproduce `portfolio.py`'s static-start, drifting independent
  sleeve capital. A virtual-sleeve ledger and parity study remain possible
  future research work beyond this phase.
- Same-day daily-close signals are rejected, but the allowed next-session
  regular-hours market order is still not the backtest's next-day-close fill.
  Execution-timing parity remains unproven.
- IEX is a single-venue feed. Paper fills may use a different best-price view,
  and a market order has no guaranteed execution price after the risk check.
- Multi-order execution is not atomic. A failure can leave a legitimate partial
  portfolio that requires operator review; the system stops later items but
  cannot roll back fills.
- Phase 4 adds a one-shot durable scheduler, stream supervisor, local alerts,
  backup/restore, and soak reports. Production host supervision, off-host
  signed audit export, external secrets management, and completed operator
  drills remain deployment responsibilities.
- Alpaca paper fills are simulated and cannot establish real queue position,
  market impact, latency slippage, fees, or live operational behavior.

## Gate beyond Phase 4

Phase 4 remains paper-only: use a fresh dedicated paper ledger, supported
OpenSSL runtime, zero unexplained reconciliation issues, operator-reviewed
limits, and the graduation criteria in `PHASE4_SOAK_GRADUATION.md`. Real
credentialed paper soak, repeated incident drills, and restore evidence are
still required. Nothing in this review authorizes a live endpoint; Phase 5 is
unimplemented and would require a separate design and explicit authorization.

## Primary broker references

- Alpaca paper/live endpoint and credential separation:
  https://docs.alpaca.markets/us/v1.1/docs/authentication-1
- Alpaca order lifecycle, including `done_for_day` and rare states:
  https://docs.alpaca.markets/us/docs/orders-at-alpaca
- Alpaca cancellation acceptance and HTTP 422 behavior:
  https://docs.alpaca.markets/us/reference/deleteorderbyorderid-1
- Alpaca paper-simulation limitations:
  https://docs.alpaca.markets/us/docs/paper-trading
- Alpaca trading calendar and early-close sessions:
  https://docs.alpaca.markets/us/v1.1/reference/getcalendar-1
- Alpaca paper websocket authorization and `trade_updates` subscription:
  https://docs.alpaca.markets/us/v1.4.2/docs/websocket-streaming
- urllib3 TLS runtime requirements:
  https://urllib3.readthedocs.io/en/latest/v2-migration-guide.html
