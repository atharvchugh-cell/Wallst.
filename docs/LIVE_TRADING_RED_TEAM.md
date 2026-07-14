# Phase 1–3 Adversarial Review

## Verdict

Reviewed and hardened on 2026-07-14. The result is suitable for offline OMS
testing and a manually supervised, exact-hash-approved Alpaca paper pilot.
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

## Residual risks and prohibited assumptions

- The Alpaca contract tests are mocked. A credentialed paper soak test has not
  yet validated real latency, rate limits, pagination volume, account resets,
  or long-running session behavior.
- Phase 3 polls REST state. It does not yet consume and supervise streaming
  trade updates, trade corrections, trade busts, or corporate-action events.
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
- Aggregation is deterministic but consumes an operator-provided snapshot.
  There is no signed automated strategy publisher or shadow-parity proof that
  the artifact exactly matches a fresh research run.
- The checked-in three-symbol files are schema illustrations, not a complete
  deployment for the repository's candidate 60/35/5 portfolio. Phase 4 must
  build and validate the full managed universe and strategy publisher.
- The required allocation policy explicitly rebalances to deployment weights;
  it does not reproduce `portfolio.py`'s static-start, drifting independent
  sleeve capital. A virtual-sleeve ledger and parity study remain Phase 4 work.
- Same-day daily-close signals are rejected, but the allowed next-session
  regular-hours market order is still not the backtest's next-day-close fill.
  Execution-timing parity remains unproven.
- IEX is a single-venue feed. Paper fills may use a different best-price view,
  and a market order has no guaranteed execution price after the risk check.
- Multi-order execution is not atomic. A failure can leave a legitimate partial
  portfolio that requires operator review; the system stops later items but
  cannot roll back fills.
- There is no scheduler, process supervisor, backup/restore drill, alert
  delivery, signed off-host audit, or shadow-parity report yet.
- Alpaca paper fills are simulated and cannot establish real queue position,
  market impact, latency slippage, fees, or live operational behavior.

## Gate to Phase 4

Phase 4 must remain paper-only: use a fresh dedicated paper ledger, supported
OpenSSL runtime, zero unexplained reconciliation issues, and operator-reviewed
limits. It must add credentialed soak/shadow evidence, streaming/recovery
supervision, alerts, backup restore, and repeated incident drills. Nothing in
this review authorizes a live endpoint; micro-live remains a separately
approved Phase 5 decision.

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
- urllib3 TLS runtime requirements:
  https://urllib3.readthedocs.io/en/latest/v2-migration-guide.html
