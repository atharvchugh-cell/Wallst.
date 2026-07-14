# Paper Broker Operations — Phase Two

## Completion boundary

Phase two is complete when the system can safely connect to, inspect, baseline,
reconcile, recover, and cancel against one Alpaca paper account. It does not
turn research signals into orders and it does not expose order submission in
the terminal. Phase three owns that boundary because execution requires a
reviewed account-level target, a current quote, an exchange calendar, and
durable equity guardrails.

`AlpacaPaperBroker` is hard-pinned to:

```text
https://paper-api.alpaca.markets
```

The module contains no live Alpaca endpoint. It accepts only
`ALPACA_PAPER_API_KEY` and `ALPACA_PAPER_API_SECRET`; generic or live credential
variable names are not read. Tests use stub sessions and never contact Alpaca.

## Implemented controls

- Account readiness checks require `ACTIVE`, USD, and no broker/user trading
  block before the adapter can submit through its programmatic contract.
- A ledger is permanently bound to its opening paper account. Credentials for
  another account fail before reconciliation, cancellation, or other mutation.
- Only active, tradable US equities, whole-share quantities, simple market/day
  or limit/day/GTC orders, and deterministic client order IDs are accepted.
- Extended-hours execution is explicitly disabled.
- The REST adapter never automatically retries an order submission.
- A broker lookup by client order ID precedes every OMS submission decision.
  Startup recovery only synchronizes; it never resubmits a broker-missing order.
- A file lock serializes submission against arm, disarm, and kill changes. Only
  the transaction that created an order may submit it, and it rechecks control
  state plus client-ID discovery inside that fence.
- `done_for_day`, `stopped`, pending-cancel, and pending-replace broker states
  remain locally active so a later update cannot bypass the one-order-per-symbol
  guard.
- A rejected cancellation re-reads broker truth. A fill/cancel race is adopted
  as a fill; an order that remains active is reported as a failed cancellation.
- The kill switch is stored before cancellation begins and re-queries broker
  open orders afterward. It reports failure and stays engaged if any order
  remains or verification fails.
- Successful Alpaca `X-Request-ID` values are included in command output and
  persisted in the ledger audit log for ledger-backed commands. Secrets and
  response bodies are never included in errors or audits.
- Redirect following and environment-proxy inheritance are disabled for the
  credentialed session. Account identity changes, duplicate position/order
  rows, stale snapshots, malformed numbers, and reconciliation transport errors
  all fail closed and disarm prior authorization.

## Runtime prerequisite

Use Python 3 with an OpenSSL-backed `ssl` module. The real HTTP session fails
closed on LibreSSL or OpenSSL older than 1.1.1 because the installed urllib3 2.x
line does not support those TLS runtimes. Check before adding credentials:

```bash
python3 -c 'import ssl; print(ssl.OPENSSL_VERSION)'
```

The current macOS system Python in this workspace reports `LibreSSL 2.8.3`, so
use a current Homebrew, pyenv, Conda, or equivalent OpenSSL-backed Python for
credentialed paper commands. Mocked tests and offline ledger commands do not
open a TLS connection.

## Credentials

Create a dedicated Alpaca paper account and keep its keys outside the
repository:

```bash
export ALPACA_PAPER_API_KEY='paper-key-here'
export ALPACA_PAPER_API_SECRET='paper-secret-here'
```

Do not place credentials in shell history, source files, `.env` files committed
to Git, reports, screenshots, or support messages. Rotating or recreating a
paper account changes its credentials and may also change the account ID; use a
new ledger rather than silently reusing another account's baseline.

## First connection

The following command reads account status, positions, open orders, and the
broker market clock. It cannot submit an order:

```bash
python3 -m src.live.paper_cli check --confirm-paper-network
```

Confirm all of these before creating a baseline:

- `endpoint` is exactly `https://paper-api.alpaca.markets`;
- `phase_two_ready` is `true`;
- `initial_baseline_ready` is `true` and has no blockers;
- the account ID is the expected paper account;
- cash, positions, and open orders match the Alpaca paper dashboard; and
- `paper_submission_command_available` is `true` only in the current combined
  Phase-1–3 codebase, and `submission_requires_preview_and_exact_hash_approval`
  is `true`. The `check` command itself cannot submit; follow the separate
  Phase-3 runbook before using the order-capable command.

## One-time baseline

Use a new absolute ledger path for each paper account. Baseline creation is
intentionally irreversible within that ledger:

```bash
python3 -m src.live.paper_cli bootstrap \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --confirm-paper-network \
  --confirm-initial-baseline
```

The command records opening cash and positions, imports known broker state, and
runs reconciliation. Exit code `0` means the reconciliation was clean; exit
code `1` means an error, readiness blocker, or reconciliation issue needs
operator review.

## Normal startup and inspection

At every process start, synchronize already-planned client IDs and then
reconcile:

```bash
python3 -m src.live.paper_cli recover \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --confirm-paper-network

python3 -m src.live.paper_cli reconcile \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --confirm-paper-network
```

`recover` never creates or resubmits an order. If the ledger has an active
order that Alpaca cannot find, reconciliation reports
`LOCAL_ACTIVE_ORDER_MISSING` and disarms the account.

Offline inspection requires no credentials and no network confirmation:

```bash
python3 -m src.live.paper_cli status \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --account-id PAPER_ACCOUNT_ID
```

## Cancellation and emergency procedures

Cancel one active order already tracked by the ledger:

```bash
python3 -m src.live.paper_cli cancel \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --order-id LOCAL_ORDER_ID \
  --reason 'operator requested cancellation' \
  --confirm-paper-network \
  --confirm-cancel
```

Emergency stop persists the kill switch first, attempts to cancel every open
paper order—including external orders—and verifies that none remain:

```bash
python3 -m src.live.paper_cli kill \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --reason 'emergency stop' \
  --confirm-paper-network \
  --confirm-cancel-open-orders
```

If network access is unavailable, immediately disarm locally. This does not
cancel broker orders, so inspect the paper dashboard separately:

```bash
python3 -m src.live.paper_cli disarm \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --account-id PAPER_ACCOUNT_ID \
  --reason 'network unavailable; manual broker review required'
```

Never treat a failed kill command as harmless. The local kill remains engaged,
but broker orders may still be open.

To engage the persistent kill without making any network call, use:

```bash
python3 -m src.live.paper_cli local-kill \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --account-id PAPER_ACCOUNT_ID \
  --reason 'broker unreachable; local emergency stop' \
  --confirm-local-kill
```

This command truthfully reports `broker_orders_canceled: false`. The normal
networked `kill` command also persists the local kill before its first network
request, so an account-endpoint failure cannot leave the ledger armed.

## Exceptional missing-order resolution

Only after the paper broker confirms that an active local client order ID is
absent may the operator close that local order:

```bash
python3 -m src.live.paper_cli abandon-missing \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --order-id LOCAL_ORDER_ID \
  --reason 'broker and dashboard confirmed client ID absent' \
  --confirm-paper-network \
  --confirm-abandon-missing
```

This does not resubmit anything. A future execution attempt must use a fresh,
reviewed target version after a clean reconciliation.

## Kill reset

Reset clears the kill switch only into a disarmed state and immediately runs
reconciliation:

```bash
python3 -m src.live.paper_cli reset-kill \
  --db /absolute/path/to/paper-ledger.sqlite3 \
  --reason 'incident reviewed and broker state verified' \
  --confirm-paper-network \
  --confirm-reset-kill
```

There is no Phase 2 arm or submit command. Phase 3 subsequently added reviewed
deployment configuration, quote/calendar validation, strategy-sleeve
aggregation, durable day-start/high-water equity, and an exact-hash-approved
paper-order flow. See `docs/LIVE_TRADING_PHASE3.md`; none of that changes this
Phase-2 operating boundary or adds a live endpoint.

## Known paper limitation

Paper fills are simulations. They do not prove real-world queue position,
market impact, latency slippage, price improvement, regulatory fees, dividends,
or operational reliability. Phase 4 requires shadow operation and recovery
drills; Phase 5 is the separately authorized micro-live rollout.
