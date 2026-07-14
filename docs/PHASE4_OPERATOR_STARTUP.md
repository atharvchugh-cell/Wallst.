# Phase 4 Operator Startup Checklist

1. Use a dedicated long-only Alpaca paper account with no open orders,
   fractions, shorts, or unmanaged positions.
2. Use a supported OpenSSL Python runtime. The macOS system Python in the
   original workspace uses LibreSSL and the adapters intentionally fail closed.
3. Keep the ledger, signing key, snapshots, and backups outside the repository;
   key and ledger modes must be `0600`.
4. Confirm the branch/commit is reviewed and the worktree satisfies the policy.
5. Copy the Phase 4 example deployment/policy; replace the paper account ID;
   do not add credentials or endpoint fields.
6. Export only `ALPACA_PAPER_API_KEY`, `ALPACA_PAPER_API_SECRET`, and
   `WSLAB_PHASE4_SIGNING_KEY_FILE`.
7. Run Phase 2 bootstrap once. Run Phase 4 reconciliation and require clean.
8. Run `health`; resolve every database, snapshot, stream, reconciliation,
   overdue-order, kill-switch, and critical-alert blocker.
9. Start the paper stream and require successful REST recovery before preparing
   or submitting any plan.
10. Keep the Alpaca paper dashboard visible. A local kill cannot cancel an
    already-open broker order while the network is unavailable.

Never proceed because a checklist item is inconvenient. Record the incident,
leave execution disarmed, and resolve it.
