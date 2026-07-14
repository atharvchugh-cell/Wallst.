# Phase 4 Reconciliation Runbook

Reconciliation compares broker cash, buying power, positions, whole-share
quantities, open orders, recent orders in every status (so unseen terminal
orders cannot hide), cumulative fills, statuses, client IDs, and local ledger
state. Every issue is latched as a durable critical alert at the reconciliation
boundary even if its caller crashes.

1. Engage the kill switch for any unexplained mismatch if it is not already
   active. Preserve broker and ledger evidence.
2. Classify the difference: timing lag, known fill, known cancellation,
   transfer/dividend/fee/reset, external order, unknown position, duplicate ID,
   missing broker order, or corrupted local state.
3. Use REST recovery to import only broker facts tied to known client IDs.
   Never edit/delete ledger rows or silently adopt an unexplained difference.
4. For external activity, record who created it and why; cancel or baseline it
   only through the explicit operator procedure appropriate to a fresh account.
5. A cash change from transfer, dividend, fee, or paper reset needs an explicit
   audited resolution; Phase 4 does not guess.
6. Re-run reconciliation. Require clean, then separately acknowledge/resolve
   the alert. Clearing an alert alone does not authorize submission.
