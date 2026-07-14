# Phase 4 Alert-Response Runbook

Severity meanings:

- `info`: evidence such as successful stream recovery.
- `warning`: degraded state that is not yet a submission blocker.
- `high`: invalid data/quote, risk rejection, overdue partial/open order, or
  backup/alert-delivery issue requiring prompt review.
- `critical`: reconciliation mismatch, disconnect/recovery failure, signature
  failure, database integrity failure, unexpected order/position/cash, kill
  switch, or missed run. Submission stays blocked.

Response:

1. Read the full health report and the broker paper dashboard.
2. For high/critical execution or broker events, disarm/kill first; do not
   acknowledge as a substitute for containment.
3. Preserve the deduplicated alert ID, occurrence count, entity, audit events,
   broker request IDs, and relevant backup.
4. Follow the stream, reconciliation, or backup runbook for the category.
5. Acknowledge with operator/note when ownership is clear. Resolve only after
   the underlying health/reconciliation condition is demonstrably corrected.
   The CLI rejects critical resolution without a clean reconciliation after
   the incident; stream and kill alerts also require recovered or clear state.
6. Critical alerts older than the configured threshold escalate durably.
   Webhook failure itself becomes a durable alert; console/SQLite remains the
   source of record.

Run `health --record-health-alerts --confirm-health-alert-write` from the host
supervisor at an interval shorter than the critical escalation threshold.
This explicit mode persists, delivers, and backs up due alerts/escalations;
repeated checks do not escalate the same alert again until another full
threshold has elapsed. Plain `health` is read-only.

Exact alert changes require explicit confirmation:

```bash
python3 -m src.live.phase4_cli alerts --db "$DB" \
  --acknowledge alert-REVIEWED_ID --operator 'operator-name' \
  --note 'ownership accepted; investigation started' --confirm-alert-change

python3 -m src.live.phase4_cli alerts --db "$DB" \
  --resolve alert-REVIEWED_ID --operator 'operator-name' \
  --note 'later clean reconciliation and broker evidence reviewed' \
  --confirm-alert-change
```

Acknowledgement and resolution are mutually exclusive. Repeats increment the
durable occurrence count without resending every duplicate. Webhook use also
requires an exact hostname allowlist and public DNS resolution; host egress
controls remain recommended.
