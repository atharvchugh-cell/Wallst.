# Phase 4 Stream-Recovery Runbook

On disconnect, stale stream, sequence regression, duplicate-content conflict,
or unknown order:

1. The system disarms and marks the stream recovering. Do not submit.
2. Inspect the Alpaca paper dashboard for open orders, fills, rejects, and
   cancellations; do not infer state from the last websocket message.
3. Restart `phase4_cli stream`. It is not marked ready until websocket auth and
   subscription succeed. Recovery fetches account, positions, open orders,
   recent all-status orders, recent fills, and each known client ID, then
   reconciles. A 45-second durable lease must remain fresh during quiet periods,
   and heartbeat reconciliation runs at least once per minute.
4. For a lost acknowledgement, locate the existing deterministic client ID.
   Never create a replacement client ID merely because the POST response was
   lost.
5. Import every cumulative partial fill. A fill while cancellation is pending
   is valid broker truth. A contradiction after a terminal state is a critical
   conflict requiring operator review.
6. Unknown external orders or positions remain mismatches. Cancel/resolve at the
   broker as appropriate, record the reason, and reconcile again.
7. Resume only when health shows connected, not recovering, clean
   reconciliation, no kill switch, and no unresolved critical alert.

An event durably recorded as `pending` but not fully applied is a crash gap. A
duplicate replay does not assume success: it disarms and requires REST
recovery. Only a clean recovery changes that event to `recovered`.

Order replacement is not supported in Phase 4. A replacement successor outside
the known deterministic client ID is treated as external and blocks submission.
