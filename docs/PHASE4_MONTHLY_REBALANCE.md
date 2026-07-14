# Phase 4 Monthly Rebalance Checklist

1. Confirm the authenticated calendar's final actual session and official
   close, including holidays, early close, or unexpected closure.
2. After the close, run `publish`; verify decision/data cutoff, next session,
   git/config/input hashes, asset IDs, 60/35/5 sleeves, aggregation, cash, and
   signature. A missing ticker is a stop, never a substitution.
3. If late, investigate why the scheduled run was missed and use explicit
   catch-up confirmation only before the expected next session closes. After
   that close the scheduler records failure and refuses publication.
4. Before the next regular session, reconcile, start the stream, and verify REST
   recovery and a clear kill switch.
5. During the regular session run `prepare-plan`; inspect every current/target/
   delta share count, quote timestamp, spread, reference, notional, turnover,
   concentration, cash, and order-count limit.
6. In observe mode stop. In shadow mode retain the automatically approved,
   finalized-voided, permanently non-submitting plan and compare it with
   expected targets.
7. In manual/supervised paper mode, approve only the exact 64-character plan
   hash with operator and reason, then explicitly run the paper command.
8. Observe sells before buys. After any unresolved order, wait for stream/REST
   truth and clean reconciliation before the next explicit resume.
9. Reconcile terminal positions/cash/orders, review alerts, verify automatic
   backup, and generate the daily soak report.
