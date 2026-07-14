# Phase 4 Paper-Soak Graduation Criteria

Paper-soak graduation is evidence for a later review, not permission for live
money. Phase 5 remains unimplemented.

Minimum recommended gate:

- at least three completed monthly rebalance cycles and 60 calendar days;
- every expected decision either published once or has an investigated,
  audited skip/delay;
- zero duplicate broker submissions and zero unexplained external orders;
- all signatures, account fingerprints, policy/deployment/input hashes, and
  universe checks pass;
- every disconnect completes REST recovery before a new submission;
- zero unresolved reconciliation, cash, position, order, database, backup, or
  critical alert issue;
- successful drills for crash-after-submit, lost acknowledgement, multi-partial
  fill, fill during cancel, restart, network outage, stale stream, corrupted
  restore candidate, active-ledger replacement refusal, and expired snapshot;
- daily/cumulative reports include reference slippage, next-close slippage when
  available, target/actual weight error, uptime, disconnects, recovery events,
  and database integrity;
- observed spreads, deviations, order sizes, turnover, cash deployment, and
  open-order age remain inside reviewed limits; and
- operator runbooks are executable by someone other than the implementer.

Paper fills cannot validate live liquidity, queue position, market impact,
real fees, halts, availability, or psychological/operational behavior with real
capital. A separate Phase 5 design, authorization, broker adapter boundary,
capital limit, and rollback decision would still be required.
