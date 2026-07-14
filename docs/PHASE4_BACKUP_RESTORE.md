# Phase 4 Backup and Restore Runbook

State-changing trading-workflow commands (including `soak-observe`) automatically create a versioned
backup in the configured directory (relative paths resolve beside the ledger).
Stream events are durable in the WAL ledger and covered by scheduled/manual
backups. To make an extra backup, use `phase4_cli backup` with explicit
confirmation.

Each backup is made with SQLite's online backup API, reopened with integrity
and foreign-key checks, hashed, and accompanied by credential-scanned policy
and deployment copies. Verify `manifest.json`, `manifest.sha256`, every config
hash, and the ledger hash before use.

The configured root must be a dedicated operator-owned `0700` directory,
separate from the ledger directory. Backup files and parent directories are
fsynced before success is recorded. The colocated manifest/hash prove
consistency and corruption detection, not hostile authenticity. For that
threat model, preserve a signed or MACed manifest digest in a separately
controlled off-host system.

Restore drill:

1. Stop scheduler, stream, and all CLI processes. Engage the local kill switch.
2. Copy the backup set to a separate test location.
3. Run `phase4_cli restore --db /active/path.sqlite3 --backup-dir
   /path/to/backup-ID --destination /separate/path.sqlite3` without a
   replacement flag. The command treats `--db` only as a safety fence and does
   not open/create/migrate it. Open the separate restore with `health`, inspect
   snapshots/plans/orders, and run offline integrity checks.
4. If disaster recovery truly requires replacing the active ledger, preserve
   the damaged file, close every connection, document operator/reason, and use
   `--confirm-replace-ledger`. Reconcile against broker truth before any reset
   or resume.
5. A failed hash, open, quick-check, foreign-key check, or config hash is a
   critical backup failure. Do not repair the manifest to match corruption.

Restore copies from a no-follow source descriptor, verifies the copied
temporary ledger's expected hash and SQLite structure, atomically replaces the
destination, and fsyncs the file and parent directory. It never needs to open
the unavailable or corrupt active database.

Drill process crash, computer restart, network outage, stale stream, lost
acknowledgement, database unavailability, materialized-state corruption, and
expired snapshot at least once during the soak.
