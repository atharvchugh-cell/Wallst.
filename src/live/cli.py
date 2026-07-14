"""Offline-only command line entry points for exercising phase one."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal

from .fake_broker import FakeBroker
from .ledger import Ledger
from .models import TargetPositionIntent, json_safe
from .oms import OrderManagementSystem
from .reconcile import Reconciler
from .risk import PreTradeRiskEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Wallst Strategy Lab execution-foundation utilities"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run one deterministic order through FakeBroker")
    demo.add_argument(
        "--confirm-fake",
        action="store_true",
        help="required acknowledgement that this is a local fake-broker simulation",
    )
    demo.add_argument("--db", default=":memory:", help="SQLite path (default: in-memory)")
    demo.add_argument("--account-id", default="FAKE-DEMO")
    demo.add_argument("--symbol", default="SPY")
    demo.add_argument("--price", type=Decimal, default=Decimal("100"))
    demo.add_argument("--target", type=Decimal, default=Decimal("10"))
    demo.add_argument("--cash", type=Decimal, default=Decimal("10000"))

    status = sub.add_parser("status", help="print a durable ledger snapshot; no broker access")
    status.add_argument("--db", required=True)
    status.add_argument("--account-id", required=True)
    return parser


def run_demo(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.confirm_fake:
        parser.error("demo requires --confirm-fake")
    now = datetime.now(timezone.utc)
    broker = FakeBroker(account_id=args.account_id, cash=args.cash)
    quote = broker.set_quote(args.symbol, args.price, spread_bps="0", as_of=now)
    with Ledger(args.db) as ledger:
        reconciler = Reconciler(ledger, broker)
        reconciler.bootstrap_positions()
        before = reconciler.reconcile()
        oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine())
        oms.arm("explicit fake-broker demo")
        starting_equity = broker.get_account().equity
        intent = TargetPositionIntent(
            account_id=args.account_id,
            strategy_id="offline-demo",
            symbol=args.symbol,
            target_quantity=args.target,
            signal_at=now,
            target_version="demo-v1",
            reference_price=args.price,
            reason="exercise phase-one controls without network access",
        )
        result = oms.process_intent(
            intent,
            quote=quote,
            market_open=True,
            day_start_equity=starting_equity,
            high_water_equity=starting_equity,
        )
        after = reconciler.reconcile()
        output = {
            "adapter": "FakeBroker (in-process; no network)",
            "pretrade_reconciliation_clean": before.clean,
            "result": json_safe({
                "intent_id": result.intent_id,
                "intent_status": result.intent_status,
                "client_order_id": result.client_order_id,
                "broker_order_id": result.broker_order_id,
                "risk_violations": result.risk_violations,
            }),
            "posttrade_reconciliation_clean": after.clean,
            "ledger": ledger.snapshot(args.account_id),
        }
        print(json.dumps(json_safe(output), indent=2, sort_keys=True))
    return 0


def run_status(args: argparse.Namespace) -> int:
    with Ledger(args.db) as ledger:
        print(json.dumps(json_safe(ledger.snapshot(args.account_id)), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return run_demo(args, parser)
    if args.command == "status":
        return run_status(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
