from datetime import date, datetime, time, timezone
from decimal import Decimal

from src.live.ledger import Ledger
from src.live.market_data import NEW_YORK, MarketCalendarDay
from src.live.models import (
    AccountSnapshot,
    BrokerOrder,
    Fill,
    OrderRequest,
    OrderStatus,
    Side,
    TargetPositionIntent,
)
from src.live.phase4_store import Phase4Store
from src.live.scheduler import SupervisedMonthlyScheduler
from src.live.soak import PaperSoakReporter


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class WeekdayCalendar:
    def get_market_calendar(self, start: date, end: date) -> tuple[MarketCalendarDay, ...]:
        rows = []
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5:
                rows.append(MarketCalendarDay(
                    cursor,
                    datetime.combine(cursor, time(9, 30), tzinfo=NEW_YORK),
                    datetime.combine(cursor, time(16), tzinfo=NEW_YORK),
                ))
            cursor = date.fromordinal(cursor.toordinal() + 1)
        return tuple(rows)


def test_scheduler_uses_exchange_month_and_surfaces_outstanding_run(tmp_path):
    # UTC has crossed into July, but New York is still on the completed June
    # month-end. Health must not jump ahead to July's decision session.
    boundary = datetime(2026, 7, 1, 0, 30, tzinfo=timezone.utc)
    clock = MutableClock(boundary)
    with Ledger(tmp_path / "scheduler.sqlite3", clock=clock) as ledger:
        store = Phase4Store(ledger)
        scheduler = SupervisedMonthlyScheduler(
            WeekdayCalendar(), store, clock=clock
        )
        action = scheduler.next_expected_action()
        assert action["decision_session"] == "2026-06-30"
        assert action["scheduler_status"] == "unclaimed"
        assert action["action"] == "publish_monthly_targets_after_official_close"

        # A persisted delayed run remains the next operator action after a
        # restart instead of being hidden behind the current month's close.
        store.claim_schedule(
            "2026-06-30", "2026-07-01", status="delayed",
            detail="manual catch-up confirmation required",
        )
        clock.value = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
        action = scheduler.next_expected_action()
        assert action["decision_session"] == "2026-06-30"
        assert action["scheduler_status"] == "delayed"
        assert action["action"] == "confirm_manual_catch_up_and_publish"


def test_scheduler_restart_walks_forward_without_inventing_fresh_ledger_history(tmp_path):
    clock = MutableClock(datetime(2026, 7, 15, 16, 0, tzinfo=timezone.utc))
    calendar = WeekdayCalendar()
    with Ledger(tmp_path / "restart.sqlite3", clock=clock) as ledger:
        store = Phase4Store(ledger)
        scheduler = SupervisedMonthlyScheduler(calendar, store, clock=clock)
        store.claim_schedule("2026-03-31", "2026-04-01", status="published")

        # April is the earliest missing decision after the durable March run;
        # jumping directly to June would silently erase two missed months.
        assert scheduler.latest_due().decision_day.trading_date == date(2026, 4, 30)

        # Even a later terminal row cannot hide an older unresolved run.
        april, _created = store.claim_schedule(
            "2026-04-30", "2026-05-01", status="delayed"
        )
        store.claim_schedule("2026-06-30", "2026-07-01", status="published")
        assert scheduler.latest_due().decision_day.trading_date == date(2026, 4, 30)
        scheduler.mark_skipped(
            april["run_id"], operator="red-team", reason="reviewed missed window"
        )
        # The later June row must not make the unrecorded May decision vanish.
        assert scheduler.latest_due().decision_day.trading_date == date(2026, 5, 29)

    with Ledger(tmp_path / "fresh.sqlite3", clock=clock) as fresh:
        scheduler = SupervisedMonthlyScheduler(
            calendar, Phase4Store(fresh), clock=clock
        )
        # No durable inception exists, so a fresh install considers only the
        # immediately preceding due month instead of fabricating April/May.
        assert scheduler.latest_due().decision_day.trading_date == date(2026, 6, 30)


def test_daily_soak_joins_old_orders_and_counts_only_that_days_stream_events(tmp_path):
    clock = MutableClock(datetime(2026, 7, 1, 16, 0, tzinfo=timezone.utc))
    with Ledger(tmp_path / "soak.sqlite3", clock=clock) as ledger:
        ledger.bootstrap_positions(
            AccountSnapshot("PAPER", "1000", "1000", "1000", clock()), []
        )
        intent, _created = ledger.create_intent(TargetPositionIntent(
            account_id="PAPER",
            strategy_id="redteam",
            symbol="AAPL",
            target_quantity=Decimal("1"),
            signal_at=clock(),
            target_version="day-one",
            reference_price=Decimal("100"),
        ))
        request = OrderRequest(
            account_id="PAPER",
            client_order_id="redteam-old-order",
            intent_id=intent["intent_id"],
            symbol="AAPL",
            side=Side.BUY,
            quantity=Decimal("1"),
            reference_price=Decimal("100"),
        )
        order = ledger.plan_order(request)
        ledger.acknowledge_order(order["order_id"], BrokerOrder(
            broker_order_id="broker-old-order",
            client_order_id=request.client_order_id,
            account_id="PAPER",
            symbol="AAPL",
            side=Side.BUY,
            quantity=Decimal("1"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.SUBMITTED,
            submitted_at=clock(),
            updated_at=clock(),
        ))

        store = Phase4Store(ledger)
        store.set_stream_state(
            "alpaca-paper-trade-updates", connected=False, recovering=True,
            disconnected=True,
        )
        store.set_stream_state(
            "alpaca-paper-trade-updates", connected=True, recovering=False,
            recovery_completed=True,
        )

        clock.value = datetime(2026, 7, 2, 16, 0, tzinfo=timezone.utc)
        ledger.acknowledge_order(order["order_id"], BrokerOrder(
            broker_order_id="broker-old-order",
            client_order_id=request.client_order_id,
            account_id="PAPER",
            symbol="AAPL",
            side=Side.BUY,
            quantity=Decimal("1"),
            filled_quantity=Decimal("1"),
            status=OrderStatus.FILLED,
            submitted_at=datetime(2026, 7, 1, 16, 0, tzinfo=timezone.utc),
            updated_at=clock(),
        ))
        ledger.record_fill(order["order_id"], Fill(
            fill_id="fill-next-day",
            broker_order_id="broker-old-order",
            client_order_id=request.client_order_id,
            account_id="PAPER",
            symbol="AAPL",
            side=Side.BUY,
            quantity=Decimal("1"),
            price=Decimal("101"),
            commission=Decimal("0"),
            occurred_at=clock(),
        ))
        store.set_stream_state(
            "alpaca-paper-trade-updates", connected=False, recovering=True,
            disconnected=True,
        )
        store.set_stream_state(
            "alpaca-paper-trade-updates", connected=True, recovering=False,
            recovery_completed=True,
        )
        store.set_stream_state(
            "alpaca-paper-trade-updates", connected=True, recovering=False,
            recovery_completed=True,
        )
        ledger.record_audit(
            "reconciliation_completed", "reconciliation", "routine",
            {"clean": True},
        )

        report = PaperSoakReporter(ledger).report("2026-07-02")
        assert report["daily"]["fills"] == 1
        assert Decimal(report["daily"]["slippage_vs_reference_quote_bps"]) == Decimal("100")
        assert report["daily"]["stream_disconnects"] == 1
        assert report["daily"]["stream_recoveries"] == 2
        assert report["daily"]["recovery_events"] == 2
        assert report["cumulative"]["stream_disconnects"] == 2
        assert report["cumulative"]["stream_recoveries"] == 3
        assert report["cumulative"]["recovery_events"] == 3
