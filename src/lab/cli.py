"""CLI surface for `--strategy strategy_lab`.

Adds the lab argument group to the main parser and dispatches lab
subcommands. Deliberately does NOT import src.cli at module level (src.cli
imports this module to register the arguments; helpers it shares are
imported lazily inside functions).

Commands (all research-only; nothing here places live orders):

  --experiment baseline      deterministic, manifest-stamped run of the
                             immutable 60/35/5 baseline through the lab engine
                             (proven identical to --strategy portfolio)
  --experiment list          print the experiment registry
  --experiment <name>|all    run registered enhancement experiments
  --experiment final         final comparison + acceptance gates (+ holdout
                             only with --lab-holdout)
  --explain-date / --explain-ticker / --show-reason
                             query the decision trace of an existing lab run
                             directory (--lab-run-dir)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .. import config
from ..portfolio import PortfolioError, parse_portfolio_weights, validate_portfolio_weights
from ..tournament import STRATEGY_REGISTRY
from .lab_config import LabConfig, LabConfigError
from .manifest import (
    atomic_write_json,
    attach_artifact_hashes,
    build_manifest,
    write_manifest,
)
from .portfolio_engine import LabEngineError, run_lab_portfolio

LAB_SNAPSHOT_SCHEMA_VERSION = 1


def add_lab_arguments(parser) -> None:
    group = parser.add_argument_group(
        "strategy lab",
        "Options for --strategy strategy_lab (ignored for other --strategy values). "
        "Research/paper-trading only.",
    )
    group.add_argument(
        "--experiment", default=None,
        help="Lab command: 'baseline' (default) for the deterministic frozen-baseline run; "
             "'list' to show the experiment registry; an experiment name or 'all' to run "
             "enhancement experiments; 'final' for the final comparison + acceptance gates.",
    )
    group.add_argument(
        "--lab-config", default=None,
        help="Path to a JSON LabConfig for ad-hoc lab runs. Unknown keys fail closed. "
             "Omit for the all-enhancements-off default.",
    )
    group.add_argument(
        "--lab-run-dir", default=None,
        help="Existing lab run directory to query with --explain-date/--explain-ticker/"
             "--show-reason. Defaults to the most recent strategy_lab run under --output-dir.",
    )
    group.add_argument("--explain-date", default=None, metavar="YYYY-MM-DD",
                       help="Explain every decision the lab run made on this date.")
    group.add_argument("--explain-ticker", default=None, metavar="TICKER",
                       help="Explain every decision involving this ticker.")
    group.add_argument("--show-reason", default=None, metavar="REASON_CODE",
                       help="Show every decision with this reason code (see src/lab/reasons.py).")
    group.add_argument(
        "--lab-holdout", action="store_true",
        help="Permit evaluation of the reserved holdout period (2024). The first holdout "
             "evaluation is recorded permanently in the run manifest. Off by default.",
    )


def _load_lab_config(args) -> LabConfig:
    if not getattr(args, "lab_config", None):
        return LabConfig()
    import json

    path = Path(args.lab_config)
    if not path.exists():
        raise LabConfigError(f"--lab-config file not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return LabConfig.from_dict(raw)


def _resolve_universe(args, pairs):
    """Shared stock universe, resolved once -- lazy import to avoid a module
    cycle with src.cli."""
    from ..cli import resolve_mean_reversion_universe

    needs_stock = any(STRATEGY_REGISTRY[name].uses_stock_universe for name, _ in pairs)
    if not needs_stock:
        return None
    return resolve_mean_reversion_universe(args)


def run_strategy_lab(args) -> int:
    from .. import universe as universe_module
    from ..data import FetchError

    try:
        if args.explain_date or args.explain_ticker or args.show_reason:
            from .explain import run_explain_query

            return run_explain_query(args)

        experiment = args.experiment or "baseline"
        if experiment == "baseline":
            return _run_baseline(args)
        if experiment == "list":
            from .experiments import print_registry

            print_registry()
            return 0
        from .experiments import run_experiment_command

        return run_experiment_command(args, experiment)
    except (LabConfigError, LabEngineError, PortfolioError, FetchError,
            universe_module.UniverseError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# --- Baseline ---------------------------------------------------------------------

def _run_baseline(args) -> int:
    """The frozen-baseline command: run the fixed 60/35/5 portfolio through
    the lab engine with EVERY enhancement off, write the standard portfolio
    artifacts plus a reproducibility manifest and a baseline snapshot.

    This is the control every experiment is compared against. The lab engine
    path is proven identical to --strategy portfolio by
    tests/test_lab_equivalence.py, so this run IS the production baseline,
    just manifest-stamped and traced."""
    from ..reporting import write_portfolio_report

    lab_config = _load_lab_config(args)
    if lab_config.any_behavior_change():
        raise LabConfigError(
            "--experiment baseline must run with every enhancement disabled; got enabled "
            f"features {lab_config.enabled_features()}. Use a named experiment (or "
            "--experiment all) to run enhancements."
        )

    fractional_shares = not args.no_fractional_shares
    pairs = parse_portfolio_weights(args.portfolio_weights)
    validate_portfolio_weights(pairs, STRATEGY_REGISTRY)
    mr_universe = _resolve_universe(args, pairs)

    recorder = None
    if lab_config.trace:
        from .trace import DecisionRecorder

        recorder = DecisionRecorder(lab_config_hash=lab_config.config_hash())

    run = run_lab_portfolio(
        pairs, args.capital, args.start, args.end, args.cost_bps, fractional_shares,
        args.refresh_cache, lab_config=lab_config,
        mr_universe=mr_universe.tickers if mr_universe else None,
        mr_universe_info=mr_universe.info if mr_universe else None,
        recorder=recorder,
    )
    pf = run.portfolio

    describe_by_strategy = {}
    for name, _weight in pairs:
        spec = STRATEGY_REGISTRY[name]
        if spec.uses_stock_universe and mr_universe is not None:
            describe_by_strategy[name] = spec.factory(universe=mr_universe.tickers).describe()
        else:
            describe_by_strategy[name] = spec.factory().describe()

    run_config = {
        "requested_start": str(pd.Timestamp(args.start).date()),
        "requested_end": str(pd.Timestamp(args.end).date()),
        "capital": args.capital,
        "cost_bps": args.cost_bps,
        "fractional_shares": fractional_shares,
        "universe_mode": args.universe,
        "weights": {name: w for name, w in pairs},
        "lab_config_hash": lab_config.config_hash(),
        "allocation_note": "static allocation: capital split once at the start, sleeve weights "
                           "drift with performance, no cash transferred between sleeves",
    }
    run_dir = write_portfolio_report(pf, run_config, describe_by_strategy, output_dir=args.output_dir)

    if run.trace is not None:
        from .trace import write_trace_artifacts

        write_trace_artifacts(run.trace, run_dir)

    snapshot = build_baseline_snapshot(run, pairs, args)
    atomic_write_json(Path(run_dir) / "baseline_snapshot.json", snapshot)

    manifest = _build_run_manifest(run, pairs, args, mr_universe, run_kind="baseline")
    attach_artifact_hashes(manifest, run_dir)
    write_manifest(manifest, run_dir)

    m = pf.metrics
    print(f"\n[strategy_lab baseline] {run_config['requested_start']} to {run_config['requested_end']}")
    for sleeve in pf.sleeves:
        print(f"  {sleeve.strategy}: ${sleeve.allocated_capital:,.2f} -> ${sleeve.final_value:,.2f} "
              f"(P&L ${sleeve.pnl_contribution:,.2f}, costs ${sleeve.cost_contribution:,.2f})")
    print(f"  PORTFOLIO: ${pf.total_capital:,.2f} -> ${m.get('final_equity', 0):,.2f}  "
          f"total_return={m.get('total_return'):.2%}  cagr={m.get('cagr'):.2%}  "
          f"maxDD={m.get('max_drawdown'):.2%}  sharpe={m.get('sharpe_ratio'):.2f}  "
          f"excess vs SPY={m.get('excess_return'):.2%}")
    print(f"[strategy_lab] artifacts written to: {run_dir}")
    print(f"[strategy_lab] lab config hash: {lab_config.config_hash()}")
    return 0


def build_baseline_snapshot(run, pairs, args) -> dict:
    """The frozen numeric reference for the immutable baseline: everything a
    later comparison needs, stamped with the config hash and window."""
    pf = run.portfolio
    from ..metrics import annual_returns

    return {
        "schema_version": LAB_SNAPSHOT_SCHEMA_VERSION,
        "lab_config_hash": run.lab_config.config_hash(),
        "weights": {name: w for name, w in pairs},
        "capital": args.capital,
        "cost_bps": args.cost_bps,
        "window": {
            "requested_start": str(pd.Timestamp(args.start).date()),
            "requested_end": str(pd.Timestamp(args.end).date()),
            "common_start": str(pf.common_start.date()),
            "common_end": str(pf.common_end.date()),
        },
        "portfolio_metrics": pf.metrics,
        "spy_metrics": pf.spy_metrics,
        "annual_returns": {
            str(y): r for y, r in annual_returns(pf.combined_result.equity_curve).items()
        },
        "sleeves": {
            s.strategy: {
                "weight": s.weight,
                "allocated_capital": s.allocated_capital,
                "final_value": s.final_value,
                "ending_weight": s.ending_weight,
                "pnl_contribution": s.pnl_contribution,
                "cost_contribution": s.cost_contribution,
                "metrics": s.metrics,
            }
            for s in pf.sleeves
        },
    }


def _build_run_manifest(run, pairs, args, mr_universe, run_kind: str,
                        seeds: dict | None = None, holdout: dict | None = None) -> dict:
    pf = run.portfolio
    strategy_params = {}
    warmup_days = {}
    for prepared in run.prepared:
        strategy_params[prepared.name] = prepared.strategy.describe()
        warmup_days[prepared.name] = prepared.warmup_days

    cache_tickers = sorted(
        {t for prepared in run.prepared for t in prepared.strategy.universe}
        | {t for prepared in run.prepared for t in prepared.strategy.signal_tickers}
        | {config.BENCHMARK_TICKER}
    )
    return build_manifest(
        run_kind=run_kind,
        lab_config_dict=run.lab_config.to_dict(),
        lab_config_hash=run.lab_config.config_hash(),
        weights=pairs,
        capital=args.capital,
        cost_bps=args.cost_bps,
        fractional_shares=not args.no_fractional_shares,
        start=str(pd.Timestamp(args.start).date()),
        end=str(pd.Timestamp(args.end).date()),
        universe_mode=args.universe,
        universe=sorted({t for prepared in run.prepared for t in prepared.strategy.universe}),
        universe_info=mr_universe.info if mr_universe else None,
        strategy_params=strategy_params,
        warmup_days=warmup_days,
        seeds=seeds,
        cash_yield_annual=run.lab_config.cash_yield.annual_rate if run.lab_config.cash_yield.enabled else 0.0,
        holdout=holdout,
        cache_tickers=cache_tickers,
        extra={
            "portfolio_result_summary": {
                "final_equity": pf.metrics.get("final_equity"),
                "common_start": str(pf.common_start.date()),
                "common_end": str(pf.common_end.date()),
            }
        },
    )
