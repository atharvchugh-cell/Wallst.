"""Strategy-intelligence lab: explainable decision tracing, attribution,
optional enhancement overlays, and a validated experiment framework.

Everything in this package is OBSERVATIONAL or OPT-IN:
  - with every lab feature disabled (the default), behavior is proven
    identical to the existing `--strategy portfolio` mode by the baseline
    equivalence tests in tests/test_lab_equivalence.py;
  - no module here changes any existing CLI mode, strategy default, or
    engine behavior;
  - research / paper-trading only -- no brokerage integration, no
    credentials, no live orders.
"""
