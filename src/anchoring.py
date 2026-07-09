"""Anchoring convention for cross-refit comparability (RQ1 prerequisite).

Both BT and lattice ability scales are translation-indeterminate; production
Arena pins mixtral-8x7b-instruct-v0.1 = 1114 Elo. For this project every fit
of EITHER method is anchored the same way:

    ANCHOR_MODEL = "gpt-4-0613" pinned to 0.0 in the method's native units.

Why gpt-4-0613 (reasoning recorded 2026-07-10, see RESEARCH_LOG):
- fixed dated checkpoint (no silent version drift, unlike -latest aliases);
- 96,284 votes — among the best-estimated models in the log;
- active 2023-06-27 through the end of the battle file (2024-08-14), so it
  exists in every cumulative fit window the rolling-refit experiment uses;
- used identically for both methods, so cross-method comparisons never mix
  anchoring conventions. (mixtral-8x7b enters 2023-12-11 — too late.)

Known caveat (pre-specified sensitivity for RQ1): single-model anchoring
propagates the anchor's own estimation noise as a common offset in every
refit. Rank metrics (Kendall tau) are immune; |delta-theta| metrics get a
median-alignment sensitivity variant (align consecutive fits on the median
shift over common incumbents) to bound that contribution.
"""

from __future__ import annotations

import pandas as pd

ANCHOR_MODEL = "gpt-4-0613"
ANCHOR_VALUE = 0.0


def anchor(theta: pd.Series, model: str = ANCHOR_MODEL, value: float = ANCHOR_VALUE) -> pd.Series:
    """Translate a fitted ability Series so theta[model] == value."""
    if model not in theta.index:
        raise KeyError(f"anchor model {model!r} not in fitted set "
                       f"({len(theta)} models) — choose a window that contains it")
    return theta + (value - theta[model])
