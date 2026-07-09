"""Unit tests for the pre-committed RQ3 pattern classifier (section 4 of
RQ3_PREANALYSIS.md). All four verdict branches must fire on constructed
window tables — in particular lattice_positive, whose end-to-end synthetic
coverage is impossible (no in-family world produces a >=MPD lattice
advantage at realistic N; the bt_positive mirror is covered end-to-end by
World P2 in scripts/10)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq3_eval import MPD_LOGLOSS, classify, pooled_estimate  # noqa: E402


def _table(deltas, n=20_000):
    return pd.DataFrame({
        "window": [f"w{k:02d}" for k in range(len(deltas))],
        "n": [n] * len(deltas),
        "mean_d_logloss": deltas,
        "mean_d_brier": [d / 2 for d in deltas],
        "mean_ll_bt": [0.5] * len(deltas),
        "mean_ll_lat": [0.5] * len(deltas),
    })


def _verdict(deltas):
    wt = _table(deltas)
    return classify(wt, pooled_estimate(wt))


def test_lattice_positive_branch():
    # 12/13 windows lattice-better, effect ~3x MPD, tight spread
    deltas = [3 * MPD_LOGLOSS + 0.2 * MPD_LOGLOSS * ((-1) ** k) for k in range(12)]
    deltas.append(-0.1 * MPD_LOGLOSS)
    v = _verdict(deltas)
    assert v["verdict"] == "lattice_positive", v


def test_bt_positive_branch():
    deltas = [-3 * MPD_LOGLOSS - 0.2 * MPD_LOGLOSS * ((-1) ** k) for k in range(12)]
    deltas.append(0.1 * MPD_LOGLOSS)
    v = _verdict(deltas)
    assert v["verdict"] == "bt_positive", v


def test_equivalence_branch_with_directional_note():
    rng = np.random.default_rng(3)
    deltas = list(0.3 * MPD_LOGLOSS + 0.05 * MPD_LOGLOSS * rng.standard_normal(13))
    v = _verdict(deltas)
    assert v["verdict"] == "equivalence", v
    assert v.get("directional_note", "").endswith("lattice")


def test_inconclusive_branch_wide_ci():
    # huge window-to-window spread -> CI spans both band edges
    deltas = [8 * MPD_LOGLOSS * ((-1) ** k) + MPD_LOGLOSS * 0.3 for k in range(13)]
    v = _verdict(deltas)
    assert v["verdict"] == "inconclusive", v


def test_heterogeneous_big_effect_without_consistency():
    # pooled effect large and CI>0 but driven by 3 giant windows; consistency fails
    deltas = [12 * MPD_LOGLOSS] * 4 + [-0.4 * MPD_LOGLOSS] * 9
    v = _verdict(deltas)
    assert v["verdict"] == "inconclusive", v
