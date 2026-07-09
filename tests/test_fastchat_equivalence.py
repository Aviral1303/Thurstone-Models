"""Regression test for the both-bad-tie treatment bug (caught by the RQ1
validation track, 2026-07-10).

fastchat's compute_mle_elo pools 'tie' AND 'tie (bothbad)' into the half-tie
weighting. Our half_tie fits must therefore run with include_both_bad=True
to be BT-baseline-equivalent. This test generates a synthetic battle set
containing BOTH tie labels and asserts our per-vote logistic fit matches
compute_mle_elo — and that forgetting the flag does NOT.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt_baseline import compute_mle_elo  # noqa: E402
from fit import fit_gaplink  # noqa: E402
from lattice_link import LogisticLink  # noqa: E402

ELO_PER_NAT = 400 / math.log(10)


@pytest.fixture(scope="module")
def battles_with_both_tie_labels():
    rng = np.random.default_rng(7)
    n_models, n = 8, 20_000
    theta = rng.normal(0, 1, n_models)
    models = np.array([f"m{k}" for k in range(n_models)])
    i = rng.integers(0, n_models, n)
    j = rng.integers(0, n_models, n)
    ok = i != j
    i, j = i[ok], j[ok]
    p = 1 / (1 + np.exp(-(theta[i] - theta[j])))
    u = rng.uniform(size=len(i))
    # 15% quality ties, 15% both-bad ties, rest decisive by BT
    winner = np.where(u < 0.15, "tie",
                      np.where(u < 0.30, "tie (bothbad)",
                               np.where(rng.uniform(size=len(i)) < p, "model_a", "model_b")))
    return pd.DataFrame({"model_a": models[i], "model_b": models[j], "winner": winner})


def _max_resid_vs_fastchat(bat, **fit_kwargs):
    ours = fit_gaplink(bat, LogisticLink(), mode="half_tie", l2=0.0, **fit_kwargs) * ELO_PER_NAT
    fc = compute_mle_elo(bat, anchor=None, tol=1e-12, max_iter=5000)
    merged = pd.DataFrame({"ours": ours, "fc": fc}).dropna()
    merged -= merged.mean()
    return float((merged["ours"] - merged["fc"]).abs().max())


def test_half_tie_with_both_bad_matches_fastchat(battles_with_both_tie_labels):
    resid = _max_resid_vs_fastchat(battles_with_both_tie_labels, include_both_bad=True)
    assert resid <= 0.1, f"max residual {resid:.3f} Elo pts vs compute_mle_elo"


def test_half_tie_without_both_bad_diverges(battles_with_both_tie_labels):
    """Guards the guard: dropping both-bad ties must produce a DIFFERENT fit,
    proving the flag actually changes the objective on mixed-label data."""
    resid = _max_resid_vs_fastchat(battles_with_both_tie_labels, include_both_bad=False)
    assert resid > 1.0, f"expected divergence without both-bad pooling, got {resid:.3f}"
