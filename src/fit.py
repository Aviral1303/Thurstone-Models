"""Per-vote direct MLE for gap-link models on pairwise battles.

Fits one ability per model by maximizing a per-vote likelihood, mirroring how
choix/fastchat fit BT — same individual-vote data, only the link differs.

Two modes (Phase 3 decisions, logs/RESEARCH_LOG.md):

- mode="half_tie" — fastchat-equivalent pseudo-likelihood. Each decisive vote
  contributes weight 2 in its direction; each tie contributes weight 1 in
  EACH direction. Outcome model is the two-outcome decisive link F(g)
  (logistic for BT; W/(W+L) for lattice). This is the fair-comparison mode:
  identical tie treatment on both sides.
- mode="native" — trinomial likelihood using the link's native (W, D, L).
  Decisive vote -> log W(gap); tie vote -> log D(gap). Lattice only (the
  logistic link has no tie mass). Used for RQ4.

Tie categories: `tie` is the dead-heat outcome. `tie (bothbad)` is EXCLUDED
by default per user decision 2026-07-09 (dead-heat = draws close together;
"both bad" = both below an absolute bar, unrelated to closeness). Pass
include_both_bad=True to revisit as a robustness check.

Gauge: abilities are mean-centered (translation is unidentified).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DECISIVE = ("model_a", "model_b")
TIE_DEADHEAT = ("tie",)
TIE_BOTH_BAD = ("tie (bothbad)",)


@dataclass
class VoteData:
    """Compact per-vote arrays: index pairs plus outcome codes."""

    models: list[str]
    i: np.ndarray  # index of model_a
    j: np.ndarray  # index of model_b
    y: np.ndarray  # 0: a wins, 1: b wins, 2: dead-heat tie, 3: both-bad tie

    @classmethod
    def from_battles(cls, battles: pd.DataFrame) -> "VoteData":
        models = sorted(set(battles["model_a"]) | set(battles["model_b"]))
        midx = {m: k for k, m in enumerate(models)}
        code = {"model_a": 0, "model_b": 1, "tie": 2, "tie (bothbad)": 3}
        y = battles["winner"].map(code)
        if y.isna().any():
            bad = battles.loc[y.isna(), "winner"].unique()
            raise ValueError(f"unrecognized winner labels: {bad}")
        return cls(
            models=models,
            i=battles["model_a"].map(midx).to_numpy(np.int64),
            j=battles["model_b"].map(midx).to_numpy(np.int64),
            y=y.to_numpy(np.int64),
        )


def _rows_half_tie(data: VoteData, include_both_bad: bool):
    """Expand votes into weighted 'winner beats loser' rows (fastchat scheme)."""
    tie_mask = data.y == 2
    if include_both_bad:
        tie_mask |= data.y == 3
    a_win, b_win = data.y == 0, data.y == 1
    w_idx = np.concatenate([data.i[a_win], data.j[b_win], data.i[tie_mask], data.j[tie_mask]])
    l_idx = np.concatenate([data.j[a_win], data.i[b_win], data.j[tie_mask], data.i[tie_mask]])
    wts = np.concatenate([
        np.full(a_win.sum(), 2.0),
        np.full(b_win.sum(), 2.0),
        np.ones(tie_mask.sum()),
        np.ones(tie_mask.sum()),
    ])
    return w_idx, l_idx, wts


def fit_gaplink(
    battles: pd.DataFrame,
    link,
    mode: str = "half_tie",
    include_both_bad: bool = False,
    l2: float = 1e-6,
    tol: float = 1e-13,
) -> pd.Series:
    """Fit abilities (higher = better) by per-vote MLE. Returns mean-centered Series."""
    data = VoteData.from_battles(battles)
    n = len(data.models)

    if mode == "half_tie":
        w_idx, l_idx, wts = _rows_half_tie(data, include_both_bad)
        norm = 1.0 / wts.sum()  # normalize objective for optimizer conditioning

        def negll(theta):
            g = theta[w_idx] - theta[l_idx]
            logf, dlogf = link.log_f_decisive(g)
            grad_rows = wts * dlogf
            grad = np.zeros(n)
            np.add.at(grad, w_idx, grad_rows)
            np.add.at(grad, l_idx, -grad_rows)
            nll = -(wts @ logf) + l2 * theta @ theta
            return (norm * nll, norm * (-grad + 2 * l2 * theta))

    elif mode == "native":
        if not hasattr(link, "log_p_tie"):
            raise ValueError("native mode requires a link with tie mass")
        dec = (data.y == 0) | (data.y == 1)
        w_idx = np.where(data.y[dec] == 0, data.i[dec], data.j[dec])
        l_idx = np.where(data.y[dec] == 0, data.j[dec], data.i[dec])
        tie_mask = data.y == 2
        if include_both_bad:
            tie_mask |= data.y == 3
        t_i, t_j = data.i[tie_mask], data.j[tie_mask]
        norm = 1.0 / (len(w_idx) + len(t_i))

        def negll(theta):
            g_dec = theta[w_idx] - theta[l_idx]
            logw, dlogw = link.log_p_win(g_dec)
            g_tie = theta[t_i] - theta[t_j]
            logd, dlogd = link.log_p_tie(g_tie)
            grad = np.zeros(n)
            np.add.at(grad, w_idx, dlogw)
            np.add.at(grad, l_idx, -dlogw)
            np.add.at(grad, t_i, dlogd)
            np.add.at(grad, t_j, -dlogd)
            nll = -(np.sum(logw) + np.sum(logd)) + l2 * theta @ theta
            return (norm * nll, norm * (-grad + 2 * l2 * theta))

    else:
        raise ValueError(f"unknown mode: {mode}")

    res = minimize(negll, np.zeros(n), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": tol, "gtol": 1e-10})
    if not res.success and "REL_REDUCTION" not in str(res.message):
        raise RuntimeError(f"MLE did not converge: {res.message}")
    theta = res.x - res.x.mean()
    return pd.Series(theta, index=data.models).sort_values(ascending=False)
