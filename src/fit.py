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

Tie categories (mode-dependent — be careful):
- half_tie mode: fastchat's compute_mle_elo POOLS 'tie' and 'tie (bothbad)',
  so faithful BT-equivalent fits must pass include_both_bad=True. (Caught by
  the RQ1 validation track 2026-07-10: fitting with both-bad dropped puts
  our BT ~13-19 Elo pts MAE from the published board vs ~0.2 when pooled.)
- native mode (RQ4): `tie` is the dead-heat outcome; `tie (bothbad)` is
  EXCLUDED by default per user decision 2026-07-09 (dead-heat = draws close
  together; "both bad" = both below an absolute bar, unrelated to
  closeness). include_both_bad=True is the robustness variant.

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
    full_output: bool = False,
):
    """Fit abilities (higher = better) by per-vote MLE. Returns mean-centered Series.

    With full_output=True returns (series, penalized_nll_total) — the total
    (unnormalized) negative log-likelihood at the optimum, for profiling
    structural parameters such as the lattice unit.
    """
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
    # The interpolated log-curves are piecewise linear, so their gradients are
    # piecewise constant; L-BFGS line searches can end ABNORMAL at kinks even
    # when the fit is done. Accept any termination whose gradient certifies
    # convergence on the normalized objective (comparable fits that report
    # CONVERGENCE stop at the same ~1e-5 gradient scale).
    if not res.success and np.max(np.abs(res.jac)) > 1e-4:
        raise RuntimeError(
            f"MLE did not converge: {res.message} (max|grad|={np.max(np.abs(res.jac)):.2e})"
        )
    theta = res.x - res.x.mean()
    series = pd.Series(theta, index=data.models).sort_values(ascending=False)
    if full_output:
        # undo the conditioning normalization; includes the (tiny) l2 term
        return series, float(res.fun / norm)
    return series


def profile_lattice_unit(
    battles: pd.DataFrame,
    units: np.ndarray,
    make_link,
    mode: str = "native",
    include_both_bad: bool = False,
    refine: bool = True,
) -> tuple[float, pd.Series, pd.DataFrame]:
    """Profile-MLE over the lattice unit (tie-band width).

    The unit enters only through the link curves, so joint MLE reduces to a
    1D profile: for each candidate unit, rebuild the link (cheap) and
    maximize over abilities. Returns (best_unit, theta_at_best, profile_df).

    `make_link` is a callable unit -> link object (kept injectable so tests
    can pass reduced-resolution links).

    If refine=True, one quadratic-interpolation step through the best grid
    point and its neighbors sharpens the estimate beyond grid resolution.
    """
    rows = []
    fits = {}
    for u in units:
        theta, nll = fit_gaplink(battles, make_link(float(u)), mode=mode,
                                 include_both_bad=include_both_bad, full_output=True)
        rows.append({"unit": float(u), "nll": nll})
        fits[float(u)] = theta
    prof = pd.DataFrame(rows).sort_values("unit").reset_index(drop=True)
    k = int(prof["nll"].idxmin())
    best_u = float(prof.loc[k, "unit"])

    if refine and 0 < k < len(prof) - 1:
        x = prof["unit"].to_numpy()[k - 1:k + 2]
        y = prof["nll"].to_numpy()[k - 1:k + 2]
        denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
        b = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0]) + x[0] ** 2 * (y[1] - y[2])) / denom
        if a > 0:
            u_star = float(-b / (2 * a))
            if x[0] < u_star < x[2]:
                theta, nll = fit_gaplink(battles, make_link(u_star), mode=mode,
                                         include_both_bad=include_both_bad, full_output=True)
                if nll < prof.loc[k, "nll"]:
                    fits[u_star] = theta
                    prof = pd.concat([prof, pd.DataFrame([{"unit": u_star, "nll": nll}])],
                                     ignore_index=True).sort_values("unit").reset_index(drop=True)
                    best_u = u_star

    return best_u, fits[best_u], prof
