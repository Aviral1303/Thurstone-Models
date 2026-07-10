"""RQ4 evaluation machinery — held-out TRINOMIAL calibration.

Mirrors rq3_eval but scores all three outcomes {A wins, B wins, tie}
(both-bad excluded per the standing decision; include_both_bad handled at
the data-filter level by the caller). Reuses rq3_eval's pooled_estimate
and classify (with mpd=MPD_RQ4).

Sign convention (RQ4_PREANALYSIS): delta = ll_davidson - ll_lattice per
vote; POSITIVE = lattice better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MPD_RQ4 = 3e-4  # nats/vote; derivation in RQ4_PREANALYSIS.md section 3

OUTCOMES = ("model_a", "model_b", "tie")


def predict_and_score_trinomial(theta: pd.Series, link, test: pd.DataFrame) -> pd.DataFrame:
    sub = test[test["winner"].isin(OUTCOMES)].copy()
    known = sub["model_a"].isin(theta.index) & sub["model_b"].isin(theta.index)
    g = (theta.reindex(sub["model_a"]).to_numpy()
         - theta.reindex(sub["model_b"]).to_numpy())
    g = np.nan_to_num(g)
    W, D, L = link.p_win(g), link.p_tie(g), link.p_loss(g)
    s = np.maximum(W + D + L, 1e-300)
    W, D, L = W / s, D / s, L / s
    p_outcome = np.select(
        [sub["winner"] == "model_a", sub["winner"] == "model_b"], [W, L], default=D)
    p_outcome = np.clip(p_outcome, 1e-12, 1.0)
    sub["scoreable"] = known.to_numpy()
    sub["logloss"] = np.where(known, -np.log(p_outcome), np.nan)
    return sub


@dataclass
class TriWindowResult:
    label: str
    n_test_votes: int
    n_in_outcome_space: int
    n_scoreable: int
    per_vote: pd.DataFrame  # columns: d_logloss, ll_dav, ll_lat


def evaluate_window_trinomial(theta_dav: pd.Series, link_dav,
                              theta_lat: pd.Series, link_lat,
                              test: pd.DataFrame, label: str) -> TriWindowResult:
    s_dav = predict_and_score_trinomial(theta_dav, link_dav, test)
    s_lat = predict_and_score_trinomial(theta_lat, link_lat, test)
    both = s_dav["scoreable"] & s_lat["scoreable"]
    per_vote = pd.DataFrame({
        "d_logloss": (s_dav["logloss"] - s_lat["logloss"])[both],
        "ll_dav": s_dav["logloss"][both],
        "ll_lat": s_lat["logloss"][both],
    })
    return TriWindowResult(label=label, n_test_votes=len(test),
                           n_in_outcome_space=len(s_dav),
                           n_scoreable=int(both.sum()), per_vote=per_vote)


def window_table_trinomial(results: list[TriWindowResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        pv = r.per_vote
        rows.append({
            "window": r.label, "n": len(pv),
            "mean_d_logloss": float(pv["d_logloss"].mean()) if len(pv) else np.nan,
            "mean_ll_dav": float(pv["ll_dav"].mean()) if len(pv) else np.nan,
            "mean_ll_lat": float(pv["ll_lat"].mean()) if len(pv) else np.nan,
        })
    return pd.DataFrame(rows)


def relabel_verdict(v: str) -> str:
    """rq3_eval.classify speaks 'bt'/'lattice'; RQ4's first method is Davidson."""
    return {"bt_positive": "davidson_positive"}.get(v, v)
