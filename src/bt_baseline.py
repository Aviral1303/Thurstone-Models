"""Bradley-Terry baseline: faithful port of fastchat's compute_mle_elo.

This is the algorithm behind the published Arena leaderboard (verified to
MAE 0.18 Elo pts in Phase 2, scripts/04). Kept as the single shared
implementation for both the replication script and synthetic cross-checks.

Tie treatment: each decisive battle contributes weight 2 in its direction;
each tie ('tie' AND 'tie (bothbad)' — fastchat pools them) contributes
weight 1 in each direction. Ratings = 400/ln(10) * theta + 1000, optionally
anchored.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def compute_mle_elo(df: pd.DataFrame, scale=400, base=10, init_rating=1000,
                    anchor=("mixtral-8x7b-instruct-v0.1", 1114.0),
                    tie_labels=("tie", "tie (bothbad)"),
                    tol=1e-6, max_iter=1000) -> pd.Series:
    """Faithful port of fastchat's compute_mle_elo (weighted LR formulation).

    Defaults (tol, max_iter) match fastchat; synthetic cross-checks pass
    tighter values so optimizer slack doesn't masquerade as objective
    disagreement.
    """
    ptbl_a_win = pd.pivot_table(df[df["winner"] == "model_a"], index="model_a",
                                columns="model_b", aggfunc="size", fill_value=0)
    ptbl_tie = pd.pivot_table(df[df["winner"].isin(list(tie_labels))],
                              index="model_a", columns="model_b", aggfunc="size", fill_value=0)
    ptbl_tie = ptbl_tie.add(ptbl_tie.T, fill_value=0)
    ptbl_b_win = pd.pivot_table(df[df["winner"] == "model_b"], index="model_a",
                                columns="model_b", aggfunc="size", fill_value=0)
    ptbl_win = (ptbl_a_win.mul(2).add(ptbl_b_win.T.mul(2), fill_value=0)
                .add(ptbl_tie, fill_value=0).fillna(0))
    all_models = ptbl_win.index.union(ptbl_win.columns)
    ptbl_win = ptbl_win.reindex(index=all_models, columns=all_models, fill_value=0)

    models = pd.Series(np.arange(len(all_models)), index=all_models)
    p = len(models)
    X, Y, W = [], [], []
    logb = math.log(base)
    # fastchat structure: for each ordered pair, one Y=1 row weighted by
    # "a beats b" count and one Y=0 row weighted by "b beats a" count.
    for m_a in all_models:
        for m_b in all_models:
            if m_a == m_b:
                continue
            w_ab = float(ptbl_win.loc[m_a, m_b])
            w_ba = float(ptbl_win.loc[m_b, m_a])
            if w_ab == 0 and w_ba == 0:
                continue
            x = np.zeros(p)
            x[models[m_a]] = +logb
            x[models[m_b]] = -logb
            X.append(x); Y.append(1.0); W.append(w_ab)
            X.append(x); Y.append(0.0); W.append(w_ba)
    X = np.asarray(X); Y = np.asarray(Y); W = np.asarray(W)
    keep = W > 0
    X, Y, W = X[keep], Y[keep], W[keep]
    lr = LogisticRegression(fit_intercept=False, C=np.inf, tol=tol, max_iter=max_iter)
    lr.fit(X, Y, sample_weight=W)
    elo = scale * lr.coef_[0] + init_rating
    s = pd.Series(elo, index=models.index)
    if anchor is not None and anchor[0] in s.index:
        s += anchor[1] - s[anchor[0]]
    return s.sort_values(ascending=False)
