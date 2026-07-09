"""RQ3 pre-analysis: recent-entrant-stratum noise & achievable effect,
with bootstrap-CALIBRATED SEs (items 1-2 of the 2026-07-11 review).

Answers: at each checkpoint regime, what is the ACTUAL noise level of the
28-day recent-entrant cohort (not the era average), and what expected
held-out delta could lattice-truth produce on the recent-entrant STRATUM
votes at that noise?

Stratum vote composition is taken from the last 28 days of TRAINING data
(votes involving >=1 recent model) — a strictly pre-test proxy for the test
window's stratum composition; no test-window data of any kind is used.
All SEs are bootstrap-calibrated (Fisher / 0.80). No held-out real
calibration number is computed.

Output: results/tables/rq3_stratum_noise.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402
from rq3_eval import RECENT_DAYS, SECONDS_PER_DAY, fisher_se_calibrated  # noqa: E402

SEED = 20260712
MPD = 4e-4
ELO_PER_NAT = 400 / np.log(10)
rng = np.random.default_rng(SEED)

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

fits_full = pd.read_csv(ROOT / "results" / "tables" / "full_fits_20240814.csv", index_col="model")
g_lat_all = (fits_full["lattice_theta"].reindex(dedup["model_a"]).to_numpy()
             - fits_full["lattice_theta"].reindex(dedup["model_b"]).to_numpy())
hist, edges = np.histogram(g_lat_all, bins=400, range=(-8, 8))
w_emp = hist / hist.sum()
centers = 0.5 * (edges[:-1] + edges[1:])


def best_logistic_scale(gen_link):
    q = np.clip(gen_link.f_decisive(centers), 1e-12, 1 - 1e-12)

    def kl_at(s):
        p = np.clip(expit(centers / s), 1e-12, 1 - 1e-12)
        return float(np.sum(w_emp * (q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p)))))

    return minimize_scalar(kl_at, bounds=(0.2, 5.0), method="bounded").x


CUTS = {
    "early_2023-08-31": pd.Timestamp("2023-09-01", tz="UTC").timestamp(),
    "mid_2023-12-31": pd.Timestamp("2024-01-01", tz="UTC").timestamp(),
    "full_2024-08-12": 1723479651.547,
}
TRUTHS = [(0.5855, 0.0), (1.2, 6.0)]
N_DRAWS = 200

rows = []
for regime, cut in CUTS.items():
    train = dedup[dedup["tstamp"] <= cut]
    first_seen = pd.concat([
        train.groupby("model_a")["tstamp"].min(),
        train.groupby("model_b")["tstamp"].min(),
    ], axis=1).min(axis=1)
    recent_models = first_seen[first_seen >= cut - RECENT_DAYS * SECONDS_PER_DAY].index
    # stratum vote composition: last 28 training days, >=1 recent model
    last28 = train[train["tstamp"] > cut - RECENT_DAYS * SECONDS_PER_DAY]
    strat = last28[last28["model_a"].isin(recent_models) | last28["model_b"].isin(recent_models)]

    th_lat = fit_gaplink(train, LatticeLink(unit=0.5855), mode="half_tie", include_both_bad=True)
    se_lat = fisher_se_calibrated(train, LatticeLink(unit=0.5855), th_lat)
    th_bt = fit_gaplink(train, LogisticLink(), mode="half_tie", include_both_bad=True)
    se_bt = fisher_se_calibrated(train, LogisticLink(), th_bt)

    se_recent = se_bt.reindex(recent_models.intersection(se_bt.index))
    print(f"[{regime}] recent models: {len(recent_models)}, stratum votes (last 28d): "
          f"{len(strat):,}; recent-cohort SE (BT, calibrated): "
          f"median {se_recent.median()*ELO_PER_NAT:.1f} Elo, max {se_recent.max()*ELO_PER_NAT:.1f} "
          f"vs era median {se_bt.median()*ELO_PER_NAT:.1f}")

    models = th_lat.index
    midx = {m: i for i, m in enumerate(models)}
    ia = strat["model_a"].map(midx).to_numpy()
    ib = strat["model_b"].map(midx).to_numpy()
    th = th_lat.to_numpy()
    g_true = th[ia] - th[ib]
    se_l = se_lat.reindex(models).to_numpy()
    se_b = se_bt.reindex(models).to_numpy()

    for (u, a) in TRUTHS:
        gen = LatticeLink(unit=u, skew_a=a, g_step=0.005)
        s_star = best_logistic_scale(gen)
        q = np.clip(gen.f_decisive(g_true), 1e-12, 1 - 1e-12)
        d_sum = 0.0
        for _ in range(N_DRAWS):
            eps_l = rng.normal(0, se_l)
            eps_b = rng.normal(0, se_b)
            p_lat = np.clip(gen.f_decisive(g_true + eps_l[ia] - eps_l[ib]), 1e-12, 1 - 1e-12)
            p_bt = np.clip(expit((g_true + eps_b[ia] - eps_b[ib]) / s_star), 1e-12, 1 - 1e-12)
            ll_lat = -(q * np.log(p_lat) + (1 - q) * np.log(1 - p_lat))
            ll_bt = -(q * np.log(p_bt) + (1 - q) * np.log(1 - p_bt))
            d_sum += float(np.mean(ll_bt - ll_lat))
        d = d_sum / N_DRAWS
        rows.append({
            "regime": regime, "truth_unit": u, "truth_skew": a,
            "n_recent_models": len(recent_models), "stratum_votes_28d": len(strat),
            "recent_cohort_median_se_elo": float(se_recent.median() * ELO_PER_NAT),
            "recent_cohort_max_se_elo": float(se_recent.max() * ELO_PER_NAT),
            "era_median_se_elo": float(se_bt.median() * ELO_PER_NAT),
            "stratum_expected_delta_x_mpd": d / MPD,
        })
        print(f"  truth u{u}/a{a}: stratum expected delta {d/MPD:+.2f}x MPD "
              f"({'lattice' if d > 0 else 'BT'} favored)")

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "tables" / "rq3_stratum_noise.csv", index=False)
print("\nDONE")
