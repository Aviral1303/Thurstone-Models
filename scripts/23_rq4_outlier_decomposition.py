"""POST-HOC DIAGNOSTIC (labeled): decompose RQ4-main's 2023-11-30 outlier
window per pair and outcome channel.

Run 2026-07-10 after Part A of scripts/22 contradicted the assumed driver
(the filter excluded pplx votes yet the window delta ROSE). Result: pplx
pairs contribute -2.8% of the window delta; the driver is the TIE channel
on near-peer same-family pairs (+407 nats on ties vs -301 on decisive).
See RQ4_FINDINGS.md section 5. Reproduces fits deterministically at the
recorded per-window parameters.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from davidson_link import DavidsonLink
from lattice_link import LatticeLink
from fit import fit_gaplink
from rq4_eval import predict_and_score_trinomial

battles = pd.read_parquet(ROOT / "data/processed/clean_battle_20240814.parquet")
battles = battles.assign(model_a=battles.model_a.replace({"chatgpt-4o-latest":"chatgpt-4o-latest-2024-08-08"}),
                         model_b=battles.model_b.replace({"chatgpt-4o-latest":"chatgpt-4o-latest-2024-08-08"}))
dedup = battles[battles.dedup_sampled]
t0 = pd.Timestamp("2023-12-01", tz="UTC").timestamp()
t1 = pd.Timestamp("2024-01-01", tz="UTC").timestamp()
train, test = dedup[dedup.tstamp <= t0], dedup[(dedup.tstamp > t0) & (dedup.tstamp <= t1)]
traj = pd.read_csv(ROOT / "results/tables/rq4_param_trajectories.csv")
r = traj[(traj.variant == "main_tie_only") & (traj.window == "2023-11-30")].iloc[0]
lk_d, lk_l = DavidsonLink(nu=r.nu_hat), LatticeLink(unit=r.unit_hat, g_step=0.005)
th_d = fit_gaplink(train, lk_d, mode="native")
th_l = fit_gaplink(train, lk_l, mode="native")
s_d = predict_and_score_trinomial(th_d, lk_d, test)
s_l = predict_and_score_trinomial(th_l, lk_l, test)
both = s_d.scoreable & s_l.scoreable
sub = s_d[both].assign(d=(s_d.logloss - s_l.logloss)[both])
sub["pair"] = sub.apply(lambda x: " vs ".join(sorted([x.model_a, x.model_b])), axis=1)
print(f"total d={sub.d.sum():.1f}; pplx share: "
      f"{sub[sub.pair.str.contains('pplx')].d.sum()/sub.d.sum():.1%}")
print(sub.groupby(sub.winner).agg(n=("d","size"), total_d=("d","sum")).round(1).to_string())
out = sub.groupby("pair").agg(n=("d","size"), total_d=("d","sum")).sort_values(
    "total_d", key=abs, ascending=False)
out.to_csv(ROOT / "results/tables/rq4_outlier_decomposition.csv")
print(out.head(8).round(2).to_string())
