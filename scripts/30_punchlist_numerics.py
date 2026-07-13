"""Punch-list numerics (labeled post-hoc robustness/diagnostics). Parts:

A  MPD sensitivity: RQ3/RQ4 verdicts across a grid of practical thresholds.
B  Leave-one-window-out pooled estimates for RQ3 and RQ4.
C  Triple concentration in the RQ2b analysis.
D  RQ1 mean/median paired tau_b difference with window-bootstrap CI.
E  Bootstrap check of the non-monotone width ordering in RQ3 leans.
F  2025-vs-2024 composition: votes per model, pool overlap.
G  Published-board CI half-widths (context for the 10-Elo threshold).
H  Diagnostic on the 2024-04-03 validation outlier: top residual models.
I  Toy worked example: probit-vs-logit ceiling on a unit-normal gap world.
J  Recalibration gain: what fixing the shared 1.46 slope would buy, x MPD.

Outputs under results/tables/: mpd_sensitivity.csv, loo_windows.csv,
punchlist_misc.csv, outlier_residuals.csv, recalibration_gain.csv
"""

import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402
from rq3_eval import classify, pooled_estimate  # noqa: E402

T = ROOT / "results" / "tables"
misc = []

def note(part, key, value):
    misc.append({"part": part, "key": key, "value": value})
    print(f"[{part}] {key}: {value}")

ELO_PER_NAT = 400 / np.log(10)

# ---------- A: MPD sensitivity ----------
w3 = pd.read_csv(T / "rq3_window_table_u0.5855.csv")
p3 = pooled_estimate(w3)
w4 = pd.read_csv(T / "rq4_window_table.csv")
w4 = w4[w4.variant == "main_tie_only"].drop(columns=["variant"])
p4 = pooled_estimate(w4)
rows = []
for elo in (3, 5, 7.5, 10, 15, 20):
    mpd = 0.5 * 0.25 * (elo * np.log(10) / 400) ** 2
    v = classify(w3, p3, mpd=mpd)
    rows.append({"experiment": "rq3_u0.5855", "threshold": f"{elo} Elo",
                 "mpd_nats": mpd, "pooled_x_mpd": p3["pooled"] / mpd,
                 "verdict": v["verdict"], "note": v.get("directional_note", "")})
for pp in (0.5, 0.75, 1.0, 1.5, 2.0):
    mpd = 0.5 * (pp / 100) ** 2 * (1 / 0.2 + 1 / 0.8)
    v = classify(w4, p4, mpd=mpd)
    rows.append({"experiment": "rq4_main", "threshold": f"{pp}pp tie",
                 "mpd_nats": mpd, "pooled_x_mpd": p4["pooled"] / mpd,
                 "verdict": v["verdict"], "note": v.get("directional_note", "")})
sens = pd.DataFrame(rows)
sens.to_csv(T / "mpd_sensitivity.csv", index=False)
print(sens.round(4).to_string(index=False))
# breakeven reading error where RQ3 CI-hi hits the band edge
elo_be = np.sqrt(abs(p3["ci_lo"]) * 8) * ELO_PER_NAT
note("A", "rq3 CI stays in band down to reading error (Elo)", round(elo_be, 2))

# ---------- B: leave-one-window-out ----------
loo = []
for name, wt, mpd in (("rq3_u0.5855", w3, 4e-4), ("rq4_main", w4, 3e-4)):
    for drop in list(wt["window"]) + [None]:
        sub = wt if drop is None else wt[wt.window != drop]
        p = pooled_estimate(sub)
        v = classify(sub, p, mpd=mpd, consistency_needed=int(np.ceil(len(sub) * 10 / 13)))
        loo.append({"experiment": name, "dropped": drop or "(none)",
                    "pooled_x_mpd": p["pooled"] / mpd,
                    "ci_lo_x_mpd": p["ci_lo"] / mpd, "ci_hi_x_mpd": p["ci_hi"] / mpd,
                    "verdict": v["verdict"]})
loo = pd.DataFrame(loo)
loo.to_csv(T / "loo_windows.csv", index=False)
r4 = loo[(loo.experiment == "rq4_main")]
note("B", "rq4 without 2023-11-30",
     r4[r4.dropped == "2023-11-30"][["pooled_x_mpd", "ci_lo_x_mpd", "ci_hi_x_mpd", "verdict"]]
     .round(3).to_dict("records")[0])
note("B", "rq4 LOO verdict counts", r4[r4.dropped != "(none)"]["verdict"].value_counts().to_dict())
note("B", "rq3 LOO verdict counts",
     loo[(loo.experiment == "rq3_u0.5855") & (loo.dropped != "(none)")]["verdict"].value_counts().to_dict())

# ---------- C: triple concentration ----------
mo = pd.read_csv(T / "rq2b_excess_models.csv")
tot = mo.groupby("model")["triples_total"].sum().sort_values(ascending=False)
slots = tot.sum()  # each triple contributes 3 slots
note("C", "distinct models in triples", len(tot))
note("C", "top-5 model share of triple slots", round(tot.head(5).sum() / slots, 3))
note("C", "top-10 model share of triple slots", round(tot.head(10).sum() / slots, 3))

# ---------- D: RQ1 delta-tau CI ----------
m = pd.read_csv(T / "rq1_metrics.csv")
s = m[(m.delta == 1) & (m.incumbents == "votes>=1000")]
piv = s.pivot(index="T", columns="method", values="kendall")
d = (piv["lattice_u0.1"] - piv["bt"]).to_numpy()
rng = np.random.default_rng(20260713)
bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
note("D", "mean dtau", f"{d.mean():+.5f}")
note("D", "mean dtau 95% CI", f"({np.percentile(bs,2.5):+.5f}, {np.percentile(bs,97.5):+.5f})")

# ---------- E: width-ordering bootstrap ----------
tabs = {u: pd.read_csv(T / f"rq3_window_table_{u}.csv").set_index("window")["mean_d_logloss"]
        for u in ("u0.1", "u0.5855", "u0.8")}
D = pd.DataFrame(tabs)
idx = rng.integers(0, len(D), size=(10000, len(D)))
means = {u: D[u].to_numpy()[idx].mean(axis=1) for u in D}
flip = np.mean(means["u0.1"] > means["u0.5855"])  # lean(u0.1) less negative than u0.5855
note("E", "P(u0.1 lean weaker than u0.5855 under window bootstrap)", round(float(flip), 3))
note("E", "P(u0.8 lean strongest)", round(float(np.mean(
    (means["u0.8"] < means["u0.1"]) & (means["u0.8"] < means["u0.5855"]))), 3))

# ---------- F: 2025 composition ----------
b24 = pd.read_parquet(ROOT / "data/processed/clean_battle_20240814.parquet")
b24 = b24[b24.dedup_sampled]
b25 = pd.read_parquet(ROOT / "data/processed/arena2025_140k.parquet")
v24 = pd.concat([b24.model_a, b24.model_b]).value_counts()
v25 = pd.concat([b25.model_a, b25.model_b]).value_counts()
note("F", "votes/model median 2024 vs 2025", f"{int(v24.median()):,} vs {int(v25.median()):,}")
note("F", "models overlapping the 2023-24 pool", f"{len(set(v25.index) & set(v24.index))} of {len(v25)}")

# ---------- G: published-board CI half-widths ----------
pub = pd.read_csv(T / "published_bt_20240813.csv")
hw = (pub["rating_q975"] - pub["rating_q025"]) / 2
note("G", "published 95% CI half-width median (Elo)", round(float(hw.median()), 2))
note("G", "published 95% CI half-width p90 (Elo)", round(float(hw.quantile(0.9)), 2))

# ---------- H: 2024-04-03 outlier diagnostic ----------
class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): pass
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("plotly") or module.startswith("_plotly"):
            return _Stub
        return super().find_class(module, name)
pkl = ROOT / "data/raw/elo_pkls/elo_results_20240403.pkl"
if not pkl.exists():
    urllib.request.urlretrieve(
        "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main/elo_results_20240403.pkl", pkl)
with open(pkl, "rb") as f:
    obj = SafeUnpickler(f).load()
node = obj["text"]["full"]
pubr = node["leaderboard_table_df"]["rating"]
ts = float(node["last_updated_tstamp"])
b24r = pd.read_parquet(ROOT / "data/processed/clean_battle_20240814.parquet")
b24r = b24r.assign(model_a=b24r.model_a.replace({"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}),
                   model_b=b24r.model_b.replace({"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}))
best = {}
for variant, frame in (("no_dedup", b24r), ("dedup", b24r[b24r.dedup_sampled])):
    sub = frame[frame.tstamp <= ts]
    th = fit_gaplink(sub, LogisticLink(), mode="half_tie", include_both_bad=True) * ELO_PER_NAT
    common = pubr.index.intersection(th.index)
    resid = (pubr.loc[common] - (th.loc[common] + (pubr.loc[common] - th.loc[common]).median()))
    best[variant] = resid
    note("H", f"{variant} MAE", round(float(resid.abs().mean()), 2))
resid = best["no_dedup"]
top = resid.abs().sort_values(ascending=False).head(8)
first_seen = pd.concat([b24r.groupby("model_a").tstamp.min(),
                        b24r.groupby("model_b").tstamp.min()], axis=1).min(axis=1)
out_h = pd.DataFrame({"residual_elo": resid.reindex(top.index).round(2),
                      "days_in_pool_at_snapshot": ((ts - first_seen.reindex(top.index)) / 86400).round(1)})
out_h.to_csv(T / "outlier_residuals.csv")
print(out_h.to_string())

# ---------- I: toy example — probit vs logit, unit-normal gaps ----------
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import norm  # noqa: E402
gs = np.linspace(-6, 6, 2401)
wts = norm.pdf(gs); wts /= wts.sum()
q = np.clip(norm.cdf(gs), 1e-12, 1 - 1e-12)
def kl_at(s):
    p = np.clip(1 / (1 + np.exp(-gs / s)), 1e-12, 1 - 1e-12)
    return float(np.sum(wts * (q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p)))))
r = minimize_scalar(kl_at, bounds=(0.3, 2.0), method="bounded")
note("I", "toy ceiling probit-truth vs best logistic, N(0,1) gaps (nats)", f"{r.fun:.2e}")
note("I", "toy best logistic scale", round(float(r.x), 4))
note("I", "toy ceiling in RQ3-MPD units", round(r.fun / 4e-4, 2))

# ---------- J: recalibration gain ----------
FINAL_CUTOFF = 1723479651.547
dedup = b24r[b24r.dedup_sampled]
month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
from sklearn.linear_model import LogisticRegression  # noqa: E402
bt = LogisticLink()
gains = []
for k in range(len(cutoffs) - 1):
    tr = dedup[dedup.tstamp <= cutoffs[k]]
    te = dedup[(dedup.tstamp > cutoffs[k]) & (dedup.tstamp <= cutoffs[k + 1])]
    th = fit_gaplink(tr, bt, mode="half_tie", include_both_bad=True)
    dec = te[te.winner.isin(("model_a", "model_b"))]
    known = dec.model_a.isin(th.index) & dec.model_b.isin(th.index)
    d2 = dec[known]
    g = th.reindex(d2.model_a).to_numpy() - th.reindex(d2.model_b).to_numpy()
    p = np.clip(bt.f_decisive(g), 1e-9, 1 - 1e-9)
    y = (d2.winner == "model_a").to_numpy(dtype=int)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(x, y)
    pr = np.clip(lr.predict_proba(x)[:, 1], 1e-9, 1 - 1e-9)
    ll_raw = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    ll_cal = -(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean()
    gains.append({"window": k, "n": len(d2), "gain_nats": ll_raw - ll_cal,
                  "gain_x_mpd": (ll_raw - ll_cal) / 4e-4})
gains = pd.DataFrame(gains)
gains.to_csv(T / "recalibration_gain.csv", index=False)
pooled_gain = float(np.average(gains.gain_nats, weights=gains.n)) / 4e-4
note("J", "in-window recalibration gain, vote-weighted (x MPD)", round(pooled_gain, 2))
note("J", "NB", "gain uses same-window refit slope; an oracle upper bound on drift cost")

pd.DataFrame(misc).to_csv(T / "punchlist_misc.csv", index=False)
print("DONE")
