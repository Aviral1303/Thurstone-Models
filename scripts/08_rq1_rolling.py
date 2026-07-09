"""RQ1 rolling-refit experiment, exactly per logs/RQ1_SPEC.md (approved
2026-07-10, with the unit=0.5855 variant added at review).

Experiment track:
  14 cumulative checkpoints (month-ends 2023-07-31..2024-07-31 UTC + the
  2024-08-12 published cutoff). Four fits per checkpoint: BT(logistic) and
  lattice at units 0.1 / 0.5855 / 0.8 — all half-tie mode, dedup_sampled,
  anchored gpt-4-0613=0. Rank metrics primary; slope-matched magnitude
  metrics secondary (anchored + median-aligned), incumbents >=1000 votes at
  T (all-models table alongside), horizons delta = 1 and 3 checkpoints.

Validation track:
  Nearest published elo_results_*.pkl per checkpoint (<=10 days). BT refit at
  each pickle's own last_updated_tstamp, dedup AND no-dedup variants,
  compared to published ratings (Spearman; MAE after median alignment on
  common models, Elo units).

Outputs: results/tables/rq1_metrics.csv, rq1_window_covariates.csv,
rq1_validation_published.csv, figures/rq1_*.png
"""

import pickle
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring import ANCHOR_MODEL, anchor  # noqa: E402
from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402

ELO_PER_NAT = 400 / np.log(10)
FINAL_CUTOFF = 1723479651.547  # published 20240813 last_updated_tstamp

# ---------------- data ----------------
battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

# ---------------- checkpoints ----------------
month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

# ---------------- links / methods ----------------
METHODS = {
    "bt": (LogisticLink(), 1.0),
    "lattice_u0.1": (LatticeLink(unit=0.1), None),
    "lattice_u0.5855": (LatticeLink(unit=0.5855), None),
    "lattice_u0.8": (LatticeLink(unit=0.8), None),
}
METHODS = {k: (lk, f if f is not None else lk.slope_match_factor()) for k, (lk, f) in METHODS.items()}
for k, (lk, f) in METHODS.items():
    print(f"{k}: slope_match_factor={f:.5f}")

# ---------------- experiment fits ----------------
fits: dict[tuple[str, int], pd.Series] = {}
votes_at: dict[int, pd.Series] = {}
for k, cut in enumerate(cutoffs):
    sub = dedup[dedup["tstamp"] <= cut]
    votes = pd.concat([sub["model_a"], sub["model_b"]]).value_counts()
    votes_at[k] = votes
    assert votes.get(ANCHOR_MODEL, 0) >= 500, f"anchor thin at {labels[k]}: {votes.get(ANCHOR_MODEL, 0)}"
    for name, (link, _) in METHODS.items():
        fits[(name, k)] = anchor(fit_gaplink(sub, link, mode="half_tie", include_both_bad=True))
    print(f"[{labels[k]}] battles={len(sub):,} models={len(votes)} "
          f"anchor_votes={votes[ANCHOR_MODEL]:,}")

# ---------------- metrics ----------------
rows = []
for k in range(len(cutoffs)):
    for delta in (1, 3):
        kd = k + delta
        if kd >= len(cutoffs):
            continue
        for inc_name, inc_set in (
            ("votes>=1000", votes_at[k][votes_at[k] >= 1000].index),
            ("all", votes_at[k].index),
        ):
            for name, (_, factor) in METHODS.items():
                th0, th1 = fits[(name, k)], fits[(name, kd)]
                inc = [m for m in inc_set if m in th0.index and m in th1.index]
                a = th0.reindex(inc).to_numpy() * factor
                b = th1.reindex(inc).to_numpy() * factor
                r0 = pd.Series(a, index=inc).rank(ascending=False)
                r1 = pd.Series(b, index=inc).rank(ascending=False)
                rank_moves = (r1 - r0).abs()
                d_anch = b - a
                d_med = d_anch - np.median(d_anch)
                rows.append({
                    "T": labels[k], "T_plus": labels[kd], "delta": delta,
                    "incumbents": inc_name, "method": name, "n_inc": len(inc),
                    "kendall": kendalltau(a, b).statistic,
                    "spearman": spearmanr(a, b).statistic,
                    "max_rank_move": float(rank_moves.max()),
                    "frac_move_gt5": float((rank_moves > 5).mean()),
                    "max_abs_dtheta_anch": float(np.max(np.abs(d_anch))),
                    "mean_abs_dtheta_anch": float(np.mean(np.abs(d_anch))),
                    "p95_abs_dtheta_anch": float(np.percentile(np.abs(d_anch), 95)),
                    "max_abs_dtheta_med": float(np.max(np.abs(d_med))),
                    "mean_abs_dtheta_med": float(np.mean(np.abs(d_med))),
                    "p95_abs_dtheta_med": float(np.percentile(np.abs(d_med), 95)),
                })
metrics = pd.DataFrame(rows)
metrics.to_csv(ROOT / "results" / "tables" / "rq1_metrics.csv", index=False)

cov_rows = []
for k in range(len(cutoffs)):
    for delta in (1, 3):
        kd = k + delta
        if kd >= len(cutoffs):
            continue
        new_models = votes_at[kd].index.difference(votes_at[k].index)
        win = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[kd])]
        n_win = len(win)
        new_votes = win["model_a"].isin(new_models) | win["model_b"].isin(new_models)
        cov_rows.append({
            "T": labels[k], "T_plus": labels[kd], "delta": delta,
            "n_entrants": len(new_models), "window_votes": n_win,
            "entrant_vote_share": float(new_votes.mean()) if n_win else np.nan,
            "n_incumbents_1000": int((votes_at[k] >= 1000).sum()),
            "n_models_at_T": len(votes_at[k]),
        })
covariates = pd.DataFrame(cov_rows)
covariates.to_csv(ROOT / "results" / "tables" / "rq1_window_covariates.csv", index=False)

# ---------------- figures ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, delta in zip(axes, (1, 3)):
    sub = metrics[(metrics.delta == delta) & (metrics.incumbents == "votes>=1000")]
    for name in METHODS:
        s = sub[sub.method == name]
        ax.plot(s["T"], s["kendall"], marker="o", label=name)
    ax.set_title(f"Kendall tau of incumbent ranking, T vs T+{delta}mo")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.set_ylabel("tau_b")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "rq1_kendall.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 5))
sub = metrics[(metrics.delta == 1) & (metrics.incumbents == "votes>=1000")]
for name in METHODS:
    s = sub[sub.method == name]
    ax.plot(s["T"], s["mean_abs_dtheta_med"] * ELO_PER_NAT, marker="o", label=name)
ax.set_title("mean |Δθ| slope-matched, median-aligned, Elo-equiv pts (δ=1mo)")
ax.tick_params(axis="x", rotation=75, labelsize=7)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "rq1_magnitude.png", dpi=150)

# ---------------- validation track ----------------
SNAPSHOTS = {  # checkpoint label -> published snapshot date
    "2023-07-31": "20230802", "2023-08-31": "20230905", "2023-09-30": "20231002",
    "2023-10-31": "20231108", "2023-11-30": "20231206", "2023-12-31": "20240109",
    "2024-01-31": "20240202", "2024-02-29": "20240305", "2024-03-31": "20240403",
    "2024-04-30": "20240501", "2024-05-31": "20240602", "2024-06-30": "20240629",
    "2024-07-31": "20240731", "2024-08-12": "20240813",
}
PKL_DIR = ROOT / "data" / "raw" / "elo_pkls"
PKL_DIR.mkdir(parents=True, exist_ok=True)
BASE = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main"


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("plotly") or module.startswith("_plotly"):
            return _Stub
        return super().find_class(module, name)


def load_published(date: str):
    """Return (ratings Series, rating_system, last_updated_tstamp or None)."""
    p = PKL_DIR / f"elo_results_{date}.pkl"
    if not p.exists():
        urllib.request.urlretrieve(f"{BASE}/elo_results_{date}.pkl", p)
    with open(p, "rb") as f:
        obj = SafeUnpickler(f).load()
    node = obj
    if isinstance(node, dict) and "text" in node:
        node = node["text"]
    if isinstance(node, dict) and "full" in node:
        node = node["full"]
    ratings = node.get("elo_rating_final")
    if ratings is None and "elo_rating_online" in node:
        ratings = pd.Series(node["elo_rating_online"])
    if isinstance(ratings, dict):
        ratings = pd.Series(ratings)
    return (ratings.astype(float),
            node.get("rating_system", "unknown"),
            node.get("last_updated_tstamp"))


val_rows = []
for label, date in SNAPSHOTS.items():
    try:
        pub, system, ts = load_published(date)
    except Exception as e:  # noqa: BLE001 — record and continue, don't kill the run
        val_rows.append({"checkpoint": label, "snapshot": date, "error": repr(e)})
        print(f"[val {date}] EXTRACTION FAILED: {e!r}")
        continue
    cut = float(ts) if ts is not None else pd.Timestamp(date, tz="US/Pacific").timestamp()
    for variant, frame in (("dedup", dedup), ("no_dedup", battles)):
        sub = frame[frame["tstamp"] <= cut]
        ours = fit_gaplink(sub, METHODS["bt"][0], mode="half_tie", include_both_bad=True) * ELO_PER_NAT
        common = pub.index.intersection(ours.index)
        a, b = pub.loc[common], ours.loc[common]
        b = b + (a - b).median()
        val_rows.append({
            "checkpoint": label, "snapshot": date, "rating_system": system,
            "variant": variant, "n_common": len(common),
            "n_pub_only": len(pub.index.difference(ours.index)),
            "spearman": spearmanr(a, b).statistic,
            "mae_elo": float((a - b).abs().mean()),
            "max_abs_elo": float((a - b).abs().max()),
        })
        print(f"[val {date}] {system:8s} {variant:8s} n={len(common):3d} "
              f"spearman={val_rows[-1]['spearman']:.5f} mae={val_rows[-1]['mae_elo']:.2f}")

validation = pd.DataFrame(val_rows)
validation.to_csv(ROOT / "results" / "tables" / "rq1_validation_published.csv", index=False)
print("\nDONE")
