"""SYNTHETIC validation of the profiled-unit MLE for the lattice tie-band.

*** Simulated data from known ground truth; not a research result. ***

World: 30 models, theta*~N(0,1), 200k votes, outcomes drawn from the lattice
native trinomial with TRUE unit = 0.5 (tie share ~13%). The profile MLE over
the unit must recover it.

Pass criteria (pre-stated):
  - fitted unit within 10% of 0.5
  - theta recovery at fitted unit: Spearman >= 0.99, RMSE_raw <= 0.10
  - profile NLL curve is convex-ish around the optimum (min is interior)

Writes results/tables/unit_profile_validation.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import spearmanr  # noqa: E402

from fit import profile_lattice_unit  # noqa: E402
from lattice_link import LatticeLink  # noqa: E402

SEED = 20260710
rng = np.random.default_rng(SEED)

N_MODELS, N_VOTES = 30, 200_000
TRUE_UNIT = 0.5

theta_true = rng.normal(0.0, 1.0, N_MODELS)
theta_true -= theta_true.mean()
models = [f"m{k:02d}" for k in range(N_MODELS)]
popularity = rng.dirichlet(np.full(N_MODELS, 0.5))

gen_link = LatticeLink(unit=TRUE_UNIT)
a = rng.choice(N_MODELS, size=N_VOTES, p=popularity)
b = rng.choice(N_MODELS, size=N_VOTES, p=popularity)
ok = a != b
a, b = a[ok], b[ok]
g = theta_true[a] - theta_true[b]
W, D = gen_link.p_win(g), gen_link.p_tie(g)
u = rng.uniform(size=len(g))
outcome = np.where(u < W, "model_a", np.where(u < W + D, "tie", "model_b"))
bat = pd.DataFrame({"model_a": np.array(models)[a],
                    "model_b": np.array(models)[b], "winner": outcome})
print(f"synthetic world: {len(bat):,} votes, true unit {TRUE_UNIT}, "
      f"tie share {np.mean(outcome == 'tie'):.4f}")

units = np.geomspace(0.1, 1.4, 12)
best_u, theta_hat, prof = profile_lattice_unit(bat, units, make_link=lambda u: LatticeLink(unit=u))
print(prof.to_string(index=False))
print(f"fitted unit: {best_u:.4f} (true {TRUE_UNIT})")

v = theta_hat.reindex(models).to_numpy()
v = v - v.mean()
sp = spearmanr(theta_true, v).statistic
rmse = float(np.sqrt(np.mean((v - theta_true) ** 2)))
print(f"theta recovery at fitted unit: spearman={sp:.5f} rmse={rmse:.5f}")

k = int(prof["nll"].idxmin())
assert 0 < k < len(prof) - 1, "profile minimum at grid edge"
assert abs(best_u - TRUE_UNIT) / TRUE_UNIT <= 0.10, f"unit recovery FAILED: {best_u}"
assert sp >= 0.99 and rmse <= 0.10, "theta recovery FAILED"

prof["fitted_unit"] = best_u
prof["true_unit"] = TRUE_UNIT
prof.to_csv(ROOT / "results" / "tables" / "unit_profile_validation.csv", index=False)
print("UNIT PROFILE VALIDATION PASSED")
