"""Sign-convention regression tests (user decision #2, 2026-07-09).

The vendored thurstone package is a race-TIME model: performances are drawn
around ability and the field MINIMUM wins, so LOWER raw ability = better.
These tests pin that convention so a silent upstream change (or our own
confusion) can't invert every ranking downstream.

The sign flip to higher-is-better happens exactly once, in
src/lattice_link.LatticeLink; the wrapper tests below pin that side,
including a hand-checked real-data subset (gpt-4-1106-preview vs vicuna-13b:
197 / 42 / 79 a-wins/b-wins/ties, counted by scripts/ ad hoc on
clean_battle_20240814 and re-verified inside the test).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from thurstone import STD_L, STD_UNIT, AbilityCalibrator, Density, UniformLattice

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink  # noqa: E402


@pytest.fixture(scope="module")
def calibrator():
    lattice = UniformLattice(L=STD_L, unit=STD_UNIT)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)
    return AbilityCalibrator(base)


def test_raw_convention_lower_ability_wins_more(calibrator):
    """Raw package: the entrant with the LOWER ability value must win more."""
    p_low, p_high = calibrator.state_prices_from_ability([-0.5, +0.5])
    assert p_low > 0.6 > 0.4 > p_high, (
        f"expected lower-ability entrant to win (race-time convention), got {p_low=} {p_high=}"
    )


def test_raw_convention_symmetric(calibrator):
    """Equal abilities -> equal win probabilities (sanity)."""
    p1, p2 = calibrator.state_prices_from_ability([0.3, 0.3])
    assert abs(p1 - p2) < 1e-9
    assert abs(p1 + p2 - 1.0) < 1e-6


def test_raw_convention_inverse_consistent(calibrator):
    """Round trip: prices favoring entrant 0 must invert to entrant 0 having
    the LOWER ability value."""
    ab = calibrator.solve_from_prices([0.75, 0.25])
    assert ab[0] < ab[1], f"inverse must respect race-time convention, got {ab}"


def test_forward_inverse_roundtrip(calibrator):
    """Forward then inverse recovers the ability gap (within lattice tolerance)."""
    true_gap = 1.0
    prices = calibrator.state_prices_from_ability([-true_gap / 2, +true_gap / 2])
    ab = calibrator.solve_from_prices(list(prices))
    recovered_gap = ab[1] - ab[0]
    assert recovered_gap == pytest.approx(true_gap, abs=0.15), (
        f"round-trip gap {recovered_gap:.3f} vs true {true_gap}"
    )
    # np.interp-based inversion should be deterministic
    ab2 = calibrator.solve_from_prices(list(prices))
    assert np.allclose(ab, ab2)


# ---- higher-is-better wrapper (src/lattice_link + src/fit) ----


@pytest.fixture(scope="module")
def link():
    return LatticeLink()


def test_wrapper_higher_ability_wins_more(link):
    """LatticeLink is higher-is-better: positive gap -> win prob > 1/2."""
    assert link.p_win(1.0) > 0.6 > 0.4 > link.p_win(-1.0)
    gs = np.linspace(-4, 4, 81)
    assert np.all(np.diff(link.p_win(gs)) > 0), "p_win must increase in the gap"


def test_wrapper_fit_recovers_direction_synthetic(link):
    """Synthetic 2-model fit: the model that wins more must come out higher."""
    rng = np.random.default_rng(0)
    n = 2000
    winner = np.where(rng.uniform(size=n) < 0.75, "model_a", "model_b")
    bat = pd.DataFrame({"model_a": "strong", "model_b": "weak", "winner": winner})
    theta = fit_gaplink(bat, link, mode="half_tie")
    assert theta["strong"] > theta["weak"]


def test_wrapper_fit_real_subset_hand_checked(link):
    """Decision #2: hand-checked real-data subset through the full wrapper.

    All 318 gpt-4-1106-preview vs vicuna-13b battles in
    clean_battle_20240814: gpt-4 wins 197, vicuna wins 42 (79 ties).
    The counts are re-verified here, then the fitted abilities must put
    gpt-4-1106-preview on top. Skipped if the parquet isn't present.
    """
    pq = ROOT / "data" / "processed" / "clean_battle_20240814.parquet"
    if not pq.exists():
        pytest.skip("clean_battle parquet not available")
    df = pd.read_parquet(pq, columns=["model_a", "model_b", "winner"])
    a, b = "gpt-4-1106-preview", "vicuna-13b"
    m = df[((df.model_a == a) & (df.model_b == b)) | ((df.model_a == b) & (df.model_b == a))]
    a_wins = int(((m.model_a == a) & (m.winner == "model_a")).sum()
                 + ((m.model_b == a) & (m.winner == "model_b")).sum())
    b_wins = int(((m.model_a == b) & (m.winner == "model_a")).sum()
                 + ((m.model_b == b) & (m.winner == "model_b")).sum())
    assert (len(m), a_wins, b_wins) == (318, 197, 42), "hand-checked counts drifted"
    for mode in ("half_tie", "native"):
        theta = fit_gaplink(m, link, mode=mode)
        assert theta[a] > theta[b], f"sign inversion in {mode} mode"
