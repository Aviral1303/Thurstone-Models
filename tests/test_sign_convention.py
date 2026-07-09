"""Sign-convention regression tests (user decision #2, 2026-07-09).

The vendored thurstone package is a race-TIME model: performances are drawn
around ability and the field MINIMUM wins, so LOWER raw ability = better.
These tests pin that convention so a silent upstream change (or our own
confusion) can't invert every ranking downstream.

Phase 3 will add tests asserting the sign-flip wrapper in src/ maps
higher-is-better abilities onto correct win probabilities, using a
hand-checked real-data subset (gpt-4-1106-preview vs vicuna-13b).
"""

import numpy as np
import pytest
from thurstone import STD_L, STD_UNIT, AbilityCalibrator, Density, UniformLattice


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
