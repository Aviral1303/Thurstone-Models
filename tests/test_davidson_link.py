"""Sanity tests for DavidsonLink (RQ4 comparator)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from davidson_link import DavidsonLink  # noqa: E402

gs = np.linspace(-6, 6, 241)


@pytest.fixture(scope="module")
def link():
    return DavidsonLink(nu=0.5)


def test_trinomial_sums_to_one(link):
    s = link.p_win(gs) + link.p_tie(gs) + link.p_loss(gs)
    assert np.allclose(s, 1.0, atol=1e-12)


def test_decisive_link_is_logistic_for_any_nu():
    for nu in (0.1, 0.5, 1.5):
        lk = DavidsonLink(nu=nu)
        w, lo = lk.p_win(gs), lk.p_loss(gs)
        assert np.allclose(w / (w + lo), 1 / (1 + np.exp(-gs)), atol=1e-12)


def test_symmetry_and_monotonicity(link):
    assert np.allclose(link.p_win(gs), link.p_loss(-gs), atol=1e-12)
    assert np.allclose(link.p_tie(gs), link.p_tie(-gs), atol=1e-12)
    assert np.all(np.diff(link.p_win(gs)) > 0)


def test_tie_share_at_zero(link):
    # P(tie|0) = nu/(2+nu)
    assert link.p_tie(0.0) == pytest.approx(0.5 / 2.5)


def test_analytic_gradients_match_numeric(link):
    eps = 1e-6
    for fn in (link.log_p_win, link.log_p_tie):
        val, grad = fn(gs)
        num = (fn(gs + eps)[0] - fn(gs - eps)[0]) / (2 * eps)
        assert np.allclose(grad, num, atol=1e-6)
