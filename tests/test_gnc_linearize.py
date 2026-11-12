import numpy as np
import pytest

from gnc.analysis import rank_test
from gnc.linearize import analytic_plane, linearize_at, numeric_jacobian, operating_points


def test_numeric_jacobian_matches_analytic():
    f = lambda x, u: np.array([x[1], -4.0 * np.sin(x[0]) - 0.3 * x[1] + 2.0 * u[0] * np.cos(x[0])])  # noqa: E731
    x0, u0 = np.array([0.4, -0.2]), np.array([0.1])
    A, B = numeric_jacobian(f, x0, u0)
    A_true = np.array([[0, 1], [-4 * np.cos(0.4) - 0.2 * np.sin(0.4), -0.3]])
    assert np.allclose(A, A_true, atol=1e-7) and np.allclose(B, [[0], [2 * np.cos(0.4)]], atol=1e-7)


@pytest.fixture(scope="module")
def setup():
    from gnc.dynamics import make_env
    from gnc.frames import launch_attitude, nose_direction
    from gnc.rocket import Rocket
    r = Rocket.load()
    q = launch_attitude(np.radians(6), np.radians(90))
    n = nose_direction(q)
    return r, make_env(r), q, n


@pytest.mark.parametrize("t,V", [(0.5, 35.0), (2.0, 180.0), (3.5, 280.0)])
def test_numeric_plane_model_matches_analytic(setup, t, V):
    r, env, q, n = setup
    x0 = np.concatenate([n * 300, n * V, q, [0, 0, 0]])
    lin = linearize_at(r, t, x0, env)
    for which in ("pitch", "yaw"):
        Aa, Ba, _ = analytic_plane(r, t, x0, env, which)
        A, B = lin[which]["A"], lin[which]["B"]
        assert np.allclose(A, Aa, rtol=1e-5, atol=1e-5 * np.abs(Aa).max())
        assert np.allclose(B, Ba, rtol=1e-6)
        assert lin[which]["coupling"] < 1e-3        # pitch/yaw decoupling holds (no roll rate)
    assert np.allclose(lin["pitch"]["short_period"], lin["yaw"]["short_period"])


def test_operating_points_ctrb_obsv():
    """Rank conclusions at every gain-schedule point (block-normalised Kalman matrices)."""
    from gnc.flight_computer import design_schedule
    from gnc.rocket import Rocket
    r = Rocket.load()
    pts = operating_points(r, design_schedule(r)[1]["t"])
    assert pts[-1]["flight"]["qbar"] > 50 * pts[0]["flight"]["qbar"]
    for p in pts:
        A, B = p["pitch"]["A"], p["pitch"]["B"]
        assert rank_test(A, B, "ctrb")["full"]
        for C in ([[0, 0, 0, 1.0]], [[0, 0, 1.0, 0]]):            # gyro only, attitude only
            o = rank_test(A, np.array(C), "obsv")                  # lateral position never observable
            assert o["rank"] == 3 and abs(o["deficient_directions"][0][0]) > 0.99
        assert rank_test(A, np.array([[1.0, 0, 0, 0], [0, 0, 0, 1.0]]), "obsv")["full"]  # + GPS position
        assert rank_test(A, np.array([[1.0, 0, 0, 0]]), "obsv")["full"]
        a1, a0, b = p["pitch"]["short_period"]
        assert a0 > 0 and b < 0          # fin-stabilised airframe; positive delta_p pitches nose-down (-q)
