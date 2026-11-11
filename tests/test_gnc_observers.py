import numpy as np
import pytest

from gnc.observers import (KalmanFilter, Luenberger, attitude_plane_model, axis_nav_model, discretize,
                           kalman_gain_dare, luenberger_gain, nav_luenberger_gains, nav_model, nees)

DT = 0.01


def _cv_system():
    """Constant-velocity + bias axis model with white acceleration noise, position measured."""
    Ac, Bc = axis_nav_model()
    return discretize(Ac, Bc, DT, np.diag([0.0, 0.5**2, 0.01**2])), np.array([[1.0, 0, 0]])


def test_luenberger_error_converges_at_designed_rate():
    (A, B, _), C = _cv_system()
    poles = [0.9, 0.92, 0.95]
    L = luenberger_gain(A, C, poles)
    assert np.allclose(np.sort(np.linalg.eigvals((np.eye(3) - L @ C) @ A).real), poles, atol=1e-8)
    obs = Luenberger(A, B, {"p": C}, {"p": L}, x0=np.zeros(3))
    x = np.array([5.0, -2.0, 0.3])
    errs = []
    for _ in range(300):
        x = A @ x
        obs.predict(np.zeros(1))
        obs.update("p", C @ x)
        errs.append(np.linalg.norm(x - obs.x))
    # noise-free: asymptotic decay rate = slowest designed pole (0.95 per step)
    rate = (errs[299] / errs[199]) ** (1 / 100)
    assert rate == pytest.approx(0.95, abs=0.01)


def test_kalman_consistent_and_better_than_measurement():
    rng = np.random.default_rng(0)
    (A, B, Q), C = _cv_system()
    R = np.array([[2.0**2]])
    kf = KalmanFilter(A, B, {"p": C}, Q, {"p": R}, np.zeros(3), np.diag([4.0, 1.0, 0.1]))
    x = np.array([0.0, 3.0, 0.1])
    Lq = np.linalg.cholesky(Q + 1e-15 * np.eye(3))
    ne, err_kf, err_meas = [], [], []
    for k in range(4000):
        x = A @ x + Lq @ rng.standard_normal(3)
        kf.predict(np.zeros(1))
        y = C @ x + 2.0 * rng.standard_normal(1)
        kf.update("p", y)
        if k > 200:
            ne.append(nees(x - kf.x, kf.P))
            err_kf.append(x[0] - kf.x[0])
            err_meas.append(y[0] - x[0])
    assert 2.6 < np.mean(ne) < 3.4                     # chi^2(3) mean = 3
    assert np.std(err_kf) < 0.5 * np.std(err_meas)


def test_dare_gain_equals_converged_kf_gain():
    (A, B, Q), C = _cv_system()
    R = np.array([[4.0]])
    L, P = kalman_gain_dare(A, C, Q, R)
    kf = KalmanFilter(A, B, {"p": C}, Q, {"p": R}, np.zeros(3), np.eye(3))
    for _ in range(20000):
        kf.predict()
        Pm = kf.P.copy()
        kf.update("p", np.zeros(1))
    assert np.allclose(Pm, P, rtol=1e-3)


def test_nav_multirate_luenberger_stable():
    """Periodic error map over one GPS period (10 predicts, baro every 2, GPS at the end) is stable."""
    Ac, Bc = axis_nav_model()
    A = discretize(Ac, Bc, DT)[0]
    Lg = nav_luenberger_gains(DT, 10, 2)
    _, _, _, C = nav_model(DT)
    A9 = np.kron(A, np.eye(3))
    M = np.eye(9)
    for k in range(1, 11):
        M = A9 @ M
        if k % 2 == 0:
            M = (np.eye(9) - Lg["baro"] @ C["baro"]) @ M
    M = (np.eye(9) - Lg["gps"] @ C["gps"]) @ M
    assert np.max(np.abs(np.linalg.eigvals(M))) < 0.99


def test_attitude_plane_model_unobservable_without_reference():
    from gnc.analysis import rank_test
    A, B, C = attitude_plane_model(DT)
    assert rank_test(A, C, "obsv")["full"]
    assert rank_test(A, np.zeros((1, 2)), "obsv")["rank"] == 0


def test_discretize_van_loan_matches_analytic():
    Ac, Bc = np.array([[0, 1.0], [0, 0]]), np.array([[0], [1.0]])
    _, _, Qd = discretize(Ac, Bc, 0.1, np.diag([0, 2.0]))
    T = 0.1
    assert np.allclose(Qd, 2.0 * np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]]))
    assert pytest.approx(Qd[0, 1]) == Qd[1, 0]
