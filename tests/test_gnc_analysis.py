import numpy as np
import scipy.linalg as sla

from gnc.analysis import gramian, gramian_summary, poles_zeros, rank_test
from gnc.observers import axis_nav_model

DI_A = np.array([[0.0, 1.0], [0.0, 0.0]])   # double integrator


def test_rank_tests_double_integrator():
    assert rank_test(DI_A, np.array([[0.0], [1.0]]), "ctrb")["full"]
    assert rank_test(DI_A, np.array([[1.0], [0.0]]), "ctrb")["rank"] == 1
    assert rank_test(DI_A, np.array([[1.0, 0.0]]), "obsv")["full"]
    r = rank_test(DI_A, np.array([[0.0, 1.0]]), "obsv")          # velocity only: position unobservable
    assert r["rank"] == 1 and np.allclose(np.abs(r["deficient_directions"][0]), [1, 0])


def test_nav_model_observability_by_sensor_set():
    A, _ = axis_nav_model()   # [p, v, b_a]
    assert rank_test(A, np.array([[1.0, 0, 0]]), "obsv")["rank"] == 3            # GPS position / baro
    assert rank_test(A, np.array([[0, 1.0, 0]]), "obsv")["rank"] == 2            # GPS velocity only
    assert rank_test(A, np.zeros((1, 3)), "obsv")["rank"] == 0                   # accelerometer only


def test_gramian_lyapunov_and_finite():
    A = np.array([[-1.0, 0.0], [1.0, -2.0]])
    B = np.array([[1.0], [0.0]])
    W = gramian(A, B, "c")
    assert np.allclose(A @ W + W @ A.T + B @ B.T, 0, atol=1e-10)
    Wt = gramian(A, B, "c", T=20.0, n_steps=4000)
    assert np.allclose(W, Wt, atol=1e-3)
    assert gramian_summary(W)["eig_min"] > 0
    Wo = gramian(DI_A, np.array([[1.0, 0.0]]), "o", T=1.0)
    assert np.allclose(Wo, [[1, 0.5], [0.5, 1 / 3]], atol=1e-4)


def test_poles_zeros():
    import control as ct
    pz = poles_zeros(ct.ss(ct.tf([1, 2], [1, 3, 2])))
    assert np.allclose(sorted(p[0] for p in pz["poles"]), [-2, -1])
    assert np.allclose(pz["zeros"][0][0], -2)
