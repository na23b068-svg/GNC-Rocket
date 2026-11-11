"""State observers: Luenberger (pole placement) and discrete Kalman filter, plus the linear models they run on.

Both observers share one interface so the flight computer can swap them:
    obs.predict(u)          x <- A x + B u
    obs.update(name, y)     correct with measurement `name` (C, and L or R, registered at construction)
    obs.x                   current estimate
Updates use the "current-estimator" form x+ = x- + L (y - C x-), so the a-priori error obeys
e_{k+1} = (I - L C) A e_k at the measurement rate; Luenberger gains place the eigenvalues of (I - L C) A.
"""
import numpy as np
import scipy.linalg as sla
from scipy.signal import place_poles


# ------------------------------------------------------------------ design functions

def discretize(Ac, Bc, dt, Qc=None):
    """Zero-order-hold discretisation; with Qc also the exact discrete process covariance (Van Loan)."""
    n, m = Ac.shape[0], Bc.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = Ac, Bc
    E = sla.expm(M * dt)
    Ad, Bd = E[:n, :n], E[:n, n:]
    if Qc is None:
        return Ad, Bd
    V = np.zeros((2 * n, 2 * n))
    V[:n, :n], V[:n, n:], V[n:, n:] = -Ac, Qc, Ac.T
    F = sla.expm(V * dt)
    Qd = F[n:, n:].T @ F[:n, n:]
    return Ad, Bd, 0.5 * (Qd + Qd.T)


def luenberger_gain(A, C, poles):
    """L such that eig((I - L C) A) = poles (discrete, current-estimator form).
    For a continuous-time predictor pass A=Ac and use eig(Ac - L C): call place directly instead."""
    return place_poles(A.T, (C @ A).T, poles).gain_matrix.T


def kalman_gain_dare(A, C, Q, R):
    """Steady-state Kalman gain (current-estimator form) and a-priori covariance from the DARE."""
    P = sla.solve_discrete_are(A.T, C.T, Q, R)
    L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    return L, P


def nees(err, P):
    return float(err @ np.linalg.solve(P, err))


# ------------------------------------------------------------------ observers

class Luenberger:
    def __init__(self, A, B, C, L, x0):
        """C, L: dicts measurement-name -> matrix."""
        self.A, self.B, self.C, self.L = A, B, C, L
        self.x = np.array(x0, float)

    def predict(self, u=None):
        self.x = self.A @ self.x + (self.B @ u if u is not None else 0.0)

    def update(self, name, y):
        nu = np.atleast_1d(y) - self.C[name] @ self.x
        self.x = self.x + self.L[name] @ nu
        return nu


class KalmanFilter:
    def __init__(self, A, B, C, Q, R, x0, P0):
        """C, R: dicts measurement-name -> matrix. Discrete A, B, Q."""
        self.A, self.B, self.C, self.Q, self.R = A, B, C, Q, R
        self.x, self.P = np.array(x0, float), np.array(P0, float)
        self.nis = {}

    def predict(self, u=None):
        self.x = self.A @ self.x + (self.B @ u if u is not None else 0.0)
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, name, y):
        C, R = self.C[name], self.R[name]
        nu = np.atleast_1d(y) - C @ self.x
        S = C @ self.P @ C.T + R
        K = np.linalg.solve(S, C @ self.P).T
        self.x = self.x + K @ nu
        IKC = np.eye(len(self.x)) - K @ C
        self.P = IKC @ self.P @ IKC.T + K @ R @ K.T   # Joseph form
        self.nis[name] = float(nu @ np.linalg.solve(S, nu))
        return nu


# ------------------------------------------------------------------ models

def axis_nav_model():
    """Per-axis navigation model, x = [p, v, b_a], u = measured acceleration (ENU, gravity removed):
    p' = v, v' = u - b_a, b_a' = 0 (random walk)."""
    Ac = np.array([[0, 1, 0], [0, 0, -1], [0, 0, 0]], float)
    Bc = np.array([[0], [1], [0]], float)
    return Ac, Bc


def nav_model(dt, sigma_a=1.0, sigma_b=0.02):
    """3-axis navigation model, x = [p_enu(3), v_enu(3), b_a(3)] (= kron(per-axis, I3)).
    sigma_a: white acceleration-input noise density (m/s^2/sqrt(Hz)) covering IMU noise AND attitude-error
    leakage of thrust; sigma_b: accel-bias random walk. Returns Ad, Bd, Qd and the measurement matrices
    C = {'gps': [p; v] (6x9), 'gps_pos': p, 'gps_vel': v, 'baro': p_U (1x9)}."""
    Ac, Bc = axis_nav_model()
    Ad1, Bd1, Qd1 = discretize(Ac, Bc, dt, np.diag([0.0, sigma_a**2, sigma_b**2]))
    I3 = np.eye(3)
    Ad, Bd, Qd = np.kron(Ad1, I3), np.kron(Bd1, I3), np.kron(Qd1, I3)
    C = {"gps": np.kron(np.array([[1.0, 0, 0], [0, 1.0, 0]]), I3),
         "gps_pos": np.kron(np.array([[1.0, 0, 0]]), I3),
         "gps_vel": np.kron(np.array([[0, 1.0, 0]]), I3),
         "baro": np.kron(np.array([[1.0, 0, 0]]), I3[2:3])}
    return Ad, Bd, Qd, C


def nav_luenberger_gains(dt, n_gps, n_baro, gps_poles=(0.5, 0.55, 0.9), baro_poles=(0.9, 0.92, 0.995)):
    """Luenberger gains for nav_model by per-axis pole placement on the discrete error map at each
    sensor's own update interval (n_gps / n_baro predict steps of dt). Pole magnitudes are per update.
    GPS measures [p, v] so each axis has 2 outputs -> placement of 3 poles; baro measures p_U only."""
    Ac, Bc = axis_nav_model()
    A_g = discretize(Ac, Bc, dt * n_gps)[0]
    A_b = discretize(Ac, Bc, dt * n_baro)[0]
    Lg1 = luenberger_gain(A_g, np.array([[1.0, 0, 0], [0, 1.0, 0]]), list(gps_poles))
    Lb1 = luenberger_gain(A_b, np.array([[1.0, 0, 0]]), list(baro_poles))
    I3 = np.eye(3)
    return {"gps": np.kron(Lg1, I3), "baro": np.kron(Lb1, I3[:, 2:3])}


def attitude_plane_model(dt):
    """Single-plane attitude estimator model, x = [theta, b_gyro], u = gyro rate:
    theta' = omega_meas - b, b' = 0; measured by an attitude reference (e.g. magnetometer tilt) y = theta.
    With the gyro as input there is no theta information without the reference (unobservable)."""
    Ac = np.array([[0, -1.0], [0, 0]])
    Bc = np.array([[1.0], [0]])
    Ad, Bd = discretize(Ac, Bc, dt)
    return Ad, Bd, np.array([[1.0, 0]])
