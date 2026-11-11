import numpy as np
import pytest

from gnc.atmosphere import G0
from gnc.dynamics import Env, aero_state, derivatives, specific_force_body
from gnc.frames import launch_attitude, nose_direction, quat_from_axis_angle, quat_normalize, quat_to_dcm
from gnc.rocket import DATA, Rocket
from gnc.sim import SimConfig, simulate


@pytest.fixture(scope="module")
def rocket():
    return Rocket.load(DATA / "rocket_params.placeholder.json")


def state(p=(0, 0, 100), v=(0, 0, 0), q=(1, 0, 0, 0), w=(0, 0, 0)):
    return np.concatenate([p, v, q, w]).astype(float)


VERT = tuple(launch_attitude(0.0, 0.0))  # nose up


def test_free_fall(rocket):
    xd = derivatives(10.0, state(q=VERT), np.zeros(2), rocket, Env())
    assert np.allclose(xd[3:6], [0, 0, -G0]) and np.allclose(xd[6:], 0)
    assert np.allclose(specific_force_body(state(q=VERT), xd[3:6]), 0)


def test_coast_drag_opposes_velocity(rocket):
    xd = derivatives(10.0, state(q=VERT, v=(0, 0, 150)), np.zeros(2), rocket, Env())
    a = aero_state(10.0, state(q=VERT, v=(0, 0, 150)), rocket, Env())
    drag = a["q_dyn"] * rocket.S * a["cd"] / rocket.m_dry
    assert np.allclose(xd[3:6], [0, 0, -G0 - drag]) and np.allclose(xd[10:], 0)
    # descending tail-first: axial force reverses sign (always opposes motion)
    xd2 = derivatives(10.0, state(q=VERT, v=(0, 0, -150)), np.zeros(2), rocket, Env())
    assert np.isclose(xd2[5], -G0 + drag)


def test_powered_vertical_flight_stays_vertical(rocket):
    xd = derivatives(1.0, state(q=VERT, v=(0, 0, 50)), np.zeros(2), rocket, Env())
    assert np.allclose(xd[3:5], 0) and xd[5] > 0 and np.allclose(xd[10:], 0)
    res = simulate(rocket, SimConfig(rail_angle_deg=0.0, sensor_noise_scale=0.0))
    X = res["x"]
    assert np.abs(X[:, 0:2]).max() < 1e-6 and np.abs(X[:, 10:13]).max() < 1e-9
    assert np.degrees(np.arccos(np.clip(quat_to_dcm(X[-1, 6:10])[2, 0], -1, 1))) < 1e-6


def test_tvc_sign(rocket):
    """Positive delta_p -> F_b z negative at the aft pivot -> negative pitch moment -> nose swings to +z_b."""
    x0 = state(q=VERT, v=(0, 0, 0.0))
    u = np.array([np.radians(2), 0.0])
    xd = derivatives(1.0, x0, u, rocket, Env())
    assert xd[11] < 0 and abs(xd[10]) < 1e-12 and abs(xd[12]) < 1e-12
    zb0 = quat_to_dcm(x0[6:10])[:, 2]
    x, dt = x0.copy(), 1e-3
    for k in range(200):
        x = x + dt * derivatives(1.0 + k * dt, x, u, rocket, Env())
        x[6:10] = quat_normalize(x[6:10])
    assert nose_direction(x[6:10]) @ zb0 > 0.0
    # positive delta_y -> thrust +y_b at the tail -> nose swings to -y_b (negative moment about z_b)
    xd = derivatives(1.0, x0, np.array([0.0, np.radians(2)]), rocket, Env())
    assert xd[12] < 0
    yb0 = quat_to_dcm(x0[6:10])[:, 1]
    x = x0.copy()
    for k in range(200):
        x = x + dt * derivatives(1.0 + k * dt, x, np.array([0.0, np.radians(2)]), rocket, Env())
    assert nose_direction(quat_normalize(x[6:10])) @ yb0 < 0.0
    # magnitude: M_z = -l_g T sin(dy) cos(dp)
    T, m, xcg, Ixx, Iyy, _ = rocket._thrust_mass(1.0)
    assert np.isclose(xd[12], -(rocket.x_gimbal - xcg) * T * np.sin(np.radians(2)) / Iyy)


def test_aero_restoring_and_damping(rocket):
    # nose tilted towards +z_b relative to velocity (w_r < 0) -> positive pitch moment (restoring)
    q = quat_from_axis_angle([0, 1, 0], -np.radians(3))   # nose x_b rotated toward +z
    x = state(q=q, v=(150, 0, 0))
    a = aero_state(10.0, x, rocket, Env())
    assert a["v_rel_b"][2] < 0 and np.isclose(a["alpha_total"], np.radians(3))
    assert derivatives(10.0, x, np.zeros(2), rocket, Env())[11] > 0
    # pitch-rate damping
    xd = derivatives(10.0, state(v=(150, 0, 0), w=(0.0, 0.5, -0.5)), np.zeros(2), rocket, Env())
    assert xd[11] < 0 and xd[12] > 0
    xd = derivatives(10.0, state(v=(150, 0, 0), w=(3.0, 0, 0)), np.zeros(2), rocket, Env())
    assert xd[10] < 0
    # wind from the side creates AoA even when the body is aligned with ground velocity
    w = Env(wind=lambda t, z: (0.0, 10.0, 0.0))
    assert aero_state(10.0, state(v=(150, 0, 0)), rocket, w)["alpha_total"] > np.radians(3)
