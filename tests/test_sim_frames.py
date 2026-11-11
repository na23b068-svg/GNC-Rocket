import numpy as np

from gnc import frames as F


def _rand_quats(n=50, seed=1):
    return F.quat_normalize(np.random.default_rng(seed).standard_normal((n, 4)))


def test_dcm_orthonormal_and_roundtrip():
    for q in _rand_quats():
        R = F.quat_to_dcm(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        q2 = F.dcm_to_quat(R)
        assert np.allclose(q2, q * np.sign(q[0]), atol=1e-10)


def test_quat_mul_matches_dcm_product_and_conj():
    a, b = _rand_quats(2, seed=3)
    assert np.allclose(F.quat_to_dcm(F.quat_mul(a, b)), F.quat_to_dcm(a) @ F.quat_to_dcm(b))
    assert np.allclose(F.quat_mul(a, F.quat_conj(a)), [1, 0, 0, 0])
    v = np.array([0.3, -1.2, 2.0])
    assert np.allclose(F.quat_rotate_inv(a, F.quat_rotate(a, v)), v)


def test_axis_angle_and_basic_rotations():
    q = F.quat_from_axis_angle([0, 1, 0], 0.3)
    assert np.allclose(F.quat_to_dcm(q), F.Ry(0.3))
    assert np.allclose(F.quat_to_dcm(F.quat_from_axis_angle([0, 0, 1], 0.2)), F.Rz(0.2))
    assert np.allclose(F.quat_to_dcm(F.quat_from_axis_angle([1, 0, 0], -0.7)), F.Rx(-0.7))
    # contract thrust direction formula
    dp, dy = 0.05, -0.03
    assert np.allclose(F.Rz(dy) @ F.Ry(dp) @ [1, 0, 0],
                       [np.cos(dy) * np.cos(dp), np.sin(dy) * np.cos(dp), -np.sin(dp)])


def test_euler_roundtrip():
    q = F.euler_zyx_to_quat(0.4, -0.2, 1.1)
    assert np.allclose(F.quat_to_euler_zyx(q), (0.4, -0.2, 1.1))


def test_quat_derivative_integrates_to_rotation():
    q, w, dt = np.array([1.0, 0, 0, 0]), np.array([0.0, 0.5, 0.0]), 1e-4
    for _ in range(10000):  # 1 s
        q = F.quat_normalize(q + dt * F.quat_derivative(q, w))
    assert np.allclose(F.quat_to_dcm(q), F.Ry(0.5), atol=1e-4)


def test_launch_attitude():
    q = F.launch_attitude(np.radians(6), np.radians(90))
    n = F.nose_direction(q)
    assert np.allclose(n, [np.sin(np.radians(6)), 0, np.cos(np.radians(6))])
    assert np.isclose(F.tilt_from_vertical(q), np.radians(6))
    el, az = F.nose_elevation_azimuth(q)
    assert np.isclose(el, np.radians(84)) and np.isclose(az, np.radians(90))
    assert np.allclose(F.quat_to_dcm(q)[:, 1], [0, 1, 0])  # y_b = North for an East launch
    assert np.allclose(F.nose_direction(F.launch_attitude(0.0, 0.0)), [0, 0, 1])
