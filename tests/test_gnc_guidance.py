import numpy as np

from gnc.guidance import LOSPathGuidance, kinematic_path_check, los_angles, steer_toward


def test_los_path_converges_to_straight_line():
    g = LOSPathGuidance([[0, 0, 0], [0, 0, 5000]], lookahead_m=100.0, k=1.5)
    out = kinematic_path_check(g, [40.0, -25.0, 0.0], [0.0, 0.0, 100.0], dt=0.01, t_max=20.0)
    ct = np.array([i["cross_track"] for i in out["info"]])
    assert ct[0] > 45 and ct[-1] < 0.5 and ct[1000:].max() < 1.0
    assert out["control_effort"] > 0


def test_waypoint_switching_and_3d_path():
    W = [[0, 0, 0], [0, 0, 300], [100, 100, 800], [100, 300, 1500]]
    g = LOSPathGuidance(W, lookahead_m=120.0, accept_radius=40.0, k=2.0)
    out = kinematic_path_check(g, [0, 0, 0], [0, 0, 200.0], dt=0.01, t_max=8.0, a_max=80.0)
    wp = [i["wp_index"] for i in out["info"]]
    assert wp[0] == 0 and wp[-1] == 2 and np.all(np.diff(wp) >= 0)
    late = [i["cross_track"] for i in out["info"][-100:]]
    assert max(late) < 15.0


def test_steer_toward_is_perpendicular_and_signed():
    a, eps = steer_toward(np.array([1.0, 0, 1.0]), np.array([0, 0, 50.0]), 2.0)
    assert abs(a @ [0, 0, 1]) < 1e-12 and a[0] > 0 and np.isclose(eps, np.pi / 4)
    assert np.isclose(np.linalg.norm(a), 2.0 * 50 * np.pi / 4)
    assert np.allclose(los_angles(np.array([0, 1.0, 1.0])), [np.pi / 2, np.pi / 4])
