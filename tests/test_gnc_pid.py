import control as ct
import numpy as np
import pytest

from gnc.analysis import closed_loop, critical_delay, delay_sweep, loop_tf, margins
from gnc.pid import PID, GainSchedule, closed_loop_poles, place_pid, poles_from_specs, specs_of

PLANT = (1.5, 180.0, -180.0)   # a1, a0, b representative of the boost at moderate qbar


@pytest.mark.parametrize("extra", [[], [-60.0]])
def test_place_pid_reproduces_poles(extra):
    poles = poles_from_specs(0.7, 12.0, 4.0) + extra
    g = place_pid(*PLANT, poles)
    got = np.sort_complex(closed_loop_poles(*PLANT, **g))
    assert np.allclose(got, np.sort_complex(np.array(poles, complex)), atol=1e-6)
    # same poles out of the python-control interconnection (independent derivation)
    p_ct = np.sort_complex(ct.poles(closed_loop(*PLANT, g)["ss"]))
    assert np.allclose(p_ct, np.sort_complex(np.array(poles, complex)), atol=1e-6)


def test_specs():
    s = specs_of(0.7, 10.0)
    assert abs(s["overshoot_pct"] - 4.6) < 0.1 and abs(s["settling_2pct_s"] - 4 / 7) < 1e-9


def test_integral_rejects_step_disturbance():
    g = place_pid(*PLANT, poles_from_specs(0.7, 12.0, 4.0))
    assert abs(ct.dcgain(closed_loop(*PLANT, g)["T_dist"])) < 1e-9
    pd = dict(g, Ki=0.0)
    ss = ct.dcgain(closed_loop(*PLANT, pd)["T_dist"])
    assert np.isclose(ss, 1.0 / (PLANT[1] + PLANT[2] * pd["Kp"]))
    assert ct.dcgain(closed_loop(*PLANT, g)["T_ref"]) == pytest.approx(1.0)


def test_delay_margin_matches_pade_bisection_and_sweep():
    g = place_pid(*PLANT, poles_from_specs(0.7, 12.0, 4.0))
    m = margins(loop_tf(*PLANT, g, tau_act=0.02))
    dc = critical_delay(*PLANT, g, tau_act=0.02)
    assert m["delay_margin_s"] == pytest.approx(dc, rel=0.05)
    rows = delay_sweep(*PLANT, g, [0.0, 0.5 * dc, 1.5 * dc], tau_act=0.02)
    assert [r["stable"] for r in rows] == [True, True, False]


def test_gain_schedule_interpolates():
    s = GainSchedule.design([0, 1], [1, 2], [10, 100], [-100, -100], poles_from_specs(0.7, 10, 4))
    k0, k1 = s(0.0), s(1.0)
    assert s(0.5)[0] == pytest.approx(0.5 * (k0[0] + k1[0]))
    assert s(-5) == k0 and s(5) == k1


def test_pid_antiwindup_and_saturation():
    pid = PID(Kp=1.0, Ki=10.0, Kd=0.0, dt=0.01, u_max=0.1)
    for _ in range(500):
        u = pid.update(1.0, 0.0)
    assert u == pytest.approx(0.1)
    assert pid.i < 0.1      # integrator frozen while saturated (would be 5.0 without anti-windup)
    u = pid.update(-0.05, 0.0)
    assert u < 0.1          # recovers immediately when the error reverses
