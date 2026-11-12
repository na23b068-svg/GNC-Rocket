"""Independent checker tests for the LQR attitude controller (gnc/lqr.py), the block-normalised rank test and
linearize.aoa_form (docs/gnc_design.md sec. 3, 4, 10). Gains and margins are rebuilt here from scratch
(scipy ARE, exact frequency response, no Pade)."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg as sla

from gnc.analysis import rank_test
from gnc.flight_computer import FlightComputer, design_schedule
from gnc.linearize import aoa_form, operating_points
from gnc.rocket import ROOT, Rocket
from gnc.sim import SimConfig, simulate

G0 = 9.80665
TH, QM, DM = np.radians(4.0), np.radians(40.0), np.radians(5.0)
Q = np.diag([1 / TH**2, 1 / QM**2, 1 / (TH * 0.3) ** 2])
R = np.array([[1 / DM**2]])


@pytest.fixture(scope="module")
def rocket():
    return Rocket.load()


@pytest.fixture(scope="module")
def tab(rocket):
    return design_schedule(rocket)[1]


def _lqr(a1, a0, b):
    """x = [theta, q, xi], xi' = theta; returns (Kp, Ki, Kd) = (K_theta, K_xi, K_q)."""
    A = np.array([[0, 1.0, 0], [-a0, -a1, 0], [1.0, 0, 0]])
    B = np.array([[0], [b], [0.0]])
    K = np.linalg.solve(R, B.T @ sla.solve_continuous_are(A, B, Q, R))[0]
    return K[0], K[2], K[1]


def _exact(a1, a0, b, Kp, Ki, Kd, tau):
    """Min phase margin and delay margin over all gain crossovers, exact e^{-s tau_d} (no Pade)."""
    w = np.logspace(-3, 4, 400001)
    s = 1j * w
    L = (Kp + Ki / s + Kd * s) * b / (s * s + a1 * s + a0) / (tau * s + 1)
    idx = np.where(np.diff(np.sign(np.abs(L) - 1)))[0]
    pm = np.array([(np.degrees(np.angle(L[j])) + 180) % 360 for j in idx])
    return pm.min(), min(np.radians(p) / w[j] for p, j in zip(pm, idx)), L


@pytest.mark.parametrize("i", [0, 4, 8])
def test_lqr_gains_from_independent_care(tab, i):
    from gnc.lqr import design_lqr_schedule
    g = design_lqr_schedule(tab)[1][i]
    Kp, Ki, Kd = _lqr(tab["a1"][i], tab["a0"][i], tab["b"][i])
    assert np.allclose([g["Kp"], g["Ki"], g["Kd"]], [Kp, Ki, Kd], rtol=1e-8)
    assert Ki == pytest.approx(-DM / (TH * 0.3), rel=1e-9)     # K_xi = -sqrt(q_xi / r) (b < 0)


@pytest.mark.parametrize("i", [0, 4, 8])
def test_lqr_no_lag_pm60_and_no_upward_gm(tab, i):
    a1, a0, b = tab["a1"][i], tab["a0"][i], tab["b"][i]
    pm, _, L = _exact(a1, a0, b, *_lqr(a1, a0, b), 0.0)
    assert pm >= 60.0
    pc = np.where(np.diff(np.sign(np.angle(L))) != 0)[0]          # -180 deg crossings
    assert all(abs(L[j]) > 1.0 for j in pc if np.cos(np.angle(L[j])) < 0)   # -180 only where |L| > 1: no upward limit


@pytest.mark.parametrize("i", [0, 2, 4])                            # t <= 2.2 s
def test_lag_margins_match_doc_early_boost(tab, i):
    a1, a0, b = tab["a1"][i], tab["a0"][i], tab["b"][i]
    pm_l, dm_l, _ = _exact(a1, a0, b, *_lqr(a1, a0, b), 0.02)
    pm_p, dm_p, _ = _exact(a1, a0, b, tab["Kp"][i], tab["Ki"][i], tab["Kd"][i], 0.02)
    assert 0.027 <= round(dm_l, 3) <= 0.030 and 43 <= round(pm_l) <= 51
    assert 0.055 <= round(dm_p, 3) <= 0.073 and 47 <= round(pm_p) <= 75


@pytest.mark.xfail(strict=True, reason="DEFECT: analysis.critical_delay returns inf for the LQR from 3.07 s "
                   "(Pade-5 loop at d_max=1 s happens to be stable; stability is not monotonic in delay). "
                   "Exact delay margin is 39-53 ms, not > 1 s as in doc sec. 10")
def test_late_burn_lqr_delay_margin_is_finite(tab):
    from gnc.lqr import compare
    rows = compare(tab)
    for i in (6, 7, 8):
        a1, a0, b = tab["a1"][i], tab["a0"][i], tab["b"][i]
        _, dm, _ = _exact(a1, a0, b, *_lqr(a1, a0, b), 0.02)
        assert 0.035 < dm < 0.06
        assert rows[i]["lqr"]["delay_margin_s"] == pytest.approx(dm, rel=0.02)


def test_late_burn_lqr_integral_pole_and_settling(tab):
    from gnc.lqr import compare
    a1, a0, b = tab["a1"][8], tab["a0"][8], tab["b"][8]
    Kp, Ki, Kd = _lqr(a1, a0, b)
    p = np.roots([1, a1 + b * Kd, a0 + b * Kp, b * Ki])
    assert abs(Kp) < 0.25 and p.real.max() == pytest.approx(-0.6, abs=0.02)
    assert compare(tab)[8]["lqr"]["settling_2pct_s"] == pytest.approx(6.3, abs=0.1)


def test_rank_test_keeps_scaled_deficient_pair_deficient():
    rng = np.random.default_rng(0)
    T = rng.standard_normal((3, 3))
    A = T @ np.diag([1e3, 1e-3, -2.0]) @ np.linalg.inv(T)
    B = 1e-6 * T @ np.array([[1.0], [1.0], [0.0]])                 # mode -2 not excited; blocks span 1e-6..1
    assert rank_test(A, B, "ctrb")["rank"] == 2
    assert rank_test(A.T, B.T, "obsv")["rank"] == 2


@pytest.mark.xfail(strict=True, reason="DEFECT (minor): block normalisation blows a roundoff-only block "
                   "(true A^2 B = 0) up to unit norm, so rank_test overstates the rank (3 instead of 2)")
def test_rank_test_roundoff_block_not_amplified():
    rng = np.random.default_rng(0)
    T = rng.standard_normal((3, 3))
    A = T @ np.array([[0, 1.0, 0], [0, 0, 0], [0, 0, -1.0]]) @ np.linalg.inv(T)
    B = T @ np.array([[0], [1.0], [0]])
    assert rank_test(A, B, "ctrb")["rank"] == 2


def test_rank_table_doc_sec4(rocket, tab):
    pts = operating_points(rocket, tab["t"])
    sets = {(0, 0, 0, 1): [3] * 9, (0, 0, 1, 0): [3] * 9, (0, 1, 0, 0): [3, 4] + [3] * 7, (1, 0, 0, 0): [4] * 9}
    for c, want in sets.items():
        assert [rank_test(p["pitch"]["A"], np.array([c], float), "obsv")["rank"] for p in pts] == want
    assert all(rank_test(p["pitch"]["A"], p["pitch"]["B"], "ctrb")["rank"] == 4 for p in pts)


def test_aoa_form_is_time_varying_transform_and_residual_near_g_over_v(rocket, tab):
    pts = operating_points(rocket, [tab["t"][i] for i in (0, 4, 8)])
    for p, tol in zip(pts, (0.4, 0.05, 0.05)):
        A, B, f = p["pitch"]["A"], p["pitch"]["B"], p["flight"]
        V, Vd = f["V"], f["Vdot"]
        T = np.diag([1, 1 / V, 1, 1])
        Am = T @ A @ np.linalg.inv(T) + np.diag([0, -Vd / V**2, 0, 0]) @ np.linalg.inv(T)   # + Tdot T^-1
        Aw, Bw = aoa_form(A, B, V, Vd)
        assert np.allclose(Aw, Am) and np.allclose(Bw, T @ B)
        ev = np.linalg.eigvals(Aw)
        lam = max(e.real for e in ev if abs(e.imag) < 1e-9)
        assert lam == pytest.approx(G0 / V, rel=tol)


def _run(rocket, att, seed=7, **cfg):
    fc = FlightComputer(attitude=att)
    simulate(rocket, SimConfig(t_max=4.5, seed=seed, **cfg), fc)
    ph = np.array(fc.log["phase"])
    return fc, np.degrees(np.abs(np.array(fc.log["att_err"])[ph == 1])).max()


def test_lqr_flight_uses_lqr_gains_and_doc_rows(rocket, tab):
    from gnc.lqr import design_lqr_schedule
    sched = design_lqr_schedule(tab)[0]
    fc, e_lqr = _run(rocket, "lqr")
    ph = np.array(fc.log["phase"])
    t = np.array(fc.log["t"])[ph == 1]
    got = np.array(fc.log["gains"])[ph == 1]
    assert np.allclose(got, np.array([sched(ti - fc.t_ld) for ti in t]), rtol=1e-12)   # same rule as the FC
    assert not np.allclose(got[0], [tab["Kp"][0], tab["Ki"][0], tab["Kd"][0]], rtol=0.1)  # not the PID gains
    _, e_pid = _run(rocket, "pid")
    assert round(e_pid, 2) == 0.21 and round(e_lqr, 2) == 0.15
    _, m_pid = _run(rocket, "pid", thrust_misalignment_rad=(0.01, 0.01))
    _, m_lqr = _run(rocket, "lqr", thrust_misalignment_rad=(0.01, 0.01))
    assert round(m_pid, 2) == 0.73 and round(m_lqr, 2) == 0.28
    _, d_pid = _run(rocket, "pid", actuator_delay_s=0.03, estimation_delay_s=0.03)
    _, d_lqr = _run(rocket, "lqr", actuator_delay_s=0.03, estimation_delay_s=0.03)
    assert d_pid < 1.5 and round(d_lqr, 2) == 6.99


_IDENT = """
import json, sys, numpy as np
from gnc.rocket import Rocket
from gnc.flight_computer import FlightComputer
from gnc.sim import SimConfig, simulate
W = np.array([[0, 0, 0], [0, 30, 300], [0, 90, 900]], float)
r = Rocket.load()
out = {}
for k, fc in (("hold", FlightComputer()), ("los", FlightComputer(guidance="los_path", waypoints=W))):
    res = simulate(r, SimConfig(t_max=3.0, seed=7), fc)
    out[k] = [res["x"].tolist(), res["u_cmd"].tolist()]
json.dump(out, open(sys.argv[1], "w"))
"""


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_default_pid_bit_identical_to_8cd9205(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    arc = subprocess.run(["git", "-C", str(ROOT), "archive", "8cd9205", "gnc"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(old)], input=arc.stdout, check=True)
    (old / "data").symlink_to(ROOT / "data")
    script = tmp_path / "ident.py"
    script.write_text(_IDENT)
    res = {}
    for tag, path in (("old", old), ("new", ROOT)):
        env = dict(os.environ, PYTHONPATH=str(path), PYTHONWARNINGS="ignore")
        subprocess.run([sys.executable, str(script), str(tmp_path / f"{tag}.json")], env=env, cwd=tmp_path, check=True)
        res[tag] = json.loads((tmp_path / f"{tag}.json").read_text())
    assert res["old"] == res["new"]
