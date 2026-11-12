"""Scenario results: schema validation (viz.build_dashboard's validator) and numeric sanity checks.

Reads the generated results/*.json (run `.venv/bin/python -m scenarios.run_all` first); no simulation here.
"""
import json
from pathlib import Path

import pytest

from viz.build_dashboard import validate_demo

RES = Path(__file__).resolve().parents[1] / "results"
ORDER = ["passive.json", "analysis.json", "attitude_control.json", "los_path.json", "observers.json",
         "robustness.json"]


def load(name):
    path = RES / name
    if not path.exists():
        pytest.fail(f"{path} missing: run .venv/bin/python -m scenarios.run_all")
    return json.loads(path.read_text())


def variant(demo, vid):
    return next(v for v in demo["variants"] if v["id"] == vid)


def test_index_order():
    assert json.loads((RES / "index.json").read_text())["demos"] == ORDER


@pytest.mark.parametrize("name", ORDER)
def test_schema(name):
    demo = load(name)
    validate_demo(demo, name)
    assert demo["id"] == name[:-5] and demo.get("group") and demo.get("description")
    for v in demo["variants"]:
        if v.get("rocket"):
            assert 15 <= 1 / (v["t"][1] - v["t"][0]) <= 30          # ~25 Hz time base
        assert v.get("metrics"), f"{name}:{v['id']} has no metrics"
    assert (RES / name).stat().st_size < 3e6


def test_analysis_sanity():
    d = load("analysis.json")
    lin = variant(d, "linearisation")["metrics"]
    assert lin["Max open-loop unstable eigenvalue [1/s]"] > 0          # drift mode unstable open loop
    assert lin["Max closed-loop Re(pole) excl. lateral position [1/s]"] < 0
    assert variant(d, "pid_schedule")["metrics"]["Max pole-placement error"] < 1e-6
    pid = variant(d, "pid_schedule")["metrics"]
    assert abs(pid["Steady-state error with integral action [deg]"]) < 0.1 * abs(
        pid["Steady-state error without integral action [deg]"])
    dm = variant(d, "delays")["metrics"]
    assert 20 < dm["Min delay margin (Pade) [ms]"] < 60
    assert variant(d, "ctrb_obsv")["metrics"]["12-state ctrb rank"] < 12   # roll / along-track uncontrollable


def test_attitude_control_sanity():
    d = load("attitude_control.json")
    m = {v["id"]: v["metrics"] for v in d["variants"]}
    nom = m["nominal"]
    err = "Max true pointing error (guided) [deg]"
    assert nom[err] < 0.5
    assert nom[err] < nom["Passive: max deviation from rail line (same window) [deg]"]   # better than passive
    assert m["disturbance"][err] < m["disturbance"]["Passive: max deviation from rail line (same window) [deg]"]
    assert abs(m["delay_15_15"][err] - nom[err]) < 0.1                                   # inside the margin
    assert m["delay_30_30"][err] > 3 * nom[err]                                          # beyond: worse
    assert m["delay_30_30"]["Control effort [deg^2 s]"] > 2 * nom["Control effort [deg^2 s]"]
    for v in m.values():
        assert v["Max gimbal [deg]"] <= 5.0 + 1e-6 and v["Apogee AGL [m]"] > 2800
    if "lqr_nominal" in m:                                  # LQR vs PID comparison (when gnc.lqr exists)
        assert m["lqr_nominal"][err] < 0.5
        assert m["lqr_delay_15_15"][err] > 2 * m["lqr_nominal"][err]                    # smaller early delay margin
        assert {t["title"][:10] for t in variant(d, "lqr_nominal")["tables"]} == {"LQR vs pol"}


def test_los_path_sanity():
    d = load("los_path.json")
    m = {v["id"]: v["metrics"] for v in d["variants"]}
    rms = "RMS cross-track (guided) [m]"
    assert m["aggressive"][rms] < 10
    assert m["aggressive"]["Gimbal >= 4.5 deg [% guided]"] > 10                        # works near the limit
    assert m["aggressive_dist"]["Gimbal saturation (|delta| = 5 deg) [% guided]"] > 20
    assert m["beyond"]["Max cross-track (guided) [m]"] > 2 * m["aggressive"]["Max cross-track (guided) [m]"]
    for vid in ("aggressive", "aggressive_dist", "beyond"):
        assert {o["kind"] for o in variant(d, vid)["objects"]} == {"waypoints", "path"}
        assert variant(d, vid)["rocket"].get("pos_est")


def test_observers_sanity():
    d = load("observers.json")
    kf, lu = variant(d, "inloop_kf")["metrics"], variant(d, "inloop_luenberger")["metrics"]
    key = "Position RMSE (rail exit -> apogee) [m]"
    assert kf[key] < lu[key]
    sw = variant(d, "noise_sweep")["metrics"]
    assert sw["KF pos RMSE @ k=1 [m]"] < sw["Luenberger pos RMSE @ k=1 [m]"]
    assert sw["Diverged runs"] == 0 and sw["Runs"] >= 60
    rows = variant(d, "noise_sweep")["tables"][0]["rows"]
    assert [r[0] for r in rows] == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    att = variant(d, "attitude_plane")["metrics"]
    assert att["KF theta RMSE @ k=1 [deg]"] < att["Luenberger theta RMSE @ k=1 [deg]"] < 1.0
    assert att["Gyro-only drift after 15 s [deg]"] > 5


def test_robustness_sanity():
    d = load("robustness.json")
    summ = variant(d, "summary")
    assert summ["metrics"]["Flights"] >= 200
    rows = summ["tables"][0]["rows"]
    nominal = [r for r in rows if r[1] == 0.0 and r[0].startswith("Wind")][0]
    assert nominal[4] == "PASS" and nominal[7] == "PASS"
    wind = variant(d, "wind")["tables"][0]["rows"]
    assert float(wind[-1][1].split()[0]) > float(wind[0][1].split()[0])                # wind hurts tracking
