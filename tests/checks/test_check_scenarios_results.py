"""Independent checker tests for the scenarios agent's results/*.json: metrics recomputed from the plotted arrays,
dashboard fitness, and the notes' claims checked against the numbers in the same JSON. JSON-only (no sims),
except one call into gnc.analysis.critical_delay. Known defects are strict xfails so they flip when fixed."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from scenarios.common import AGGRESSIVE_WP, BEYOND_WP
from viz.build_dashboard import validate_demo

RES = Path(__file__).resolve().parents[2] / "results"


def load(name):
    p = RES / name
    if not p.exists():
        pytest.skip(f"{p} missing: run .venv/bin/python -m scenarios.run_all")
    return json.loads(p.read_text())


def var(demo, vid):
    return next(v for v in demo["variants"] if v["id"] == vid)


def ser(v, pid, name):
    p = next(p for p in v["plots"] if p["id"] == pid)
    s = next(s for s in p["series"] if s["name"] == name)
    return np.array(s["x"], float), np.array([np.nan if y is None else y for y in s["y"]], float)


def ev(v, prefix):
    return next(e["t"] for e in v["events"] if e["label"].lower().startswith(prefix))


DEMOS = [n for n in json.loads((RES / "index.json").read_text())["demos"]] if (RES / "index.json").exists() else []


@pytest.mark.parametrize("name", DEMOS)
def test_dashboard_fitness(name):
    d = load(name)
    validate_demo(d, name)
    for v in d["variants"]:
        t = [e["t"] for e in v["events"]]
        assert t == sorted(t), f"{v['id']}: events out of order"
        rk = v.get("rocket")
        if rk:
            q = np.array(rk["quat"], float)
            assert np.abs(np.linalg.norm(q, axis=1) - 1).max() < 1e-4
            if name != "passive.json":     # every flight-computer run has a navigation filter
                assert rk.get("pos_est"), f"{v['id']}: filter runs but no pos_est"
        for p in v["plots"]:
            if p["xlabel"] == "t [s]" and p["id"] not in {"step_0", "step_4", "step_8", "step_u_0", "step_u_4",
                                                            "step_u_8"}:
                xs = np.concatenate([np.array(s["x"], float) for s in p["series"]])
                assert xs.min() >= 0 and xs.max() <= (max(v["t"]) + 1 if v["t"] else 60), (v["id"], p["id"])



@pytest.mark.parametrize("vid,W", [("aggressive", AGGRESSIVE_WP), ("aggressive_dist", AGGRESSIVE_WP),
                                   ("beyond", BEYOND_WP)])
def test_los_metrics_match_plotted_arrays(vid, W):
    v = var(load("los_path.json"), vid)
    m = v["metrics"]
    t0, t1 = ev(v, "rail"), ev(v, "tvc off")
    x, e = ser(v, "xte", "total |e|")
    g = (x >= t0) & (x <= t1 + 0.011)
    assert np.sqrt(np.mean(e[g] ** 2)) == pytest.approx(m["RMS cross-track (guided) [m]"], rel=0.02)
    assert e[g].max() == pytest.approx(m["Max cross-track (guided) [m]"], rel=0.02)
    xg, up = ser(v, "gimbal", "pitch delta_p")
    _, uy = ser(v, "gimbal", "yaw delta_y")
    gg = (xg >= t0) & (xg <= t1)
    u = np.abs(np.column_stack([up, uy]))[gg]
    assert 100 * np.mean(np.any(u >= 4.5, axis=1)) == pytest.approx(m["Gimbal >= 4.5 deg [% guided]"], abs=3)
    assert 100 * np.mean(np.any(u >= 5 - 1e-3, axis=1)) == pytest.approx(
        m["Gimbal saturation (|delta| = 5 deg) [% guided]"], abs=3)
    assert ser(v, "effort", "effort")[1][-1] == pytest.approx(m["Control effort [deg^2 s]"], rel=1e-3)
    wp = next(o for o in v["objects"] if o["kind"] == "waypoints")["pos"]
    assert wp == W == next(o for o in v["objects"] if o["kind"] == "path")["pos"]
    P = np.array(v["rocket"]["pos"])
    for i, w in enumerate(np.array(W[1:-1], float)):   # 25 Hz pos vs 1 kHz metric: within 1 m
        key = next(k for k in m if k.startswith(f"WP{i + 1} "))
        assert np.min(np.linalg.norm(P - w, axis=1)) == pytest.approx(m[key], abs=1.0)


def test_los_note_numbers():
    d = load("los_path.json")
    assert var(d, "beyond")["metrics"]["WP5 (14, 27, 450) pass distance [m]"] == pytest.approx(16, abs=1.5)
    agg, dist = var(d, "aggressive")["metrics"], var(d, "aggressive_dist")["metrics"]
    assert agg["RMS cross-track (guided) [m]"] < 10 < dist["RMS cross-track (guided) [m]"]
    assert dist["Gimbal saturation (|delta| = 5 deg) [% guided]"] > 33          # "over a third"
    env = var(d, "envelope")
    for r in env["tables"][0]["rows"]:
        frac = r[3]
        assert r[-1] == ("flyable" if frac >= 80 else "marginal" if frac >= 50 else "NOT flyable")


@pytest.mark.parametrize("vid", ["nominal", "disturbance", "delay_15_15", "delay_30_30", "lqr_nominal",
                                 "lqr_disturbance", "lqr_delay_15_15"])
def test_attitude_metrics_match_plots(vid):
    v = var(load("attitude_control.json"), vid)
    m = v["metrics"]
    t0, t1 = ev(v, "rail"), ev(v, "tvc off")
    x, ep = ser(v, "att_err", "pitch error (est.)")
    _, ey = ser(v, "att_err", "yaw error (est.)")
    g = (x >= t0) & (x <= t1 + 0.011)
    assert np.abs(np.r_[ep[g], ey[g]]).max() == pytest.approx(m["Max attitude error, controller [deg]"], abs=2e-3)
    xs, cp = ser(v, "gimbal_p", "actual")
    _, cy = ser(v, "gimbal_y", "actual")
    gg = (xs >= t0) & (xs <= t1)
    u = np.abs(np.column_stack([cp, cy]))[gg]
    assert 100 * np.mean(np.any(u >= 5 - 1e-3, axis=1)) == pytest.approx(
        m["Gimbal saturation [% of guided time]"], abs=2)


@pytest.mark.parametrize("vid", ["inloop_kf", "inloop_luenberger"])
def test_observer_rmse_matches_plot(vid):
    v = var(load("observers.json"), vid)
    p = next(p for p in v["plots"] if p["id"] == "pos_err")
    s = next(s for s in p["series"] if "this flight" in s["name"])
    x, e = np.array(s["x"]), np.array(s["y"])
    assert np.sqrt(np.mean(e[x >= ev(v, "rail")] ** 2)) == pytest.approx(
        v["metrics"]["Position RMSE (rail exit -> apogee) [m]"], rel=0.02)


def test_robustness_counts_and_verdicts():
    d = load("robustness.json")
    rows = var(d, "summary")["tables"][0]["rows"]
    assert var(d, "summary")["metrics"]["Flights"] == 250 == 2 * 5 * len(rows)
    for r in rows:
        ok, n = map(int, r[3].split("/"))
        assert (r[4] == "PASS") == (ok == n)
        ok, n = map(int, r[6].split("/"))
        assert (r[7] == "PASS") == (ok == n)


# ------------------------------------------------------------------------------------------- known defects
@pytest.mark.xfail(strict=True, reason="analysis linearisation notes still call the frozen-coefficient real "
                                       "root a physical 'thrusting weathercock divergence' (docs/gnc_design.md "
                                       "section 3 says it is mostly an artefact; residual ~g/V)")
def test_linearisation_notes_drift_not_physical():
    notes = var(load("analysis.json"), "linearisation")["notes"].lower()
    assert "divergence" not in notes or "artefact" in notes


@pytest.mark.xfail(strict=True, reason="notes say numeric Jacobian matches analytic to ~1e-10; metric is 1.8e-6")
def test_linearisation_notes_accuracy_matches_metric():
    v = var(load("analysis.json"), "linearisation")
    claimed = float(re.search(r"to ~(1e-\d+)", v["notes"]).group(1))
    assert v["metrics"]["Max numeric vs analytic rel. diff"] <= 10 * claimed


@pytest.mark.xfail(strict=True, reason="delay_30_30 notes: 66 ms 'exceeds the delay margin at every operating "
                                       "point', but the PID Pade margin is 66.8 / 72.6 ms at 1.27 / 1.72 s")
def test_delay_30_30_note_vs_margin_table():
    rows = var(load("analysis.json"), "delays")["tables"][0]["rows"]
    assert "every operating point" not in var(load("attitude_control.json"), "delay_30_30")["notes"] or \
        all(66 > r[5] for r in rows)


@pytest.mark.xfail(strict=True, reason="gnc.analysis.critical_delay(d_max=1.0) returns inf when the Pade loop "
                                       "is stable again at 1 s, although it goes unstable at ~40 ms")
def test_critical_delay_default_bracket():
    from gnc import analysis as an, lqr
    from gnc.flight_computer import design_schedule
    from gnc.rocket import Rocket
    tab = design_schedule(Rocket.load())[1]
    i = 6                                              # t = 3.07 s
    g = {k: lqr.compare(tab)[i]["lqr"][k] for k in ("Kp", "Ki", "Kd")}
    a = (tab["a1"][i], tab["a0"][i], tab["b"][i])
    assert an.critical_delay(*a, g, 0.02, d_max=0.3) < 0.1     # really unstable below 100 ms
    assert np.isfinite(an.critical_delay(*a, g, 0.02))
