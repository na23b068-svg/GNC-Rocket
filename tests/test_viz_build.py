import copy
import json
import re

import numpy as np
import pytest

from viz import build_dashboard as bd
from viz import mock_results


@pytest.fixture(scope="module")
def mock_dir(tmp_path_factory):
    return mock_results.write_mock(tmp_path_factory.mktemp("mock"))


def _payload(html):
    m = re.search(r'<script id="gnc-data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1), parse_constant=lambda c: pytest.fail(f"non-JSON constant {c} embedded"))


def test_build_mock(mock_dir, tmp_path):
    out = bd.build(mock_dir, tmp_path / "d.html")
    html = out.read_text()
    assert not re.search(r"<script[^>]+src=[\"']?https?:", html), "must not load external scripts"
    assert "plotly.js v" in html or "Plotly" in html
    ids = [d["id"] for d in _payload(html)["demos"]]
    index = json.loads((mock_dir / "index.json").read_text())["demos"]
    assert len(ids) == len(index) == 5
    assert {"passive", "los_path", "ascent_orbital", "docking_lvlh", "linear_analysis"} == set(ids)
    for i in ids:
        assert i in html
    assert out.stat().st_size < bd.WARN_BYTES


def test_cli(mock_dir, tmp_path, capsys):
    assert bd.main(["--results", str(mock_dir), "--out", str(tmp_path / "x.html")]) == 0
    assert bd.main(["--results", str(tmp_path / "nope")]) == 2
    assert "index.json" in capsys.readouterr().err


def _demo():
    return copy.deepcopy(mock_results.waypoint_demo())


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d.pop("title"), "missing required key 'title'"),
    (lambda d: d.update(variants=[]), "at least one variant"),
    (lambda d: d["variants"][0]["rocket"]["pos"].pop(), "rocket.pos: has 612 rows but t has 613"),
    (lambda d: d["variants"][0]["rocket"]["quat"].pop(), "rocket.quat: has"),
    (lambda d: d["variants"][0]["rocket"].update(pos_est=[[0, 0]] * len(d["variants"][0]["t"])), "rocket.pos_est"),
    (lambda d: d["variants"][0].pop("t"), ".t"),
    (lambda d: d["variants"][0]["plots"][0]["series"][0]["y"].pop(), "plots[0].series[0]"),
    (lambda d: d["variants"][0].update(tables=[{"columns": ["a", "b"], "rows": [[1]]}]), "tables[0].rows[0]"),
    (lambda d: d["variants"][0]["objects"][0].pop("kind"), "objects[0]"),
    (lambda d: d["variants"][1].update(id="nominal"), "duplicate variant id"),
])
def test_validation_rejects(mutate, needle):
    d = _demo()
    mutate(d)
    with pytest.raises(bd.ResultsError) as e:
        bd.validate_demo(d, "results/los_path.json")
    assert "results/los_path.json" in str(e.value) and needle in str(e.value)


def test_validation_accepts_mock_and_optional_fields():
    for demo in mock_results.all_demos():
        bd.validate_demo(demo)
    d = _demo()
    for k in ("objects", "events", "plots", "tables", "notes", "metrics"):
        d["variants"][0].pop(k, None)
    d["variants"][0]["rocket"] = {"pos": d["variants"][0]["rocket"]["pos"]}
    bd.validate_demo(d)


def test_missing_listed_file(tmp_path):
    (tmp_path / "index.json").write_text('{"demos": ["gone.json"]}')
    with pytest.raises(bd.ResultsError, match="gone.json"):
        bd.load_results(tmp_path)


def test_downsampling():
    n = 20001
    t = np.linspace(0, 800, n).tolist()
    v = {"id": "v", "t": t, "rocket": {"pos": [[0, 0, x] for x in t], "quat": [[1, 0, 0, 0]] * n},
         "objects": [{"kind": "path", "pos": [[0, 0, 0], [1, 1, 1]]}],
         "plots": [{"series": [{"x": t, "y": t}]}]}
    out = bd.downsample_variant(v, 5000, 3000)
    m = len(out["t"])
    assert m <= 5001 and out["t"][0] == 0 and out["t"][-1] == 800
    assert len(out["rocket"]["pos"]) == len(out["rocket"]["quat"]) == m
    assert out["rocket"]["pos"][-1][2] == 800
    assert out["objects"][0]["pos"] == [[0, 0, 0], [1, 1, 1]]
    s = out["plots"][0]["series"][0]
    assert len(s["x"]) == len(s["y"]) <= 3001 and s["x"][-1] == 800
    assert len(v["t"]) == n, "input must not be modified"
    short = {"id": "s", "t": [0, 1], "rocket": {"pos": [[0, 0, 0]] * 2}}
    assert bd.downsample_variant(short)["t"] == [0, 1]


def test_nan_becomes_null(mock_dir, tmp_path):
    html = bd.build(mock_dir, tmp_path / "d.html").read_text()
    demos = _payload(html)["demos"]   # parse_constant fails on NaN/Infinity
    ana = next(d for d in demos if d["id"] == "linear_analysis")
    assert ana["variants"][0]["metrics"]["Phase margin [deg]"] is None
    assert ana["variants"][0]["plots"][1]["series"][1]["y"][-1] is None


def test_mock_quaternion_nose_follows_velocity():
    v = mock_results.waypoint_demo()["variants"][0]
    p, q = np.array(v["rocket"]["pos"]), np.array(v["rocket"]["quat"])
    vel = np.gradient(p, np.array(v["t"]), axis=0)
    w, x, y, z = q.T
    nose = np.column_stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    vh = vel / np.linalg.norm(vel, axis=1, keepdims=True)
    assert np.median(np.sum(nose * vh, axis=1)) > 0.99


@pytest.mark.parametrize("mutate, needle", [
    (lambda v: v["scene"].update(glyph="boat"), "scene.glyph"),
    (lambda v: v["scene"].update(views2d=[{"x": 5, "y": 0}]), "views2d[0]"),
    (lambda v: v["objects"][1].update(radius=-1), "objects[1].radius"),
    (lambda v: v["objects"][1].pop("center"), "objects[1].center"),
])
def test_scene_validation_rejects(mutate, needle):
    d = copy.deepcopy(mock_results.docking_demo())
    mutate(d["variants"][0])
    with pytest.raises(bd.ResultsError, match="docking_lvlh.json") as e:
        bd.validate_demo(d, "results/docking_lvlh.json")
    assert needle in str(e.value)


def test_build_includes_live_guidance_demos(mock_dir, tmp_path):
    out = bd.build(mock_dir, tmp_path / "with_guid.html", include_guidance=True)
    ids = {d["id"] for d in _payload(out.read_text())["demos"]}
    assert {"guidance_primitives", "guidance_los_kinematic", "guidance_intercept"}.issubset(ids)


def test_guidance_intercept_demo_reports_finite_miss(mock_dir, tmp_path):
    """The 4(i) dashboard demo must actually intercept, not just be present."""
    out = bd.build(mock_dir, tmp_path / "with_guid.html", include_guidance=True)
    demo = next(d for d in _payload(out.read_text())["demos"] if d["id"] == "guidance_intercept")
    for v in demo["variants"]:
        m = v["metrics"]
        misses = [val for k, val in m.items() if k == "Miss distance [m]" or k.endswith("miss [m]")]
        assert misses, (v["id"], list(m))
        for miss in misses:
            assert np.isfinite(miss) and miss < 1.0, (v["id"], miss)
