"""Independent checks of viz.build_dashboard schema validation (malformed -> clear error; edge cases accepted)."""
import copy
import json

import pytest

from viz.build_dashboard import ResultsError, load_results, render_html, validate_demo

T = [0.0, 1.0, 2.0]
POS = [[0, 0, 0], [1, 0, 10], [2, 0, 20]]


def base():
    return {"id": "d", "title": "D", "variants": [{
        "id": "v", "t": list(T), "rocket": {"pos": copy.deepcopy(POS), "quat": [[1, 0, 0, 0]] * 3},
        "objects": [], "events": [], "plots": [], "metrics": {}, "tables": []}]}


def v0(d):
    return d["variants"][0]


def _set(path, val):
    def f(d):
        v = v0(d)
        v["rocket"].update(pos=val) if path == "pos" else None
    return f


BAD = {
    "missing id": (lambda d: d.pop("id"), "id"),
    "pos len": (lambda d: v0(d)["rocket"].update(pos=POS[:2]), "rocket.pos"),
    "moving len": (lambda d: v0(d).update(objects=[{"name": "m", "kind": "moving", "pos": POS[:2]}]), "objects[0].pos"),
    "sphere no radius": (lambda d: v0(d).update(objects=[{"name": "s", "kind": "sphere", "center": [0, 0, 0]}]), "radius"),
    "nan t": (lambda d: v0(d).update(t=[0.0, float("nan"), 2.0]), "NaN"),
    "non-monotonic t": (lambda d: v0(d).update(t=[0.0, 2.0, 1.0]), "non-decreasing"),
    "axis 5": (lambda d: v0(d).update(scene={"views2d": [{"title": "x", "x": 0, "y": 5}]}), "views2d"),
    "unknown glyph": (lambda d: v0(d).update(scene={"glyph": "ufo"}), "glyph"),
    "quat width": (lambda d: v0(d)["rocket"].update(quat=[[1, 0, 0]] * 3), "quat"),
    "event no label": (lambda d: v0(d).update(events=[{"t": 1}]), "label"),
    "series len": (lambda d: v0(d).update(plots=[{"id": "p", "series": [{"x": [1, 2], "y": [1]}]}]), "series"),
    "unknown kind": (lambda d: v0(d).update(objects=[{"kind": "blob", "pos": []}]), "kind"),
    "sphere neg radius": (lambda d: v0(d).update(objects=[{"kind": "sphere", "center": [0, 0, 0], "radius": -1}]), "radius"),
    "t not list": (lambda d: v0(d).update(t="abc"), "t"),
    "pos strings": (lambda d: v0(d)["rocket"].update(pos=[["a", "b", "c"]] * 3), "pos"),
    "table row": (lambda d: v0(d).update(tables=[{"title": "t", "columns": ["a", "b"], "rows": [[1]]}]), "rows"),
}


@pytest.mark.parametrize("name", BAD)
def test_malformed_rejected_with_field(name):
    mut, needle = BAD[name]
    d = base()
    mut(d)
    with pytest.raises(ResultsError) as e:
        validate_demo(d, "foo.json")
    msg = str(e.value)
    assert "foo.json" in msg and needle in msg, msg


def _ok(mut):
    d = base()
    mut(d)
    validate_demo(d, "x.json")
    render_html([d])


GOOD = {
    "rocket null": lambda d: v0(d).update(rocket=None, t=None) or v0(d).pop("t"),
    "rocket null with t": lambda d: v0(d).update(rocket=None),
    "minimal": lambda d: d.update(variants=[{"id": "v", "t": T, "rocket": {"pos": POS}}]),
    "empty objects": lambda d: v0(d).update(objects=[]),
    "no optional": lambda d: [v0(d).pop(k) for k in ("objects", "events", "plots", "metrics", "tables")],
    "null quat": lambda d: v0(d)["rocket"].update(quat=None),
    "equal t": lambda d: v0(d).update(t=[0.0, 1.0, 1.0]),
    "sphere": lambda d: v0(d).update(objects=[{"name": "s", "kind": "sphere", "center": [0, 0, 0], "radius": 200}]),
    "analysis only": lambda d: d.update(variants=[{"id": "v", "metrics": {"a": 1}, "plots": [
        {"id": "p", "title": "p", "series": [{"name": "s", "x": [1, 2], "y": [1, None]}]}]}]),
    "single sample": lambda d: v0(d).update(t=[0.0], rocket={"pos": [[0, 0, 0]]}),
    "nan in pos": lambda d: v0(d)["rocket"].update(pos=[[0, 0, 0], [float("nan")] * 3, [2, 0, 0]]),
}


@pytest.mark.parametrize("name", GOOD)
def test_valid_edge_cases_accepted(name):
    _ok(GOOD[name])


def test_load_results_messages(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({"demos": ["a.json"]}))
    with pytest.raises(ResultsError, match="a.json"):
        load_results(tmp_path)
    (tmp_path / "a.json").write_text("{bad")
    with pytest.raises(ResultsError, match="a.json"):
        load_results(tmp_path)
    (tmp_path / "a.json").write_text(json.dumps(base()))
    assert len(load_results(tmp_path)) == 1


def test_script_injection_safe():
    d = base()
    d["title"] = "</script><script>alert(1)</script>"
    assert "</script><script>alert" not in render_html([d])
