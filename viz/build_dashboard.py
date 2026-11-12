"""Build the self-contained offline GNC dashboard from a results directory.

    python -m viz.build_dashboard [--results results] [--out GNC_Dashboard.html] [--open]

Reads ``<results>/index.json`` and the demo files it lists (schema: ARCHITECTURE.md, "Results JSON schema"),
validates them, downsamples over-long series, and writes one HTML file with plotly.js and the data embedded.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import warnings
import webbrowser
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
MAX_TIME_POINTS = 5000      # per variant time base (25 Hz * 200 s)
MAX_SERIES_POINTS = 5000    # per plot series
WARN_BYTES = 25e6


class ResultsError(ValueError):
    """A results file is missing a required field or has inconsistent array shapes."""


# ----------------------------------------------------------------------------------------------- validation
def _fail(src, where, msg):
    raise ResultsError(f"{src}: {where}: {msg}")


def _arr(src, where, value, ncols, nrows=None):
    """Check `value` is an (nrows x ncols) numeric array (None/NaN allowed as entries)."""
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        _fail(src, where, f"must be a list of [{ncols} numbers] rows")
    if a.ndim != 2 or a.shape[1] != ncols:
        if a.size == 0 and nrows in (None, 0):
            return
        _fail(src, where, f"must be a list of [{ncols} numbers] rows, got array of shape {a.shape}")
    if nrows is not None and a.shape[0] != nrows:
        _fail(src, where, f"has {a.shape[0]} rows but t has {nrows} samples (must be time-aligned)")


def _req(src, where, d, key, types):
    if not isinstance(d, dict):
        _fail(src, where, "must be an object")
    if key not in d:
        _fail(src, where, f"missing required key '{key}'")
    if not isinstance(d[key], types):
        _fail(src, f"{where}.{key}", "must be " + {str: "a string", list: "a list"}.get(types, "a number"))
    return d[key]


KINDS = ("waypoints", "path", "moving", "sphere")
GLYPHS = ("rocket", "spacecraft")


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _scene(src, w, sc):
    """Optional variant.scene (ARCHITECTURE.md): frame, axis_labels, views2d, ground, glyph, glyph_length_m."""
    if sc is None:
        return
    if not isinstance(sc, dict):
        _fail(src, w, "must be an object")
    if sc.get("frame") is not None and not isinstance(sc["frame"], str):
        _fail(src, f"{w}.frame", "must be a string")
    lab = sc.get("axis_labels")
    if lab is not None and not (isinstance(lab, list) and len(lab) == 3 and all(isinstance(x, str) for x in lab)):
        _fail(src, f"{w}.axis_labels", "must be a list of 3 strings")
    views = sc.get("views2d")
    if views is not None:
        if not isinstance(views, list) or not views:
            _fail(src, f"{w}.views2d", "must be a non-empty list of {title, x, y}")
        ax = lambda a: isinstance(a, int) and not isinstance(a, bool) and 0 <= a <= 2
        for i, vw in enumerate(views):
            if not (isinstance(vw, dict) and (ax(vw.get("x")) or vw.get("x") == "downrange") and ax(vw.get("y"))):
                _fail(src, f"{w}.views2d[{i}]", 'needs x = 0, 1, 2 or "downrange" and y = 0, 1 or 2')
    if sc.get("ground") is not None and not isinstance(sc["ground"], bool):
        _fail(src, f"{w}.ground", "must be true or false")
    if sc.get("glyph") is not None and sc["glyph"] not in GLYPHS:
        _fail(src, f"{w}.glyph", f"must be one of {', '.join(GLYPHS)}, got {sc['glyph']!r}")
    gl = sc.get("glyph_length_m")
    if gl is not None and not (_num(gl) and gl > 0):
        _fail(src, f"{w}.glyph_length_m", "must be a positive number")


def validate_demo(demo, src="<demo>"):
    """Raise ResultsError (naming file and field) if `demo` does not follow the results schema."""
    _req(src, "demo", demo, "id", str)
    _req(src, "demo", demo, "title", str)
    variants = _req(src, "demo", demo, "variants", list)
    if not variants:
        _fail(src, "variants", "must contain at least one variant")
    for key in ("group", "description"):
        if demo.get(key) is not None and not isinstance(demo[key], str):
            _fail(src, key, "must be a string")
    seen = set()
    for k, v in enumerate(variants):
        w = f"variants[{k}]"
        vid = _req(src, w, v, "id", str)
        w = f"variants[{k}] ('{vid}')"
        if vid in seen:
            _fail(src, w, "duplicate variant id")
        seen.add(vid)
        rocket = v.get("rocket")
        t = v.get("t")
        n = None
        if t is not None or rocket is not None:
            if not isinstance(t, list):
                _fail(src, f"{w}.t", "missing or not a list (required when rocket is given)")
            try:
                ta = np.asarray(t, dtype=float) if t else np.zeros(0)
            except (TypeError, ValueError):
                _fail(src, f"{w}.t", "must be a flat list of numbers")
            if ta.ndim != 1 or any(isinstance(x, bool) for x in t):
                _fail(src, f"{w}.t", "must be a flat list of numbers")
            if not np.all(np.isfinite(ta)):
                _fail(src, f"{w}.t", "contains NaN/inf/null")
            if np.any(np.diff(ta) < 0):
                _fail(src, f"{w}.t", "must be non-decreasing")
            n = len(ta)
        if rocket is not None:
            if not isinstance(rocket, dict):
                _fail(src, f"{w}.rocket", "must be an object or null")
            if n == 0:
                _fail(src, f"{w}.t", "is empty but rocket is given")
            if "pos" not in rocket:
                _fail(src, f"{w}.rocket", "missing required key 'pos'")
            _arr(src, f"{w}.rocket.pos", rocket["pos"], 3, n)
            for key, nc in (("quat", 4), ("pos_est", 3)):
                if rocket.get(key) is not None:
                    _arr(src, f"{w}.rocket.{key}", rocket[key], nc, n)
        _scene(src, f"{w}.scene", v.get("scene"))
        for j, o in enumerate(v.get("objects") or []):
            ow = f"{w}.objects[{j}]"
            kind = _req(src, ow, o, "kind", str)
            if kind not in KINDS:
                _fail(src, f"{ow}.kind", f"unknown kind '{kind}' (expected one of {', '.join(KINDS)})")
            if kind == "sphere":
                c = o.get("center")
                if not (isinstance(c, list) and len(c) == 3 and all(_num(x) for x in c)):
                    _fail(src, f"{ow}.center", "must be [x, y, z] (3 finite numbers)")
                if not (_num(o.get("radius")) and o["radius"] > 0):
                    _fail(src, f"{ow}.radius", "must be a positive number")
                continue
            _req(src, ow, o, "pos", list)
            if kind == "moving" and n is None:
                _fail(src, ow, "kind 'moving' needs the variant time base t")
            _arr(src, f"{ow}.pos", o["pos"], 3, n if kind == "moving" else None)
        for j, e in enumerate(v.get("events") or []):
            ew = f"{w}.events[{j}]"
            _req(src, ew, e, "t", (int, float))
            _req(src, ew, e, "label", str)
        for j, p in enumerate(v.get("plots") or []):
            pw = f"{w}.plots[{j}]"
            for i, s in enumerate(_req(src, pw, p, "series", list)):
                sw = f"{pw}.series[{i}]"
                x = _req(src, sw, s, "x", list)
                y = _req(src, sw, s, "y", list)
                if len(x) != len(y):
                    _fail(src, sw, f"x has {len(x)} points but y has {len(y)}")
        if v.get("metrics") is not None and not isinstance(v["metrics"], dict):
            _fail(src, f"{w}.metrics", "must be an object {name: value}")
        for j, tb in enumerate(v.get("tables") or []):
            tw = f"{w}.tables[{j}]"
            cols = _req(src, tw, tb, "columns", list)
            for i, row in enumerate(_req(src, tw, tb, "rows", list)):
                if not isinstance(row, list) or len(row) != len(cols):
                    _fail(src, f"{tw}.rows[{i}]", f"must be a list of {len(cols)} cells (one per column)")


# ----------------------------------------------------------------------------------------------- downsampling
def _stride_idx(n, max_points):
    if n <= max_points:
        return None
    idx = np.arange(0, n, math.ceil(n / max_points))
    return idx if idx[-1] == n - 1 else np.append(idx, n - 1)


def _take(lst, idx):
    return [lst[i] for i in idx]


def downsample_variant(v, max_time=MAX_TIME_POINTS, max_series=MAX_SERIES_POINTS):
    """Return a copy of variant `v` with time-aligned rocket arrays and plot series thinned to the limits."""
    v = dict(v)
    idx = _stride_idx(len(v.get("t") or []), max_time)
    if idx is not None:
        v["t"] = _take(v["t"], idx)
        if v.get("rocket"):
            v["rocket"] = {k: (_take(a, idx) if k in ("pos", "quat", "pos_est") and a is not None else a)
                           for k, a in v["rocket"].items()}
        if v.get("objects"):
            v["objects"] = [dict(o, pos=_take(o["pos"], idx)) if o.get("kind") == "moving" else o for o in v["objects"]]
    plots = []
    for p in v.get("plots") or []:
        series = []
        for s in p["series"]:
            si = _stride_idx(len(s["x"]), max_series)
            series.append(s if si is None else dict(s, x=_take(s["x"], si), y=_take(s["y"], si)))
        plots.append(dict(p, series=series))
    if plots:
        v["plots"] = plots
    return v


# ----------------------------------------------------------------------------------------------- embedding
def _clean(o):
    """NaN/inf -> None (JSON null, plotly gap); floats rounded to 6 significant digits to keep the file small."""
    if isinstance(o, float):
        return float(f"{o:.6g}") if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(x) for x in o]
    return o


def load_results(results_dir):
    results_dir = Path(results_dir)
    index_path = results_dir / "index.json"
    if not index_path.is_file():
        raise ResultsError(f"{index_path}: not found (run the scenarios first, or try --results viz/mock "
                           f"after `python -m viz.mock_results`)")
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as e:
        raise ResultsError(f"{index_path}: invalid JSON: {e}") from None
    files = index.get("demos") if isinstance(index, dict) else None
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise ResultsError(f"{index_path}: must be {{\"demos\": [\"<file>.json\", ...]}}")
    demos, ids = [], set()
    for name in files:
        path = results_dir / name
        if not path.is_file():
            raise ResultsError(f"{index_path}: lists '{name}' but {path} does not exist")
        try:
            demo = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ResultsError(f"{path}: invalid JSON: {e}") from None
        validate_demo(demo, str(path))
        if demo["id"] in ids:
            raise ResultsError(f"{path}: duplicate demo id '{demo['id']}'")
        ids.add(demo["id"])
        demos.append(demo)
    return demos


def render_html(demos, source="results", max_time=MAX_TIME_POINTS, max_series=MAX_SERIES_POINTS):
    demos = [dict(d, variants=[downsample_variant(v, max_time, max_series) for v in d["variants"]])
             for d in demos]
    payload = {"demos": _clean(demos), "source": str(source),
               "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    # '<' only ever occurs inside JSON strings, so < is a safe escape that can't close the <script> block
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__PLOTLY_JS__*/", get_plotlyjs(), 1).replace("__GNC_DATA__", data, 1)
    return html


def build(results_dir="results", out="GNC_Dashboard.html", open_browser=False,
          max_time=MAX_TIME_POINTS, max_series=MAX_SERIES_POINTS):
    demos = load_results(results_dir)
    html = render_html(demos, results_dir, max_time, max_series)
    out = Path(out)
    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size
    if size > WARN_BYTES:
        warnings.warn(f"{out} is {size / 1e6:.1f} MB (> {WARN_BYTES / 1e6:.0f} MB); "
                      f"consider lowering --max-points", stacklevel=2)
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results", help="results directory containing index.json")
    ap.add_argument("--out", default="GNC_Dashboard.html", help="output HTML file")
    ap.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    ap.add_argument("--max-points", type=int, default=MAX_TIME_POINTS,
                    help="max samples per time base / plot series (longer ones are downsampled)")
    a = ap.parse_args(argv)
    try:
        out = build(a.results, a.out, a.open, a.max_points, a.max_points)
    except ResultsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
