"""Independent browser checks: offline, geometry (nose vector, sphere radius, views2d axes), toggle/keys."""
import json
import math

import pytest

pw = pytest.importorskip("playwright.sync_api")
from viz.build_dashboard import render_html

N = 101
T = [0.5 * i for i in range(N)]
H = math.sqrt(0.5)
POS = [[10 * i, 5 * i, 3 * i] for i in range(N)]
SPH = {"name": "Keepout", "kind": "sphere", "center": [300, 200, 100], "radius": 200}


def _var(id, q, **kw):
    v = {"id": id, "t": T, "rocket": {"pos": POS, "quat": [q] * N}, "events": [{"t": 10, "label": "e"}]}
    v.update(kw)
    return v


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    demos = [
        {"id": "g", "title": "G", "variants": [
            # ARCHITECTURE: identity => nose (body x) = East. Ry(-90) => nose Up. Rz(90)Ry(-90)... => nose Up still;
            _var("east", [1, 0, 0, 0], objects=[SPH]),
            _var("up", [H, 0, -H, 0]),
            _var("north", [H, 0, 0, H]),    # yaw +90 about Up: East -> North
            _var("views", [1, 0, 0, 0], objects=[SPH],
                 scene={"views2d": [{"title": "V1", "x": 1, "y": 2}, {"title": "V2", "x": 2, "y": 0}]}),
        ]},
        {"id": "n", "title": "N", "variants": [{"id": "v", "rocket": None, "metrics": {"a": 1}}]},
    ]
    f = tmp_path_factory.mktemp("h") / "d.html"
    f.write_text(render_html(demos))
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome")
        except Exception as e:  # pragma: no cover
            pytest.skip(f"chrome unavailable: {e}")
        ctx = b.new_context(offline=True, viewport={"width": 1400, "height": 900})
        reqs, errs = [], []
        ctx.route("**/*", lambda r: r.continue_() if r.request.url.startswith("file:") else (reqs.append(r.request.url), r.abort()))
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(f.as_uri())
        pg.wait_for_function("window.GNC")
        yield pg, reqs, errs
        b.close()


def _nose(pg, vj):
    pg.evaluate(f"GNC.selectVariant({vj}); GNC.setView('3d'); GNC.seek(25)")
    pg.wait_for_timeout(300)
    return pg.evaluate("""()=>{const t=document.getElementById('scene-plot').data.find(t=>t.type==='mesh3d'&&t.name==='Rocket');
        const d=[t.x[24]-t.x[25],t.y[24]-t.y[25],t.z[24]-t.z[25]];const m=Math.hypot(...d);return d.map(x=>x/m)}""")


@pytest.mark.parametrize("vj,exp", [(0, (1, 0, 0)), (1, (0, 0, 1)), (2, (0, 1, 0))])
def test_nose_vector(page, vj, exp):
    got = _nose(page[0], vj)
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, exp)), got


def test_sphere_radius_3d_and_2d(page):
    pg = page[0]
    pg.evaluate("GNC.selectVariant(0); GNC.setView('3d')")
    pg.wait_for_timeout(300)
    r = pg.evaluate("""()=>{const t=document.getElementById('scene-plot').data.find(t=>t.name==='Keepout');const f=a=>a.filter(v=>v!=null);
        return [Math.min(...f(t.x)),Math.max(...f(t.x)),Math.min(...f(t.z)),Math.max(...f(t.z))]}""")
    assert [round(x) for x in r] == [100, 500, -100, 300]
    pg.evaluate("GNC.setView('2d')")
    pg.wait_for_timeout(300)
    w = pg.evaluate("""()=>[...document.querySelectorAll('#view2d .js-plotly-plot')].map(g=>{const t=g.data.find(t=>t.name==='Keepout');
        const f=a=>a.filter(v=>v!=null);return [Math.max(...f(t.x))-Math.min(...f(t.x)),Math.max(...f(t.y))-Math.min(...f(t.y))]})""")
    assert all(abs(a - 400) < 1 and abs(b - 400) < 1 for a, b in w), w


def test_views2d_axes(page):
    pg = page[0]
    pg.evaluate("GNC.selectVariant(3); GNC.setView('2d')")
    pg.wait_for_timeout(300)
    r = pg.evaluate("""()=>[...document.querySelectorAll('#view2d .js-plotly-plot')].map(g=>{const t=g.data.find(t=>t.name==='Keepout');
        return [Math.min(...t.x.filter(v=>v!=null)),Math.min(...t.y.filter(v=>v!=null))]})""")
    assert r == [[0, -100], [-100, 100]]   # V1: x=North(200-200), y=Up(100-200); V2: x=Up, y=East(300-200)


def test_toggle_disabled_without_rocket_and_keys(page):
    pg = page[0]
    pg.evaluate("GNC.selectDemo(1)")
    assert pg.evaluate("[...document.querySelectorAll('#view-toggle button')].every(b=>b.disabled)")
    pg.keyboard.press("Space")
    assert not pg.evaluate("GNC.S.playing")
    pg.evaluate("GNC.selectDemo(0)")
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("GNC.S.i") == 1


def test_offline_no_errors(page):
    _, reqs, errs = page
    assert reqs == [] and errs == []
