"""Load the mock dashboard in headless Chrome, click through everything, and fail on any JS error.

Skipped when playwright or a Chrome/Chromium install is missing. Screenshots go to viz/screenshots/.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from viz import build_dashboard as bd
from viz import mock_results

sync_api = pytest.importorskip("playwright.sync_api")
SHOTS = Path(__file__).resolve().parents[1] / "viz" / "screenshots"


def _launch(p):
    for kw in ({"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    pytest.skip("no Chrome/Chromium available for playwright")


def _quat_ref(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    if shutil.which("google-chrome") is None and shutil.which("chromium") is None:
        pytest.skip("no browser")
    d = tmp_path_factory.mktemp("dash")
    html = bd.build(mock_results.write_mock(d / "mock"), d / "dash.html")
    SHOTS.mkdir(exist_ok=True)
    with sync_api.sync_playwright() as p:
        b = _launch(p)
        pg = b.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(html.resolve().as_uri())
        pg.wait_for_function("window.GNC && window.GNC.ctx()")
        pg.errors = errors
        yield pg
        b.close()


def test_quat_matches_python_reference(page):
    rng = np.random.default_rng(0)
    for q in [[1, 0, 0, 0], [np.cos(0.3), 0, np.sin(0.3), 0], *rng.normal(size=(5, 4)).tolist()]:
        R = np.array(page.evaluate("q => GNC.quatR(q)", list(map(float, q))))
        np.testing.assert_allclose(R, _quat_ref(q), atol=1e-12)
    # a 90 deg pitch about body/ENU y turns the nose (body x) from East to Down
    R = np.array(page.evaluate("q => GNC.quatR(q)", [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]))
    np.testing.assert_allclose(R[:, 0], [0, 0, -1], atol=1e-12)


def test_click_through_everything(page):
    demos = page.evaluate("() => [...document.querySelectorAll('.demo-btn')].map(b => b.dataset.id)")
    assert demos == ["passive", "los_path", "ascent_orbital", "docking_lvlh", "linear_analysis"]
    for k, did in enumerate(demos):
        page.click(f".demo-btn[data-id='{did}']")
        nvar = page.evaluate("() => GNC.ctx().demo.variants.length")
        for j in range(nvar):
            if nvar > 1:
                page.click(f"#variant-tabs .tab:nth-child({j + 1})")
            has = page.evaluate("() => !!GNC.ctx().rocket")
            assert page.is_visible("#main-view") == has
            assert page.is_disabled("#view-toggle button[data-view='3d']") == (not has)
            n_plots = page.evaluate("() => GNC.ctx().v.plots.length")
            assert page.locator("#plots .js-plotly-plot").count() == n_plots
            if has:
                for view in ("2d", "3d"):
                    page.click(f"#view-toggle button[data-view='{view}']")
                    page.wait_for_timeout(300)
                    assert page.is_visible("#view3d" if view == "3d" else "#view2d")
                    page.evaluate("() => GNC.seek(GNC.ctx().t[Math.floor(GNC.ctx().N * 0.6)])")
                    page.wait_for_timeout(300)
                    page.screenshot(path=str(SHOTS / f"{k}_{did}_{j}_{view}.png"), full_page=True)
                page.click("#speed-seg button[data-speed='5']")
                t0 = page.evaluate("() => (GNC.seek(GNC.ctx().t[0]), GNC.S.simT)")
                page.keyboard.press("Space")
                page.wait_for_timeout(1000)
                page.keyboard.press("Space")
                assert page.evaluate("() => GNC.S.playing") is False
                assert page.evaluate("() => GNC.S.simT") > t0 + 2, "playback did not advance"
                page.fill("#time-slider", "3")      # scrub
                assert page.evaluate("() => GNC.S.i") == 3
                page.keyboard.press("ArrowRight")
                page.click("#event-chips .chip:nth-child(3)")
            else:
                page.screenshot(path=str(SHOTS / f"{k}_{did}_{j}.png"), full_page=True)
    page.click("#theme-btn")
    page.click(".demo-btn[data-id='los_path']")
    page.click("#view-toggle button[data-view='3d']")
    page.click("#scale-toggle button[data-scale='stretched']")
    page.evaluate("() => GNC.seek(14)")
    page.wait_for_timeout(300)
    page.screenshot(path=str(SHOTS / "light_los_path_3d_stretched.png"))
    page.click("#theme-btn")
    assert page.errors == [], page.errors


def test_narrow_viewport_no_horizontal_scroll(page):
    page.set_viewport_size({"width": 390, "height": 850})
    page.click(".demo-btn[data-id='passive']")
    page.wait_for_timeout(300)
    assert page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth + 1")
    page.screenshot(path=str(SHOTS / "phone_passive.png"), full_page=True)
    page.set_viewport_size({"width": 1500, "height": 1000})
    assert page.errors == [], page.errors
