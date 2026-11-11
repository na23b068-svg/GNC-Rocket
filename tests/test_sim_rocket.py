import numpy as np
import pytest

from gnc.rocket import DATA, Rocket, read_eng


@pytest.fixture(scope="module")
def ph():
    return Rocket.load(DATA / "rocket_params.placeholder.json")


@pytest.fixture(scope="module", params=["placeholder", "default"])
def rocket(request):
    return Rocket.load(DATA / "rocket_params.placeholder.json") if request.param == "placeholder" else Rocket.load()


def test_impulse_integration(rocket):
    _, t, F = read_eng(rocket.eng_path)
    assert np.isclose(rocket.total_impulse, np.trapezoid(F, t))
    tt = np.linspace(0, rocket.burn_time, 20001)
    assert np.isclose(np.trapezoid(rocket.thrust(tt), tt), rocket.total_impulse, rtol=1e-4)
    assert np.isclose(rocket.cumulative_impulse(rocket.burn_time), rocket.total_impulse)
    assert rocket.thrust(-0.1) == 0 and rocket.thrust(rocket.burn_time + 0.1) == 0


def test_mass_properties(rocket):
    tt = np.linspace(0, rocket.burn_time + 1, 500)
    m = rocket.mass(tt)
    assert np.all(np.diff(m) <= 1e-12)
    assert np.isclose(m[0], rocket.m_wet) and np.isclose(m[-1], rocket.m_dry)
    assert np.isclose(rocket.cg(0.0), rocket.cg_wet) and np.isclose(rocket.cg(100.0), rocket.cg_dry)
    assert np.isclose(rocket.Iyy(0.0), rocket.Iyy_wet) and np.isclose(rocket.Ixx(100.0), rocket.Ixx_dry)
    t = np.linspace(0, rocket.burn_time, 20001)
    assert np.isclose(np.trapezoid(rocket.mdot(t), t), rocket.m_wet - rocket.m_dry, rtol=1e-4)
    # half impulse -> half propellant burned
    th = t[np.searchsorted(rocket.cumulative_impulse(t), 0.5 * rocket.total_impulse)]
    assert np.isclose(rocket.mass(th), 0.5 * (rocket.m_wet + rocket.m_dry), atol=0.01)


def test_aero_lookup(ph):
    a = ph.params["aero"]
    assert np.isclose(ph.cd(0.0), a["cd_power_off"][0]) and np.isclose(ph.cd(0.0, True), a["cd_power_on"][0])
    assert np.isclose(ph.cn_alpha(0.45), 0.5 * (a["cn_alpha"][1] + a["cn_alpha"][2]))
    assert np.isclose(ph.cp(10.0), a["cp_m"][-1])
    assert 1.5 < ph.stability_cal(0.0, 0.3) < 4.0


def test_placeholder_schema_matches_contract(ph):
    p = ph.params
    for k in ["name", "length_m", "diameter_m", "reference_area_m2", "motor", "mass", "gimbal_station_m", "aero",
              "payload", "launch", "openrocket_reference"]:
        assert k in p
    assert np.isclose(ph.m_wet - ph.m_dry, p["motor"]["propellant_mass_kg"])
    assert np.isclose(ph.total_impulse, p["motor"]["total_impulse_Ns"], rtol=1e-3)
