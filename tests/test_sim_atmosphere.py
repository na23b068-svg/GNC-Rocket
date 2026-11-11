import numpy as np

from gnc.atmosphere import WindModel, geopotential, isa, pressure_altitude


def test_isa_sea_level():
    T, p, rho, a = isa(0.0)
    assert np.isclose(T, 288.15) and np.isclose(p, 101325.0)
    assert np.isclose(rho, 1.2250, atol=1e-4) and np.isclose(a, 340.294, atol=1e-2)


def test_isa_11km_geopotential():
    h = 11000.0 * 6356766.0 / (6356766.0 - 11000.0)   # geometric altitude of H = 11 km
    assert np.isclose(geopotential(h), 11000.0)
    T, p, rho, a = isa(h)
    assert np.isclose(T, 216.65, atol=1e-6)
    assert np.isclose(p, 22632.06, rtol=1e-4)
    assert np.isclose(rho, 0.36392, rtol=1e-3)
    assert np.isclose(a, 295.07, atol=0.02)


def test_isa_continuous_and_inverse():
    for h in [0, 850, 3000, 10999, 11001, 15000, 19000]:
        assert np.isclose(pressure_altitude(isa(h)[1]), h, atol=1e-3)
    assert np.isclose(isa(20000 - 1e-3)[1], isa(20000 + 1e-3)[1], rtol=1e-6)
    assert isa(25000)[1] < isa(20000)[1] and isa(60000)[0] < isa(50000)[0]


def test_wind_model():
    assert WindModel()(3.0, 100.0) == (0.0, 0.0, 0.0)
    w = WindModel((-5.0, 2.0, 0.0), shear_exponent=1 / 7, ref_alt_m=10.0)
    assert np.allclose(w(0, 0.0), (-5, 2, 0)) and np.allclose(w(0, 10.0), (-5, 2, 0))
    assert np.isclose(w(0, 1280.0)[0], -5 * 128 ** (1 / 7))
    g1 = WindModel(gust={"std_mps": 2.0, "tau_s": 1.0}, t_max=2000, seed=4)
    g2 = WindModel(gust={"std_mps": 2.0, "tau_s": 1.0}, t_max=2000, seed=4)
    s = np.array([g1(t, 0) for t in np.arange(0, 2000, 0.1)])
    assert np.allclose(s, [g2(t, 0) for t in np.arange(0, 2000, 0.1)])
    assert np.allclose(s.std(axis=0), [2.0, 2.0, 1.0], rtol=0.15)
    oc = WindModel(gust={"one_cos": [{"t0": 1.0, "duration": 2.0, "amplitude": [0, 4, 0]}]})
    assert np.allclose(oc(2.0, 0), (0, 4, 0)) and np.allclose(oc(0.5, 0), (0, 0, 0))
