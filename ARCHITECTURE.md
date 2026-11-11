# GNC Project — Architecture & Interface Contract

OE5310 GNC project: **thrust-vector-controlled (TVC) M-class sounding rocket**, 10,000 ft AGL apogee target,
launched from Midland, TX (lat 31.99 N, lon 102.08 W, ~850 m ASL), 18 ft (5.4864 m) rail at 6 deg from vertical.
Assignment brief: `GNC_Project_Assignment.pdf`. Python only (no MATLAB/Simulink).

Several agents work on this repo in parallel. **Every agent owns specific files (see Ownership). Do not edit files you
do not own; if you need a change in someone else's file, say so in your final report.** Do **not** run `git commit`;
the orchestrator commits at milestones. Use the project venv: `.venv/bin/python`, `.venv/bin/pytest`
(numpy, scipy, matplotlib, plotly, python-control, jpype1, pytest already installed; the system python is broken —
never use it). If you must `pip install` something extra, install into `.venv` and append it to `requirements.txt`.

## Conventions (binding for everyone)
- SI units everywhere; angles in **radians** internally (degrees only in labels/plots).
- Inertial frame: local flat-earth **ENU** with origin at the launch pad base. x = East, y = North, z = Up.
  Gravity `g = 9.80665` m/s^2 along -z (altitude variation of g may be ignored).
- Body frame: x_b along the rocket's longitudinal axis pointing **to the nose**; y_b, z_b lateral (right-handed).
- Attitude quaternion **scalar-first** `q = [w, x, y, z]`, rotating body-frame vectors into ENU: `v_enu = R(q) @ v_b`.
- Truth state vector (13): `x = [p_enu(3), v_enu(3), q(4), omega_b(3)]` (omega = body angular rate, rad/s).
- Rocket stations (CG, CP, gimbal pivot, ...) are measured in metres **from the nose tip, positive aft**.
- TVC input `u = [delta_p, delta_y]` (gimbal deflection angles, rad). Thrust unit vector in body frame:
  `d_b = Rz(delta_y) @ Ry(delta_p) @ [1,0,0] = [cos(dy)cos(dp), sin(dy)cos(dp), -sin(dp)]`,
  thrust force `F_b = T(t) * d_b`, applied at the gimbal pivot (nozzle throat station from rocket_params.json).
  Moment about CG: `M_b = r_gimbal_b x F_b`, with `r_gimbal_b = [-(x_gimbal - x_cg(t)), 0, 0]`.
  Gimbal limit default ±5 deg each axis (configurable). Roll is not controllable by a single gimbal; roll is
  simulated but only passively damped.
- Linearisations should be computed **numerically from `gnc.dynamics`** (finite-difference Jacobians) so sign
  conventions can't drift; analytic reduced models are welcome for insight and must agree with the numeric ones.

## Code layout & ownership
| Path | Owner | Contents |
|---|---|---|
| `openrocket/`, `data/motor.eng`, `data/rocket_params.json`, `data/openrocket_flight.csv`, `figures/openrocket_*`, `docs/rocket_design.md` | rocket-designer | OpenRocket model (.ork), headless OpenRocket tooling, motor curve, extracted parameters, OpenRocket reference sim |
| `gnc/__init__.py`, `gnc/frames.py`, `gnc/atmosphere.py`, `gnc/rocket.py`, `gnc/dynamics.py`, `gnc/sensors.py`, `gnc/sim.py`, `data/*placeholder*`, `scenarios/passive_flight.py`, `docs/sim_model.md`, `tests/test_sim_*.py` | sim-core | 6-DOF nonlinear simulator, environment, sensor models |
| `gnc/linearize.py`, `gnc/analysis.py`, `gnc/pid.py`, `gnc/observers.py`, `gnc/guidance.py`, `gnc/flight_computer.py`, `docs/gnc_design.md`, `tests/test_gnc_*.py` | gnc-designer | linearisation, ctrb/obsv, PID-by-pole-placement, Luenberger & Kalman, LOS & PN guidance, flight computer |
| `viz/`, `tests/test_viz_*.py` | dashboard | interactive dashboard (single self-contained HTML) |
| `scenarios/` (except passive_flight.py), `results/` | scenarios (phase 2) | every demo the brief asks for → results JSON |
| `ARCHITECTURE.md`, `README.md`, `run_all.py`, `.gitignore` | orchestrator | |

## `data/rocket_params.json` schema (written by rocket-designer; sim-core ships `data/rocket_params.placeholder.json` with the same schema)
```jsonc
{
  "name": "string",
  "length_m": 0.0, "diameter_m": 0.0, "reference_area_m2": 0.0,
  "motor": {"designation": "AeroTech M....", "manufacturer": "...", "eng_file": "data/motor.eng",
            "total_impulse_Ns": 0.0, "burn_time_s": 0.0, "propellant_mass_kg": 0.0, "motor_total_mass_kg": 0.0},
  "mass": {"dry_kg": 0.0, "wet_kg": 0.0,                       // whole rocket incl. motor casing; wet = dry + propellant
           "cg_dry_m": 0.0, "cg_wet_m": 0.0,                   // from nose
           "Ixx_dry": 0.0, "Ixx_wet": 0.0,                     // roll inertia about CG, kg m^2
           "Iyy_dry": 0.0, "Iyy_wet": 0.0},                    // transverse inertia about CG (Izz = Iyy)
  "gimbal_station_m": 0.0,                                     // TVC pivot, from nose
  "aero": {"mach": [], "cd_power_off": [], "cd_power_on": [],  // axial drag coefficient vs Mach
           "cn_alpha": [],                                     // normal-force slope per rad vs Mach
           "cp_m": [],                                         // CP station from nose vs Mach
           "cmq": -0.0, "clp": -0.0},                          // pitch & roll damping derivatives (ref length = diameter)
  "payload": {"mass_kg": 4.0, "station_m": 0.0},
  "launch": {"lat_deg": 31.99, "lon_deg": -102.08, "elevation_m": 850.0,
             "rail_length_m": 5.4864, "rail_angle_deg": 6.0, "rail_azimuth_deg": 90.0},
  "openrocket_reference": {"apogee_m_agl": 0.0, "max_velocity_mps": 0.0, "max_mach": 0.0,
                           "off_rail_velocity_mps": 0.0, "stability_cal_off_rail": 0.0, "time_to_apogee_s": 0.0}
}
```
Mass properties vary with propellant burn: interpolate dry↔wet linearly in burned propellant fraction
(burned fraction = cumulative impulse / total impulse).

## Simulator API (sim-core)
```python
from gnc.rocket import Rocket            # Rocket.load(path=None) -> real json if present else placeholder
from gnc.sim import SimConfig, simulate
cfg = SimConfig(...)                      # dataclass, all fields defaulted
res = simulate(rocket, cfg, controller=None)   # controller=None -> passive, gimbal fixed at 0
```
`SimConfig` fields (at least): `dt` (integration step), `t_max`, `stop_at` ("apogee" | "ground"),
`wind` (steady ENU vector m/s) + `gust` settings, `thrust_misalignment_rad` (2-vector bias),
`sensor_noise_scale` (multiplies all sensor std-devs, 0 → noiseless), per-sensor std overrides,
`actuator_tau_s` (first-order servo lag), `actuator_rate_limit_rad_s`, `actuator_delay_s`, `estimation_delay_s`,
`gimbal_limit_rad`, `seed`, `recovery` (bool: drogue at apogee, main at 300 m AGL, for passive full-flight plots).

**Controller protocol** (implemented by gnc-designer in `gnc/flight_computer.py`, called by `simulate` at `controller.dt`):
```python
class Controller:
    dt: float                                         # e.g. 0.01 s
    def reset(self, rocket, cfg) -> None: ...
    def update(self, t: float, meas: dict) -> np.ndarray   # returns commanded [delta_p, delta_y] (rad)
    log: dict[str, list]                               # anything per-step (estimates, guidance cmds, ...)
```
`meas` keys (sensor outputs with Gaussian noise + optional bias; a key is `None` when that sensor has no new sample
this step): `imu_accel` (body specific force, m/s^2), `imu_gyro` (body rate, rad/s), `baro_alt` (m AGL from
pressure), `gps_pos` (ENU m), `gps_vel` (ENU m/s), `mag` (body-frame unit field vector), plus `t_launch_detect`.
Nominal rates: IMU 200 Hz, baro 50 Hz, mag 50 Hz, GPS 10 Hz. `meas["truth"]` (true state) is included **only for
logging/evaluation**; controllers must not use it for feedback unless `cfg.perfect_nav` is True.

`simulate` returns a dict of numpy arrays: `t`, `x` (N×13), `u_cmd`, `u_act` (N×2), `thrust`, `mass`, `cg`, `mach`,
`alpha` (total AoA), `q_dyn`, `events` (dict name → time: liftoff, rail_exit, burnout, apogee, ...), `meas_log`,
`controller_log`, and the `cfg` used.

## Results JSON schema (written by scenarios, consumed by viz)
`results/index.json` = `{"demos": ["passive.json", ...]}` (display order). Each `results/<demo>.json`:
```jsonc
{
  "id": "los_path", "group": "Path following", "title": "LOS waypoint path following",
  "description": "1-3 short paragraphs, plain text",
  "variants": [{
    "id": "nominal", "label": "No disturbance",
    "t": [],                                             // common time base, downsampled (~25 Hz)
    "rocket": {"pos": [[x,y,z]], "quat": [[w,x,y,z]],    // truth; may be null for analysis-only demos
               "pos_est": [[x,y,z]]},                    // optional, navigation estimate
    "objects": [{"name": "Waypoints", "kind": "waypoints", "pos": [[x,y,z]]},    // static points
                {"name": "Reference path", "kind": "path", "pos": [[x,y,z]]}],  // static polyline
    "events": [{"t": 0.0, "label": "Rail exit"}],
    "plots": [{"id": "xte", "title": "Cross-track error", "xlabel": "t [s]", "ylabel": "m",
               "series": [{"name": "cross-track", "x": [], "y": [], "mode": "lines"}]}],
    "metrics": {"RMS cross-track error [m]": 0.0},
    "tables": [{"title": "Controllability", "columns": ["..."], "rows": [["..."]]}],
    "notes": "optional plain text"
  }]
}
```
Optional variant field `"scene"` (lets the same dashboard draw non-launch-site scenes; absent = launch-site defaults):
```jsonc
"scene": {"frame": "ENU" | "LVLH" | "LAUNCH",          // label only
          "axis_labels": ["East [m]", "North [m]", "Up [m]"],
          "views2d": [{"title": "Side view", "x": "downrange" | 0 | 1 | 2, "y": 0 | 1 | 2}],   // axis indices into pos
          "ground": true,                              // draw z = 0 ground plane in 3D
          "glyph": "rocket" | "spacecraft",
          "glyph_length_m": 40}
```
Extra object kinds: `"moving"` (pos time-aligned with `t`, drawn as trail + current-time marker) and
`"sphere"` (`center` [x,y,z] + `radius`, drawn as a translucent wireframe sphere, e.g. a keep-out zone).

Dashboard (`viz/`): `python -m viz.build_dashboard` reads `results/` and writes **one self-contained HTML file**
`GNC_Dashboard.html` (plotly.js embedded, works offline) with: demo selector, variant selector, a **2D/3D toggle**
(2D = altitude-vs-downrange side view + ground track; 3D = 3D scene with trajectory, waypoints/path and an animated rocket
oriented by `quat`), a time slider / play button, and the `plots`, `metrics`, `tables` panels.

## Testing
Every module gets pytest tests under `tests/`. `.venv/bin/pytest -q` must pass from the repo root.
Independent checker agents will run and extend these tests after each worker finishes.

## Scope note
Rocket moving-target intercept (PN / pursuit of a moving target) is **out of scope** for this vehicle and must not be
implemented anywhere in this repo. Problem statement (i) of the brief is pending a decision by the user.
