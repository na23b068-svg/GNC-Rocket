# Rocket design: TVC M-class sounding rocket (OpenRocket 23.09)

The rocket is a single stage, 6 in (156.7 mm) fiberglass airframe, 2.747 m long, flying an **AeroTech M1845NT
(Blue Thunder, RMS-98/7680)**. On the 4 m/s Midland reference sim it reaches **3,067 m AGL (10,064 ft)**. The
gimbal is at the nozzle throat. Thrust vectoring has authority for the 4.4 s burn. Fins keep the coasting flight
passively stable.

Model: `openrocket/tvc_sounding_rocket.ork`. It opens in the OpenRocket 23.09 GUI with the motor configured. Two
simulations are stored with their results: the 4 m/s reference run and a zero-wind run.
Reproduce: `openrocket/setup.sh` downloads the jar if it is missing and runs `openrocket/run_openrocket.py`.
The motor trade study is `openrocket/trade_study.py`, which writes `openrocket/trade_study.csv`.

## Key numbers (OpenRocket reference sim)

| Quantity | Value | Requirement |
|---|---|---|
| Apogee AGL, 4 m/s wind (reference) | **3,067 m (10,064 ft)**, +0.6 % | 2,950–3,140 m |
| Apogee AGL, zero wind | 3,139 m (10,298 ft), +3.0 % | noted |
| Off-rail velocity (5.486 m rail) | **30.6 m/s (100.3 ft/s)** at t = 0.37 s | ≥ 30.5 m/s |
| Static margin at rail exit (AoA 0, rail-exit Mach and CG) | **2.51 cal** | 2–3 cal |
| Stability during ascent, OpenRocket, zero wind | 2.51 → 3.61 cal (max at burnout) | 1.5–4 |
| Stability during ascent, OpenRocket, 4 m/s wind | 1.62 (rail exit, about 7.5° wind AoA) → 3.61 cal | 1.5–4 |
| Burn time / burnout | 4.44 s (OpenRocket estimate) / 4.73 s (end of curve) | ≥ 4 s preferred |
| Max velocity / Mach | 298 m/s / **M 0.89** | |
| Max acceleration (powered) | 100 m/s² (10.2 g) | |
| Time to apogee | 24.3 s | |
| Apogee drift / landing | 942 m E / 247 m E of the pad | |
| Descent | drogue 0.91 m at apogee (about 20 m/s), main 2.44 m at 300 m AGL (5.4 m/s at landing) | |

## Launch conditions

- **Site.** Midland, TX: 31.99 N, 102.08 W, 850 m ASL. The city is at about 847 m and the Midland
  International airport at 875 m. Atmosphere is OpenRocket ISA.
- **Rail.** 18 ft (5.4864 m) rail, 6° from vertical, tilted toward the east (azimuth 90°).
- **Wind.** Steady 4 m/s **from the east**, with zero turbulence, so the rail points into the wind. In ENU this is
  `[-4, 0, 0]` m/s, and it is stored as `openrocket_reference.wind_enu_mps`.
- **Integrator.** OpenRocket RK4, 10 ms step.
- **Wind direction in the GUI.** The `.ork` format does not store wind direction. OpenRocket's default of 90°
  (wind from the east) is the value used here, so the GUI run reproduces the reference sim.

## Airframe and mass budget

Stations are measured from the nose tip. The components are those of the OpenRocket model.

| Component | Mass [kg] | Station fwd–aft [m] |
|---|---|---|
| Nose cone: Von Kármán (LD-Haack), 0.65 m (4.1 cal), 2.5 mm fiberglass, 0.15 m shoulder | 1.35 | 0.000–0.650 |
| Payload bay tube, 0.35 m | 0.68 | 0.650–1.000 |
| **Payload mass block** (just below the nose cone) | **4.00** | 0.680–0.880 (CG 0.780) |
| Payload bulkhead (9.5 mm birch ply) | 0.11 | 0.990 |
| Main parachute bay tube, 0.45 m | 0.87 | 1.000–1.450 |
| **Recovery: main parachute** (2.44 m toroidal, CD 2.2) | 0.85 | 1.030–1.280 |
| **Recovery: main harness** (9 m 11 mm tubular nylon + links) | 0.40 | 1.300–1.420 |
| Avionics bay / switch band tube, 0.20 m, plus 2 bulkheads | 0.39 + 0.22 | 1.450–1.650 |
| **Avionics** (flight computer, IMU, GPS, batteries, sled) | 1.40 | 1.475–1.675 |
| Drogue bay tube, 0.30 m | 0.58 | 1.650–1.950 |
| **Recovery: drogue** (0.91 m, CD 1.5) and **harness** | 0.25 + 0.30 | 1.680–1.910 |
| Booster / fin can tube, 0.797 m | 1.55 | 1.950–2.747 |
| Motor mount / gimbal sleeve, 3 centering rings, thrust plate | 0.55 + 0.11 + 0.15 | 2.115–2.747 |
| **TVC gimbal mount and 2 servos** (around the nozzle) | 1.20 | 2.607–2.727 |
| Fins (4) | 1.29 | 2.437–2.737 |
| Rail buttons (2 × 1515) | ~0 | 2.150, 2.687 |
| **Motor** AeroTech M1845NT: 6.682 kg loaded, 3.772 kg propellant | 6.68 | 2.150–2.747 |
| **Total: wet / dry (dry includes the empty motor case)** | **22.92 / 19.15** | CG 1.778 / 1.646 |

Inertias about the CG come from OpenRocket's mass calculator:

- Wet: Iyy = 13.75 kg m², Ixx = 0.0866 kg m².
- Dry: Iyy = 11.61 kg m², Ixx = 0.0821 kg m².

**Fins.** Four clipped-delta fins:

- Root chord 300 mm, tip chord 100 mm, span 155 mm, leading-edge sweep 200 mm (52°).
- 6.35 mm (1/4 in) G10 fiberglass, airfoiled cross-section, 8 mm fillets.
- Planform area 0.031 m² per fin.
- Trailing edge 10 mm forward of the aft end, so the gimbal ring and nozzle are clear.
- Placed at 45° to the rail buttons.

**Gimbal pivot.** `gimbal_station_m` is 2.697 m: the nozzle throat, taken as 50 mm forward of the nozzle exit. The
motor sits flush with the aft end. The lever arm about the CG is 0.92 m wet and 1.05 m dry.

## Motor selection (GNC trade)

The trade study flew every AeroTech and Cesaroni M motor between 5 and 10.3 kNs that is in OpenRocket's database.
Each motor flew in this airframe, with the mount sized to the motor, under the Midland reference conditions.
Of the 67 motors flown, only the first three rows below meet both the apogee band and the 30.5 m/s rail rule.

| Motor | Ø | I [Ns] | Burn [s] | Apogee [m] | Off-rail [m/s] | Verdict |
|---|---|---|---|---|---|---|
| **AeroTech M1845NT** | 98 | 8,093 | **4.44** | 3,067 | 30.6 | **chosen**: AeroTech, longest burn that meets everything |
| Cesaroni 8429M2020-P | 75 | 8,429 | 4.17 | 3,062 | 33.2 | meets everything; backup if more rail margin is wanted |
| Cesaroni 8634M6400-P | 98 | 8,634 | 1.36 | 3,115 | 54.3 | meets everything, but 1.4 s burn leaves almost no TVC time |
| AeroTech M2000R | 98 | 9,186 | 4.58 | 3,248 | 28.6 | rail velocity too low, apogee too high |
| AeroTech M1850W | 75 | 7,366 | 5.41 | 2,657 | 29.4 | apogee −13 %, rail velocity too low |
| Cesaroni 7545M1590-P | 75 | 7,561 | 4.74 | 2,796 | 31.3 | apogee −8 % |
| Cesaroni 9977M2245-P | 75 | 9,978 | 4.24 | 3,483 | 32.6 | apogee +14 % (needs about 3 kg of ballast, which breaks the rail rule) |
| AeroTech M1419W / M1315W | 98/75 | 7,674 / 6,645 | 7.1 / 5.8 | 2,823 / 2,443 | 22.0 / 27.0 | long burn, but too little thrust to leave the rail |
| AeroTech M2500T | 98 | 9,573 | 3.88 | 3,450 | 33.8 | apogee too high, burn < 4 s |

Long-burn motors (W, H and DM propellants with 5–13 s burns) average only about 0.7–1.4 kN. They cannot reach
30.5 m/s on an 18 ft rail with a 23 kg rocket. The M1845NT is the best compromise: about 1.8 kN average, with a
2.1–2.4 kN plateau for the first 2.3 s.

The airframe was then tuned so the M1845NT meets both rules. The trims were the fin thickness, the main parachute
size, and avionics and TVC masses of 1.4 and 1.2 kg. The off-rail velocity of 30.6 m/s leaves only a small margin.
If the real build comes out more than about 0.3 kg heavy, either:

- trim mass, or
- switch to the Cesaroni M2020 (33 m/s off the rail, with a 0.27 s shorter burn).

**Thrust curve.** `data/motor.eng` is the RASP file that OpenRocket's database holds for the M1845NT. That is the
thrustcurve.org manufacturer file, simfile `5f4294d20002e900000006d3`, and it was checked point by point against the
thrustcurve.org download. It integrates to 8,093 Ns. The certified impulse listed on thrustcurve.org is 8,307 Ns.
The simulator should use the curve.

*Note:* OpenRocket's database lists "8212M1630-P" under Cesaroni. thrustcurve.org lists it as an AMW Tamed Tiger, so
it was left out of the study.

## How the parameters were extracted

### Driving OpenRocket

OpenRocket 23.09 runs headless in-process through JPype, in `openrocket/orkit.py`:

- **Start-up.** Same Guice modules as `SwingStartup`: `GuiModule` and `PluginModule`, then `Application.setInjector`
  and `startLoader()`.
- **Headless databases.** The blocking database providers try to open a Swing progress dialog, so with
  `java.awt.headless=true` the script polls until the motor and preset databases have loaded.
- **Logging.** `openrocket/logback.xml` silences the debug logging.
- **Building.** The rocket is built with the OpenRocket component API: `NoseCone`, `BodyTube`, `InnerTube` (the motor
  mount), `TrapezoidFinSet`, `MassComponent`, `Parachute`, `ShockCord`, `Bulkhead`, `CenteringRing` and `RailButton`.
  The motor comes from the built-in database.
- **Simulation and saving.** `Simulation.simulate()` runs the flight and `GeneralRocketSaver` saves the `.ork`.
- **Reload check.** The test suite reloads the `.ork` with `GeneralRocketLoader` and checks that it reproduces the
  same apogee.

### What each output comes from

- **Mass, CG and inertia.** `MassCalculator.calculateLaunch` gives the wet values and `calculateBurnout` gives the
  dry values (burnout includes the empty motor case). Inertias are taken about the CG.
- **Aero tables.**
  - Source: `BarrowmanCalculator.getAerodynamicForces` at AoA = 0 on a Mach grid of 0–1.5 in steps of 0.05.
  - Reynolds number: ISA at 2,350 m ASL, the middle of the ascent. The M = 0 row is evaluated at M = 0.02, because
    the friction model needs V > 0.
  - `cd_power_off` is OpenRocket's total CD: friction + pressure + base.
  - `cn_alpha` is CNα per radian. `cp_m` is the CP station.
  - `cd_power_on = cd_power_off − CD_base·(d_exit/d)²`, with an estimated 65 mm nozzle exit. OpenRocket 23.09
    applies full base drag even while thrusting, so its own reference sim effectively flies power-off drag
    throughout. The correction is about 5–7 % of CD, so a 6-DOF that uses it will reach an apogee slightly higher
    than OpenRocket's.
- **Damping derivatives.** These use strip theory and OpenRocket's own normal-force slopes. They are averaged over
  M 0.1–0.9, with reference length d. The convention is
  `M = q̄·S·d·C·(ω·d/2V)`, with ω = q for pitch and p for roll.
  - **cmq = −880.** This is `−2 Σ CNα_i ((x_cp,i − x_cg)/d)²` over OpenRocket's per-component force analysis
    (nose cone and fins), with x_cg the mean of the wet and dry CG. It ranges from −946 to −850 across Mach.
  - **clp = −33.7.** This is `−2 N CNα_1fin y_eff²/d²`, with CNα_1fin = CNα_finset/2 taken from OpenRocket
    (includes body interference). y_eff² is the chord-weighted mean square fin radius. It ranges from −36.5 to −32.2.
  - OpenRocket's internal roll-damping model gives −97. That value uses the 2-D lift slope 2π/β, so it overstates
    damping for these low-aspect-ratio (AR 1.55) fins. It is shown in the script output for comparison only.
- **Rail-exit static margin.** `(CP(M_rail, AoA 0) − CG(t_rail))/d`. OpenRocket's in-flight "stability" column
  uses the actual AoA. With 4 m/s wind that AoA is about 7.5° at rail exit, so the column reads lower: 1.62 cal.
- **Flight data.** `data/openrocket_flight.csv` has 28 columns from the reference simulation, in SI units: time,
  altitude AGL, east and north position, vertical, lateral and total velocity, accelerations, Mach, AoA,
  orientation, rates, stability, CG, CP, thrust, drag, CD, base CD, mass, inertias, p, T and wind. It runs from
  ignition to ground hit.
- **Figures.** `figures/openrocket_profile.png`, `openrocket_trajectory.png` and `openrocket_stability.png`.

## Deviations and caveats

- **Off-rail velocity margin.** It is 30.6 m/s against the 30.5 m/s requirement, so the margin is thin; see the
  motor selection notes above.
- **Zero-wind apogee.** At 3,139 m it sits exactly at the +3 % edge. The 4 m/s reference run, which the requirement
  applies to, is at +0.6 %.
- **Gimbal clearance.** OpenRocket cannot model a gimballed motor. The motor is modelled fixed and on-axis, so the
  OpenRocket flight is the "gimbal at 0" passive case. A ±5° gimbal about the throat needs about ±50 mm of lateral
  clearance at the forward end of the motor. The 6 in airframe has about 27 mm, so a real build would need either
  a pivot nearer the motor's mid-length or a smaller gimbal limit. That is a mechanical issue only; it does not
  change the parameters here.
- **Non-schema keys.** `openrocket_reference` carries extra keys beyond the schema: `wind_enu_mps`,
  `apogee_m_agl_zero_wind`, `burnout_time_s`, `max_accel_mps2`, `stability_cal_min_ascent` and
  `stability_cal_max_ascent`. `aero` also carries `damping_convention`, matching sim-core's placeholder.
