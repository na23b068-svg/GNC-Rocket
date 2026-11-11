# 6-DOF simulation model (sim-core)

Code: `gnc/frames.py`, `gnc/atmosphere.py`, `gnc/rocket.py`, `gnc/dynamics.py`, `gnc/sensors.py`, `gnc/sim.py`.
Conventions are in `ARCHITECTURE.md` (ENU inertial frame at the pad, body $x$ to the nose, scalar-first
quaternion $q$ with $v_{ENU} = R(q)\,v_b$, stations from the nose tip, positive aft).

## State and equations of motion

$x = [\,p,\ v,\ q,\ \omega_b\,] \in \mathbb{R}^{13}$, input $u = [\delta_p, \delta_y]$ (gimbal angles).

$$
\dot p = v,\qquad
\dot v = \frac{1}{m(t)} R(q)\,F_b - g\,\hat z,\qquad
\dot q = \tfrac12\, q \otimes [0,\ \omega_b],\qquad
J(t)\,\dot\omega_b = M_b - \omega_b \times J(t)\,\omega_b
$$

with $J = \mathrm{diag}(I_{xx}, I_{yy}, I_{yy})$ about the instantaneous CG and $g = 9.80665$ m/s² (flat Earth,
constant $g$). Variable mass enters through $m(t)$, $x_{cg}(t)$, $J(t)$ and jet damping; the
$\dot J\omega$ term and Coriolis effects of internal flow other than jet damping are neglected.

**Mass properties.** Burned fraction $f(t) = I(t)/I_{tot}$ with $I(t) = \int_0^t T\,dt$ (exact for the
piecewise-linear RASP curve). $m, x_{cg}, I_{xx}, I_{yy}$ are interpolated linearly between wet ($f=0$) and dry
($f=1$) values; $\dot m = T(t)\, m_{prop}/I_{tot}$.

## Forces and moments (body frame, about the CG)

Air-relative velocity $v_r = R^T (v - w(t, z))$, $V = |v_r|$, $\bar q = \tfrac12\rho V^2$,
$M = V/a$, total angle of attack $\alpha_t = \operatorname{atan2}(\sqrt{v_{r,y}^2 + v_{r,z}^2},\ v_{r,x})$.
Reference area $S = \pi d^2/4$ (from params), reference length $d$.

| Term | Model |
|---|---|
| Thrust | $F_T = T(t)\,[\cos\delta_y\cos\delta_p,\ \sin\delta_y\cos\delta_p,\ -\sin\delta_p]$, $\delta$ = command + misalignment bias |
| Thrust moment | $M_T = r_g \times F_T$, $r_g = [-(x_g - x_{cg}),0,0]$ → $M_{T} = [0,\ \ell_g F_{T,z},\ -\ell_g F_{T,y}]$ |
| Axial drag | $F_x = -\bar q S\, C_D(M)\, v_{r,x}/V$ ($= -\bar q S C_D\cos\alpha_t$; $C_D$ power-on while $T>0$) |
| Normal force | $F_{y,z} = -\bar q S\, C_{N\alpha}(M)\,[v_{r,y}, v_{r,z}]/V$, i.e. $|F_N| = \bar q S C_{N\alpha}\sin\alpha_t$ in the plane of $\alpha_t$ (valid for any attitude) |
| Restoring moment | $F_N$ at $x_{cp}(M)$: $M = [0,\ \ell_{cp} F_z,\ -\ell_{cp} F_y]$, $\ell_{cp} = x_{cp} - x_{cg}$ |
| Pitch/yaw damping | $M_{y,z} = \bar q S d\, C_{mq}\, \frac{(q, r)\, d}{2V}$ |
| Roll damping | $L = \bar q S d\, C_{lp}\, \frac{p\, d}{2V}$ |
| Jet damping | $M_{y,z} = -\dot m\, \ell_g^2\, (q, r)$ |
| Gravity | $-m g \hat z$ (ENU) |

Sign check (tested): $\delta_p > 0$ ⇒ $F_{T,z} < 0$ at the aft pivot ⇒ $M_y < 0$ ⇒ the nose swings towards
$+z_b$; $\delta_y > 0$ ⇒ $M_z < 0$ ⇒ nose swings towards $-y_b$.
On the rail the rocket points along $x_b$, $y_b = \hat z \times \hat h$ (horizontal, North for an East launch),
$z_b = x_b\times y_b$ (up-range); hence a positive body pitch rate $q$ tilts the nose further down-range.

Aerodynamic tables ($C_D$ on/off, $C_{N\alpha}$, $x_{cp}$ vs Mach) come from OpenRocket via
`data/rocket_params.json` and are linearly interpolated (clamped outside the table). Roll is not
controllable (single gimbal) and is only passively damped.

## Environment

* **ISA 1976** to 84.8 km; geometric → geopotential altitude $H = R_E h/(R_E + h)$; site elevation 850 m ASL is
  added to $z$. $\rho$ and $a$ feed the aerodynamics; pressure feeds the barometer.
* **Wind** $w(t,z) = w_0\,(\max(z, z_{ref})/z_{ref})^{n} + w_g(t) + \sum$ 1-cos gusts. $w_0$ is the steady ENU
  wind at $z_{ref}=10$ m, $n$ the power-law shear exponent (1/7 typical, 0 = uniform). Turbulence $w_g$ is a
  seeded first-order Gauss–Markov process per axis (std $\sigma$ horizontal, $\sigma/2$ vertical, correlation
  time $\tau$), pre-generated on a 0.02 s grid and linearly interpolated so RK4 sees a smooth signal.
  Discrete gusts: $A\,\tfrac12(1-\cos(2\pi (t-t_0)/T_g))$.

## Simulation (`simulate`)

* Fixed-step **RK4**, $\Delta t = 1$ ms during rail/powered/coast; quaternion renormalised each step.
* **Rail**: translation constrained to the rail direction (6° from vertical towards azimuth 90°), no rotation, no
  friction, no sliding back. **Liftoff** = first instant the along-rail net force is positive (thrust > weight
  component + drag); **rail exit** when the travelled distance reaches the rail length (5.4864 m).
* **Actuator chain** (controller only): command → transport delay → clip to gimbal limit → first-order lag
  $\tau$ (exact discretisation) → rate limit → clip. $u$ is held constant over each RK4 step; a command computed
  at $t$ reaches the actuator input at $t + D + \Delta t$ ($D$ = `actuator_delay_s`; the extra $\Delta t$ is the
  zero-order-hold step).
* **Controller** called at `controller.dt` from $t=0$ with the `meas` dict (latest new sample per sensor since the
  last call, delivered `estimation_delay_s` after it was taken, `None` if none).
* **Events**: liftoff, rail_exit, burnout, apogee ($v_z$ sign change, interpolated), drogue (= apogee),
  main ($z \le 300$ m AGL), landing ($z = 0$, interpolated).
* **Recovery** (optional): after apogee a 3-DOF point mass with drag area $C_DA$ (drogue 0.55 m² ≈ 25 m/s,
  main 10 m² ≈ 5.5 m/s) drifting with the wind, $\Delta t = 0.05$ s; attitude frozen. Without recovery and
  `stop_at="ground"`, the 6-DOF ballistic flight continues to impact.

## Sensor models

Output = truth + constant bias + $\mathcal N(0, (k\,\sigma)^2)$, $k$ = `sensor_noise_scale`; independent seeded
streams per sensor (deterministic for a given `seed`).

| Sensor | Truth | σ (default) | Rate | Typical part |
|---|---|---|---|---|
| Accelerometer | specific force $R^T(\dot v + g\hat z)$ (incl. rail reaction) | 0.08 m/s² /axis | 200 Hz | ICM-42688 / BMI088 |
| Gyro | $\omega_b$ | 0.002 rad/s /axis (0.11 °/s) | 200 Hz | same IMU |
| Barometer | ISA $p(z)$, inverted to altitude AGL | 6 Pa (≈0.5 m) | 50 Hz | MS5611 / BMP388 |
| GPS position | $p$ (ENU) | 1.5, 1.5, 3.0 m (E, N, U) | 10 Hz | u-blox M8/M9 |
| GPS velocity | $v$ (ENU) | 0.15 m/s /axis | 10 Hz | u-blox M8/M9 |
| Magnetometer | $R^T b$, $b$ unit field (decl. 9.5° E, incl. 58°) | 0.005 /axis, re-normalised | 50 Hz | LIS3MDL / MMC5983 |

`meas["t_launch_detect"]` is the time of the first delivered accelerometer sample with $f_x > 2g$ (all samples
since the previous controller call are scanned). On the pad and rail the accelerometer includes the rail/pad
reaction (reads $+g$ along up at rest).

## Validation

* Unit tests (`tests/test_sim_*.py`): quaternion algebra, ISA at 0 and 11 km, impulse/mass integration,
  free fall, drag direction, symmetric vertical flight stays vertical, TVC sign and magnitude, restoring and
  damping signs, sensor noise statistics and rates, determinism, controller timing, actuator limits/delays,
  recovery events.
* **Cross-check vs OpenRocket** (`figures/passive_openrocket.png`, same 4 m/s wind as the OpenRocket run):
  apogee 3098 m vs 3067 m (+1.0 %), time to apogee 24.36 vs 24.24 s, max speed 300.1 vs 297.9 m/s (+0.7 %),
  off-rail speed 30.56 vs 30.57 m/s. Zero-wind apogee: 3168 m AGL (10 395 ft).

## Assumptions and limitations

Flat, non-rotating Earth with constant $g$; rigid airframe (no bending/slosh); linear normal-force slope
($C_{N\alpha}\sin\alpha_t$, no body-lift nonlinearity at large $\alpha$) and Mach-only aero tables (no Reynolds or
roughness dependence beyond what OpenRocket baked in); axial drag independent of $\alpha$ except the
$\cos\alpha_t$ projection; no fin cant / roll forcing; thrust independent of ambient pressure; nozzle at the gimbal
station; recovery descent is point-mass with instantaneous deployment (no opening shock); sensor noise is white
(no GPS correlated errors/latency, no baro transonic error, no IMU scale-factor/misalignment errors).
