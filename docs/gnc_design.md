# GNC design — TVC M-class sounding rocket

Vehicle: 6 in TVC sounding rocket, AeroTech M1845NT (8093 N s, 4.73 s burn), 3168 m AGL (10 394 ft) passive
apogee, Midland TX, 18 ft rail at 6 deg toward East. All numbers below are from `data/rocket_params.json` and the
6-DOF model in `gnc/dynamics.py`. Each number is reproducible with the functions named next to it.

## 1. Mapping to the brief

| Brief item | Where | Section |
|---|---|---|
| Linearise nonlinear dynamics about operating points | `gnc/linearize.py` (numeric error-state Jacobian + analytic model) | 3 |
| Controllability / observability checks | `gnc/analysis.py::rank_test, gramian` | 4 |
| PID tuned by pole placement, transient specs, with/without disturbance | `gnc/pid.py::place_pid, GainSchedule`; `analysis.closed_loop` | 5 |
| LQR comparison (additional) | `gnc/lqr.py` | 10 |
| Closed loop with estimation & actuation delay | `analysis.loop_tf, margins, delay_sweep, critical_delay` (Pade) | 6 |
| Luenberger vs Kalman under Gaussian noise, varying variance | `gnc/observers.py`; nav filter in `gnc/flight_computer.py` | 7 |
| LOS guidance, path following of waypoints, with/without disturbance | `gnc/guidance.py::LOSPathGuidance` | 8 |
| Metrics: cross-track, LOS angle, control effort | flight-computer `log` | 8, 10 |
| Robustness: bias, disturbances, noise levels | sim `SimConfig` knobs + table | 9 |

Target tracking / intercept of a moving target (brief item 4(i)) is handled separately and is not part of this
vehicle's GNC design. No PN or pursuit guidance is implemented here.

## 2. Architecture and the key physical constraint

```
 sensors (IMU 200 Hz, baro 50 Hz, mag 50 Hz, GPS 10 Hz, Gaussian noise + bias)
   -> NAVIGATION  attitude: strapdown gyro (bias averaged on the rail) + magnetometer correction (Luenberger on SO(3))
                  position/velocity: 9-state [p, v, b_a] linear filter, KF or Luenberger, accel as input
   -> GUIDANCE    "none" = attitude hold | "los_path" = 3D lookahead LOS -> lateral accel a_cmd (perp. to v)
   -> accel -> attitude: + gravity compensation, AoA alpha_c = |a| / (f_x + qbar S CNa / m), alpha_c <= alpha_hold
   -> CONTROL     body-frame attitude error -> pitch PID (delta_p), yaw PID (delta_y), gain-scheduled vs time
   -> gimbal (+-5 deg, 20 ms servo lag, 150 deg/s rate limit, in the simulator)
```
Pitch (about y_b) and yaw (about z_b) are decoupled: the coupling norm in the numeric 12-state Jacobian is
< 2e-5 of the in-plane block. Roll cannot be controlled by a single gimbal. Roll is only passively damped
(clp), and because errors are formed in the body frame, roll does not corrupt the loops.

**TVC has authority only while the motor burns.** Control runs from rail exit (0.37 s, 31 m/s) to the start of
the thrust tail-off, `tvc_end_time` = last time T >= 0.5 * mean thrust = **3.97 s**. After that the gimbal is
centred and the rocket coasts ballistically to apogee (~24.7 s). The fins make the airframe statically stable
(2.5 cal off the rail). The gimbal therefore has to beat the aerodynamic restoring moment to hold any angle of
attack. The AoA it can hold with 5 deg is alpha_hold = |b| delta_max / a0:

| t [s] | 0.37 | 0.82 | 1.27 | 1.72 | 2.17 | 2.62 | 3.07 | 3.52 | 3.97 |
|---|---|---|---|---|---|---|---|---|---|
| qbar [kPa] | 0.5 | 2.8 | 7.1 | 13.7 | 22.0 | 30.6 | 37.4 | 43.2 | 47.6 |
| alpha_hold [deg] | 29 | 29 | 18 | 8.7 | 5.0 | 2.7 | 1.8 | 1.4 | 0.7 |
| max lateral accel [m/s^2] | 50 | 64 | 56 | 38 | 30 | 20 | 16 | 14 | 8 |

(`design_schedule(rocket)[1]["alpha_hold_deg"], ["a_lat_max"]`). **Consequence for guidance:** all large
manoeuvres must happen in the first ~2 s. Above ~30 kPa the TVC can only trim. A path is feasible only if its
required lateral acceleration stays under this envelope. Guidance clips alpha_c to 60 % of alpha_hold and keeps
40 % in reserve for regulation.

## 3. Modelling and linearisation

**Nonlinear model** (sim-core, `gnc/dynamics.py`): 13-state rigid body with variable mass. The terms are thrust
through the gimbal pivot, axial drag, normal force CNa(M) at CP(M), pitch damping cmq, jet damping and gravity.

**Numeric linearisation** (`linearize.error_state_jacobian`). Error state e = [dp, dv, dphi, domega] (12) with
the attitude perturbation as a body-frame rotation vector, q = q0 (x) exp(dphi/2). The quaternion norm constraint
therefore never enters the Jacobian. Error kinematics:
dphi_dot = 2 vec(q0* (x) (q_dot(x) - q_dot0 (x) dq)). The Jacobian is taken by central differences straight
through `dynamics.derivatives`, so sign conventions cannot drift. dp and dv are then rotated into the nominal body
axes (`to_body_axes`, including the -[omega0 x] term), which splits the system into two 4-state planes:

pitch: x = [z_b, w, theta, q], u = delta_p;  yaw: x = [y_b, v, psi, r], u = delta_y (mirror image).

**Analytic plane model** (`linearize.analytic_plane`), with N = qbar S CNa, F_x = T - qbar S Cd,
l_cp = x_cp - x_cg (static margin), l_g = x_gimbal - x_cg:
```
z'     = w
w'     = -(N/(mV)) w - ((F_x + N)/m) theta - (T/m) delta
theta' = q
q'     = -(l_cp N/(I V)) w - (l_cp N / I) theta + (qbar S d^2 cmq/(2 V I) - mdot l_g^2 / I) q - (l_g T / I) delta
```
This matches the numeric Jacobian to ~1e-10 relative at every tested point (`tests/test_gnc_linearize.py`).
Attitude-loop reduction: theta'' + a1 theta' + a0 theta = b delta, with a1 = -q'_q (aero + jet damping),
a0 = -q'_theta (weathercock stiffness, > 0) and b = q'_delta (< 0: positive delta_p pitches the nose
"down-range-negative").

**Operating points** (`operating_points`): 9 points from rail exit to the TVC end on the passive, noiseless,
calm-air nominal trajectory. Eigenvalues of the frozen pitch-plane model, in the raw states [z_b, w, theta, q]
and in AoA form [z_b, w/V, theta, q] (`linearize.aoa_form`):

| t [s] | 0.37 | 0.82 | 1.27 | 2.17 | 3.07 | 3.97 |
|---|---|---|---|---|---|---|
| short period | -1.0 +- 2.3j | -1.1 +- 4.5j | -1.3 +- 7.2j | -1.8 +- 13.6j | -2.2 +- 19.3j | -2.4 +- 23.2j |
| real mode, frozen w-model | +1.56 | +1.22 | +0.91 | +0.50 | +0.23 | +0.06 |
| Vdot / V | 2.81 | 1.29 | 0.88 | 0.47 | 0.20 | 0.03 |
| real mode, AoA form | +0.21 | +0.13 | +0.084 | +0.048 | +0.037 | +0.032 |
| g / V | 0.32 | 0.14 | 0.087 | 0.049 | 0.037 | 0.033 |
| lateral position | 0 | 0 | 0 | 0 | 0 | 0 |

**The positive real eigenvalue of the frozen w-model is mostly a frozen-coefficient artefact, not a physical
divergence.** Freezing the coefficients holds a lateral *velocity* w fixed while the boost accelerates the
rocket (Vdot/V up to 2.8 /s at rail exit). In reality the same w becomes a shrinking AoA w/V. Rewriting the
state as w/V adds the time-varying term d(w/V)/dt = w'/V - (Vdot/V)(w/V). The eigenvalue then drops to
0.21 -> 0.03 /s, which is approximately g/V. That residual is the genuine but slow gravity-turn divergence of
the flight-path angle: near-vertical flight is an unstable equilibrium with rate g cos(phi)/V. Its time
constant is 5 s to 30 s, longer than the boost. An independent nonlinear check (`tests/checks/`) agrees. Seeded
along the frozen eigenvector, the nonlinear attitude perturbation stays flat (1e-3 -> 1.9e-3 rad, no further
growth after 0.5 s), while the frozen model predicts 4.8x growth by 1 s. The short-period pair is the same in
both forms. The attitude design (section 5) uses only the short-period reduction (a1, a0, b), so it does not
depend on this mode. The frozen 4-state model is used only as a conservative stability cross-check.

## 4. Controllability and observability

`analysis.rank_test` computes the Kalman rank on the **block-normalised** Kalman matrix: every block C A^k
(obsv) or A^k B (ctrb) is scaled to unit norm. This does not change rank or deficient directions, but it removes
the |A|^k growth. With |A| ~ 500 at max qbar the raw blocks span ~8 orders of magnitude, and an unscaled SVD test
wrongly gave rank 2 for gyro-only at 3.07 s and later. Numerical rank uses a relative singular-value tolerance
of 1e-8, and the condition number is reported. Pitch plane, numeric model, all 9 schedule points
(t = 0.37 ... 3.97 s, tested in `tests/test_gnc_linearize.py`):

| Sensor set (output) | rank at the 9 points | cond (block-normalised) | Comment |
|---|---|---|---|
| ctrb (delta_p) | 4 at all | 3 ... 60 | controllable everywhere; harder to steer at high qbar |
| gyro q | 3 at all | 1e16 ... 3e24 | z_b unobservable. theta is observable from q only through the aero stiffness a0 |
| attitude angle theta | 3 at all | 3e9 ... 1e13 | z_b unobservable |
| GPS velocity | 3 (4 at 0.82 s) | 7e7 ... 1e13 | z_b acts only through the air-density gradient, ~8 orders weaker: practically unobservable. At 0.82 s it sits just above the 1e-8 tolerance |
| GPS position | 4 at all | 1 ... 4 | full |
| gyro + GPS position | 4 at all | 5 ... 9e2 | full |

Navigation model per axis, [p, v, b_a] with the accelerometer as input (`observers.axis_nav_model`): GPS position
or baro -> rank 3. GPS velocity only -> rank 2 (position unobservable). Accelerometer only -> rank 0 (pure dead
reckoning, drift unbounded). Attitude-plane estimator [theta, b_gyro] with the gyro as input
(`observers.attitude_plane_model`): an attitude reference gives rank 2, gyro alone gives rank 0. Hence the
magnetometer correction and the rail-averaged gyro bias. Gramians are available in `analysis.gramian`
(Lyapunov for Hurwitz A, finite-horizon quadrature otherwise, as needed here because of the gravity-turn and
lateral-position modes).

## 5. Attitude control: PID by pole placement

Control law (gyro supplies the rate, so the derivative acts on the measurement and needs no differentiation):
delta = Kp e + Ki int(e) - Kd q_meas, e = theta_cmd - theta. With the plant from section 3 the closed-loop
characteristic polynomial is

  s^3 + (a1 + b Kd) s^2 + (a0 + b Kp) s + b Ki  =  (s + p)(s^2 + 2 zeta wn s + wn^2)

which gives **Kd = (2 zeta wn + p - a1)/b, Kp = (wn^2 + 2 zeta wn p - a0)/b, Ki = p wn^2 / b**
(`pid.place_pid`). A 4-pole variant with a first-order derivative filter is also solved in closed form
(tau_f = 1/(c3 - a1)). Tests check that both reproduce the requested poles exactly, both through the
characteristic polynomial and through an independent python-control interconnection.

**Specs and pole choice.** zeta = 0.7 (4.6 % overshoot of the dominant pair), integral pole p = 4 rad/s, and
wn_i = max(8, 0.8 sqrt(a0_i)). The floor 0.8 sqrt(a0) tracks the airframe's own weathercock frequency. With a
fixed wn = 8 the PID would have to *cancel* the aerodynamic stiffness at high qbar (Kp > 0 there), and the
upper gain margin would fall to 1.6 dB at 3.97 s. With the floor it stays >= 8.4 dB (with 15 ms extra delay)
and >= 14.7 dB (servo lag only). Settling (2 %) of the dominant pair is 0.71 s at wn = 8 and 0.31 s at
wn = 18.5.

Gain schedule (`flight_computer.design_schedule`, interpolated linearly in time since launch). Margins include
the 20 ms servo lag. DM = smallest loop delay that destabilises the loop (Pade order 5, bisection):

| t | qbar kPa | a1 | a0 | b | wn | Kp | Ki | Kd | PM deg | GM dB | DM ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.37 | 0.5 | 0.33 | 3.2 | -148.9 | 8.0 | -0.709 | -1.719 | -0.0998 | 47 | (-17.8) | 55 |
| 0.82 | 2.8 | 0.69 | 18.1 | -159.6 | 8.0 | -0.568 | -1.604 | -0.0909 | 52 | inf | 60 |
| 1.27 | 7.1 | 1.09 | 49.8 | -177.4 | 8.0 | -0.333 | -1.443 | -0.0795 | 61 | inf | 67 |
| 1.72 | 13.7 | 1.51 | 105.1 | -183.4 | 8.2 | -0.044 | -1.467 | -0.0762 | 75 | inf | 73 |
| 2.17 | 22.0 | 1.93 | 185.2 | -184.9 | 10.9 | 0.031 | -2.564 | -0.0936 | 74 | 28.8 | 56 |
| 2.62 | 30.6 | 2.28 | 282.4 | -149.9 | 13.4 | 0.176 | -4.823 | -0.1370 | 72 | 19.4 | 46 |
| 3.07 | 37.4 | 2.54 | 373.0 | -135.2 | 15.5 | 0.353 | -7.062 | -0.1708 | 70 | 16.8 | 40 |
| 3.52 | 43.2 | 2.74 | 460.4 | -125.9 | 17.2 | 0.553 | -9.363 | -0.2009 | 69 | 15.5 | 36 |
| 3.97 | 47.6 | 2.87 | 536.8 | -77.7 | 18.5 | 1.152 | -17.694 | -0.3486 | 68 | 14.7 | 33 |

(Gains are in rad/rad, rad/(rad s) and rad/(rad/s). Their sign is negative because b < 0. A negative GM in dB is
a gain-*reduction* margin.)

**Verification on the full model.** `analysis.plane_pid_closed_loop` closes the PID around the full 4-state plane
model, including its positive real eigenvalue (the conservative frozen-coefficient one from section 3). Every
point is stable: the dominant pair lands near the design (e.g. -9.1 +- 8.2j at 2.17 s), and the integrator pole
merges with that real mode into slow, well-damped poles (-0.1 ... -1.5). The only pole left at 0 is lateral
position, which the guidance loop closes.

**With and without disturbance.** A step disturbance moment (thrust misalignment, or a wind-induced AoA trim)
enters as angular acceleration d. The steady-state attitude error is **0 with integral action** and
d/(a0 + b Kp) without it (`closed_loop(...)["T_dist"]` DC gain: 0 vs 5.6e-3 rad per rad/s^2 at 2.17 s). In
the 6-DOF sim a 0.57 deg thrust misalignment raises the peak hold error from 0.21 deg to 0.73 deg, and it
trims out. Step response at 2.17 s with servo lag: 0 % overshoot, 0.97 s settling. The designed poles
(-7.62 +- 7.77j, -4) are placed exactly on the plant model *without* servo lag. Adding the 20 ms lag (a fourth
pole) moves them to -5.85 +- 1.68j and -20.1 +- 15.3j, still well damped. The response shape also reflects
derivative-on-measurement and the PI zero.

## 6. Delays (actuation and estimation)

In this single SISO loop a transport delay of the command (actuator) and a delay of the measurements
(estimation) both add the same e^{-s tau}, so they are analysed together as the loop delay
tau = tau_act + tau_est, with a Pade approximant (`loop_tf(..., delay, pade_order)`). Pole migration at
2.17 s (servo lag 20 ms):

| loop delay [ms] | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| max Re(pole) | -5.9 | -4.9 | -4.3 | -3.9 | -3.0 | -0.9 | +0.5 |
| dominant zeta | 0.96 | 0.89 | 0.84 | 0.81 | 0.12 | 0.04 | unstable |
| phase margin [deg] | 74 | 61 | 48 | 34 | 21 | 8 | -5 |

Delay margin PM/w_c agrees with the Pade bisection within 5 % (`tests/test_gnc_pid.py`). The tightest point is
the end of the boost (33 ms) because wn is highest there. These continuous-time margins exclude the 100 Hz
controller's sample-and-hold (~5 ms equivalent delay) and the IMU sample age (up to 5 ms at 200 Hz), together
~5-8 ms of hidden loop delay. That is why 40 ms of explicit delay already shows strong oscillatory growth.
The 6-DOF sim (attitude hold, seed 7) agrees. With 15 ms actuator + 15 ms estimation delay the hold error stays
at 0.21 deg and the late-burn gimbal peak-to-peak is unchanged (0.66 vs 0.65 deg). With 30 + 30 ms, which
exceeds the margin late in the burn, the oscillation grows from about 3.5 s: gimbal peak 4.8 deg at 3.93 s,
late-burn peak-to-peak 8.6 deg. The divergence is cut short by the TVC end at 3.97 s. It is not a saturated
limit cycle.

## 7. Observers: Luenberger vs Kalman

**Attitude.** The strapdown quaternion starts from the surveyed rail attitude. The gyro bias is averaged while
the rocket sits on the rail, where it physically cannot rotate (0.37 s, about 74 samples). In flight a
magnetometer correction w_c = k_mag (m_meas x m_pred) is added. This is a Luenberger observer on SO(3) whose
error pole is -k_mag = -0.2 rad/s in the observable directions. The axis along the local field is unobservable
from the magnetometer, which is why gyro integration dominates. Result: attitude-estimate error <= 0.08 deg
nominal, 0.21 deg with gyro/mag/accel/GPS biases, 0.31 deg at 4x noise.

**Navigation** (what guidance consumes). x = [p_enu, v_enu, b_a] (9). Input u = R(q_hat) f_imu + g.
GPS (pos+vel, 10 Hz) and baro (z, 50 Hz) updates (`observers.nav_model`, ZOH + Van Loan discretisation).
- **Kalman**: Q from the acceleration-input noise sigma_a = 0.3 m/s^2/sqrt(Hz), which covers IMU noise *and*
  attitude-error leakage of the ~100 m/s^2 thrust, plus a bias walk of 0.02. R comes from the sensor datasheet
  stds; the baro R is converted from Pa to m at the current density. Joseph-form update. GPS innovation NIS is
  logged: its mean is 4.5 to 5.9 against an expected 6 across the noise sweep, so the filter is consistent and
  slightly conservative. sigma_a = 1.0 gives NIS ~3.9 (too conservative, worse velocity).
- **Luenberger**: fixed gains by pole placement of the discrete error map (I - L C) A at each sensor's own
  update interval. GPS poles 0.8/0.85/0.97 per 0.1 s, baro poles 0.99/0.992/0.999 per 0.02 s. The multirate
  periodic error map (one GPS period) has spectral radius < 0.99 (tested). Faster baro poles amplified noise
  (2 m/s vertical-velocity RMS).

Noise-variance sweep, 15 s flight, attitude hold, seeds 1-3 (`sensor_noise_scale` multiplies every std).
Absolute RMS values are for seed 1. Ratios (Luenberger / KF) and NIS give the range over seeds 1-3:

| noise scale | KF pos RMS E/N/U [m] | Luen. pos RMS E/N/U [m] | pos ratio L/KF E / N / U | vel ratio L/KF E / N / U | KF NIS (6) |
|---|---|---|---|---|---|
| 0.5 | 0.10 / 0.09 / 0.05 | 0.20 / 0.19 / 0.30 | 1.3-2.1 / 1.6-2.7 / 5.9-8.8 | 0.48-0.55 / 0.46-0.58 / 1.06-1.22 | 4.5-4.7 |
| 1 | 0.17 / 0.16 / 0.10 | 0.40 / 0.38 / 0.60 | 1.3-2.3 / 1.7-2.8 / 5.9-8.8 | 0.63-0.70 / 0.61-0.75 / 0.97-1.10 | 5.2-5.4 |
| 2 | 0.33 / 0.31 / 0.20 | 0.80 / 0.75 / 1.21 | 1.3-2.4 / 1.8-2.9 / 5.9-8.8 | 0.88-0.92 / 0.86-0.97 / 1.18-1.34 | 5.7-5.9 |
| 4 | 0.65 / 0.61 / 0.41 | 1.60 / 1.50 / 2.42 | 1.3-2.5 / 1.8-2.9 / 6.0-8.9 | 1.22-1.29 / 1.11-1.25 / 1.41-1.86 | 5.9-6.1 |

Both estimators stay stable at every level, and their errors scale roughly linearly with the noise std.
- **Position:** the KF is better on every axis, seed and level, by **1.3x to 8.9x**. The largest gain is
  vertical, where it weights baro against GPS by their variances. The smallest is East on seed 2, only 1.3x.
- **Horizontal velocity:** the fixed-gain Luenberger is *better*, by about 2x at 0.5x noise, 1.3-1.6x at
  nominal noise and 1.1x at 2x noise. Only at 4x noise does the KF win, by 1.1-1.3x.
- **Interpretation:** together with NIS below 6 at low noise, this says sigma_a = 0.3 is conservative for
  attitude hold. The KF trusts the IMU-propagated velocity less than it could. The value was kept on purpose
  as margin for manoeuvring flight, where attitude-error leakage of the thrust is larger, and was not retuned.

## 8. Guidance: 3D LOS path following

`LOSPathGuidance(waypoints, lookahead_m, lookahead_time, accept_radius, k)`. On the active segment
W_i -> W_{i+1} with unit tangent t:
- along-track s = (p - W_i).t, cross-track vector e = (p - W_i) - s t. Logged: |e|, plus a horizontal part
  (perpendicular to t in the ground plane) and a vertical part.
- lookahead point P = W_i + (s + Delta) t, Delta = max(lookahead_m, lookahead_time |v|). The LOS vector is
  P - p, logged as LOS azimuth/elevation (the 3D form of the classical LOS heading angle).
- command: rotate the velocity vector toward the LOS at rate k * eps (eps = angle between v and LOS),
  a_cmd = k |v| eps e_perp, which is perpendicular to v.
- switching: next segment when |p - W_{i+1}| < R_accept or s >= segment length (the last segment extends).

Acceleration to attitude (flight computer): add gravity compensation, a_lat = a_cmd - g_perp(v). Then the AoA
command is alpha_c = |a_lat| / (f_x + qbar S CNa / m), where f_x is the measured axial specific force. Thrust
tilt and airframe lift both turn the velocity vector. alpha_c is clipped to 60 % of alpha_hold (section 2),
and the commanded nose direction is n = cos(alpha_c) v_hat + sin(alpha_c) a_lat_hat.

Kinematic tests: converges onto a straight line from 47 m off (< 0.5 m), and switches through a 3D dog-leg.
6-DOF (`tests/test_gnc_flight_computer.py::test_los_path_flight`), path (0,0,0) -> (0,30,300) -> (0,90,900)
(a northward bend, cross-range to the 6 deg East rail), seed 7: cross-track RMS 5.2 m nominal, 7.1-10.7 m with
10 m/s wind + 3 m/s gusts + 1/7 shear depending on wind direction, 5.3 m with 0.57 deg thrust misalignment per
axis, 6.4 m with sensor biases (section 9). Paths that bend after
~2.5 s (qbar > 30 kPa) are not flyable: the cross-track error grows once alpha_hold falls to ~2 deg, which is
the TVC authority limit, not a tuning issue. After `tvc_end_time` the guidance keeps logging, but the vehicle
coasts ballistically.

## 9. Robustness summary

6-DOF, simulator at commit 1f3a255, `SimConfig(t_max=4.5, seed=7, ...)`. Path = (0,0,0) -> (0,30,300) ->
(0,90,900). Wind = 10 m/s steady from the stated direction (vector shown, ENU, blowing *toward*), plus gust
{std 3 m/s, tau 1.5 s} and 1/7 power-law shear. Misalignment = `thrust_misalignment_rad=(0.01, 0.01)`
(0.57 deg per axis). Gimbal statistics use the actual (post-servo) gimbal angle between rail exit and the TVC end.

| case | hold: err max [deg] | hold: gimbal max [deg] / sat. frac | LOS path: gimbal max / sat. frac | LOS path: cross-track RMS [m] |
|---|---|---|---|---|
| nominal | 0.21 | 1.7 / 0 | 3.5 / 0 | 5.2 |
| sensor biases (gyro 0.01 rad/s, accel 0.2 m/s^2, mag 0.02, GPS 2 m) | 0.20 | 1.6 / 0 | 4.7 / 0 | 6.4 |
| wind (10, 0, 0) toward E | 1.81 | 5.0 / 0.49 | 5.0 / 0.43 | 10.7 |
| wind (-10, 0, 0) toward W | 1.24 | 5.0 / 0.05 | 5.0 / 0.13 | 7.1 |
| wind (0, 10, 0) toward N | 1.63 | 5.0 / 0.01 | 5.0 / 0.25 | 8.9 |
| wind (0, -10, 0) toward S | 2.05 | 5.0 / 0.48 | 5.0 / 0.42 | 9.6 |
| thrust misalignment (0.01, 0.01) rad | 0.73 | 1.8 / 0 | 3.3 / 0 | 5.3 |
| delay 15 + 15 ms | 0.21 | 1.8 / 0 | 4.0 / 0 | 5.2 |
| delay 30 + 30 ms (> late-burn delay margin) | 1.15 | 4.8 / 0 (growing oscillation after 3.5 s) | 5.0 / 0 | 5.3 |
| 4x sensor noise | 0.21 | 1.7 / 0 | 3.5 / 0 | 5.1 |

Attitude hold in a crosswind spends authority fighting weathercocking (AoA = wind/V is large at low speed), and
the gimbal saturates for up to half of the guided phase. A velocity-vector (guidance) mode is the better choice
in wind.

## 10. LQR vs pole-placement PID

`gnc/lqr.py`, enabled with `FlightComputer(attitude="lqr")`; the default `"pid"` is unchanged and bit-identical.
The LQR uses the same operating points and short-period model, augmented with the integral of the attitude
error: x = [theta, q, xi], xi' = theta - theta_c, delta = -K x, K = R^-1 B^T P from the continuous ARE
(`ct.lqr`). On this model, state feedback with integral action has exactly the PID structure, with
derivative on the gyro rate: Kp = K_theta, Kd = K_q, Ki = K_xi. The loop broken at the plant input is therefore
the same C(s)G(s), and both designs are compared with the same tools (`lqr.compare`, which reuses
`analysis.closed_loop / margins / critical_delay` with the same 20 ms servo lag). Gain scheduling is the same
(9 points, linear in time since launch).

**Weights** (Bryson's rule): Q = diag(1/theta_max^2, 1/q_max^2, 1/xi_max^2), R = 1/delta_max^2. The starting
point theta_max = 1 deg, q_max = 10 deg/s, xi_max = theta_max * 0.5 s, delta_max = 5 deg gave a very fast loop
(real pole at -74 to -90 rad/s) with only 8-20 ms delay margin under the servo lag. **Final:** theta_max =
4 deg, q_max = 40 deg/s, xi_max = theta_max * 0.3 s, delta_max = 5 deg (gimbal limit). These maximise the
worst-case delay margin among the tried weights while keeping >= 43 deg phase margin with lag.

Comparison at every schedule point (servo lag 20 ms in poles, step, PM, GM, DM. "PM no lag" is at the plant
input without lag. DM = smallest destabilising loop delay. "inf" for DM means > 1 s, the bisection limit):

| t [s] | qbar [kPa] | ctrl | Kp | Ki | Kd | slowest pole Re | overshoot % | settling 2 % [s] | PM deg | PM no lag | GM dB | DM ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.37 | 0.5 | PID | -0.709 | -1.719 | -0.0998 | -4.20 | 30.3 | 1.03 | 47 | 65 | (-17.8) | 55 |
| 0.37 | 0.5 | LQR | -1.772 | -4.167 | -0.1964 | -3.59 | 17.2 | 1.09 | 43 | 73 | (-23.1) | 28 |
| 0.82 | 2.8 | PID | -0.568 | -1.604 | -0.0909 | -4.39 | 19.5 | 0.96 | 52 | 70 | inf | 60 |
| 0.82 | 2.8 | LQR | -1.666 | -4.167 | -0.1868 | -3.58 | 13.4 | 1.04 | 44 | 75 | inf | 28 |
| 1.27 | 7.1 | PID | -0.333 | -1.443 | -0.0795 | -5.13 | 2.3 | 0.70 | 61 | 79 | inf | 67 |
| 1.27 | 7.1 | LQR | -1.492 | -4.167 | -0.1741 | -3.51 | 7.0 | 0.89 | 45 | 77 | inf | 27 |
| 1.72 | 13.7 | PID | -0.044 | -1.467 | -0.0762 | -5.13 | 0.1 | 0.94 | 75 | 93 | inf | 73 |
| 1.72 | 13.7 | LQR | -1.248 | -4.167 | -0.1629 | -3.24 | 0.0 | 0.47 | 48 | 80 | inf | 28 |
| 2.17 | 22.0 | PID | 0.031 | -2.564 | -0.0936 | -5.85 | 0.0 | 0.96 | 74 | 87 | 28.8 | 56 |
| 2.17 | 22.0 | LQR | -0.978 | -4.167 | -0.1518 | -2.74 | 0.0 | 1.18 | 51 | 83 | inf | 30 |
| 2.62 | 30.6 | PID | 0.176 | -4.823 | -0.1370 | -6.35 | 0.0 | 0.98 | 72 | 83 | 19.4 | 46 |
| 2.62 | 30.6 | LQR | -0.649 | -4.167 | -0.1414 | -1.90 | 0.0 | 1.95 | 58 | 87 | inf | 36 |
| 3.07 | 37.4 | PID | 0.353 | -7.062 | -0.1708 | -5.78 | 0.0 | 0.98 | 70 | 82 | 16.8 | 40 |
| 3.07 | 37.4 | LQR | -0.472 | -4.167 | -0.1327 | -1.41 | 0.0 | 2.72 | 63 | 91 | inf | inf |
| 3.52 | 43.2 | PID | 0.553 | -9.363 | -0.2009 | -5.48 | 0.0 | 0.98 | 69 | 80 | 15.5 | 36 |
| 3.52 | 43.2 | LQR | -0.364 | -4.167 | -0.1261 | -1.10 | 0.0 | 3.53 | 66 | 95 | inf | inf |
| 3.97 | 47.6 | PID | 1.152 | -17.694 | -0.3486 | -5.35 | 0.0 | 0.98 | 68 | 80 | 14.7 | 33 |
| 3.97 | 47.6 | LQR | -0.200 | -4.167 | -0.1118 | -0.60 | 0.0 | 6.33 | 81 | 91 | inf | inf |

**Reading the table.**
- **Margins without lag:** the LQR has the guaranteed >= 60 deg phase margin (73-95 deg here) and infinite
  upward gain margin at every point (tested).
- **Low and mid dynamic pressure (<= 2.2 s):** the LQR closes a faster rate loop (|Kd| 2x larger), so with the
  20 ms servo lag its delay margin is only 27-30 ms, against 55-73 ms for the PID. Bryson weights do not see
  the lag or the delay.
- **High dynamic pressure:** the LQR "lets the airframe work". Kp shrinks toward 0 because the fins already
  provide the stiffness that the cost would otherwise pay for. Margins grow (inf GM, delay margin > 1 s from
  3.07 s), but the integral mode becomes slow (-0.6 rad/s at 3.97 s, 6.3 s settling). The PID instead pins the
  dominant poles at wn >= 0.8 sqrt(a0) and settles in ~1 s at every point.

6-DOF, seed 7, t_max 4.5 s (same cases as section 9; error = max attitude-control error in the guided phase,
effort = integral of |delta_act|^2 dt over the TVC window in rad^2 s):

| case | PID err max [deg] | LQR err max [deg] | PID effort | LQR effort | PID / LQR LOS cross-track RMS [m] |
|---|---|---|---|---|---|
| attitude hold, nominal | 0.21 | 0.15 | 6e-4 | 3e-4 | - |
| hold, misalignment (0.01, 0.01) rad | 0.73 | 0.28 | 1.2e-3 | 6e-4 | - |
| hold, wind (0, 10, 0) + gust + shear | 1.63 | 0.76 | 1.1e-2 | 6.3e-3 | - |
| hold, delay 15 + 15 ms | 0.21 | 0.77 | 6e-4 | 2.1e-3 | - |
| hold, delay 30 + 30 ms | 1.15 | 6.99 (diverges) | 1.6e-3 | 2.6e-2 | - |
| LOS path, nominal | 3.39 | 3.29 | 5.1e-3 | 1.7e-3 | 5.2 / 4.4 |
| LOS path, misalignment | 3.33 | 3.27 | 5.9e-3 | 2.2e-3 | 5.3 / 4.4 |
| LOS path, wind (0, 10, 0) + gust + shear | 3.34 | 3.27 | 2.1e-2 | 1.1e-2 | 8.9 / 7.5 |
| LOS path, delay 30 + 30 ms | 3.78 | 9.31 | 1.8e-2 | 3.6e-2 | 5.3 / 10.1 |

**Verdict.**
- **LQR strengths:** it is smoother (about half the gimbal effort) and rejects misalignment and wind better
  without delay, with 15-18 % lower cross-track error.
- **LQR weakness:** it is much more delay-sensitive in the early boost. With 15 + 15 ms delay its hold error
  is already 0.77 deg against 0.21 deg for the PID, and at 30 + 30 ms it diverges while the PID stays bounded.
- **Choice:** the pole-placement PID stays the default because its delay robustness is the binding
  requirement (servo + sampling + estimation delays). A natural improvement is an LQR with the servo lag in
  the design model, or loop-transfer recovery.

## 11. API for scenario scripts

```python
from gnc.flight_computer import FlightComputer, design_schedule, tvc_end_time
fc = FlightComputer(guidance="none" | "los_path", nav="kf" | "luenberger", dt=0.01,
                    waypoints=None, los_k=1.5, lookahead_m=150.0, lookahead_time=0.0, accept_radius=40.0,
                    gravity_comp=True, a_max=60.0, alpha_max_deg=10.0, hold="rail" | "vertical",
                    zeta=0.7, wn=8.0, p_int=4.0, kappa=0.8, schedule=None, k_mag=0.2,
                    nav_sigma_a=0.3, nav_sigma_b=0.02, gps_poles=(0.8, 0.85, 0.97), baro_poles=(0.99, 0.992, 0.999),
                    attitude="pid" | "lqr")
res = simulate(rocket, SimConfig(...), fc);  fc.log / res["controller_log"]
```
`log` (one entry per 100 Hz step): `t, phase` (0 rail / 1 guided / 2 coast), `pos_est, vel_est, quat_est,
accel_bias_est, gyro_bias_est, nav_std` (KF 1-sigma, NaN for Luenberger), `u_cmd, att_err` (pitch, yaw rad),
`rate, a_cmd, alpha_cmd, nose_cmd, gains` (Kp, Ki, Kd), `nis_gps` (None if no GPS update). With "los_path" it
also logs `wp_index, cross_track, cross_track_h, cross_track_v, along_track, los_az, los_el, heading_error,
dist_to_wp`. Control effort = sum(u_cmd^2) * dt.
Analysis helpers: `linearize.operating_points / linearize_at / analytic_plane`, `analysis.rank_test / gramian /
margins / delay_sweep / critical_delay / closed_loop / plane_pid_closed_loop / step_metrics`,
`pid.place_pid / poles_from_specs / closed_loop_poles`, `lqr.attitude_lqr / design_lqr_schedule / compare`,
`linearize.aoa_form`, `observers.*` (incl. `attitude_plane_model` for an
offline single-plane Luenberger-vs-KF demo), `guidance.kinematic_path_check`.
