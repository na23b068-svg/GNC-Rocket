# Agent handoff log — OE5310 GNC project

Read this first, then `ARCHITECTURE.md` (binding interface contract) and `GNC_Project_Assignment.pdf` (the brief).
Last updated 2026-11-12 by the orchestrating Claude session.

## What this is
Course project (OE5310, IIT Madras): GNC for an autonomous vehicle, Python only. Vehicle: **thrust-vector-controlled
M-class sounding rocket** (AeroTech M1845NT, ~10,000 ft apogee, Midland TX, 18 ft rail at 6°). Owner is busy with exams;
agents do the work.

## Hard rules
- **Scope:** brief item 4(i) "track and intercept a moving target (PN/LOS)" is implemented as kinematic guidance in
  `gnc/guidance.py` (`InterceptGuidance`, PN and LOS pursuit) and shown in the dashboard via `viz/guidance_demos.py`.
  It is exercised on a point-mass model, not on the 6-DOF vehicle. The earlier prohibition on PN/pursuit/moving-target
  code is lifted; the scope-checking tests were removed in `b587e2e`.
- Python only, venv at `.venv/` (`.venv/bin/python`, `.venv/bin/pytest`). **System python is broken** (numpy 2 vs apt scipy).
- Every worker gets an **independent checker agent**; checker tests live in `tests/checks/test_check_*.py`.
- Only the orchestrator commits; commit at each milestone (end messages with the Co-Authored-By line used in git log).
- Subagent models: **Sonnet for non-STEM work** (UI/HTML/docs formatting), Opus for physics/control/verification.
  Keep agent count lean (user rejected "optional extra parallelism").
- The `ponytail` plugin (lazy-senior-dev mode, level full) is injected into every subagent via its SubagentStart hook.

## Layout (master, `~/Documents/GNC Project`)
| Path | What |
|---|---|
| `openrocket/` | headless OpenRocket 23.09 via JPype (`setup.sh` fetches jar to gitignored `bin/`), `.ork`, trade study |
| `data/` | `motor.eng`, `rocket_params.json` (schema in ARCHITECTURE.md), `openrocket_flight.csv`, placeholders |
| `gnc/` sim | `frames, atmosphere, rocket, dynamics, sensors, sim` — 6-DOF, RK4 1 ms, rail, recovery, sensors (IMU/baro/GPS/mag) |
| `gnc/` GNC | `linearize, analysis, pid, lqr, observers, guidance, flight_computer` |
| `scenarios/` | one module per demo → `results/<demo>.json`; `python -m scenarios.run_all` (~75 s) |
| `viz/` | `python -m viz.build_dashboard --results results --out GNC_Dashboard.html` (offline single file, 3b1b theme) |
| `docs/` | `rocket_design.md`, `sim_model.md`, `gnc_design.md` (report source material) |
| `tests/` | `pytest -q` from repo root; `tests/checks/` = independent checker tests |

Key numbers: apogee 3,168 m AGL no wind (OpenRocket 3,139 m; sim within +1%), off-rail 30.6 m/s, stability 2.51 cal,
burn 4.73 s, **TVC authority only until ~3.97 s** (≈600–800 m altitude). PID = default attitude controller (better
delay margin); LQR available via `FlightComputer(attitude="lqr")`.

## Status at handoff
Done & committed on `master` (HEAD `76b06c6`): design, sim, GNC (PID + LQR), 6 scenario demos (passive, analysis,
attitude_control PID-vs-LQR, aggressive LOS path + envelope, observers KF-vs-Luenberger, robustness Monte Carlo),
dashboard + `GNC_Dashboard.html`, all checkers run.

**In progress — uncommitted, agents were stopped (usage limit). Do not `git checkout`/reset these files:**
1. `gnc/analysis.py` (modified): gnc-designer's fix of `critical_delay` (was bisecting on [0, d_max] → false `inf`;
   true LQR late-burn delay margins 38.9/39.8/52.9 ms at 3.07/3.52/3.97 s). **Remaining for that task:**
   - finish/verify `critical_delay` (exact: min over gain crossovers of ((∠L+180°) mod 360°)/ω, or scan up from 0);
   - `rank_test` `_kalman_matrix`: treat a block as zero if its norm < n·eps·‖A‖·‖prev raw block‖ (roundoff block
     currently normalised to unit norm → overstated rank);
   - `docs/gnc_design.md` §10: regenerate LQR late-burn delay-margin rows + "High dynamic pressure" bullet, delete
     "inf means > 1 s"; state LQR is reduced-order (θ, q, ξ) = a second way to pick PID gains, α/w feedback dropped
     (no α sensor), ≥60° PM guaranteed on design model only (41.4° on frozen 4-state model at 3.97 s); "≈ g/V" →
     "of order g/V"; also update §3 accordingly. Don't change FlightComputer default behaviour.
2. Scenarios text/rubric fixes (nothing written yet) — from the scenarios checker:
   - attitude_control disturbance/LQR notes: PID saturation is seed-dependent (3.9–48.7 %), report a 5-seed range;
   - `scenarios/analysis.py:83-84,116,118-120`: drift mode is a frozen-coefficient artefact (≈V̇/V; AoA-form residual
     0.21→0.03 /s, of order g/V) — not "thrusting weathercock divergence"; accuracy text from `max_diff` (≤2e-6);
   - `attitude_control.py:144-147`: 30+30 ms exceeds the margin "from ~2 s on" (amplitudes 0.04° @1 s, 0.25° @2.5 s);
     `analysis.py:376-378`: mid-boost margin is 56 ms;
   - `los_path.py:207`: "below ~30 m", envelope metric string must come from the sweep data;
   - `robustness.py:153`: 0.01 rad/s = 0.57°/s ≈ 8° in 15 s;
   - analysis ctrb_obsv: GPS-vel rank 4 at 0.82 s; "along-track position"→velocity `u`; gyro-only unobservable list
     must match the 4 lost directions; `cond` plot missing attitude-θ series;
   - **rubric gaps:** add LOS **rate** plot + metrics to los_path; add Luenberger column to accel/gyro-bias robustness sweeps;
   - drop the `d_max=0.3` workaround in `scenarios/attitude_control.py:168` once (1) is in.
3. Then: flip the strict-xfail markers that start XPASSing (`tests/checks/test_check_scenarios_results.py` ×4 —
   one, `test_critical_delay_default_bracket`, already XPASSes; `tests/checks/test_check_lqr_design.py` ×2) into plain
   tests; `python -m scenarios.run_all`; full `pytest -q`; rebuild `GNC_Dashboard.html`; commit.
4. Then: `REPORT.md` outline (≤10-page report per brief §8): (1) 6-DOF model + OpenRocket validation,
   (2) linearisation + ctrb/obsv, (3) PID pole placement vs LQR + delay margins, (4) KF vs Luenberger noise sweeps,
   (5) LOS waypoint guidance (note 4(i) excluded and why), (6) results tables + conclusions. Pull numbers from
   `results/*.json` and the docs, not from memory.

Known, accepted (user said don't fix): payload mass block overlaps nose-cone shoulder in the `.ork`; off-rail margin
0.07 m/s (30.35 m/s if aft-button exit convention). Dashboard nits: 3D LOS view is a needle at "True scale" (click
Stretched); stride downsampling can drop single-sample spikes.

## Orbital fork (paused)
Branch `orbital-iss`, worktree `~/Documents/GNC Project - Orbital ISS` (.venv symlinked). Two-stage Raptor-3 launch
vehicle to the ISS plane + cooperative ISS docking (starts in orbit, soft contact ≤0.1 m/s, keep-out sphere/corridor).
Only its `ARCHITECTURE.md` is committed. Four agents (vehicle-design, ascent-sim, ascent-gnc, rendezvous) were stopped
early; uncommitted partial files: `data/`, `gnc/analysis.py`, `gnc/observers.py`, `gnc/pid.py` — unreviewed.
User deprioritised it; resume only if asked.

## Helpful facts
- Rocket params changes invalidate the cached gain schedule per process (designs cached per rocket file).
- `FlightComputer` kwargs/log keys: see docstring in `gnc/flight_computer.py`; `simulate` API in `gnc/sim.py`.
- Headless browser checks: playwright with `channel="chrome"` (system Chrome) works; tests rewrite `viz/screenshots/`
  on every run — discard that churn rather than committing it.
- Graph of the codebase: `graphify-out/` (if present) — query it before broad file reads.
