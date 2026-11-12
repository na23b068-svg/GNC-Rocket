# Graph Report - GNC Project  (2026-11-12)

## Corpus Check
- Large corpus: 106 files · ~510,367 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 835 nodes · 2567 edges · 24 communities (19 shown, 5 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 82 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- 6-DOF Dynamics & Aero
- Package & TVC Timing
- Dashboard Builder
- Flight Computer & Sim Loop
- Rocket Design Checks
- ISA Atmosphere & Wind
- Rocket Mass & Thrust Model
- Project Docs & Handoff
- Kalman & Luenberger Observers
- Dashboard Browser Tests
- OpenRocket Headless Driver
- Margins & Delay Analysis
- LQR vs PID Comparison
- Passive Flight Scenario
- Gain Schedule Design
- LOS Path Guidance
- Pole-Placement PID Tuning
- Ctrb/Obsv Rank & Gramians
- LQR Weights (Bryson)
- Discrete PID
- OpenRocket Setup Script

## God Nodes (most connected - your core abstractions)
1. `simulate()` - 63 edges
2. `SimConfig` - 56 edges
3. `Rocket` - 48 edges
4. `FlightComputer` - 30 edges
5. `design_schedule()` - 29 edges
6. `quat_to_dcm()` - 29 edges
7. `isa()` - 25 edges
8. `derivatives()` - 23 edges
9. `operating_points()` - 23 edges
10. `series()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `lqr_available()` --uses--> `FlightComputer`  [INFERRED]
  scenarios/attitude_control.py → gnc/flight_computer.py
- `build()` --calls--> `Rocket`  [INFERRED]
  openrocket/orkit.py → gnc/rocket.py
- `rocket()` --uses--> `Rocket`  [INFERRED]
  scenarios/common.py → gnc/rocket.py
- `main()` --uses--> `Rocket`  [INFERRED]
  scenarios/passive_flight.py → gnc/rocket.py
- `rocket()` --uses--> `Rocket`  [INFERRED]
  tests/checks/test_check_gnc_design.py → gnc/rocket.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GNC pipeline: navigation, guidance, attitude control, actuation** — docs_gnc_design_attitude_nav, docs_gnc_design_kf_vs_luenberger, docs_gnc_design_los_guidance, docs_gnc_design_pid_pole_placement, docs_sim_model_actuator_chain [INFERRED 0.85]
- **OpenRocket to params to simulator data flow** — docs_rocket_design_openrocket_jpype, docs_rocket_design_aero_extraction, architecture_rocket_params_schema, docs_sim_model_forces_moments [INFERRED 0.85]

## Communities (24 total, 5 thin omitted)

### Community 0 - "6-DOF Dynamics & Aero"
Cohesion: 0.05
Nodes (60): aero_state(), derivatives(), Env, make_env(), specific_force_body(), dcm_to_quat(), euler_zyx_to_quat(), launch_attitude() (+52 more)

### Community 1 - "Package & TVC Timing"
Cohesion: 0.08
Nodes (51): tvc_end_time(), cstr(), _empty(), gains_at(), main(), pole_list(), v_ctrb_obsv(), v_delays() (+43 more)

### Community 2 - "Dashboard Builder"
Cohesion: 0.05
Nodes (45): _nose(), page(), test_nose_vector(), _var(), base(), _ok(), test_load_results_messages(), test_malformed_rejected_with_field() (+37 more)

### Community 3 - "Flight Computer & Sim Loop"
Cohesion: 0.06
Nodes (47): FlightComputer, nominal_trajectory(), SimConfig, simulate(), _exact_margins(), _late_pp(), _los(), _Poison (+39 more)

### Community 4 - "Rocket Design Checks"
Cohesion: 0.07
Nodes (43): barrowman(), eng_curve(), flight(), isa(), ork_xml(), point_mass(), test_apogee_point_mass(), test_barrowman_cp_cna_low_mach() (+35 more)

### Community 5 - "ISA Atmosphere & Wind"
Cohesion: 0.09
Nodes (25): g(), geopotential(), isa(), _layer_p(), pressure_altitude(), WindModel, _core(), rail_derivatives() (+17 more)

### Community 6 - "Rocket Mass & Thrust Model"
Cohesion: 0.07
Nodes (16): read_eng(), Rocket, _vec(), rocket(), test_motor_impulse_and_mass_flow(), rocket(), rocket(), tab() (+8 more)

### Community 7 - "Project Docs & Handoff"
Cohesion: 0.07
Nodes (41): Independent checker agents, Agent handoff log, Hard rules (scope, venv, checkers, commits), Uncommitted in-progress work (critical_delay, scenario fixes), Orbital ISS fork (paused), REPORT.md outline, Controller protocol (reset/update/log, meas dict), Binding conventions (SI, ENU, quaternion, body frame) (+33 more)

### Community 8 - "Kalman & Luenberger Observers"
Cohesion: 0.10
Nodes (19): attitude_plane_model(), axis_nav_model(), discretize(), kalman_gain_dare(), KalmanFilter, Luenberger, luenberger_gain(), nav_luenberger_gains() (+11 more)

### Community 9 - "Dashboard Browser Tests"
Cohesion: 0.10
Nodes (24): _launch(), page(), _quat_ref(), test_quat_matches_python_reference(), mock_dir(), all_demos(), analysis_demo(), analysis_variant() (+16 more)

### Community 10 - "OpenRocket Headless Driver"
Cohesion: 0.10
Nodes (18): build(), _bulkhead(), find_motor(), load_and_run(), m_motors(), make_sim(), _mass(), _material() (+10 more)

### Community 11 - "Margins & Delay Analysis"
Cohesion: 0.15
Nodes (13): attitude_plant(), closed_loop(), critical_delay(), delay_sweep(), loop_tf(), margins(), pid_tf(), poles_zeros() (+5 more)

### Community 12 - "LQR vs PID Comparison"
Cohesion: 0.14
Nodes (14): compare(), design_lqr_schedule(), _exact(), _lqr(), rocket(), tab(), test_default_pid_bit_identical_to_8cd9205(), test_lag_margins_match_doc_early_boost() (+6 more)

### Community 13 - "Passive Flight Scenario"
Cohesion: 0.15
Nodes (12): figure_2d(), figure_3d(), figure_openrocket(), load_openrocket_csv(), main(), _mark_events(), metrics(), openrocket_comparison() (+4 more)

### Community 14 - "Gain Schedule Design"
Cohesion: 0.14
Nodes (12): plane_pid_closed_loop(), design_schedule(), aoa_form(), operating_points(), closed_loop_poles(), test_doc_claim_gyro_only_rank3_late_boost(), test_aoa_form_is_time_varying_transform_and_residual_near_g_over_v(), test_rank_table_doc_sec4() (+4 more)

### Community 15 - "LOS Path Guidance"
Cohesion: 0.17
Nodes (8): kinematic_path_check(), los_angles(), LOSPathGuidance, steer_toward(), _unit(), test_los_path_converges_to_straight_line(), test_steer_toward_is_perpendicular_and_signed(), test_waypoint_switching_and_3d_path()

### Community 16 - "Pole-Placement PID Tuning"
Cohesion: 0.18
Nodes (9): GainSchedule, place_pid(), poles_from_specs(), specs_of(), test_delay_margin_matches_pade_bisection_and_sweep(), test_gain_schedule_interpolates(), test_integral_rejects_step_disturbance(), test_place_pid_reproduces_poles() (+1 more)

### Community 17 - "Ctrb/Obsv Rank & Gramians"
Cohesion: 0.19
Nodes (10): _cond(), gramian(), gramian_summary(), _kalman_matrix(), rank_test(), test_rank_test_keeps_scaled_deficient_pair_deficient(), test_gramian_lyapunov_and_finite(), test_nav_model_observability_by_sensor_set() (+2 more)

### Community 18 - "LQR Weights (Bryson)"
Cohesion: 0.44
Nodes (4): attitude_lqr(), augmented_model(), bryson(), test_gain_solves_care()

## Knowledge Gaps
- **8 isolated node(s):** `Agent file ownership table`, `Dashboard (viz, GNC_Dashboard.html)`, `Orbital ISS fork (paused)`, `Independent checker agents`, `Attitude estimator (strapdown gyro + magnetometer Luenberger on SO(3))` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 239 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Rocket` connect `Rocket Mass & Thrust Model` to `6-DOF Dynamics & Aero`, `Package & TVC Timing`, `Flight Computer & Sim Loop`, `Rocket Design Checks`, `OpenRocket Headless Driver`, `LQR vs PID Comparison`, `Passive Flight Scenario`, `Gain Schedule Design`, `LQR Weights (Bryson)`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `simulate()` (e.g. with `.step()` and `log()`) actually correct?**
  _`simulate()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Agent file ownership table`, `Dashboard (viz, GNC_Dashboard.html)`, `Orbital ISS fork (paused)` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `6-DOF Dynamics & Aero` be split into smaller, more focused modules?**
  _Cohesion score 0.051287128712871284 - nodes in this community are weakly interconnected._
- **Why does `simulate()` connect `Flight Computer & Sim Loop` to `6-DOF Dynamics & Aero`, `Package & TVC Timing`, `ISA Atmosphere & Wind`, `LQR vs PID Comparison`, `Passive Flight Scenario`, `Gain Schedule Design`, `LQR Weights (Bryson)`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `SimConfig` (e.g. with `flight()` and `test_actuator_delay_measured()`) actually correct?**
  _`SimConfig` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `Package & TVC Timing` be split into smaller, more focused modules?**
  _Cohesion score 0.08401084010840108 - nodes in this community are weakly interconnected._