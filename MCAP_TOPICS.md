# MCAP Topics Reference

This document lists every ROS 2 topic written into the MCAP files produced
by `mpc_examples`. The logging is centralised in `UnifiedMcapLogger`, a thin
facade over `mav_flight_review::MCAPLogger`:

- C++: [examples_cpp/src/framework/unified_mcap_logger.cpp](examples_cpp/src/framework/unified_mcap_logger.cpp)
- Python: [examples_py/examples_py/framework/unified_mcap_logger.py](examples_py/examples_py/framework/unified_mcap_logger.py)
- Underlying writer: [thirdparty/mav_flight_review/include/mav_flight_review/mcap_logger.hpp](thirdparty/mav_flight_review/include/mav_flight_review/mcap_logger.hpp)

A run produces one MCAP per case under
`simulator_logs/<run_id>/{cpp,py}/<controller>_<generator>.mcap`. The
schemas embedded in the MCAP make each file self-describing, so any ROS 2
tooling (Foxglove, `ros2 bag`, `mcap-cli`, `mav_flight_review`) can replay
or analyse it without extra metadata.

## Conventions

- **Frames**: world frame is `earth` (ENU); body frame is
  `drone0/base_link`. State pose/linear-velocity, motion-reference pose
  and motion-reference linear-velocity live in `earth`. Body-frame angular
  velocity (state and command) and the actuator commands live in body.
- **Units**: SI throughout. Position in m, linear velocity in m/s, angular
  velocity in rad/s, thrust in N, motor speeds in rad/s, time in s.
- **Quaternion order**: `[w, x, y, z]` (Eigen / `geometry_msgs/Quaternion`
  layout).
- **Time mode**: `SIMULATION` ([config.hpp:75](thirdparty/mav_flight_review/include/mav_flight_review/config.hpp#L75)).
  Timestamps in the MCAP are simulated seconds, **not** POSIX wall-clock.
- **Logging rate**: every per-step topic is written once per outer-loop
  step, i.e. at `controller_dt` (100 Hz by default). Metadata is emitted
  once at `t = 0`. `/clock` is auto-emitted alongside each `save_*` call
  (throttleable via `LoggerConfig::clock_min_period_s`).

## Topic catalogue

### Vehicle state (earth / body)

Written together by `MCAPLogger::save_state` ([mcap_logger.hpp:165](thirdparty/mav_flight_review/include/mav_flight_review/mcap_logger.hpp#L165)).

| Topic | Type | Rate | Content |
| --- | --- | --- | --- |
| `/drone0/self_localization/pose` | `geometry_msgs/msg/PoseStamped` | `controller_dt` | Current vehicle pose in `earth` |
| `/drone0/self_localization/twist` | `geometry_msgs/msg/TwistStamped` | `controller_dt` | Linear velocity (`earth`) + angular velocity (body) |
| `/drone0/sensor_measurements/odom` | `nav_msgs/msg/Odometry` | `controller_dt` | Aggregate of pose + twist (zero covariance) |

### Motion reference

The built-in `motion_reference/pose` channel of `mav_flight_review` is
remapped to `motion_reference/trajectory` so the stepwise waypoint target
can take its own dedicated channel
([unified_mcap_logger.cpp:51](examples_cpp/src/framework/unified_mcap_logger.cpp#L51),
[unified_mcap_logger.py:129](examples_py/examples_py/framework/unified_mcap_logger.py#L129)).

| Topic | Type | Rate | Content |
| --- | --- | --- | --- |
| `/drone0/motion_reference/trajectory` | `geometry_msgs/msg/PoseStamped` | `controller_dt` | Smooth generator sample consumed by the controller (computation delay applied) |
| `/drone0/motion_reference/twist` | `geometry_msgs/msg/TwistStamped` (linear only) | `controller_dt` | Linear-velocity reference from the generator (same delay applied) |
| `/drone0/motion_reference/position` | `geometry_msgs/msg/Vector3` | `controller_dt` | Active waypoint target — stepwise, no delay; identical across all controller × generator combinations |

### Actuator command (outer-loop output)

Thrust + body rates written by `MCAPLogger::save_actuation`; rotor speeds
written via `save_float64_multi_array`.

| Topic | Type | Rate | Content |
| --- | --- | --- | --- |
| `/drone0/actuator_command/thrust` | `as2_msgs/msg/Thrust` | `controller_dt` | Commanded collective thrust [N] |
| `/drone0/actuator_command/twist` | `geometry_msgs/msg/TwistStamped` (angular only) | `controller_dt` | Commanded body rates (frame `drone0/base_link`) |
| `/drone0/actuator_command/motor_speeds` | `std_msgs/msg/Float64MultiArray` (`dim=[4]`) | `controller_dt` | Per-rotor speed allocation (4 motors) |

### Compute times and applied delays (microseconds)

Defined as constants in [unified_mcap_logger.cpp:18-22](examples_cpp/src/framework/unified_mcap_logger.cpp#L18-L22).

| Topic | Type | Rate | Content |
| --- | --- | --- | --- |
| `/mpc_examples/controller_compute_time_us` | `std_msgs/msg/Float64` | `controller_dt` | Wall-clock duration of `IController::compute` |
| `/mpc_examples/generator_update_time_us` | `std_msgs/msg/Float64` | `controller_dt` | Wall-clock duration of `ITrajectoryGenerator::update` |
| `/mpc_examples/generator_eval_time_us` | `std_msgs/msg/Float64` | `controller_dt` | Wall-clock duration of `ITrajectoryGenerator::evaluate` |
| `/mpc_examples/controller_delay_applied_us` | `std_msgs/msg/Float64` | `controller_dt` | Delay actually applied by `DelayBuffer` to the controller command |
| `/mpc_examples/generator_delay_applied_us` | `std_msgs/msg/Float64` | `controller_dt` | Delay actually applied to the generator sample |

### Scheduler state

| Topic | Type | Rate | Content |
| --- | --- | --- | --- |
| `/mpc_examples/waypoint_index` | `std_msgs/msg/Int32` | `controller_dt` | Active waypoint index from `WaypointScheduler` |
| `/mpc_examples/hover_active` | `std_msgs/msg/Int32` (0/1) | `controller_dt` | `1` once the mission is over and the hold-after-mission phase begins |
| `/mpc_examples/max_speed` | `std_msgs/msg/Float64` | `controller_dt` | `sim_config.max_speed` (constant per run, single source of truth) |

### Run metadata (single-shot at `t = 0`)

Emitted once in the `UnifiedMcapLogger` constructor
([unified_mcap_logger.cpp:74-77](examples_cpp/src/framework/unified_mcap_logger.cpp#L74-L77),
[unified_mcap_logger.py:152-155](examples_py/examples_py/framework/unified_mcap_logger.py#L152-L155)).

| Topic | Type | Content |
| --- | --- | --- |
| `/mpc_examples/metadata/controller_name` | `std_msgs/msg/String` | E.g. `pid`, `mpc_position`, `mpc_trajectory` |
| `/mpc_examples/metadata/generator_name` | `std_msgs/msg/String` | E.g. `waypoints`, `jerk_limited`, `gcopter`, `dynamic`, `mav_traj_gen` |
| `/mpc_examples/metadata/run_id` | `std_msgs/msg/String` | `YYYYmmdd_HHMMSS` |
| `/mpc_examples/metadata/language` | `std_msgs/msg/String` | `cpp` or `py` |

### Simulated clock (auto-emitted)

| Topic | Type | Rate | Content |
| --- | --- | --- | --- |
| `/clock` | `rosgraph_msgs/msg/Clock` | Alongside every `save_*` (throttle via `LoggerConfig::clock_min_period_s`) | Simulated wall clock matching the per-row `t` |

## Notes

- The C++ and Python loggers publish **the same topics with the same
  types**. Keeping them in lockstep is intentional — downstream tooling
  (`mav_flight_review` metrics, the dashboard, plotters) treats `cpp/` and
  `py/` runs identically.
- `UnifiedMcapLogger` is only instantiated when
  `sim_config.output_format == "mcap"` (the default). With
  `output_format: csv` the framework switches to `UnifiedCsvLogger` and
  this document does not apply.
- The remapping of the built-in `motion_reference/pose` channel to
  `/drone0/motion_reference/trajectory` happens at logger construction
  ([unified_mcap_logger.cpp:51](examples_cpp/src/framework/unified_mcap_logger.cpp#L51)).
  The free `pose` slot is then reused for the stepwise waypoint target via
  `motion_reference/position` (`Vector3`).
- Frame summary: `self_localization/*` and `motion_reference/*` are in
  `earth`, except angular-velocity components of state/command twists,
  which are in `drone0/base_link`. `actuator_command/*` are body-frame.
- If a new topic is added to `unified_mcap_logger.{cpp,py}`, please update
  the corresponding table here so the document stays a faithful reference.
