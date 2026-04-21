# mpc_examples

Comparative quadcopter controller examples. Each example wires a controller
against a reference generator and the shared simulator, producing CSV logs,
plots, and performance metrics with a common format so the runs can be
compared directly.

The examples are built on a shared OOP framework (see
[examples/framework/](examples/framework)) with adapter layers for the
concrete controllers and reference generators. [examples/README.md](examples/README.md)
documents the `IController` / `ITrajectoryGenerator` interfaces and gives
the step-by-step recipe for adding a new controller or a new generator. One
unified example main is produced per (controller, generator) combination
(12 in total):

| Controller \\ Generator | Waypoints | Jerk-limited | GCopter | Dynamic |
|---|---|---|---|---|
| Cascade PID + geometric   | `pid_waypoints` | `pid_jerk_limited` | `pid_gcopter` | `pid_dynamic` |
| Position MPC (acados)     | `mpc_position_waypoints` | `mpc_position_jerk_limited` | `mpc_position_gcopter` | `mpc_position_dynamic` |
| Trajectory MPC (acados)   | `mpc_trajectory_waypoints` | `mpc_trajectory_jerk_limited` | `mpc_trajectory_gcopter` | `mpc_trajectory_dynamic` |

Each binary lives under `build/examples/mpc_examples_run_<combination>` and
produces a CSV log in the shared 44-column format. Every combination also
ships a Python twin at `examples/examples/<combination>/run_example.py` that
consumes the same YAML configs and writes the same CSV schema through a
pure-Python mirror of the framework
([examples/framework/python/](examples/framework/python/)) and the adapters
([examples/adapters/python/](examples/adapters/python/)). Adding a new
controller or generator requires one new C++ adapter (+ its Python mirror)
plus a ~30-line main in each language.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Control Loop                                 │
│                                                                  │
│  ┌──────────────┐  thrust + ω_ref  ┌─────────────────────────┐   │
│  │  Controller  │ ──────────────►  │      mav_simulator      │   │
│  │  (MPC / PID) │                  │                         │   │
│  │              │ ◄── position ─── │  INDI (500Hz)           │   │
│  │   ref ──►    │     velocity     │  Physics model (1000Hz) │   │
│  └──────────────┘     attitude     │  IMU (500Hz)            │   │
│         ▲                          └─────────────────────────┘   │
│         │                                                        │
│  ┌──────┴───────┐                                                │
│  │  Reference   │  waypoints / dynamic trajectory                │
│  │  generator   │                                                │
│  └──────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘
```

Three nested rates:

| Loop | Frequency | Timestep | Responsibility |
|------|-----------|----------|----------------|
| Outer controller (MPC / PID) | 100 Hz | 0.01 s | thrust + angular velocity |
| INDI + IMU | 500 Hz | 0.002 s | Rates → motor commands + sensor update |
| Physics model | 1000 Hz | 0.001 s | Rigid-body dynamics integration |

The outer command is held constant (zero-order hold) across the 5 INDI steps
between two outer-controller calls. This is the standard deployment pattern
for real hardware.

## Dependencies

### System packages (Ubuntu 22.04, ROS 2 Humble host)

```bash
sudo apt install build-essential cmake libeigen3-dev libyaml-cpp-dev \
                 libgtest-dev pybind11-dev python3-pybind11 \
                 python3-numpy python3-matplotlib python3-yaml \
                 python3-pytest
```

### Python packages (user-space)

```bash
pip3 install --user casadi acados-template jinja2 tqdm
```

`numpy`, `matplotlib`, `pyyaml`, and `pytest` come from the apt step above.

### acados

Follow the [acados installation guide](https://docs.acados.org/installation/index.html):

```bash
git clone https://github.com/acados/acados.git -b v0.5.3
cd acados
git submodule update --recursive --init
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```

Export the required paths (put in `~/.bashrc`):

```bash
export ACADOS_SOURCE_DIR="<path_to_acados>"     # e.g. ~/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
export PYTHONPATH=$PYTHONPATH:$ACADOS_SOURCE_DIR/interfaces/acados_template/
```

Install the tera renderer (code generation backend):

```bash
# Download from https://github.com/acados/tera_renderer/releases/tag/v0.0.34
# Copy the binary to $ACADOS_SOURCE_DIR/bin/t_renderer and make it executable
chmod +x $ACADOS_SOURCE_DIR/bin/t_renderer
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RPS98/mpc_examples.git
cd mpc_examples
git submodule update --recursive --init
```

### 2. Generate the acados C code (once)

```bash
bash generate.sh
```

Produces `examples/acados_position_mpc/` and `examples/acados_trajectory_mpc/`.
Rerun only after editing `configs/solver_definitions/solver_definition_mpc_*.yaml`.

### 3. Build everything

```bash
bash build.sh
```

One `cmake` invocation builds:

- every C++ library, example binary, and test target;
- every pybind11 Python module of the thirdparty submodules
  (`mav_simulator`, `dynamic_trajectory_generator`, `trajectory_generator_jerk_limited`,
  `gcopter_lib`);
- a mirror of every Python package (compiled `.so` and pure-Python sources) under
  [build/python/](build/python/) so a single `PYTHONPATH` entry exposes everything.

**No `pip install` step is required** for the thirdparty bindings. The launch
scripts in `scripts/` prepend `build/python/` to `PYTHONPATH` automatically.

### 4. Run all tests (optional)

```bash
ctest --test-dir build --output-on-failure
```

Covers C++ tests from the top-level targets and from every submodule
(`mav_simulator` + its libs, `trajectory_generator_jerk_limited`, `gcopter_lib`,
`dynamic_trajectory_generator`). Python test suites are registered through
CTest too — see [`python/tests/`](python/tests/) for the mpc-level suites.

## Running the examples

Each combination produces one executable
`build/examples/mpc_examples_run_<controller>_<generator>` (12 in total).

### Run all 12 at once

```bash
./scripts/run_all.sh                  # C++ binaries (default)
./scripts/run_all.sh --lang=py        # Python twins (drops *_py_log.csv)
./scripts/run_all.sh --lang=both      # Both, side by side
```

The helper runs every combination with the default configs baked into its
CLI. C++ runs produce `simulator_logs/<controller>_<generator>_log.csv`;
Python runs produce `simulator_logs/<controller>_<generator>_py_log.csv`. The
`py` backend requires that `./build.sh` has run at least once so the
pure-Python framework and adapters are symlinked under `build/python/`.

### Compare aggregate metrics across combinations

```bash
python3 scripts/compare_all.py
python3 scripts/compare_all.py --filter mpc_position
python3 scripts/compare_all.py --out summary.csv
```

`compare_all.py` invokes `examples/utils/compute_metrics.py` on every CSV
produced by `run_all.sh` and prints a side-by-side table of the aggregate
metrics (RMSE, jerk energy, settling time, etc.).

### Generate plots for one or many runs

```bash
python3 scripts/plot_all.py                           # every *_log.csv in simulator_logs/
python3 scripts/plot_all.py --filter pid mpc_position # only subset
python3 scripts/plot_all.py --pairs \
  "mpc_trajectory_gcopter,pid_gcopter;mpc_position_waypoints,mpc_position_waypoints_py"
```

`plot_all.py` wraps `examples/utils/plot_results.py` in batch mode, writing
PNGs to `simulator_logs/plots/<combination>/` per run plus
`simulator_logs/plots/_pair_<a>__vs__<b>/` for each explicit comparison pair.
Use `plot_results.py` directly on an individual CSV when iterating.

### Running a single combination manually

```bash
export PYTHONPATH="$(pwd)/build/python:${PYTHONPATH}"

# C++ binary
./build/examples/mpc_examples_run_mpc_trajectory_gcopter \
  -c configs/simulation/config_example.yaml \
  -s configs/simulation/config_simulator.yaml \
  -k configs/controllers/config_mpc_trajectory.yaml \
  -t configs/generators/config_gcopter.yaml \
  -f simulator_logs/mpc_trajectory_gcopter_log.csv

# Python twin (same CLI, same YAMLs)
python3 examples/examples/mpc_trajectory_gcopter/run_example.py \
  -c configs/simulation/config_example.yaml \
  -s configs/simulation/config_simulator.yaml \
  -k configs/controllers/config_mpc_trajectory.yaml \
  -t configs/generators/config_gcopter.yaml \
  -f simulator_logs/mpc_trajectory_gcopter_py_log.csv
```

All 12 binaries (and their Python twins) share the same CLI: `-c/-s` for
simulator-side YAMLs, `-k` for the controller config, `-t` for the trajectory
generator config, and `-f` for the output CSV. The lone exception is
`pid_waypoints`, which shares a single YAML between controller and
generator and uses `-p` in place of `-k`/`-t`.

## Configuration

All YAML files live under [configs/](configs/).

### Shared by every example

- **[configs/simulation/config_example.yaml](configs/simulation/config_example.yaml)** — top-level
  sim parameters: total time, hover time, timesteps (`model_dt`,
  `controller_dt`, `mpc_dt`, `pid_dt`), reference speed, `path_facing`, and the
  waypoint list.
  *Timestep constraint*: `model_dt` must divide `controller_dt`, and
  `controller_dt` must divide both `mpc_dt` and `pid_dt` exactly.
- **[configs/simulation/config_simulator.yaml](configs/simulation/config_simulator.yaml)** — physical
  model (mass, inertia, motor geometry), IMU noise model, and INDI controller
  gains (cascade inner loop of `mav_simulator`).

### Controller-specific

Controller configs (passed via `-k`):

| File | Consumed by | Purpose |
|---|---|---|
| [configs/controllers/config_pid.yaml](configs/controllers/config_pid.yaml) | `pid_*` binaries | Cascade PID gains + geometric controller gains + `v_max` / `d_max` |
| [configs/controllers/config_mpc.yaml](configs/controllers/config_mpc.yaml) | `mpc_position_*` binaries | Q, R, state + input bounds for position MPC |
| [configs/controllers/config_mpc_trajectory.yaml](configs/controllers/config_mpc_trajectory.yaml) | `mpc_trajectory_*` binaries | Q, Qe, R for trajectory MPC |

Generator configs (passed via `-t`):

| File | Consumed by | Purpose |
|---|---|---|
| [configs/generators/config_waypoints.yaml](configs/generators/config_waypoints.yaml) | `*_waypoints` | `d_max`, `reach_threshold` |
| [configs/generators/config_jerk_limited.yaml](configs/generators/config_jerk_limited.yaml) | `*_jerk_limited` | Accel/jerk bounds + reach threshold (speed comes from `sim_config.max_speed`) |
| [configs/generators/config_gcopter.yaml](configs/generators/config_gcopter.yaml) | `*_gcopter` | Drone params/limits + GCOPTER optimiser tuning (`max_velocity` comes from `sim_config.max_speed`) |
| [configs/generators/config_dynamic.yaml](configs/generators/config_dynamic.yaml) | `*_dynamic` | Placeholder; travel speed comes from `sim_config.max_speed` |

### Acados solver generation

| File | Used by |
|---|---|
| [configs/solver_definitions/solver_definition_mpc_position.yaml](configs/solver_definitions/solver_definition_mpc_position.yaml) | `generate.sh` → `examples/acados_position_mpc/` |
| [configs/solver_definitions/solver_definition_mpc_trajectory.yaml](configs/solver_definitions/solver_definition_mpc_trajectory.yaml) | `generate.sh` → `examples/acados_trajectory_mpc/` |

The MPC state is `[x, y, z, qw, qx, qy, qz, vx, vy, vz]` (NX=10) and the control
is `[thrust, ωx, ωy, ωz]` (NU=4).

## Repository layout

```
mpc_examples/
├── CMakeLists.txt                                 # Top-level build (sets PYBIND_PY_MIRROR_ROOT)
├── build.sh                                       # Configure + build everything
├── generate.sh                                    # Regenerate the acados C code
├── scripts/                                       # run_all.sh + compare_all.py + plot_all.py
├── configs/                                       # Shared + per-adapter YAMLs
├── examples/
│   ├── acados_position_mpc/                       # generated by generate.sh
│   ├── acados_trajectory_mpc/                     # generated by generate.sh
│   ├── framework/                                 # IController, ITrajectoryGenerator, WaypointsSimulator
│   ├── adapters/                                  # Bridges to each third-party library
│   │   ├── controllers/{pid_geometric,mpc_position,mpc_trajectory}/
│   │   └── trajectory_generators/{waypoint_reference,jerk_limited,gcopter,dynamic}/
│   ├── examples/                                  # One main per (controller, generator) x 12
│   └── utils/
│       ├── utils.hpp                              # C++: CsvLogger, geometry helpers
│       ├── example_config_utils.hpp               # C++: shared sim_config loader
│       ├── utils.py                               # Python: CsvLogger, geometry helpers
│       ├── config_utils.py                        # Python: config loading
│       ├── plot_results.py                        # Visualisation
│       └── compute_metrics.py                     # Per-segment performance metrics
├── thirdparty/
│   ├── mav_simulator/                             # C++ simulator + mavpy (Python)
│   ├── dynamic_trajectory_generator/              # Polynomial dynamic trajectory
│   ├── trajectory_generator_jerk_limited/         # Jerk-limited S-curve generator
│   ├── gcopter_lib/                               # GCOPTER-based trajectory optimiser
│   └── mpc/                                       # MPC acados core + position/trajectory controllers
└── simulator_logs/                                # CSV logs + plots from the scripts
```

After `build.sh` runs, the centralized Python mirror is populated at:

```
build/python/
├── mavpy/                                # model, sensors.imu, controllers, simulator
├── dynamic_trajectory_generator_py/
├── trajectory_generator_jerk_limited/
├── gcopterpy/                            # gcopterpy.trajectory
├── mpc_acados_core/                      # symlink → thirdparty/mpc/mpc_acados_core
├── mpc_acados_position/                  # symlink → thirdparty/mpc/controllers/position/…
├── mpc_acados_trajectory/                # symlink → thirdparty/mpc/controllers/trajectory/…
├── mpc_examples_framework/               # symlink → examples/framework/python/mpc_examples_framework
└── mpc_examples_adapters/                # symlink → examples/adapters/python/mpc_examples_adapters
```

A single `export PYTHONPATH=$(pwd)/build/python:$PYTHONPATH` exposes all of them.

## CSV log format

Every example logger writes one row per INDI step (500 Hz):

| Column | Description |
|--------|-------------|
| `time` | Simulation time (s) |
| `x, y, z` | Position (m, world frame) |
| `qw, qx, qy, qz` | Orientation quaternion (scalar-first) |
| `roll, pitch, yaw` | Euler angles (rad) |
| `vx, vy, vz` | Linear velocity (m/s, world frame) |
| `wx, wy, wz` | Angular velocity (rad/s, body frame) |
| `x_ref, y_ref, z_ref` | Position reference (m) |
| `qw_ref, qx_ref, qy_ref, qz_ref` | Orientation reference quaternion |
| `roll_ref, pitch_ref, yaw_ref` | Reference Euler angles (rad) |
| `thrust` | Outer-controller thrust command (N) |
| `wx_cmd, wy_cmd, wz_cmd` | Outer-controller angular velocity command (rad/s, body) |
| `motor_w0…motor_w3` | Motor angular velocities (rad/s) |

`examples/utils/compute_metrics.py` produces two companion files next to each
log (`*_metrics.csv`, `*_segments.csv`) with the aggregate and per-waypoint
performance indicators printed by the launch scripts.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
