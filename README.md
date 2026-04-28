# mpc_examples

Comparative quadrotor controller examples. Three outer-loop controllers (PID
geometric, position MPC, trajectory MPC) are paired against five reference
generators (waypoints, jerk-limited, GCOPTER, dynamic polynomial, polynomial
via `mav_trajectory_generation_lib`) on a shared simulator (`mav_simulator`,
500 Hz INDI + 1000 Hz rigid-body model). Every combination uses the same
configuration files, the same telemetry schema and the same metrics tooling
so the runs can be compared directly.

The repository is organised around a small OOP framework:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Control Loop                                 │
│                                                                  │
│  ┌──────────────┐  thrust + ω_ref  ┌─────────────────────────┐   │
│  │  Controller  │ ──────────────►  │      mav_simulator      │   │
│  │  (MPC / PID) │                  │  INDI (500Hz)           │   │
│  │              │ ◄── position ─── │  Physics model (1000Hz) │   │
│  │   ref ──►    │     velocity     │  IMU (500Hz)            │   │
│  └──────────────┘     attitude     └─────────────────────────┘   │
│         ▲                                                        │
│         │                                                        │
│  ┌──────┴───────┐                                                │
│  │  Reference   │   waypoint, polynomial, GCOPTER, dynamic       │
│  │  generator   │                                                │
│  └──────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘
```

Three nested rates:

| Loop | Frequency | Timestep | Responsibility |
|------|-----------|----------|----------------|
| Outer controller (MPC / PID) | 100 Hz | 0.01 s | thrust + angular rates |
| INDI + IMU                   | 500 Hz | 0.002 s | rates → motor commands + sensor update |
| Physics model                | 1000 Hz | 0.001 s | rigid-body integration |

The outer command is held constant (zero-order hold) across the 5 INDI steps
between two outer-controller calls, matching the standard deployment pattern
on real hardware.

## Combination matrix

The factory layer accepts 3 controllers × 5 generators (PID is compatible
with every generator; the two MPCs are family-specific):

| Controller \ Generator | `waypoints` | `jerk_limited` | `gcopter` | `dynamic` | `mav_traj_gen` |
|---|---|---|---|---|---|
| `pid` (cascade + geometric) | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mpc_position` (acados)     | ✓ | — | — | — | — |
| `mpc_trajectory` (acados)   | — | ✓ | ✓ | ✓ | ✓ |

The default `configs/simulation/config_example.yaml` enables ten cases
(every entry that is meaningful in either binary's scope). Disable any case
by flipping its `enabled: false` flag.

> **Known limitation** — `mpc_trajectory + dynamic` is borderline: the
> dynamic generator emits aggressive accelerations under the default p2p
> config that drive the acados QP solver into `ACADOS_MINSTEP` (status 4),
> particularly under `parallel: true` + `*_delay_mode: measured`. The
> entry is left enabled so the failure is visible in `run_all`'s summary
> table; the per-case single script remains operational for opt-in
> experiments. Switching to `controller_delay_mode: fixed` and/or
> `parallel: false` recovers the C++ run; the Python solver is more
> sensitive and may still report FAILED.

## Repository layout

```
mpc_examples/
├── CMakeLists.txt                  # Top-level build (Python mirror, pytest registration)
├── build.sh                        # Configure + build everything (acados → cmake)
├── configs/
│   ├── controllers/                # Per-controller YAMLs (pid, mpc, mpc_trajectory)
│   ├── generators/                 # Per-generator YAMLs (5)
│   └── simulation/                 # config_example + config_simulator
├── examples_cpp/
│   ├── include/                    # framework/, controllers/, generators/, utils/
│   ├── src/                        # implementations + run_*_examples.cpp entry points
│   └── tests/                      # GoogleTest per adapter and per framework module
├── examples_py/
│   ├── examples_py/                # Pure-Python mirror of examples_cpp
│   │   ├── framework/              # IController / ITrajectoryGenerator + helpers
│   │   ├── controllers/            # 3 controllers
│   │   ├── generators/             # 5 generators
│   │   └── runs/                   # run_position_examples.py / run_trajectory_examples.py
│   └── tests/                      # pytest suites mirroring examples_cpp/tests
├── libs/                           # Generated acados solvers (acados_*_mpc/)
├── scripts/
│   ├── run_all.sh                  # Dispatch every enabled case (cpp / py / both)
│   ├── single/                     # One-shot launchers per (controller × generator)
│   ├── compute_metrics.sh          # Per-run metrics aggregation
│   └── plot.sh                     # Dashboard from a run directory
├── thirdparty/                     # 7 git submodules (see below)
└── simulator_logs/                 # Run outputs (gitignored)
```

After `bash build.sh` runs, the centralised Python mirror is populated at
`build/python/`:

```
build/python/
├── mavpy/                              # mav_simulator pybinds (model, sensors, controllers, simulator)
├── dynamic_trajectory_generator_py/    # generator pybind
├── trajectory_generator_jerk_limited/  # generator pybind
├── gcopterpy/                          # GCOPTER pybind (gcopterpy.trajectory)
├── mav_trajectory_generation_py/       # polynomial generator pybind
├── mav_flight_review/                  # MCAP / CSV logger + metrics + plotter
├── mpc_acados_core/                    # MPC core
├── mpc_acados_position/                # MPC position bindings
├── mpc_acados_trajectory/              # MPC trajectory bindings
└── examples_py/                        # The pure-Python showcase
```

A single `export PYTHONPATH=$(pwd)/build/python:$PYTHONPATH` exposes all of
them. The `scripts/run_*.sh` helpers do this automatically.

## Submodules

Seven git submodules live under `thirdparty/`:

| Submodule | Purpose |
|---|---|
| `mav_simulator` | quadrotor simulator (rigid-body + INDI) and `mavpy.*` bindings |
| `mpc` | acados-based position and trajectory MPC controllers |
| `dynamic_trajectory_generator` | polynomial dynamic trajectory (asynchronous replanner) |
| `gcopter_lib` | GCOPTER polytope SFC trajectory optimiser |
| `trajectory_generator_jerk_limited` | jerk-limited S-curve generator |
| `mav_trajectory_generation_lib` | ROS-free facade around ETH-ASL `mav_trajectory_generation` (degree-10 polynomial) |
| `mav_flight_review` | MCAP / CSV telemetry backend + matplotlib viewer |

The MPC state is `[x, y, z, qw, qx, qy, qz, vx, vy, vz]` (NX=10) and the
control vector is `[thrust, ωx, ωy, ωz]` (NU=4).

## Dependencies

### System packages (Ubuntu 22.04, ROS 2 Humble host)

```bash
sudo apt install build-essential cmake libeigen3-dev libyaml-cpp-dev \
                 libgtest-dev pybind11-dev python3-pybind11 \
                 python3-numpy python3-matplotlib python3-yaml \
                 python3-pytest libnlopt-cxx-dev libnlopt-dev \
                 libgoogle-glog-dev
```

`fastcdr` is pulled from `/opt/ros/humble` when present; install
`ros-humble-fastcdr` if not.

### Python packages (user-space)

```bash
pip3 install --user casadi acados-template jinja2 tqdm
```

`numpy`, `matplotlib`, `pyyaml` and `pytest` come from the apt step above.

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

## Build

```bash
git clone https://github.com/RPS98/mpc_examples.git
cd mpc_examples
git submodule update --init --recursive

bash build.sh        # acados codegen (one-shot) + full CMake build
```

The single CMake invocation builds:

- every C++ library, executable and gtest target;
- every pybind11 module of the seven thirdparty submodules;
- a mirror of every Python package under [build/python/](build/python/).

No `pip install` step is required for the bundled submodules.

## Run

Two entry-point binaries land under `build/examples_cpp/` and one Python
runner per family lives in `examples_py/examples_py/runs/`:

| Family | C++ binary | Python module |
|---|---|---|
| Position-class generators | `position_examples` | `examples_py.runs.run_position_examples` |
| Trajectory-class generators | `trajectory_examples` | `examples_py.runs.run_trajectory_examples` |

### Run every enabled case

```bash
./scripts/run_all.sh                      # C++ + Python in parallel
./scripts/run_all.sh --lang=cpp           # C++ only
./scripts/run_all.sh --lang=py            # Python only
./scripts/run_all.sh --no-show --no-save  # CI / headless
```

Outputs land under `simulator_logs/<run_id>/{cpp,py}/` (run_id is
auto-generated as `YYYYmmdd_HHMMSS`). `run_all.sh` also invokes the
metrics aggregator and dashboard from `mav_flight_review`.

### Run a single combination

The wrappers under `scripts/single/` cover every (controller × generator)
combination — 20 scripts total, one C++ and one Python launcher per
enabled combo:

```bash
./scripts/single/pid_waypoints_cpp.sh
./scripts/single/mpc_position_waypoints_py.sh
./scripts/single/pid_gcopter_cpp.sh
./scripts/single/pid_jerk_limited_cpp.sh
./scripts/single/pid_dynamic_cpp.sh
./scripts/single/pid_mav_traj_gen_cpp.sh
./scripts/single/mpc_trajectory_gcopter_py.sh
./scripts/single/mpc_trajectory_jerk_limited_py.sh
./scripts/single/mpc_trajectory_dynamic_py.sh
./scripts/single/mpc_trajectory_mav_traj_gen_cpp.sh
# ... and the matching `_py.sh` / `_cpp.sh` companions
```

Each single script also runs `compute_metrics`, `print_summary` and the
`mav_flight_review` plotter against the freshly produced run directory,
so a single invocation gives you the MCAP, the per-segment analysis CSV,
the metrics summary printed in the terminal and the figures saved under
`<run_dir>/plots/`.

To call a binary directly:

```bash
./build/examples_cpp/trajectory_examples \
  -c configs/simulation/config_example.yaml \
  -s configs/simulation/config_simulator.yaml \
  --only-controller mpc_trajectory \
  --only-generator  mav_traj_gen \
  --output-dir simulator_logs/manual_run
```

## Tests

`ctest` covers both gtest and pytest:

```bash
ctest --test-dir build --output-on-failure              # everything
ctest --test-dir build --output-on-failure -E pytest    # only gtest
ctest --test-dir build --output-on-failure -L pytest    # only pytest
```

The C++ side runs **14 gtest binaries** under `examples_cpp/tests/`
(3 controllers + 5 generators + 6 framework modules including an
end-to-end `WaypointsSimulator` smoke test) plus the internal acados /
mav_simulator / gcopter / mav_trajectory_generation_lib suites. The
Python side registers **6 pytest suites**: `pytest_examples_py` covers
every adapter and every framework module of the showcase, plus one
suite per thirdparty submodule that ships its own `pybind/tests/`
tree (`pytest_mav_flight_review`, `pytest_gcopter_lib`,
`pytest_dynamic_trajectory_generator`,
`pytest_trajectory_generator_jerk_limited`,
`pytest_mav_trajectory_generation_lib`).

Each pytest suite runs with
`PYTHONPATH=${PYBIND_PY_MIRROR_ROOT}:$PYTHONPATH`, so a successful
`bash build.sh` is the only prerequisite (no `pip install`).

## Configuration

All YAML files live under [configs/](configs/).

### Shared (every example)

- [configs/simulation/config_example.yaml](configs/simulation/config_example.yaml) —
  total time, hover time, timesteps (`model_dt`, `controller_dt`, `mpc_dt`,
  `pid_dt`), reference speed, `path_facing`, the waypoint list, and the
  `runs[]` matrix.
  *Timestep contract*: `model_dt | controller_dt`, and `controller_dt | mpc_dt`
  and `controller_dt | pid_dt` (exact division).
- [configs/simulation/config_simulator.yaml](configs/simulation/config_simulator.yaml) —
  physical model (mass, inertia, motor geometry), IMU noise model and INDI
  cascade gains.

### Controller-specific

| File | Consumed by |
|---|---|
| [config_pid.yaml](configs/controllers/config_pid.yaml) | `pid` |
| [config_mpc.yaml](configs/controllers/config_mpc.yaml) | `mpc_position` |
| [config_mpc_trajectory.yaml](configs/controllers/config_mpc_trajectory.yaml) | `mpc_trajectory` |

### Generator-specific

| File | Consumed by |
|---|---|
| [config_waypoints.yaml](configs/generators/config_waypoints.yaml) | `waypoints` |
| [config_waypoints_mpc.yaml](configs/generators/config_waypoints_mpc.yaml) | `waypoints` (MPC-position tuning) |
| [config_jerk_limited.yaml](configs/generators/config_jerk_limited.yaml) | `jerk_limited` |
| [config_gcopter.yaml](configs/generators/config_gcopter.yaml) | `gcopter` |
| [config_dynamic.yaml](configs/generators/config_dynamic.yaml) | `dynamic` |
| [config_mav_traj_gen.yaml](configs/generators/config_mav_traj_gen.yaml) | `mav_traj_gen` |

`max_speed` is a **single source of truth** in
`config_example.yaml:sim_config.max_speed`. Generators do NOT declare it in
their own YAML; they receive it via `ExampleConfig` in `initialize()`.

## Telemetry

The default output format is **MCAP** (ROS 2-compatible) served by
[mav_flight_review](thirdparty/mav_flight_review/). Each run produces

```
simulator_logs/<run_id>/{cpp,py}/<controller>_<generator>.mcap
```

with topics for ground-truth state, references, control commands, IMU,
motor speeds and run metadata. Switch to plain CSV by setting
`sim_config.output_format: csv` in `config_example.yaml`.

The viewer/dashboard ships with `mav_flight_review` under
`thirdparty/mav_flight_review/pybind/python/mav_flight_review/`. After
`bash build.sh`, with `PYTHONPATH` pointing to `build/python/`:

```bash
python3 -m mav_flight_review.compute_metrics --run-dir simulator_logs/<run_id>
python3 -m mav_flight_review.print_summary  --run-dir simulator_logs/<run_id>
python3 -m mav_flight_review.cli            --run-dir simulator_logs/<run_id>
```

`scripts/run_all.sh` (and every `scripts/single/*.sh`) calls these
automatically. The third command saves `run.png`, `run_3d.png`,
`run_extras.png` and `run_metrics.png` under `<run_dir>/plots/` and
opens the matplotlib windows; pass `MPLBACKEND=Agg` (or rely on
`--no-show` flags propagated by the runner wrappers) for headless
runs.

## Adding a new adapter

The framework is designed so a new controller or generator only requires a
new translation unit + factory entry. Check the adapter base classes:

- [examples_cpp/include/framework/controller_base.hpp](examples_cpp/include/framework/controller_base.hpp) (`IController`)
- [examples_cpp/include/framework/trajectory_generator_base.hpp](examples_cpp/include/framework/trajectory_generator_base.hpp) (`ITrajectoryGenerator`)

Then mirror the same shape in
[examples_py/examples_py/framework/](examples_py/examples_py/framework/) and
register your key in
[examples_cpp/src/framework/factories_*.cpp](examples_cpp/src/framework/) and
[examples_py/examples_py/framework/factories.py](examples_py/examples_py/framework/factories.py).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
