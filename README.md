# mpc_examples

Comparative quadrotor controller showcase. Three outer-loop controllers
(PID geometric, position MPC, trajectory MPC) paired against five
reference generators (waypoints, jerk-limited, GCOPTER, dynamic
polynomial, polynomial via `mav_trajectory_generation_lib`) on the same
simulator (`mav_simulator`, 500 Hz INDI + 1000 Hz rigid-body).

```
┌──────────────────────────────────────────────────────────────────┐
│  ┌──────────────┐  thrust + ω_ref  ┌─────────────────────────┐   │
│  │  Controller  │ ──────────────►  │      mav_simulator      │   │
│  │  (MPC / PID) │                  │  INDI (500 Hz)          │   │
│  │              │ ◄── position ─── │  Physics (1000 Hz)      │   │
│  │  ref ──►     │     velocity     │  IMU (500 Hz)           │   │
│  └──────────────┘     attitude     └─────────────────────────┘   │
│         ▲                                                        │
│  ┌──────┴───────┐                                                │
│  │  Reference   │  waypoint, jerk-limited, GCOPTER, dynamic,     │
│  │  generator   │  mav_traj_gen                                  │
│  └──────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘
```

| Loop | Frequency | Responsibility |
|---|---|---|
| Outer (MPC / PID) | 100 Hz | thrust + body rates |
| INDI + IMU | 500 Hz | rates → motors + sensors |
| Physics | 1000 Hz | rigid-body integration |

For architecture, factories, internal contracts and gotchas, see
[CLAUDE.md](CLAUDE.md). A list of every MCAP topic emitted by the
unified logger is in [MCAP_TOPICS.md](MCAP_TOPICS.md).

## Combination matrix

PID is compatible with every generator; the two MPCs are family-specific.

| Controller \ Generator | `waypoints` | `jerk_limited` | `gcopter` | `dynamic` | `mav_traj_gen` |
|---|---|---|---|---|---|
| `pid` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mpc_position` | ✓ | — | — | — | — |
| `mpc_trajectory` | — | ✓ | ✓ | ✓ | ✓ |

Each `runs[]` entry in
[configs/simulation/config_example.yaml](configs/simulation/config_example.yaml)
declares its `controller_config` and `generator_config` explicitly, so
the YAML is the single source of truth about which tuning every
combination uses.

## Build

```bash
git clone https://github.com/RPS98/mpc_examples.git
cd mpc_examples
git submodule update --init --recursive
bash build.sh
```

`build.sh` regenerates the acados solvers (one-shot if the `.so` files
are missing) and builds every C++ library, executable, gtest and
pybind11 module under `build/`. The Python mirror is published to
`build/python/`.

A single export exposes the bundled Python packages:

```bash
export PYTHONPATH=$(pwd)/build/python:$PYTHONPATH
```

`scripts/run_*.sh` and `scripts/single/*` add this automatically.

### Dependencies

System (Ubuntu 22.04 + ROS 2 Humble host):

```bash
sudo apt install build-essential cmake libeigen3-dev libyaml-cpp-dev \
                 libgtest-dev pybind11-dev python3-pybind11 \
                 python3-numpy python3-matplotlib python3-yaml \
                 python3-pytest libnlopt-cxx-dev libnlopt-dev \
                 libgoogle-glog-dev
pip3 install --user casadi acados-template jinja2 tqdm
```

[acados](https://docs.acados.org/installation/index.html) v0.5.3 with
`ACADOS_WITH_QPOASES=ON`. Export `ACADOS_SOURCE_DIR`, the matching
`LD_LIBRARY_PATH` and `PYTHONPATH`. Place the `t_renderer` binary
under `${ACADOS_SOURCE_DIR}/bin/`.

## Run

Two C++ binaries land in `build/examples_cpp/` and one Python module
per family lives in `examples_py/examples_py/runs/`:

| Family | C++ binary | Python module |
|---|---|---|
| Position-class | `position_examples` | `examples_py.runs.run_position_examples` |
| Trajectory-class | `trajectory_examples` | `examples_py.runs.run_trajectory_examples` |

### Every enabled case

```bash
./scripts/run_all.sh                      # C++ + Python in parallel
./scripts/run_all.sh --lang=cpp           # C++ only
./scripts/run_all.sh --lang=py            # Python only
./scripts/run_all.sh --no-show --no-save  # CI / headless
```

Outputs go to `simulator_logs/<run_id>/{cpp,py}/`. `run_id` defaults
to `YYYYmmdd_HHMMSS`; `--output-dir` overrides it.

### Single combination

The wrappers under `scripts/single/` cover every (controller × generator)
pair in C++ and Python. They run regardless of the `enabled` flag in
`runs[]`:

```bash
./scripts/single/pid_waypoints_cpp.sh
./scripts/single/mpc_position_waypoints_py.sh
./scripts/single/mpc_trajectory_gcopter_py.sh
./scripts/single/pid_jerk_limited_cpp.sh
# ... 20 scripts total
```

### Direct binary invocation

```bash
./build/examples_cpp/trajectory_examples \
  -c configs/simulation/config_example.yaml \
  -s configs/simulation/config_simulator.yaml \
  --only-controller mpc_trajectory \
  --only-generator  mav_traj_gen \
  --output-dir simulator_logs/manual_run
```

Passing both `--only-controller` and `--only-generator` overrides the
matching entry's `enabled` flag (the entry must exist in `runs[]` so
its config files are picked up).

### Evaluation mission

`configs/simulation/config_example_evaluate.yaml` carries the
controller-comparison campaign (mirrored from
`project_controller_pmpc/simulation/config/mission_evaluate_keep.yaml`):

- 11 waypoints written with the symbolic tokens **`D`** (horizontal
  distance, m) and **`H`** (vertical step, m), declared in
  `sim_config.evaluate.{distance,height}`. Resolved by the loader at
  startup; takeoff offset comes from `sim_config.takeoff_height`.
- Tuning targeted at the project_controller_pmpc 2026-05 sweep (PID
  `kp_xy=1.2`, gcopter `velocity_weight=500`, ...). MPC controllers
  keep their mpc_examples 2026-04 sweep weights — the more aggressive
  project_controller_pmpc weights triggered ACADOS_MINSTEP on this
  plant. See yaml comments for the rationale.

```bash
./build/examples_cpp/position_examples \
  -c configs/simulation/config_example_evaluate.yaml \
  -s configs/simulation/config_simulator.yaml
./build/examples_cpp/trajectory_examples \
  -c configs/simulation/config_example_evaluate.yaml \
  -s configs/simulation/config_simulator.yaml
```

To retarget the mission, edit `evaluate.distance` /
`evaluate.height` in metres; the tokens get rescaled automatically.

## Tests

```bash
ctest --test-dir build --output-on-failure              # everything
ctest --test-dir build --output-on-failure -E pytest    # gtest only
ctest --test-dir build --output-on-failure -L pytest    # pytest only
```

The C++ side runs a gtest binary per adapter and per framework module
plus the internal acados / mav_simulator / gcopter /
mav_trajectory_generation_lib suites. The Python side registers a
pytest suite per submodule that ships its own `pybind/tests/`.

A successful `bash build.sh` is the only prerequisite.

## Telemetry & analysis

Every run writes an MCAP under
`simulator_logs/<run_id>/{cpp,py}/<controller>_<generator>.mcap`
(switch to CSV with `sim_config.output_format: csv`).

[`mav_flight_review`](thirdparty/mav_flight_review/) processes those
MCAPs into the per-segment + total `<bag>_analysis.csv`, with both
the legacy metrics (rise, settle, RMSE, jerk energy) and the
project-specific extras (settling_time_5pct_s, sse_post_settle_m,
time_at_saturation_pct_seg, overshoot_pct, path_length_efficiency,
cte_*, hover_*, mean_cruise_speed_m_s, peak_speed_m_s,
speed_violation_pct_seg).

```bash
# Per-run aggregation: walks <run_dir>/{cpp,py}/ and writes
# <run_dir>/metrics/summary.csv and one <stem>_analysis.csv per MCAP.
python3 -m mav_flight_review.compute_metrics --run-dir simulator_logs/<run_id>

# Plot dashboard.
python3 -m mav_flight_review.cli --run-dir simulator_logs/<run_id>

# Per-waypoint comparison across an arbitrary set of runs (sim or real),
# driven by a YAML config with `runs: [{name, csv}, ...]` (or via
# repeated --run NAME=PATH on the CLI). One column group per run, one
# row per waypoint, plus a trailing total row.
python3 -m mav_flight_review.compare --config my_compare.yaml
python3 -m mav_flight_review.compare --output /tmp/out \
    --run pid=/path/to/pid_analysis.csv \
    --run mpc=/path/to/mpc_analysis.csv
```

`scripts/run_all.sh` and `scripts/single/*.sh` invoke
`compute_metrics` and `cli` automatically.

## Adding a new adapter

A new controller or generator is a new translation unit + factory
entry. Start from the base classes:

- [examples_cpp/include/framework/controller_base.hpp](examples_cpp/include/framework/controller_base.hpp)
- [examples_cpp/include/framework/trajectory_generator_base.hpp](examples_cpp/include/framework/trajectory_generator_base.hpp)

Mirror the same shape under
[examples_py/examples_py/framework/](examples_py/examples_py/framework/)
and register your key in
[examples_cpp/src/framework/factories_*.cpp](examples_cpp/src/framework/)
and
[examples_py/examples_py/framework/factories.py](examples_py/examples_py/framework/factories.py).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
