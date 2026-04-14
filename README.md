# mpc_examples

Integrated example that simulates a quadcopter following waypoints by combining two independent projects:

- **[mav_simulator](https://github.com/RPS98/mav_simulator)** — C++ quadcopter physics simulator with INDI controller and IMU. Converts thrust + angular velocity commands into motor speeds and simulates rigid-body dynamics.
- **[position_mpc](https://github.com/RPS98/mpc)** — Position MPC using [acados](https://docs.acados.org). Converts position references into thrust + angular velocity commands.

Both a **C++ example** and a **Python example** are provided.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Control Loop (100 Hz)                        │
│                                                                  │
│  ┌──────────────┐  thrust + ω_ref  ┌─────────────────────────┐  │
│  │  position_mpc│ ──────────────►  │      mav_simulator      │  │
│  │  (MPC 100Hz) │                  │                         │  │
│  │              │ ◄── position ─── │  INDI (500Hz)           │  │
│  │  pos_ref ──► │     velocity     │  Physics model (1000Hz) │  │
│  └──────────────┘     attitude     │  IMU (500Hz)            │  │
│                                    └─────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

Three nested rates:

| Loop | Frequency | Timestep | Responsibility |
|------|-----------|----------|----------------|
| MPC | 100 Hz | 0.01 s | Solve OCP → thrust + angular velocity |
| INDI + IMU | 500 Hz | 0.002 s | Rates → motor commands + sensor update |
| Physics model | 1000 Hz | 0.001 s | Rigid-body dynamics integration |

The MPC command (thrust + angular velocity) is held constant (zero-order hold) across the 5 INDI steps between each MPC call. This is the standard deployment pattern for real hardware.

## Dependencies

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

Export the required paths:

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

### mav_simulator (Python bindings)

```bash
cd thirdparty/mav_simulator
pip3 install pybind/
```

### position_mpc (Python package)

```bash
cd thirdparty/position_mpc
pip3 install .
```

### C++ build dependencies

- CMake ≥ 3.5
- C++17 compiler
- Eigen3
- yaml-cpp (`sudo apt install libyaml-cpp-dev`)

## Installation

#### 1. Clone the repository

```bash
git clone https://github.com/RPS98/mpc_examples.git
cd mpc_examples
git submodule update --recursive --init
```

#### 2. Install dependencies

Follow the [acados](#acados) section above, then run the build script:

```bash
bash build.sh
```

This will:
- Install `mav_simulator` Python bindings (`mavpy`)
- Install `position_mpc` Python package (`mpc_position`)
- Generate the acados C code (`examples/acados_position_mpc/`)
- Build the C++ example (`build/examples/mpc_examples_run_example`)

## Running the examples

All commands are run from the repository root.

### C++ example

```bash
./run_example_cpp.sh
```

### Python example

```bash
./run_example_py.sh
```

Both scripts run the simulation, write `mpc_log.csv`, and launch the plotter automatically.

To run with custom config files:

```bash
./build/examples/mpc_examples_run_example \
  -c config_example.yaml \
  -s config_simulator.yaml \
  -m config_mpc.yaml \
  -f mpc_log.csv

python3 examples/run_example.py \
  -c config_example.yaml \
  -s config_simulator.yaml \
  -m config_mpc.yaml \
  -f mpc_log.csv
```

### Plotting results

```bash
python3 examples/utils/plot_results.py -f mpc_log.csv
```

This generates three figures:
- 3D trajectory with drone visualizations at regular intervals
- Position, orientation, and velocity tracking vs reference
- Control inputs (thrust, angular velocity), speed magnitude, and motor angular velocities

## Configuration

Three YAML files control the example (all at the repository root):

### `config_example.yaml`

Top-level simulation parameters:

```yaml
sim_config:
  sim_time: 30.0        # Total trajectory time (s)
  hover_time: 2.0       # Extra hover time after the last waypoint (s)
  model_dt: 0.001       # Physics timestep — 1000 Hz
  controller_dt: 0.002  # INDI + IMU timestep — 500 Hz
  mpc_dt: 0.01          # MPC timestep — 100 Hz
  max_speed: 1.0        # Max reference advance speed (m/s)
  path_facing: true     # Align yaw toward the next waypoint
  waypoints:
    - [0.0, 0.0, 1.0]
    - [2.0, 0.0, 1.0]
    - [2.0, 2.0, 1.5]
    - [0.0, 2.0, 1.5]
    - [0.0, 0.0, 1.0]
```

**Timestep constraint**: `model_dt` must divide `controller_dt`, and `controller_dt` must divide `mpc_dt` exactly.

### `config_simulator.yaml`

Physical model parameters (mass, inertia, motor geometry), IMU noise model, and INDI controller gains. Corresponds to the vehicle used by `mav_simulator`.

### `config_mpc.yaml`

MPC cost weights (Q, R), control and state constraint bounds, and paths to the generated solver files. The MPC state is `[x, y, z, qw, qx, qy, qz, vx, vy, vz]` (NX=10) and the control is `[thrust, ωx, ωy, ωz]` (NU=4).

### `solver_definition_mpc_position.yaml`

Acados solver settings: prediction horizon (N=30, tf=2.0 s), integrator type, QP solver, and export directory. Used only by `generate.sh`.

## Repository structure

```
mpc_examples/
├── CMakeLists.txt
├── build.sh                               # Install deps, generate code, build C++
├── generate.sh                            # Generate acados C code only
├── run_example_cpp.sh                     # Run C++ example + plot
├── run_example_py.sh                      # Run Python example + plot
├── config_example.yaml                    # Sim time, dt, waypoints
├── config_simulator.yaml                  # mav_simulator parameters
├── config_mpc.yaml                        # MPC gains and constraints
├── solver_definition_mpc_position.yaml    # Acados solver definition
└── examples/
    ├── run_example.cpp                    # C++ integrated example
    ├── run_example.py                     # Python integrated example
    ├── CMakeLists.txt
    ├── acados_position_mpc/               # Generated by generate.sh
    └── utils/
        ├── utils.hpp         # C++: CsvLogger, geometry helpers
        ├── yaml_utils.hpp    # C++: config loading
        ├── utils.py          # Python: CsvLogger, geometry helpers
        ├── config_utils.py   # Python: config loading
        └── plot_results.py   # Visualization script
└── thirdparty/
    ├── mav_simulator/        # Submodule: quadcopter simulator
    └── position_mpc/         # Local copy: position MPC
```

## CSV log format

The logger writes one row per controller step (500 Hz):

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
| `thrust` | MPC thrust command (N) |
| `wx_cmd, wy_cmd, wz_cmd` | MPC angular velocity command (rad/s, body frame) |
| `motor_w0…motor_w3` | Motor angular velocities (rad/s) |

## License

BSD-3-Clause. See [LICENSE](LICENSE).
