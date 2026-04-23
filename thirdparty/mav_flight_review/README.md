# mav_flight_mcap

ROS 2 Humble-compatible MCAP flight logger, **standalone** (no ROS 2 runtime
required). Writes bags with `message_encoding="cdr"` and
`schema_encoding="ros2msg"`, byte-compatible with
[`rosbag2`](https://github.com/ros2/rosbag2) on x86_64.

Two components:

1. **Core C++ library + pybind bindings** — typed `save_*()` methods for every
   topic mandated by the aerostack2 spec, plus typed extras
   (`std_msgs/Int32|String|Float64|Float64MultiArray`, `geometry_msgs/Vector3`).
2. **`mav_flight_viewer`** — a matplotlib dashboard that reads the MCAP (no ROS
   2 needed, no `rclpy`) and plots position, velocity, angular rates, thrust,
   reference tracking error, |v| modulus and any extra scalar topics.

## What it logs (defaults)

Following the aerostack2 drone0 spec:

| Topic | Type | Use |
|---|---|---|
| `/drone0/motion_reference/pose`       | `geometry_msgs/msg/PoseStamped`     | Desired position + yaw |
| `/drone0/motion_reference/twist`      | `geometry_msgs/msg/TwistStamped`    | Max linear speed per axis |
| `/drone0/motion_reference/trajectory` | `as2_msgs/msg/TrajectorySetpoints`  | Full trajectory with yaw per point |
| `/drone0/actuator_command/thrust`     | `as2_msgs/msg/Thrust`               | Commanded thrust (N) |
| `/drone0/actuator_command/twist`      | `geometry_msgs/msg/TwistStamped`    | Commanded body rates |
| `/drone0/self_localization/pose`      | `geometry_msgs/msg/PoseStamped`     | Current pose (earth) |
| `/drone0/self_localization/twist`     | `geometry_msgs/msg/TwistStamped`    | Current linear + body angular |
| `/drone0/sensor_measurements/odom`    | `nav_msgs/msg/Odometry`             | Current state + zero covariance |
| `/clock`                              | `rosgraph_msgs/msg/Clock`           | Auto-emitted with every save |

Topic names are defaults; they can be overridden via `LoggerConfig` or setter
methods **before** `start()`. Extras are registered with `add_*_topic(...)`.

## Installing the dependencies

Dependency resolution follows three tiers, in this order: (1) system, (2)
`thirdparty/<dep>/` vendored, (3) `FATAL_ERROR` pointing here. `cmake/mfm_require_dep.cmake`
implements the policy.

**System packages (Ubuntu 22.04):**

```bash
sudo apt install \
  cmake g++ \
  libeigen3-dev \
  liblz4-dev libzstd-dev \
  ros-humble-fastcdr      # just the CDR library, no ROS 2 runtime
# Optional, for the pybind bindings:
sudo apt install pybind11-dev python3-dev python3-pybind11
# Optional, for tests:
sudo apt install libgtest-dev
# Optional, for the viewer:
pip install matplotlib mcap numpy
```

`ros-humble-fastcdr` is **only the serialization library** — the rest of the
ROS 2 stack (rclcpp, ament, rosidl) is not required.

**Vendored fallback:** if `fastcdr` is not on the system, clone it under
`thirdparty/fastcdr/`:

```bash
cd thirdparty && git clone --depth 1 --branch 1.0.29 \
    https://github.com/eProsima/Fast-CDR.git fastcdr
```

`thirdparty/mcap/` (Foxglove MCAP headers) is vendored already. Fast-CDR and
MCAP are **Apache-2.0** and **BSD-3-Clause** respectively, both compatible
with this project's BSD-3-Clause license.

## Building (C++)

```bash
cmake -B build -DMAV_FLIGHT_MCAP_BUILD_EXAMPLES=ON -DMAV_FLIGHT_MCAP_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Outputs: `build/libmav_flight_mcap.a`, the example
`build/examples/basic_cpp_logging`, and three gtest binaries.

## Using the C++ API

```cpp
#include <Eigen/Core>
#include "mav_flight_mcap/mcap_logger.hpp"

using namespace mav_flight_mcap;

LoggerConfig cfg;
cfg.file_path = "/tmp/flight.mcap";
cfg.time_mode = TimeMode::SIMULATION;  // or GLOBAL

MCAPLogger logger(cfg);
logger.add_float64_topic("/drone0/debug/solve_time_us");
logger.start();

// Single-topic saves
logger.save_pose_state(t, pos, /*quat_wxyz=*/quat);
// Aggregate saves (one call writes multiple topics)
logger.save_state(t, pos_earth, quat_wxyz, linear_earth, angular_body);
logger.save_actuation(t, thrust_n, angular_cmd_body);
logger.save_position_reference(t, pos_ref, quat_ref, max_speed);
// Extras
logger.save_float64("/drone0/debug/solve_time_us", t, 1.4e-3);

logger.close();
```

Quaternion order on this API is `[w, x, y, z]` throughout.

## Building and installing the Python bindings

```bash
# Direct CMake (fastest; for development)
cd pybind && cmake -B build && cmake --build build -j
# The .so lands in pybind/python/mav_flight_mcap/_logger_cpp*.so

# Or: pip install (produces a wheel via scikit-build-core)
pip install ./pybind
```

Run the bindings' test suite:

```bash
PYTHONPATH=pybind/python python3 -m pytest pybind/tests/
```

## Using the Python API

```python
import numpy as np
from mav_flight_mcap import LoggerConfig, MCAPLogger, TimeMode, TrajectoryPoint

cfg = LoggerConfig()
cfg.file_path = "/tmp/flight.mcap"
cfg.time_mode = TimeMode.SIMULATION

logger = MCAPLogger(cfg)
logger.add_float64_topic("/drone0/debug/solve_time_us")
logger.start()

logger.save_state(t,
                  pos_earth=np.array([1, 2, 3]),
                  quat_wxyz=np.array([1, 0, 0, 0]),
                  linear_earth=np.array([1, 0, 0]),
                  angular_body=np.array([0, 0, 0.5]))
logger.save_actuation(t, thrust=9.81, angular_command_body=np.array([0.1, 0.1, 0.1]))
logger.save_float64("/drone0/debug/solve_time_us", t, 1.4e-3)

logger.close()
# Or use it as a context manager (extras must be registered before entering):
```

Full example: `examples/basic_python_logging.py`.

## Visualising a bag

```bash
pip install ./viewer
mav-view samples/flight_cpp.mcap
# Save to PNG without a GUI:
mav-view samples/flight_cpp.mcap --save /tmp/flight.png
```

The dashboard shows position, linear velocity and |v| modulus, angular
velocity (body), commanded angular velocity, thrust, position and speed
references, position tracking error, a topic overview, and any extra scalar
topics.

Topic mapping can be overridden if your bag uses different names:

```bash
mav-view bag.mcap --pose-state /my_drone/pose --thrust-command /my_drone/thrust
```

The viewer does not depend on the C++ library; it only needs
`pip install mcap matplotlib numpy`.

## Repository layout

```
mav_flight_mcap/
├── CMakeLists.txt              # top-level build with the deps policy
├── cmake/mfm_require_dep.cmake # system -> vendored -> FATAL helper
├── include/mav_flight_mcap/    # public C++ API (headers)
├── src/                        # core implementation
├── examples/                   # C++ and Python end-to-end examples
├── tests/                      # gtest (15 tests)
├── pybind/                     # pybind11 bindings + wrapper + 6 pytest tests
├── viewer/                     # matplotlib dashboard + CDR decoder + 4 tests
├── thirdparty/                 # mcap/ (vendored), fastcdr/ (optional vendor)
└── deprecated/                 # previous Protobuf/Rerun implementation
```

## How it works under the hood

- **No ROS 2 dependency:** the library embeds every needed `.msg` schema as
  `constexpr` strings (`include/mav_flight_mcap/ros2/schemas.hpp`) and
  serializes payloads with Fast-CDR forced to
  `LITTLE_ENDIANNESS` / `DDS_CDR`. `mcap::McapWriter` produces the container
  with `schema_encoding="ros2msg"` / `message_encoding="cdr"` exactly as
  rosbag2 Humble does.
- **Time:** `TimeMode::SIMULATION` zeroes time on the first save; `GLOBAL`
  passes through the user-supplied seconds. `/clock` is auto-emitted with
  every `save_*` call (rate-limited by `clock_min_period_s` if desired).
- **Schemas:** registered once per type and reused across topics.
- **CDR compatibility:** the viewer's pure-Python `CdrReader`
  (`viewer/src/mav_flight_viewer/cdr_decoder.py`) accepts bytes produced by
  Fast-CDR and vice versa — the cross-stack tests in `viewer/tests/` lock
  this down.

## Verifying a bag against ROS 2 Humble

This repo's tests already validate ROS-2-compatible MCAP structure and
roundtrip CDR. If you have a machine with ROS 2 Humble + `rosbag2_storage_mcap`:

```bash
ros2 bag info samples/flight_cpp.mcap
ros2 bag play samples/flight_cpp.mcap
ros2 topic echo /drone0/self_localization/pose  # in another terminal
```

Topic names, types and payload fields must match the defaults above.

## License

BSD-3-Clause (see `LICENSE`). Vendored dependencies keep their original
licenses: MCAP is BSD-3-Clause (Foxglove), Fast-CDR is Apache-2.0 (eProsima).

## Deprecated code

The previous implementation (custom Protobuf schema + Rerun viewer) lives
under `deprecated/` for reference. It is not built by the current CMake.
