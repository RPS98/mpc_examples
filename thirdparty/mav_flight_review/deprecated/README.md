# Deprecated code

This directory holds the previous `mav_flight_mcap` implementation, preserved
verbatim for reference. It is **no longer built, tested or maintained**.

## Why it was deprecated

The previous implementation logged flight data into MCAP using a **custom
Protobuf schema** (`proto/mav_flight.proto` — `Vec3`, `Quaternion`, `State`,
`Reference`, `Actuation`, `Scalar`) and visualised it with a **Rerun.io**
viewer. Both choices made the logs incompatible with the standard ROS 2
Humble tooling (`rosbag2`, `ros2 bag play`, `mcap info`'s schema decoding,
Foxglove Studio with ROS 2 layouts, etc.).

The current implementation (in the repository root) replaces both:

- **Format**: MCAP with `message_encoding="cdr"` + `schema_encoding="ros2msg"`,
  byte-compatible with `rosbag2` bags produced on ROS 2 Humble. No dependency
  on a ROS 2 installation.
- **Types**: `geometry_msgs/{PoseStamped,TwistStamped,Vector3}`,
  `nav_msgs/Odometry`, `as2_msgs/{Thrust,TrajectorySetpoints}`,
  `rosgraph_msgs/Clock`, and extras
  (`std_msgs/{Int32,String,Float64,Float64MultiArray}`).
- **Visualisation**: matplotlib dashboard (ported from the sibling project
  `mav_flight_logger`), launched via `mav-view <path.mcap>`. No Rerun.
- **Python API**: `pybind11` bindings (`_logger_cpp.so`) with a typed wrapper
  (`numpy.typing.NDArray[np.float64]`), replacing the pure-Python
  `mcap-protobuf-support` recorder.

## Contents

| Path                        | What it was                                          |
|-----------------------------|------------------------------------------------------|
| `CMakeLists.txt`            | Old top-level build (Protobuf + MCAP vendored)       |
| `include/mav_flight_mcap.hpp` | Old `MCAPRecorder` C++ header-only class           |
| `proto/mav_flight.proto`    | Custom Protobuf schema                               |
| `examples/`                 | `basic_cpp_logging.cpp`, `basic_python_logging.py`   |
| `python/`                   | `mav_flight_mcap` (Protobuf recorder) + `mav_flight_viewer` (Rerun) |
| `tests/`                    | `test_recorder.py`, `test_viewer.py` (roundtrip, Rerun blueprint) |
| `samples/`                  | Example `.mcap` outputs (Protobuf) and a test CSV    |
| `launch.sh`                 | Helper that opened a sample MCAP in Rerun            |
| `README_old.md`             | Former top-level README                              |

## If you really need to build it

```bash
cd deprecated
cmake -B build -DCMAKE_PREFIX_PATH=$(realpath ../thirdparty/mcap/include)
cmake --build build
```

Expect build failures — the legacy paths (e.g. `thirdparty/mcap/include`
relative to the old root) no longer resolve.
