# mav_flight_mcap

MCAP logger and [rerun.io](https://rerun.io) dashboard for quadrotor flight
data. Two companion libraries:

- **`mav_flight_mcap`** — one C++ header and one Python module, API-parity, that
  write a `.mcap` file with protobuf-encoded messages (state, reference,
  actuation, user-declared extras).
- **`mav_flight_viewer`** — Python-only, reads an MCAP produced by either
  recorder and streams it into the rerun viewer with a default dashboard
  (3D world view + synchronized time-series panels).

The schema, topic names and channel layout are identical between C++ and
Python; the same `.mcap` can be written by either and opened identically.

## Quick start (Python only, ~30 s)

```bash
pip install -e python/
python examples/basic_python_logging.py /tmp/flight_py.mcap
mav-view /tmp/flight_py.mcap
```

1. installs the `mav_flight_mcap` + `mav_flight_viewer` packages and the
   `mav-view` CLI,
2. writes `/tmp/flight_py.mcap` (1001 state/reference/actuation messages plus
   two extra scalar channels),
3. spawns the rerun viewer with a default dashboard:
   - 3D world view: drone mesh (procedural) with time-varying `Transform3D`,
     reference ghost, trajectory polyline, body-frame arrows and motor discs.
   - Time-series grid: position / velocity / angular velocity / orientation
     (RPY) / actuation (thrust + body-rate cmd) / motor speeds / tracking
     error / extras.
   - Rerun's native timeline cursor + hover readout are enabled on every
     view, as well as pan / zoom / slice-over-time.

## Quick start (C++)

The MCAP C++ library is vendored under
[thirdparty/mcap/](thirdparty/mcap/) (Foxglove v1.4.1, MIT license), so no
MCAP install step is required. Only the following system packages are needed:

```bash
sudo apt install cmake g++ libeigen3-dev libprotobuf-dev protobuf-compiler \
                 liblz4-dev libzstd-dev
```

Then build and run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/examples/basic_cpp_logging /tmp/flight_cpp.mcap
mav-view /tmp/flight_cpp.mcap
```

The C++ log writes the exact same schema and channels as the Python log — the
dashboard is identical. Byte-for-byte the files differ only in the ordering
of internal MCAP chunks; the decoded message stream is equivalent.

In your own CMake project:

```cmake
find_package(mav_flight_mcap CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE mav_flight_mcap::mav_flight_mcap)
```

## Recorder API

Python ([python/src/mav_flight_mcap/recorder.py](python/src/mav_flight_mcap/recorder.py)):

```python
from mav_flight_mcap import MCAPRecorder

with MCAPRecorder("flight.mcap", n_motors=4,
                  extra_fields=["battery_voltage", "solve_time_us"]) as rec:
    rec.save(
        t, position, orientation, linear_velocity, angular_velocity,
        reference_position, reference_velocity, reference_orientation,
        reference_angular_velocity,
        thrust, command_angular_velocity, motor_angular_velocity,
        extras={"battery_voltage": 12.1, "solve_time_us": 850.3},
    )
```

C++ ([include/mav_flight_mcap.hpp](include/mav_flight_mcap.hpp)):

```cpp
#include <mav_flight_mcap.hpp>

mav_flight_mcap::MCAPRecorder recorder(
    "flight.mcap", /*n_motors=*/4,
    /*extra_fields=*/{"battery_voltage", "solve_time_us"});

recorder.save(t, position, orientation,
              linear_velocity, angular_velocity,
              reference_position, reference_velocity,
              reference_orientation, reference_angular_velocity,
              thrust, command_angular_velocity, motor_angular_velocity,
              /*extras=*/{12.1, 850.3});
```

SI units throughout. Quaternions are `[w, x, y, z]` (body → world). Angular
velocities are body frame; linear velocities are world frame. Accelerations
are **not** logged — they are derived in the viewer if needed, via rerun's
internal scalar pipeline or a subclass override.

## MCAP layout

| Topic                       | Schema                                  |
|-----------------------------|-----------------------------------------|
| `/drone/state`              | `mav_flight_mcap.State`                 |
| `/drone/reference`          | `mav_flight_mcap.Reference`             |
| `/drone/actuation`          | `mav_flight_mcap.Actuation`             |
| `/drone/extras/<name>`      | `mav_flight_mcap.Scalar` (one per field)|

Protobuf schema definitions: [proto/mav_flight.proto](proto/mav_flight.proto).
`log_time` and `publish_time` are `int(time * 1e9)` nanoseconds.

## Customizing the dashboard

Every rerun log call sits behind an override-friendly method of
`MCAPViewer`. A typical customization:

```python
import rerun as rr
from mav_flight_viewer import MCAPViewer

class MyViewer(MCAPViewer):
    def _log_state(self, state):
        super()._log_state(state)
        # Extra derived quantity: horizontal speed.
        vx, vy = state.linear_velocity.x, state.linear_velocity.y
        rr.log("scalars/velocity/horizontal", rr.Scalars((vx * vx + vy * vy) ** 0.5))

MyViewer("/tmp/flight.mcap").show()
```

To change the layout, override `blueprint()` to return a different
`rerun.blueprint.Blueprint`. The default is in
[dashboard.py](python/src/mav_flight_viewer/dashboard.py).

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Scope

In scope:
- MCAP logging with runtime-extensible scalar schema.
- Default rerun dashboard (3D world + time-series grid).
- Subclass-based customization of the viewer.

Out of scope:
- CSV export (see the sibling CSV project for that).
- Real-time / follow-tail playback.
- Non-rerun backends.
- Python bindings of the C++ recorder.

## License

BSD-3-Clause.
