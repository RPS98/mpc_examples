# mav_trajectory_generation_lib

Pure C++ (ROS-free) wrapper around ETH-ASL
[mav_trajectory_generation](https://github.com/ethz-asl/mav_trajectory_generation),
with optional Python bindings (pybind11) and an end-to-end CLI example.

The upstream core (under [`mav_trajectory_generation/`](./mav_trajectory_generation/))
is included as an **immutable git submodule**. This repository only adds an
outer CMake build, a small C++ facade, Python bindings, tests and a sample —
no upstream file is modified.

## Layout

```
mav_trajectory_generation_lib/
├── CMakeLists.txt                 # top-level build
├── cmake/                         # glog + NLopt dependency shims (find_package | FetchContent)
├── mav_trajectory_generation/     # upstream submodule (UNTOUCHED)
├── include/mav_trajectory_generation_cpp/
│   ├── types.hpp                  # OptimizationConfig, Solver, TrajectorySample
│   └── trajectory_generator.hpp   # Pimpl facade
├── src/trajectory_generator.cpp   # Linear + Nonlinear dispatch on the upstream core
├── pybind/                        # pybind11 bindings + `mav_trajectory_generation_py`
├── example/                       # CLI: run_example.cpp, run_example.py, sample YAMLs
└── tests/                         # GoogleTest facade tests
```

## Scope

* **Included** (compiled into the facade): `polynomial`, `vertex`, `segment`,
  `trajectory`, `rpoly`, `motion_defines`, `timing`,
  `polynomial_optimization_{linear,nonlinear}`.
* **Excluded from the build** (still present in the submodule, untouched):
  `trajectory_sampling.cpp` and `io.cpp`. Both transitively depend on
  `mav_msgs::EigenTrajectoryPoint` (a ROS-only type). The facade samples
  trajectories natively through `Trajectory::evaluate(t, derivative_order)`.

## Requirements

* C++17 compiler, CMake ≥ 3.16.
* [Eigen 3](https://eigen.tuxfamily.org/), [yaml-cpp](https://github.com/jbeder/yaml-cpp).
* `glog` and `NLopt`. If not installed system-wide, they are fetched
  automatically via CMake `FetchContent` (see [`cmake/`](./cmake/)).
* Optional: [`pybind11`](https://github.com/pybind/pybind11) + Python ≥ 3.8 for
  the Python bindings.
* Optional: GoogleTest for the C++ test suite.

## Build

```bash
git clone --recurse-submodules <this-repo>
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_PYBIND=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful options:

| Option | Default | Effect |
| --- | --- | --- |
| `BUILD_TESTING` | `ON` (via `CTest`) | Build + register the GoogleTest suite. |
| `BUILD_EXAMPLES` | `OFF` | Build [`example/run_example`](./example/run_example.cpp). |
| `BUILD_PYBIND` | `OFF` | Build the `_mav_trajectory_generation_bindings` pybind11 module and mirror the `mav_trajectory_generation_py` package into the build tree. |

## Using from another CMake project

```cmake
add_subdirectory(path/to/mav_trajectory_generation_lib)
target_link_libraries(my_app PRIVATE mav_trajectory_generation_cpp)
```

The library target is `mav_trajectory_generation_cpp` (public header search
path: `include/`). The upstream core is an internal detail, exported as the
static target `mav_trajectory_generation_core`. The name intentionally
differs from `mav_trajectory_generation`, so this library can coexist with
other projects that also vendor the upstream (e.g. `dynamic_trajectory_generator`).

## C++ API snippet

```cpp
#include <mav_trajectory_generation_cpp/trajectory_generator.hpp>

using namespace mav_trajectory_generation_cpp;

OptimizationConfig cfg;              // defaults: Linear, minimise snap, a_max = 4 m/s^2
cfg.solver = Solver::Nonlinear;      // switch to NLopt-backed solver

TrajectoryGenerator gen(cfg);
std::vector<Waypoint> waypoints = {
    EndWaypoint(Eigen::Vector3d(0, 0, 1)),   // start at rest
    Waypoint(Eigen::Vector3d(5, 2, 1.5)),    // intermediate (free vel/acc)
    EndWaypoint(Eigen::Vector3d(8, 0, 2)),   // end at rest
};
if (gen.generate(waypoints, /*max_speed=*/3.0)) {
  const auto s = gen.evaluate(0.5 * gen.duration());
  // s.position, s.velocity, s.acceleration
}

// The same generator can be reused: chain a second trajectory to the final
// state of the first (smooth velocity continuity at the seam).
Waypoint chained_start(Eigen::Vector3d(8, 0, 2));
chained_start.velocity = gen.evaluate(gen.maxTime()).velocity;
gen.generate({chained_start, EndWaypoint(Eigen::Vector3d(0, 0, 1))}, 3.0);
```

## Python API snippet

```python
import numpy as np
from mav_trajectory_generation_py import (
    EndWaypoint, GeneratorConfig, Trajectory, Waypoint,
)

cfg = GeneratorConfig(solver="nonlinear", a_max=4.0)
traj = Trajectory(cfg)
traj.generate(
    [
        EndWaypoint(np.array([0, 0, 1])),
        Waypoint(np.array([5, 2, 1.5])),
        EndWaypoint(np.array([8, 0, 2])),
    ],
    max_speed=3.0,
)
point = traj.evaluate(0.5 * traj.duration)
print(point.position, point.velocity, point.acceleration)
```

## Python install

```bash
pip install ./pybind
```

The package uses [scikit-build-core](https://scikit-build-core.readthedocs.io/)
and ships the native module plus the required shared libraries with
`RPATH="$ORIGIN"`, so the resulting wheel is self-contained.

Additional details in [`pybind/README.md`](./pybind/README.md).

## CLI example

```bash
cmake --build build --target run_example
cd example
./run_example_cpp.sh     # plans all trajectories in config_example.yaml
                         # and produces one CSV + one pair of plots each.
```

The example declares three back-to-back trajectories to exercise:
1. a multi-waypoint path with endpoints at rest,
2. a chained trajectory that inherits the final velocity of the previous one, and
3. a single-waypoint case that gracefully fails (`generate()` returns `false`).

The same `TrajectoryGenerator` instance is reused across all three calls —
it is not destroyed between trajectories. See
[`example/run_example.cpp`](./example/run_example.cpp) for details. A Python
mirror lives at [`example/run_example.py`](./example/run_example.py) and
produces byte-identical CSVs when launched via `./run_example_py.sh`.

The Python mirror produces a numerically identical CSV:

```bash
PYTHONPATH=build/pybind/python python example/run_example.py \
    example/config_example.yaml \
    example/config_trajectory.yaml
```

## Upstream drift

If the ETH-ASL submodule ever adds or renames `.cpp` files, the list
`MTG_CORE_SRC` in [`CMakeLists.txt`](./CMakeLists.txt) must be updated to
keep the facade in sync. Nothing else in the upstream tree should need
editing.

## License

Apache-2.0 (matches the upstream ETH-ASL core). See [`LICENSE`](./LICENSE).
The upstream algorithmic code is authored by the ETH-ASL team (Achtelik,
Burri, Oleynikova, Bähnemann, Popović); this wrapper is a thin outer layer
contributed on top.
