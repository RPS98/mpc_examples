# Vendored eProsima Fast-CDR (placeholder)

This directory is the **vendored fallback** for
[eProsima Fast-CDR](https://github.com/eProsima/Fast-CDR), used when the
library is not found on the system.

It is empty on purpose. The top-level CMake (`mfm_require_dep` helper)
searches dependencies in this order:

1. `find_package(fastcdr CONFIG)` on the system (includes
   `/opt/ros/humble/lib/cmake/fastcdr/` when `ros-humble-fastcdr` is
   installed via apt — this is **just the serialization library**, not a
   full ROS 2 runtime).
2. `thirdparty/fastcdr/CMakeLists.txt` (this directory).
3. Otherwise: `FATAL_ERROR` pointing to the top-level `README.md`.

## How to populate

Clone a Humble-compatible release:

```bash
cd thirdparty
rm -rf fastcdr
git clone --depth 1 --branch 1.0.29 https://github.com/eProsima/Fast-CDR.git fastcdr
```

Recommended CMake flags for a minimal vendored build (set automatically by
the top-level `CMakeLists.txt`):

- `COMPILE_TOOLS=OFF`
- `BUILD_TESTING=OFF`

License: Apache-2.0 (compatible with this project's BSD-3-Clause).
