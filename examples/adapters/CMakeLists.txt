# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

# Adapters bridge framework interfaces (IController, ITrajectoryGenerator) to
# concrete third-party libraries. Each adapter is its own target; the per-example
# mains link only the pair they need.

add_subdirectory(controllers/pid_geometric)
add_subdirectory(controllers/mpc_position)
add_subdirectory(controllers/mpc_trajectory)
add_subdirectory(trajectory_generators/waypoint_reference)
add_subdirectory(trajectory_generators/jerk_limited)
add_subdirectory(trajectory_generators/gcopter)
add_subdirectory(trajectory_generators/dynamic)

# ---------------------------------------------------------------------------
# Pure-Python mirror: expose examples/adapters/python/mpc_examples_adapters
# under ${PYBIND_PY_MIRROR_ROOT}/mpc_examples_adapters. Same pattern used by
# thirdparty/mpc and examples/framework; keeps a single PYTHONPATH entry.
# ---------------------------------------------------------------------------
if(DEFINED PYBIND_PY_MIRROR_ROOT)
  file(MAKE_DIRECTORY "${PYBIND_PY_MIRROR_ROOT}")

  set(_ad_src "${CMAKE_CURRENT_SOURCE_DIR}/python/mpc_examples_adapters")
  set(_ad_dst "${PYBIND_PY_MIRROR_ROOT}/mpc_examples_adapters")

  if(EXISTS "${_ad_dst}" OR IS_SYMLINK "${_ad_dst}")
    file(REMOVE_RECURSE "${_ad_dst}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_ad_src}" "${_ad_dst}"
    RESULT_VARIABLE _ad_symlink_rc
  )
  if(NOT _ad_symlink_rc EQUAL 0)
    message(WARNING
      "mpc_examples_adapters: failed to symlink ${_ad_src} -> ${_ad_dst} (rc=${_ad_symlink_rc})")
  else()
    message(STATUS "mpc_examples_adapters: mirrored -> ${_ad_dst}")
  endif()
endif()
