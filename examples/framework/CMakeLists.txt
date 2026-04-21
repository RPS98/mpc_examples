# Copyright 2025 Universidad Politécnica de Madrid
# Author: Rafael Perez-Segui <r.psegui@upm.es>
# SPDX-License-Identifier: BSD-3-Clause

# mpc_examples_framework: shared OOP scaffolding (IController, ITrajectoryGenerator,
# WaypointsSimulator) consumed by every controller+generator example main.

add_library(mpc_examples_framework STATIC
  src/waypoints_simulator.cpp
)

target_include_directories(mpc_examples_framework PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/..   # exposes "utils/..." shared headers
)

target_link_libraries(mpc_examples_framework PUBLIC
  mav_simulator   # INTERFACE target providing mav_model / simulator headers + Eigen
  yaml-cpp        # consumed via utils/example_config_utils.hpp
)

target_compile_features(mpc_examples_framework PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Pure-Python mirror: expose examples/framework/python/mpc_examples_framework
# under ${PYBIND_PY_MIRROR_ROOT}/mpc_examples_framework via a live symlink, so
# ``PYTHONPATH=${PYBIND_PY_MIRROR_ROOT}`` is enough for the 12 run_example.py
# mains to import the framework (same pattern used by thirdparty/mpc).
# ---------------------------------------------------------------------------
if(DEFINED PYBIND_PY_MIRROR_ROOT)
  file(MAKE_DIRECTORY "${PYBIND_PY_MIRROR_ROOT}")

  set(_fw_src "${CMAKE_CURRENT_SOURCE_DIR}/python/mpc_examples_framework")
  set(_fw_dst "${PYBIND_PY_MIRROR_ROOT}/mpc_examples_framework")

  if(EXISTS "${_fw_dst}" OR IS_SYMLINK "${_fw_dst}")
    file(REMOVE_RECURSE "${_fw_dst}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_fw_src}" "${_fw_dst}"
    RESULT_VARIABLE _fw_symlink_rc
  )
  if(NOT _fw_symlink_rc EQUAL 0)
    message(WARNING
      "mpc_examples_framework: failed to symlink ${_fw_src} -> ${_fw_dst} (rc=${_fw_symlink_rc})")
  else()
    message(STATUS "mpc_examples_framework: mirrored -> ${_fw_dst}")
  endif()
endif()
