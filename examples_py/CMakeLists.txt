# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Python half of the mpc_examples showcase. The package lives at
# ``examples_py/examples_py/`` (pure Python, no pybind11 of its own).
# CMake only has to:
#   1. Publish the package under ${PYBIND_PY_MIRROR_ROOT}/examples_py via a
#      live symlink, so a single ``PYTHONPATH=${PYBIND_PY_MIRROR_ROOT}``
#      exposes both the thirdparty bindings and this package.
#   2. Register the unittest suite under examples_py/tests/ with CTest (via
#      register_pytest_suite() defined at the top level).

if(DEFINED PYBIND_PY_MIRROR_ROOT)
  file(MAKE_DIRECTORY "${PYBIND_PY_MIRROR_ROOT}")

  set(_ep_src "${CMAKE_CURRENT_SOURCE_DIR}/examples_py")
  set(_ep_dst "${PYBIND_PY_MIRROR_ROOT}/examples_py")

  if(EXISTS "${_ep_dst}" OR IS_SYMLINK "${_ep_dst}")
    file(REMOVE_RECURSE "${_ep_dst}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_ep_src}" "${_ep_dst}"
    RESULT_VARIABLE _ep_symlink_rc
  )
  if(NOT _ep_symlink_rc EQUAL 0)
    message(WARNING
      "examples_py: failed to symlink ${_ep_src} -> ${_ep_dst} "
      "(rc=${_ep_symlink_rc})")
  else()
    message(STATUS "examples_py: mirrored -> ${_ep_dst}")
  endif()
endif()
