# Copyright 2025 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# mfm_require_dep: locate a dependency following the project policy:
#   1. find_package(<SYSTEM_PKG>) on the system (standard CMake paths plus
#      any extra PATHS caller supplies through EXTRA_PATHS).
#   2. add_subdirectory(<VENDORED_DIR>) if it contains a CMakeLists.txt.
#   3. FATAL_ERROR pointing to README.md.
#
# Usage:
#   mfm_require_dep(
#     NAME          fastcdr
#     SYSTEM_PKG    fastcdr
#     VENDORED_DIR  ${CMAKE_SOURCE_DIR}/thirdparty/fastcdr
#     EXTRA_PATHS   /opt/ros/humble /opt/ros/iron   # optional
#     CONFIG                                        # use CONFIG mode
#   )
function(mfm_require_dep)
  set(options CONFIG)
  set(oneValueArgs NAME SYSTEM_PKG VENDORED_DIR)
  set(multiValueArgs EXTRA_PATHS)
  cmake_parse_arguments(MFM "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT MFM_NAME OR NOT MFM_SYSTEM_PKG)
    message(FATAL_ERROR "mfm_require_dep: NAME and SYSTEM_PKG are required.")
  endif()

  set(_config_arg "")
  if(MFM_CONFIG)
    set(_config_arg CONFIG)
  endif()

  # 1. System
  find_package(${MFM_SYSTEM_PKG} ${_config_arg} QUIET)
  if(${MFM_SYSTEM_PKG}_FOUND)
    message(STATUS "mfm: using system ${MFM_NAME}")
    return()
  endif()

  # 1b. Extra paths (e.g. /opt/ros/humble for ros-humble-fastcdr)
  if(MFM_EXTRA_PATHS)
    find_package(${MFM_SYSTEM_PKG} ${_config_arg} QUIET
                 PATHS ${MFM_EXTRA_PATHS} NO_DEFAULT_PATH)
    if(${MFM_SYSTEM_PKG}_FOUND)
      message(STATUS "mfm: using ${MFM_NAME} from extra path")
      # Re-export to parent scope (find_package already did, but be explicit).
      set(${MFM_SYSTEM_PKG}_FOUND TRUE PARENT_SCOPE)
      return()
    endif()
  endif()

  # 2. Vendored
  if(MFM_VENDORED_DIR AND EXISTS "${MFM_VENDORED_DIR}/CMakeLists.txt")
    message(STATUS "mfm: using vendored ${MFM_NAME} at ${MFM_VENDORED_DIR}")
    add_subdirectory(${MFM_VENDORED_DIR}
                     ${CMAKE_BINARY_DIR}/_vendor/${MFM_NAME}
                     EXCLUDE_FROM_ALL)
    return()
  endif()

  # 3. FATAL
  message(FATAL_ERROR
    "mav_flight_mcap: dependency '${MFM_NAME}' not found on the system, "
    "and no vendored fallback at '${MFM_VENDORED_DIR}'. "
    "See README.md for installation instructions.")
endfunction()
