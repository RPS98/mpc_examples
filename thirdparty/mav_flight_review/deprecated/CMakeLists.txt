cmake_minimum_required(VERSION 3.16)
project(mav_flight_mcap LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(Protobuf REQUIRED)

# MCAP: vendored header-only library under thirdparty/mcap (Foxglove v1.4.1).
# MCAP is header-only + STB-style: exactly one translation unit must define
# MCAP_IMPLEMENTATION before including <mcap/mcap.hpp>.
set(_mcap_vendored_include
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mcap/include)
set(_mcap_impl_file ${CMAKE_CURRENT_BINARY_DIR}/mcap_impl.cpp)
file(WRITE ${_mcap_impl_file}
  "#define MCAP_IMPLEMENTATION\n"
  "#include <mcap/mcap.hpp>\n")

add_library(mav_flight_mcap_vendored STATIC ${_mcap_impl_file})
set_target_properties(mav_flight_mcap_vendored PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(mav_flight_mcap_vendored PUBLIC
  ${_mcap_vendored_include})

# MCAP needs lz4 and zstd at link time.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(LZ4  QUIET liblz4)
  pkg_check_modules(ZSTD QUIET libzstd)
endif()
if(LZ4_FOUND)
  target_include_directories(mav_flight_mcap_vendored PUBLIC ${LZ4_INCLUDE_DIRS})
  target_link_libraries(mav_flight_mcap_vendored PUBLIC ${LZ4_LIBRARIES})
else()
  target_link_libraries(mav_flight_mcap_vendored PUBLIC lz4)
endif()
if(ZSTD_FOUND)
  target_include_directories(mav_flight_mcap_vendored PUBLIC ${ZSTD_INCLUDE_DIRS})
  target_link_libraries(mav_flight_mcap_vendored PUBLIC ${ZSTD_LIBRARIES})
else()
  target_link_libraries(mav_flight_mcap_vendored PUBLIC zstd)
endif()

# ----------------------------------------------------------------------------
# Generated protobuf C++ sources
# ----------------------------------------------------------------------------
set(_proto_src ${CMAKE_CURRENT_SOURCE_DIR}/proto/mav_flight.proto)
set(_proto_out_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${_proto_out_dir})

set(_proto_cc ${_proto_out_dir}/mav_flight.pb.cc)
set(_proto_h  ${_proto_out_dir}/mav_flight.pb.h)

add_custom_command(
  OUTPUT ${_proto_cc} ${_proto_h}
  COMMAND protobuf::protoc
          --cpp_out=${_proto_out_dir}
          --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
          ${_proto_src}
  DEPENDS ${_proto_src} protobuf::protoc
  COMMENT "Generating protobuf C++ sources for mav_flight.proto"
  VERBATIM)

add_library(mav_flight_mcap_proto STATIC ${_proto_cc} ${_proto_h})
target_include_directories(mav_flight_mcap_proto PUBLIC ${_proto_out_dir})
target_link_libraries(mav_flight_mcap_proto PUBLIC protobuf::libprotobuf)
set_target_properties(mav_flight_mcap_proto PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------
# Header-only recorder target
# ----------------------------------------------------------------------------
add_library(mav_flight_mcap INTERFACE)
target_include_directories(mav_flight_mcap INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mav_flight_mcap INTERFACE
  Eigen3::Eigen
  protobuf::libprotobuf
  mav_flight_mcap_proto
  mav_flight_mcap_vendored)

add_library(mav_flight_mcap::mav_flight_mcap ALIAS mav_flight_mcap)

# ----------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------
option(MAV_FLIGHT_MCAP_BUILD_EXAMPLES "Build example executables" ON)
if(MAV_FLIGHT_MCAP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
