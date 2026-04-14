cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME mpc_examples)
project(${PROJECT_NAME})

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# set Release as default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_subdirectory(thirdparty/mav_simulator)

add_subdirectory(examples)

# Enable testing
include(CTest)