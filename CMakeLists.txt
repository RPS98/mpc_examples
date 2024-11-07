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

add_subdirectory(thirdparty/mpc)
add_subdirectory(thirdparty/multirotor_simulator)

# find dynamic_trajectory_generator
if(${dynamic_trajectory_generator_FOUND})
  MESSAGE(STATUS "dynamic_trajectory_generator found")
else(${dynamic_trajectory_generator_FOUND})
  MESSAGE(STATUS "dynamic_trajectory_generator not found")
  include(FetchContent)
  FetchContent_Declare(
    dynamic_trajectory_generator
    GIT_REPOSITORY https://github.com/miferco97/dynamic_trajectory_generator.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(dynamic_trajectory_generator)
endif(${dynamic_trajectory_generator_FOUND})

# find dependencies
set(PROJECT_DEPENDENCIES
)

foreach(DEPENDENCY ${PROJECT_DEPENDENCIES})
  find_package(${DEPENDENCY} REQUIRED)
endforeach()

include_directories(
  examples
)

set(SOURCE_CPP_FILES
  examples/integrator_example.cpp
)

# Add executable for each example
file(GLOB_RECURSE EXAMPLES_SOURCE "mpc_examples/*_example.cpp")
foreach(EXAMPLE_FILE ${EXAMPLES_SOURCE})
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)
  MESSAGE(STATUS "Adding example: ${EXAMPLE_NAME}")

  add_executable(${PROJECT_NAME}_${EXAMPLE_NAME} ${EXAMPLE_FILE})

  target_link_libraries(${PROJECT_NAME}_${EXAMPLE_NAME}
    ${PROJECT_DEPENDENCIES}
    acados_mpc
    multirotor_simulator
    dynamic_trajectory_generator
  )
endforeach()
