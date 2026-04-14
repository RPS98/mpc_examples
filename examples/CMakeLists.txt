# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

# Add generated acados MPC code
add_subdirectory(acados_position_mpc)

# Integrated example executable
add_executable(${PROJECT_NAME}_run_example run_example.cpp)

target_include_directories(${PROJECT_NAME}_run_example PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}_run_example
  mav_simulator
  acados_position_mpc
  yaml-cpp
)
