find_package(yaml-cpp REQUIRED)

add_executable(run_example run_example.cpp)
target_link_libraries(run_example PRIVATE
  mav_trajectory_generation_cpp
  yaml-cpp)
target_compile_options(run_example PRIVATE -O3 -Wall -Wextra)
