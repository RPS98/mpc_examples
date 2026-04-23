// Minimal example: write an MCAP flight log from C++ that the Python viewer
// (mav-view) can open identically. Simulates a helical trajectory at 100 Hz
// for 10 s with two extra scalar fields.

#include <mav_flight_mcap.hpp>

#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
  const std::string output =
      (argc > 1) ? argv[1] : "flight_cpp.mcap";

  constexpr int n_motors       = 4;
  constexpr double dt          = 0.01;    // 100 Hz
  constexpr double duration    = 10.0;
  constexpr double radius      = 2.0;
  constexpr double ascent_rate = 0.2;     // [m/s]
  constexpr double hover_w     = 500.0;   // [rad/s]

  const std::vector<std::string> extra_fields = {"solve_time_us",
                                                 "waypoint_index"};

  mav_flight_mcap::MCAPRecorder recorder(output, n_motors, extra_fields);

  const int n_steps = static_cast<int>(duration / dt);
  for (int k = 0; k <= n_steps; ++k) {
    const double t     = k * dt;
    const double theta = 0.5 * t;

    const Eigen::Vector3d position(radius * std::cos(theta),
                                   radius * std::sin(theta),
                                   1.0 + ascent_rate * t);
    const Eigen::Vector3d linear_velocity(
        -radius * 0.5 * std::sin(theta), radius * 0.5 * std::cos(theta),
        ascent_rate);
    const Eigen::Vector3d angular_velocity(0.0, 0.0, 0.5);
    const Eigen::Quaterniond orientation =
        mav_flight_mcap::eulerToQuaternion(0.0, 0.0, theta);

    const Eigen::Vector3d reference_position  = position;
    const Eigen::Vector3d reference_velocity  = linear_velocity;
    const Eigen::Quaterniond reference_orientation =
        mav_flight_mcap::eulerToQuaternion(0.0, 0.0, theta + 0.05);
    const Eigen::Vector3d reference_angular_velocity(0.0, 0.0, 0.5);

    const double thrust = 9.81;
    const Eigen::Vector3d command_angular_velocity(0.0, 0.0, 0.5);

    Eigen::VectorXd motor_w(n_motors);
    motor_w << hover_w, hover_w, hover_w, hover_w;

    const std::vector<double> extras = {
        850.0 + 10.0 * std::sin(3.0 * t),          // solve_time_us
        static_cast<double>(k / 200),              // waypoint_index
    };

    recorder.save(t, position, orientation, linear_velocity, angular_velocity,
                  reference_position, reference_velocity,
                  reference_orientation, reference_angular_velocity, thrust,
                  command_angular_velocity, motor_w, extras);

    if (k % 100 == 0) {
      mav_flight_mcap::printProgress(static_cast<double>(k) /
                                     static_cast<double>(n_steps));
    }
  }
  std::cout << std::endl
            << "Wrote " << n_steps + 1 << " steps to " << output << std::endl;
  return 0;
}
