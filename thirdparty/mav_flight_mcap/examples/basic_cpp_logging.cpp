// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file basic_cpp_logging.cpp
 *
 * End-to-end example: synthesise a 10 s helical trajectory, log every topic
 * declared in the aerostack2 spec plus one extra debug scalar, and write the
 * result to an MCAP file readable by rosbag2 Humble tooling.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "mav_flight_mcap/mcap_logger.hpp"

namespace {

Eigen::Vector4d yawToQuatWxyz(double yaw) {
  const double half = 0.5 * yaw;
  return Eigen::Vector4d{std::cos(half), 0.0, 0.0, std::sin(half)};
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out_path = (argc > 1) ? argv[1] : "/tmp/flight_cpp.mcap";

  using namespace mav_flight_mcap;

  LoggerConfig cfg;
  cfg.file_path = out_path;
  cfg.time_mode = TimeMode::SIMULATION;

  MCAPLogger logger(cfg);
  logger.add_float64_topic("/drone0/debug/solve_time_us");
  logger.add_int32_topic("/drone0/debug/waypoint_index");
  logger.start();

  // Helical trajectory: radius 2 m, angular rate 1 rad/s, ascent 0.2 m/s.
  constexpr double kDt       = 0.01;         // 100 Hz
  constexpr double kT        = 10.0;         // 10 s
  constexpr double kRadius   = 2.0;
  constexpr double kOmega    = 1.0;
  constexpr double kAscent   = 0.2;
  constexpr double kMaxSpeed = 3.0;

  // Pre-build a sparse trajectory setpoint list (5 waypoints along the helix).
  std::vector<TrajectoryPoint> setpoints;
  for (int k = 0; k < 5; ++k) {
    const double theta = k * (kOmega * kT) / 5.0;
    TrajectoryPoint p;
    p.id           = "wp" + std::to_string(k);
    p.position     = {kRadius * std::cos(theta),
                      kRadius * std::sin(theta),
                      1.0 + kAscent * theta / kOmega};
    p.twist        = {0, 0, 0};
    p.acceleration = {0, 0, 0};
    p.yaw_angle    = static_cast<float>(theta);
    setpoints.push_back(p);
  }

  const int n_steps = static_cast<int>(kT / kDt);
  for (int i = 0; i <= n_steps; ++i) {
    const double t     = i * kDt;
    const double theta = kOmega * t;

    const Eigen::Vector3d pos{kRadius * std::cos(theta),
                              kRadius * std::sin(theta),
                              1.0 + kAscent * t};
    const Eigen::Vector3d vel{-kRadius * kOmega * std::sin(theta),
                              kRadius * kOmega * std::cos(theta),
                              kAscent};
    const Eigen::Vector3d ang_body{0.0, 0.0, kOmega};
    const Eigen::Vector4d q_wxyz = yawToQuatWxyz(theta);

    // State aggregate: pose_state + twist_state + odom_state in one call.
    logger.save_state(t, pos, q_wxyz, vel, ang_body);

    // References: position + max speed; plus full trajectory every 1 s.
    logger.save_position_reference(t, pos, q_wxyz,
                                   Eigen::Vector3d::Constant(kMaxSpeed));
    if (i % 100 == 0) {
      logger.save_trajectory_reference(t, setpoints);
    }

    // Control output: thrust + angular command.
    logger.save_actuation(t, 9.81 * 1.0, ang_body);

    // Debug extras.
    logger.save_float64("/drone0/debug/solve_time_us", t, 800.0 + 10.0 * i);
    logger.save_int32("/drone0/debug/waypoint_index", t,
                      static_cast<int32_t>(i / 200));

    if (i % 100 == 0) {
      std::cout << "  t = " << t << " s  pos = (" << pos.x() << ", "
                << pos.y() << ", " << pos.z() << ")\n";
    }
  }

  logger.close();
  std::cout << "Wrote " << out_path << "\n";
  return 0;
}
