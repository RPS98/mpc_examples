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
//    * Neither the name of the Universidad Politécnica de Madrid nor the
//      names of its contributors may be used to endorse or promote products
//      derived from this software without specific prior written permission.
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
 * @file waypoints_simulator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "framework/waypoints_simulator.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/types.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::framework {

namespace {

std::string describeFieldsMask(ReferenceFieldMask mask) {
  std::string out;
  auto append = [&](const char* name, ReferenceField f) {
    if (hasField(mask, f)) {
      if (!out.empty()) out += ", ";
      out += name;
    }
  };
  append("position", ReferenceField::kPosition);
  append("velocity", ReferenceField::kVelocity);
  append("acceleration", ReferenceField::kAcceleration);
  return out.empty() ? "<none>" : out;
}

bool isFiniteVector3(const Eigen::Vector3d& v) {
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

bool isFiniteQuaternion(const Eigen::Quaterniond& q) {
  return std::isfinite(q.w()) && std::isfinite(q.x()) && std::isfinite(q.y()) &&
         std::isfinite(q.z());
}

double meanOfTimesUs(const std::vector<double>& seconds) {
  if (seconds.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (double x : seconds) {
    sum += x;
  }
  return sum / static_cast<double>(seconds.size()) * 1e6;
}

}  // namespace

WaypointsSimulator::WaypointsSimulator(
    std::unique_ptr<IController> controller,
    std::unique_ptr<ITrajectoryGenerator> traj_gen,
    const ExampleConfig& example_cfg,
    const mav_simulator::SimulatorParameters& simulator_params,
    const std::string& output_csv)
    : controller_(std::move(controller)),
      traj_gen_(std::move(traj_gen)),
      example_cfg_(example_cfg),
      output_csv_(output_csv),
      sim_(simulator_params) {
  if (!controller_) {
    throw std::invalid_argument("WaypointsSimulator: controller must not be null.");
  }
  if (!traj_gen_) {
    throw std::invalid_argument("WaypointsSimulator: trajectory generator must not be null.");
  }
  if (example_cfg_.waypoints.empty()) {
    throw std::invalid_argument("WaypointsSimulator: example_cfg.waypoints is empty.");
  }

  checkCompatibility_();
}

void WaypointsSimulator::checkCompatibility_() const {
  const ReferenceFieldMask required = controller_->requiredReferenceFields();
  const ReferenceFieldMask provided = traj_gen_->providedReferenceFields();
  const ReferenceFieldMask missing  = required & ~provided;
  if (missing != 0u) {
    std::cerr << "[WaypointsSimulator] Warning: generator '" << traj_gen_->name()
              << "' does not produce field(s) required by controller '" << controller_->name()
              << "': " << describeFieldsMask(missing)
              << ". Missing fields will be held at zero; controller may not converge as expected."
              << std::endl;
  }
}

void WaypointsSimulator::run() {
  // --- Simulator setup ------------------------------------------------------
  sim_.arm();
  sim_.setControlMode(mav_simulator::ControlMode::RATES);

  const mav_model::State& initial_state = sim_.getState();
  controller_->initialize(initial_state, example_cfg_);
  traj_gen_->initialize(example_cfg_.waypoints, initial_state, example_cfg_);

  // --- Timing parameters ----------------------------------------------------
  const double model_dt      = example_cfg_.model_dt;
  const double controller_dt = example_cfg_.controller_dt;
  const double outer_dt      = controller_->controlPeriod();
  const double hover_time    = example_cfg_.hover_time;
  const double max_sim_time  = example_cfg_.sim_time + hover_time;
  const bool benchmark       = example_cfg_.benchmark;
  const bool silent          = example_cfg_.silent;
  const double max_speed     = example_cfg_.max_speed;

  const int N_samples   = controller_->referenceHorizonSize();
  const double dt_h     = controller_->referenceHorizonDt();
  std::vector<ReferenceSample> refs(static_cast<std::size_t>(N_samples));

  // --- Logging --------------------------------------------------------------
  std::unique_ptr<CsvLogger> logger;
  if (!benchmark) {
    logger = std::make_unique<CsvLogger>(output_csv_);
  }

  // --- Benchmark buffers ----------------------------------------------------
  const auto est_outer_steps = static_cast<std::size_t>(max_sim_time / outer_dt) + 1U;
  const auto est_indi_steps  = static_cast<std::size_t>(max_sim_time / controller_dt) + 1U;
  std::vector<double> controller_times;
  std::vector<double> indi_times;
  std::vector<double> imu_times;
  std::vector<double> model_times;
  controller_times.reserve(est_outer_steps);
  indi_times.reserve(est_indi_steps);
  imu_times.reserve(est_indi_steps);
  model_times.reserve(est_indi_steps);

  std::cout << (benchmark ? "[BENCHMARK] " : "") << "Controller   : " << controller_->name()
            << "\n"
            << "Generator    : " << traj_gen_->name() << "\n"
            << "Output file  : " << output_csv_ << "\n"
            << "Waypoints    : " << example_cfg_.waypoints.size() << "\n"
            << "Horizon      : N=" << N_samples << ", dt_h=" << dt_h << " s\n"
            << "Outer period : " << outer_dt << " s\n"
            << "Max sim time : " << max_sim_time << " s\n\n";
  std::cout << "Waypoint 1/" << example_cfg_.waypoints.size() << ": "
            << example_cfg_.waypoints.front().transpose() << "\n";

  // Initial log entry at t=0 with zero references (pre-arm snapshot).
  if (logger) {
    logger->save(0.0, Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                 Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                 Eigen::Quaterniond::Identity(), 0.0, Eigen::Vector3d::Zero(),
                 Eigen::Matrix<double, 4, 1>::Zero(), 0.0, 0, false, max_speed);
  }

  // --- Main loop ------------------------------------------------------------
  ControlCommand cmd;
  ReferenceSample ref0_snapshot;
  bool hover_active     = false;
  double hover_end_time = max_sim_time;
  int last_wp_index     = traj_gen_->currentWaypointIndex();

  const auto wall_start = std::chrono::high_resolution_clock::now();
  double t              = 0.0;

  while (t < max_sim_time + 1e-9) {
    if (hover_active && t >= hover_end_time - 1e-9) {
      break;
    }

    // Read current ground-truth state
    const mav_model::State& state        = sim_.getState();
    const Eigen::Vector3d position       = state.getPositionVector();
    const Eigen::Quaterniond orientation = state.getOrientationVector();
    const Eigen::Vector3d velocity       = state.getLinearVelocityVector();

    if (!isFiniteVector3(position) || !isFiniteQuaternion(orientation) ||
        !isFiniteVector3(velocity)) {
      std::cerr << "Non-finite state at t=" << t << " s\n";
      break;
    }

    // Refresh generator and sample the controller horizon
    traj_gen_->update(t, state);
    for (int k = 0; k < N_samples; ++k) {
      refs[static_cast<std::size_t>(k)] = traj_gen_->evaluate(t + k * dt_h);
    }
    ref0_snapshot = refs.front();

    // Compute control command
    cmd = controller_->computeCommand(state, refs);
    const double solve_us = controller_->lastSolveTimeMicros();
    controller_times.push_back(solve_us * 1e-6);

    // Notify waypoint advances (generator-driven)
    const int current_wp = traj_gen_->currentWaypointIndex();
    if (current_wp != last_wp_index) {
      if (!silent) {
        std::cout << "\nWaypoint " << current_wp + 1 << "/" << example_cfg_.waypoints.size()
                  << ": " << example_cfg_.waypoints[static_cast<std::size_t>(current_wp)].transpose();
      }
      last_wp_index = current_wp;
    }

    // Transition to hover once the generator reports the mission finished
    if (!hover_active && traj_gen_->isFinished(t)) {
      hover_active   = true;
      hover_end_time = t + hover_time;
      if (!silent) {
        std::cout << "\nMission finished. Hovering for " << hover_time << " s";
      }
    }

    const Eigen::Quaterniond ref_orientation = eulerToQuaternion(0.0, 0.0, ref0_snapshot.yaw);

    // --- INDI + physics inner loop (zero-order hold on outer command) -----
    sim_.setReferenceRates(cmd.thrust_n, cmd.angular_rate);

    double t_ctrl = t;
    while (t_ctrl < t + outer_dt - 1e-9) {
      const auto indi_t0 = std::chrono::high_resolution_clock::now();
      sim_.updateController(controller_dt);
      const auto indi_t1 = std::chrono::high_resolution_clock::now();

      sim_.updateImu(controller_dt);
      const auto indi_t2 = std::chrono::high_resolution_clock::now();

      double t_model = t_ctrl;
      while (t_model < t_ctrl + controller_dt - 1e-9) {
        sim_.updateModel(model_dt);
        t_model += model_dt;
      }
      const auto indi_t3 = std::chrono::high_resolution_clock::now();

      indi_times.push_back(std::chrono::duration<double>(indi_t1 - indi_t0).count());
      imu_times.push_back(std::chrono::duration<double>(indi_t2 - indi_t1).count());
      model_times.push_back(std::chrono::duration<double>(indi_t3 - indi_t2).count());

      t_ctrl += controller_dt;

      if (logger) {
        const mav_model::State& s = sim_.getState();
        logger->save(t_ctrl, s.getPositionVector(), s.getOrientationVector(),
                     s.getLinearVelocityVector(), s.getAngularVelocityVector(),
                     ref0_snapshot.position, ref_orientation, cmd.thrust_n, cmd.angular_rate,
                     s.getMotorAngularVelocityVector(), solve_us,
                     traj_gen_->currentWaypointIndex(), hover_active, max_speed);
      }
    }

    t += outer_dt;
    if (!silent) {
      printProgress(t / max_sim_time);
    }
  }

  const auto wall_end = std::chrono::high_resolution_clock::now();
  const double real_time_s = std::chrono::duration<double>(wall_end - wall_start).count();

  stats_ = BenchmarkStats{
      .simulated_time_s   = t,
      .real_time_s        = real_time_s,
      .sim_speedup        = (real_time_s > 0.0) ? (t / real_time_s) : 0.0,
      .controller_mean_us = meanOfTimesUs(controller_times),
      .indi_mean_us       = meanOfTimesUs(indi_times),
      .imu_mean_us        = meanOfTimesUs(imu_times),
      .model_mean_us      = meanOfTimesUs(model_times),
      .controller_steps   = controller_times.size(),
      .indi_steps         = indi_times.size(),
  };
}

void WaypointsSimulator::printBenchmark() const {
  std::cout << "\n\nSimulated time      : " << stats_.simulated_time_s << " s\n"
            << "Real time           : " << stats_.real_time_s << " s\n"
            << "Sim speedup         : " << stats_.sim_speedup << "x\n"
            << "Controller compute  : " << stats_.controller_mean_us << " µs (avg, "
            << stats_.controller_steps << " steps)\n"
            << "INDI controller     : " << stats_.indi_mean_us << " µs (avg)\n"
            << "IMU update          : " << stats_.imu_mean_us << " µs (avg)\n"
            << "Physics model       : " << stats_.model_mean_us << " µs (avg, per INDI step)\n";
}

}  // namespace mpc_examples::framework
