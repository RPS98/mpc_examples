// Copyright 2025 Universidad Politecnica de Madrid
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
//    * Neither the name of the Universidad Politecnica de Madrid nor the names
//      of its contributors may be used to endorse or promote products derived
//      from this software without specific prior written permission.
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
 * @file run_example.cpp
 *
 * MPC + MAV Simulator integrated example.
 *
 * Simulates a quadcopter following a series of waypoints using three nested
 * control loops running at different rates:
 *
 *   - MPC (100 Hz, mpc_dt):
 *       Reads position/velocity/orientation from the simulator, sets the
 *       position reference, and solves the OCP to get thrust + angular
 *       velocity commands.
 *
 *   - INDI + IMU (500 Hz, controller_dt):
 *       Converts thrust + angular velocity into motor commands (INDI) and
 *       updates the IMU noise model.
 *
 *   - Physics model (1000 Hz, model_dt):
 *       Integrates rigid-body dynamics with the current motor commands.
 *
 * The MPC output is held constant (zero-order hold) across the INDI sub-steps.
 *
 * Usage:
 *   ./mpc_examples_run_example \
 *     -c config_example.yaml \
 *     -s config_simulator.yaml \
 *     -m config_mpc.yaml \
 *     -f mpc_log.csv
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acados_mpc/acados_mpc.hpp"
#include "acados_mpc/acados_mpc_yaml.hpp"
#include "mav_simulator/mav_simulator.hpp"
#include "mav_simulator/simulator_yaml.hpp"

#include "utils/utils.hpp"
#include "utils/yaml_utils.hpp"

namespace mpc_examples {

// ─── Conversion helpers ──────────────────────────────────────────────────────

inline Eigen::Vector3d toEigenVector3(const std::array<double, 3>& v) { return {v[0], v[1], v[2]}; }

inline Eigen::Quaterniond toEigenQuaternion(const std::array<double, 4>& q) {
  // Quaternion convention: [w, x, y, z] (scalar-first)
  return {q[0], q[1], q[2], q[3]};
}

// ─── Orientation helpers ──────────────────────────────────────────────────────

/**
 * @brief Compute desired orientation for the given waypoint.
 *
 * When path_facing is enabled the drone yaws to face the direction of travel.
 * The heading is computed from the XY component of the direction vector; yaw
 * is left unchanged if the drone is already within 0.1 m of the waypoint to
 * avoid discontinuities.
 *
 * @param waypoint        Target waypoint (world frame).
 * @param current_position Current drone position (world frame).
 * @param current_orientation Current drone orientation.
 * @param path_facing     Whether to align yaw with the direction of travel.
 * @return Desired orientation quaternion.
 */
Eigen::Quaterniond getDesiredOrientation(const Eigen::Vector3d& waypoint,
                                         const Eigen::Vector3d& current_position,
                                         const Eigen::Quaterniond& current_orientation,
                                         const bool path_facing) {
  if (!path_facing) {
    // No yaw tracking: keep north-facing (identity) orientation
    return Eigen::Quaterniond::Identity();
  }
  const Eigen::Vector3d diff = waypoint - current_position;
  if (diff.head<2>().norm() < 0.1) {
    // Close enough to waypoint: hold current orientation to avoid spin
    return current_orientation;
  }
  return computePathFacing(diff);
}

/**
 * @brief Set stage-dependent position references from current position to waypoint.
 *
 * Stage k gets:
 *   p_ref_k = p0 + min((k+1) * v_ref * dt_horizon, L) * d_hat,
 * where p0 is the current position, L = ||goal - p0|| and d_hat is the unit
 * direction toward the goal. The terminal stage is naturally clamped to the
 * goal once the accumulated distance reaches L.
 */
void setProgressiveReferences(acados_mpc::MPCData* mpc_data,
                              const Eigen::Vector3d& current_position,
                              const Eigen::Vector3d& goal_position,
                              const Eigen::Quaterniond& desired_orientation,
                              const double v_ref,
                              const double dt_horizon,
                              const int prediction_steps) {
  const Eigen::Vector3d delta = goal_position - current_position;
  const double distance       = delta.norm();

  if (distance < 1e-9) {
    mpc_data->p_params.setDesiredPosition(
        {goal_position.x(), goal_position.y(), goal_position.z()});
  } else {
    const Eigen::Vector3d direction = delta / distance;
    for (int stage = 0; stage <= prediction_steps; ++stage) {
      const double s_k                     = std::min((stage + 1) * v_ref * dt_horizon, distance);
      const Eigen::Vector3d stage_position = current_position + s_k * direction;
      mpc_data->p_params.setDesiredPosition(
          {stage_position.x(), stage_position.y(), stage_position.z()}, stage);
    }
  }

  mpc_data->p_params.setDesiredOrientation({desired_orientation.w(), desired_orientation.x(),
                                            desired_orientation.y(), desired_orientation.z()});
}

// ─── Speed constraint ────────────────────────────────────────────────────────

/**
 * @brief Override the nonlinear constraint upper bound uh = (soft_speed_margin * max_speed)².
 *
 * The soft penalty starts penalizing when ||v|| exceeds soft_speed_margin * max_speed.
 * Called after configureMpcFromYaml() to replace the default uh from the config file.
 */
void updateSpeedConstraint(acados_mpc::MPC& mpc,
                           const double soft_speed_margin,
                           const double max_speed) {
  if constexpr (acados_mpc::NonlinearConstraintBounds::Nh > 0) {
    const double soft_speed = soft_speed_margin * max_speed;
    const std::array<double, acados_mpc::NonlinearConstraintBounds::Nh> uh = {
        {soft_speed * soft_speed}};
    mpc.getNonlinearConstraintBounds()->setUh(uh);
    mpc.updateNonlinearConstraintBounds();
  }
}

// ─── Finite-value guards ─────────────────────────────────────────────────────

bool isFiniteVector3(const Eigen::Vector3d& value) {
  return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

bool isFiniteQuaternion(const Eigen::Quaterniond& value) {
  return std::isfinite(value.w()) && std::isfinite(value.x()) && std::isfinite(value.y()) &&
         std::isfinite(value.z());
}

// ─── Main simulation loop ─────────────────────────────────────────────────────

void run(const ExampleArgs& args) {
  // ── Load configuration ─────────────────────────────────────────────────────
  const ExampleConfig example_cfg = loadExampleConfig(args.example_config_path);
  const mav_simulator::SimulatorParameters sim_params =
      mav_simulator::loadSimulatorParametersFromYaml(args.simulator_config_path);
  const double mpc_soft_speed_margin = loadMpcSoftSpeedMargin(args.mpc_config_path, 1.0);

  // ── Derive loop step counts ────────────────────────────────────────────────
  // All rates must satisfy: model_dt | controller_dt | mpc_dt
  const double model_dt      = example_cfg.model_dt;       // 1000 Hz
  const double controller_dt = example_cfg.controller_dt;  // 500 Hz
  const double mpc_dt        = example_cfg.mpc_dt;         // 100 Hz
  const double sim_time      = example_cfg.sim_time;
  const double total_time    = sim_time + example_cfg.hover_time;

  const int controller_steps_per_mpc   = static_cast<int>(std::round(mpc_dt / controller_dt));
  const int model_steps_per_controller = static_cast<int>(std::round(controller_dt / model_dt));

  // ── Create simulator ───────────────────────────────────────────────────────
  mav_simulator::Simulator sim(sim_params);
  sim.arm();
  // RATES mode: the simulator expects (thrust, angular_velocity) as input.
  // The INDI controller converts these into motor commands each controller step.
  sim.setControlMode(mav_simulator::ControlMode::RATES);

  // ── Create MPC ─────────────────────────────────────────────────────────────
  acados_mpc::MPC mpc;
  acados_mpc::configureMpcFromYaml(mpc, args.mpc_config_path);
  updateSpeedConstraint(mpc, mpc_soft_speed_margin, example_cfg.max_speed);
  // mpc_data gives direct read/write access to state, parameters and actuation
  acados_mpc::MPCData* mpc_data = mpc.getData();
  const int prediction_steps    = mpc.getPredictionSteps();
  const double dt_horizon       = mpc.getPredictionTimeStep();

  // ── Logger ─────────────────────────────────────────────────────────────────
  CsvLogger logger(args.output_file);

  // ── Waypoint tracking ──────────────────────────────────────────────────────
  const std::vector<Eigen::Vector3d>& waypoints = example_cfg.waypoints;
  std::size_t wp_index                          = 0;
  const double v_ref                            = example_cfg.max_speed;

  // ── MPC output (held between MPC steps via zero-order hold) ───────────────
  double mpc_thrust                    = 0.0;
  Eigen::Vector3d mpc_angular_velocity = Eigen::Vector3d::Zero();
  // These are logged at the controller rate using the last MPC solution
  Eigen::Vector3d ref_position_log       = Eigen::Vector3d::Zero();
  Eigen::Quaterniond ref_orientation_log = Eigen::Quaterniond::Identity();

  // ── Timing statistics ──────────────────────────────────────────────────────
  std::vector<double> mpc_times;
  mpc_times.reserve(static_cast<std::size_t>(total_time / mpc_dt) + 1U);

  // ── Print summary ──────────────────────────────────────────────────────────
  std::cout << "=== MPC + Simulator Example ===" << std::endl;
  std::cout << "Total time       : " << total_time << " s" << std::endl;
  std::cout << "MPC dt           : " << mpc_dt << " s (" << 1.0 / mpc_dt << " Hz)" << std::endl;
  std::cout << "MPC horizon      : N=" << prediction_steps << ", dt_h=" << dt_horizon
            << " s, tf=" << prediction_steps * dt_horizon << " s" << std::endl;
  std::cout << "Controller dt    : " << controller_dt << " s (" << 1.0 / controller_dt << " Hz)"
            << std::endl;
  std::cout << "Model dt         : " << model_dt << " s (" << 1.0 / model_dt << " Hz)" << std::endl;
  std::cout << "Ctrl steps / MPC : " << controller_steps_per_mpc << std::endl;
  std::cout << "Model steps / Ctrl: " << model_steps_per_controller << std::endl;
  std::cout << "Waypoints        : " << waypoints.size() << std::endl;

  // Initial log entry at t=0 (drone at origin, all zeros)
  logger.save(0.0, Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), 0.0,
              Eigen::Vector3d::Zero(), Eigen::Matrix<double, 4, 1>::Zero());

  // ── Simulation loop ────────────────────────────────────────────────────────
  double t = 0.0;
  while (t < total_time + 1e-9) {
    // ── MPC step (runs at mpc_dt = 100 Hz) ──────────────────────────────────
    const auto mpc_start = std::chrono::high_resolution_clock::now();

    // Read current ground-truth state from the physics model
    const mav_model::State& state        = sim.getState();
    const Eigen::Vector3d position       = state.getPositionVector();
    const Eigen::Quaterniond orientation = state.getOrientationVector();
    const Eigen::Vector3d velocity       = state.getLinearVelocityVector();

    const Eigen::Vector3d desired_position = waypoints[wp_index];
    const Eigen::Quaterniond desired_orientation =
        getDesiredOrientation(desired_position, position, orientation, example_cfg.path_facing);

    // Cache for logging (shared across the INDI sub-steps below)
    ref_position_log    = desired_position;
    ref_orientation_log = desired_orientation;

    // Pack state into the MPC data structure
    mpc_data->state.setPosition({position.x(), position.y(), position.z()});
    mpc_data->state.setOrientation(
        {orientation.w(), orientation.x(), orientation.y(), orientation.z()});
    mpc_data->state.setLinearVelocity({velocity.x(), velocity.y(), velocity.z()});

    // Set online parameters (updated each call because the reference moves)
    setProgressiveReferences(mpc_data, position, desired_position, desired_orientation, v_ref,
                             dt_horizon, prediction_steps);

    // Guard against NaN/Inf propagation before feeding the solver
    if (!isFiniteVector3(position) || !isFiniteQuaternion(orientation) ||
        !isFiniteVector3(velocity) || !isFiniteVector3(desired_position) ||
        !isFiniteQuaternion(desired_orientation)) {
      std::cerr << "\nInvalid (non-finite) values detected before MPC solve at time " << t << " s"
                << std::endl;
      break;
    }

    // Solve the OCP (SQP-RTI: one linearisation + one QP per call)
    const int mpc_status = mpc.solve();
    const auto mpc_end   = std::chrono::high_resolution_clock::now();
    mpc_times.push_back(std::chrono::duration<double>(mpc_end - mpc_start).count());

    if (mpc_status != 0) {
      std::cerr << "\nMPC solver failed with status " << mpc_status << " at time " << t << " s"
                << std::endl;
      std::cerr << "State position: " << position.transpose() << std::endl;
      std::cerr << "State velocity: " << velocity.transpose() << std::endl;
      std::cerr << "State orientation [w x y z]: " << orientation.w() << " " << orientation.x()
                << " " << orientation.y() << " " << orientation.z() << std::endl;
      std::cerr << "Reference position: " << desired_position.transpose() << std::endl;
      std::cerr << "Reference orientation [w x y z]: " << desired_orientation.w() << " "
                << desired_orientation.x() << " " << desired_orientation.y() << " "
                << desired_orientation.z() << std::endl;
      break;
    }

    // Extract the first control action from the MPC solution
    mpc_thrust           = mpc_data->actuation.getThrust();
    mpc_angular_velocity = toEigenVector3(mpc_data->actuation.getAngularVelocity());

    // Advance waypoint index when the drone is within 0.1 m of the target
    const double error = (position - desired_position).norm();
    if (error < 0.1 && wp_index < waypoints.size() - 1U) {
      ++wp_index;
      std::cout << "\n  -> Waypoint " << wp_index << ": " << waypoints[wp_index].transpose();
    }

    // ── INDI + model sub-steps (zero-order hold on MPC output) ──────────────
    // The MPC command is fixed for this block; INDI recalculates motor speeds
    // at each controller step to track the angular velocity reference.
    sim.setReferenceRates(mpc_thrust, mpc_angular_velocity);

    for (int ctrl_step = 0; ctrl_step < controller_steps_per_mpc; ++ctrl_step) {
      // INDI: thrust + ω_ref → motor angular velocity commands
      sim.updateController(controller_dt);
      // IMU: advance gyro/accelerometer noise model
      sim.updateImu(controller_dt);

      // Physics model inner loop (smaller timestep for numerical accuracy)
      for (int model_step = 0; model_step < model_steps_per_controller; ++model_step) {
        sim.updateModel(model_dt);
      }

      // Log at controller rate (500 Hz)
      const mav_model::State& s = sim.getState();
      const double log_time     = t + (ctrl_step + 1) * controller_dt;
      logger.save(log_time, s.getPositionVector(), s.getOrientationVector(),
                  s.getLinearVelocityVector(), s.getAngularVelocityVector(), ref_position_log,
                  ref_orientation_log, mpc_thrust, mpc_angular_velocity,
                  s.getMotorAngularVelocityVector());
    }

    t += mpc_dt;
    printProgress(t / total_time);
  }

  // ── Print statistics ───────────────────────────────────────────────────────
  std::cout << "\n\n=== Simulation finished ===" << std::endl;
  std::cout << "Simulated time      : " << t << " s" << std::endl;
  std::cout << "MPC avg solve time  : " << computeMean(mpc_times) * 1000.0 << " ms" << std::endl;
  if (!mpc_times.empty()) {
    std::cout << "MPC real-time factor: " << mpc_dt / computeMean(mpc_times) << std::endl;
  }
}

}  // namespace mpc_examples

int main(int argc, char** argv) {
  const mpc_examples::ExampleArgs args = mpc_examples::parseArguments(argc, argv);
  mpc_examples::run(args);
  return 0;
}
