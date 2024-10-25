/*!*******************************************************************************************
 *  \file       multirotor_simulator_traj_gen_test.cpp
 *  \brief      Class test
 *  \authors    Rafael Pérez Seguí
 *
 *  \copyright  Copyright (c) 2022 Universidad Politécnica de Madrid
 *              All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************************/

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "dynamic_trajectory_generator/dynamic_trajectory.hpp"
#include "dynamic_trajectory_generator/dynamic_waypoint.hpp"
#include "multirotor_simulator.hpp"

#include "acados_mpc/acados_mpc.hpp"

#include "utils/ms_example_utils.hpp"

namespace acados_mpc_examples {

using DynamicTrajectory = dynamic_traj_generator::DynamicTrajectory;
using DynamicWaypoint   = dynamic_traj_generator::DynamicWaypoint;

#define SIM_CONFIG_PATH "examples/ms_simulation_config.yaml"

void traj_generator_ref_to_mpc_ref(const dynamic_traj_generator::References& references,
                                   acados_mpc::MPCData* mpc_data,
                                   int index) {
  if (index == MPC_N) {
    mpc_data->reference_end.set_data(0, references.position.x());
    mpc_data->reference_end.set_data(1, references.position.y());
    mpc_data->reference_end.set_data(2, references.position.z());
    mpc_data->reference_end.set_data(3, 1.0);
    mpc_data->reference_end.set_data(4, 0.0);
    mpc_data->reference_end.set_data(5, 0.0);
    mpc_data->reference_end.set_data(6, 0.0);
    mpc_data->reference_end.set_data(7, references.velocity.x());
    mpc_data->reference_end.set_data(8, references.velocity.y());
    mpc_data->reference_end.set_data(9, references.velocity.z());
    return;
  } else if (index > MPC_N) {
    throw std::out_of_range("Index out of range.");
  }
  mpc_data->reference.set_data(index, 0, references.position.x());
  mpc_data->reference.set_data(index, 1, references.position.y());
  mpc_data->reference.set_data(index, 2, references.position.z());
  mpc_data->reference.set_data(index, 3, 1.0);
  mpc_data->reference.set_data(index, 4, 0.0);
  mpc_data->reference.set_data(index, 5, 0.0);
  mpc_data->reference.set_data(index, 6, 0.0);
  mpc_data->reference.set_data(index, 7, references.velocity.x());
  mpc_data->reference.set_data(index, 8, references.velocity.y());
  mpc_data->reference.set_data(index, 9, references.velocity.z());
}

void print_progress_bar(float progress) {
  int bar_width = 70;
  std::cout << "[";
  int pos = bar_width * progress;
  for (int i = 0; i < bar_width; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
}

void test_mpc_controller(CsvLogger& logger,
                         acados_mpc::MPC& mpc,
                         multirotor::Simulator<double, 4>& simulator,
                         std::unique_ptr<DynamicTrajectory>& trajectory_generator,
                         const YamlData& yaml_data) {
  // Get simulation parameters
  double sim_max_time = yaml_data.sim_time;
  double dt           = yaml_data.dt;

  // MPC Parameters
  acados_mpc::MPCData* mpc_data = mpc.get_data();
  int prediction_steps          = mpc.get_prediction_steps();
  double tf                     = mpc.get_prediction_time_step();

  // Set control mode
  simulator.set_control_mode(multirotor::ControlMode::ACRO);
  simulator.set_reference_yaw_angle(0.0);

  // Simulation
  double max_time = trajectory_generator->getMaxTime();
  double min_time = trajectory_generator->getMinTime();
  double t        = 0.0;  // seconds
  logger.save(t, simulator);

  // Initialize dynamic trajectory generator
  dynamic_traj_generator::References references;
  references.position = simulator.get_dynamics_const().get_state().kinematics.position;
  references.velocity = simulator.get_dynamics_const().get_state().kinematics.linear_velocity;
  references.acceleration =
      simulator.get_dynamics_const().get_state().kinematics.linear_acceleration;
  double roll_ref, pitch_ref, yaw_ref;
  quaternion_to_Euler(simulator.get_dynamics_const().get_state().kinematics.orientation, roll_ref,
                      pitch_ref, yaw_ref);

  // Time measurement
  const std::size_t n_iterations = static_cast<std::size_t>(yaml_data.sim_time / yaml_data.dt);
  std::vector<double> mpc_times;
  mpc_times.reserve(n_iterations);
  std::vector<double> sim_times;
  sim_times.reserve(n_iterations);
  std::vector<double> total_times;
  total_times.reserve(n_iterations);

  for (double t = 0; t < yaml_data.sim_time; t += yaml_data.dt) {
    print_progress_bar(t / yaml_data.sim_time);
    auto iter_start = std::chrono::high_resolution_clock::now();

    double t_eval = t;
    dynamic_traj_generator::References init_references;
    bool init_ref = true;

    // yref from 0 to N-1 (N steps) and yref_N from N to N
    for (int i = 0; i < prediction_steps + 1; i++) {
      if (t_eval >= max_time) {
        t_eval = max_time - yaml_data.dt;
      } else if (t_eval <= min_time) {
        t_eval = min_time;
      }
      trajectory_generator->evaluateTrajectory(t_eval, references);
      traj_generator_ref_to_mpc_ref(references, mpc_data, i);
      t_eval += tf;
      if (init_ref) {
        init_references = references;
        init_ref        = false;
      }
    }

    // Solve MPC
    mpc_data->state.set_data(0, simulator.get_dynamics_const().get_state().kinematics.position.x());
    mpc_data->state.set_data(1, simulator.get_dynamics_const().get_state().kinematics.position.y());
    mpc_data->state.set_data(2, simulator.get_dynamics_const().get_state().kinematics.position.z());
    mpc_data->state.set_data(3,
                             simulator.get_dynamics_const().get_state().kinematics.orientation.w());
    mpc_data->state.set_data(4,
                             simulator.get_dynamics_const().get_state().kinematics.orientation.x());
    mpc_data->state.set_data(5,
                             simulator.get_dynamics_const().get_state().kinematics.orientation.y());
    mpc_data->state.set_data(6,
                             simulator.get_dynamics_const().get_state().kinematics.orientation.z());
    mpc_data->state.set_data(
        7, simulator.get_dynamics_const().get_state().kinematics.linear_velocity.x());
    mpc_data->state.set_data(
        8, simulator.get_dynamics_const().get_state().kinematics.linear_velocity.y());
    mpc_data->state.set_data(
        9, simulator.get_dynamics_const().get_state().kinematics.linear_velocity.z());
    auto mpc_start = std::chrono::high_resolution_clock::now();
    mpc.solve();
    auto mpc_end = std::chrono::high_resolution_clock::now();

    // Simulate
    auto sim_start = std::chrono::high_resolution_clock::now();
    // Set references
    const double thrust                    = mpc_data->control.data[0];
    const Eigen::Vector3d angular_velocity = Eigen::Vector3d(
        mpc_data->control.data[1], mpc_data->control.data[2], mpc_data->control.data[3]);
    simulator.set_reference_acro(thrust, angular_velocity);

    // Update simulator
    simulator.update_controller(dt);
    simulator.update_dynamics(dt);
    simulator.update_imu(dt);
    simulator.update_inertial_odometry(dt);
    auto sim_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> mpc_duration   = mpc_end - mpc_start;
    std::chrono::duration<double> sim_duration   = sim_end - sim_start;
    std::chrono::duration<double> total_duration = sim_end - iter_start;
    mpc_times.push_back(mpc_duration.count());
    sim_times.push_back(sim_duration.count());
    total_times.push_back(total_duration.count());

    // Update references for debugging
    simulator.set_reference_trajectory(init_references.position, init_references.velocity,
                                       init_references.acceleration);

    logger.save(t, simulator);
  }
  logger.close();
  std::cout << "Simulation finished." << std::endl;

  // Print time measurements and its average
  double mpc_avg_time = std::accumulate(mpc_times.begin(), mpc_times.end(), 0.0) / mpc_times.size();
  double sim_avg_time = std::accumulate(sim_times.begin(), sim_times.end(), 0.0) / sim_times.size();
  double total_avg_time =
      std::accumulate(total_times.begin(), total_times.end(), 0.0) / total_times.size();
  std::cout << "MPC average time: " << mpc_avg_time << " s" << std::endl;
  std::cout << "Simulator average time: " << sim_avg_time << " s" << std::endl;
  std::cout << "Total average time: " << total_avg_time << " s" << std::endl;
}
}  // namespace acados_mpc_examples

int main(int argc, char** argv) {
  // Logger
  std::string file_name = "ms_mpc_log.csv";
  acados_mpc_examples::CsvLogger logger(file_name);

  // Params
  acados_mpc_examples::YamlData yaml_data;
  acados_mpc_examples::read_yaml_params("examples/ms_simulation_config.yaml", yaml_data);

  // Initialize simulator
  multirotor::Simulator simulator = multirotor::Simulator(yaml_data.simulator_params);
  simulator.enable_floor_collision(yaml_data.floor_height);
  simulator.arm();

  // Initialize MPC
  acados_mpc::MPC mpc = acados_mpc::MPC();

  // Initialize trajectory generator
  auto trajectory_generator = acados_mpc_examples::get_trajectory_generator(
      Eigen::Vector3d::Zero(), yaml_data.waypoints, yaml_data.trajectory_generator_max_speed);

  acados_mpc_examples::test_mpc_controller(logger, mpc, simulator, trajectory_generator, yaml_data);
  return 0;
}
