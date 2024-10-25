// Copyright 2024 Universidad Politécnica de Madrid
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
 * @file example_utils.hpp
 *
 * Acados MPC examples using Acados Sim solver utility functions implementation.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef EXAMPLE_UTILS_HPP_
#define EXAMPLE_UTILS_HPP_

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "acados_mpc/acados_mpc.hpp"
#include "acados_mpc/acados_sim_solver.hpp"
#include "dynamic_trajectory_generator/dynamic_trajectory.hpp"
#include "dynamic_trajectory_generator/dynamic_waypoint.hpp"
#include "multirotor_simulator.hpp"

namespace acados_mpc_examples {

using DynamicTrajectory = dynamic_traj_generator::DynamicTrajectory;
using DynamicWaypoint   = dynamic_traj_generator::DynamicWaypoint;

void quaternion_to_Euler(const Eigen::Quaterniond& _quaternion,
                         double& roll,
                         double& pitch,
                         double& yaw) {
  // roll (x-axis rotation)
  Eigen::Quaterniond quaternion = _quaternion.normalized();

  double sinr = 2.0 * (quaternion.w() * quaternion.x() + quaternion.y() * quaternion.z());
  double cosr = 1.0 - 2.0 * (quaternion.x() * quaternion.x() + quaternion.y() * quaternion.y());
  roll        = std::atan2(sinr, cosr);

  // pitch (y-axis rotation)
  double sinp = 2.0 * (quaternion.w() * quaternion.y() - quaternion.z() * quaternion.x());
  if (std::abs(sinp) >= 1.0)
    pitch = std::copysign(M_PI / 2, sinp);  // use 90 degrees if out of range
  else
    pitch = std::asin(sinp);

  // yaw (z-axis rotation)
  double siny = 2.0 * (quaternion.w() * quaternion.z() + quaternion.x() * quaternion.y());
  double cosy = 1.0 - 2.0 * (quaternion.y() * quaternion.y() + quaternion.z() * quaternion.z());
  yaw         = std::atan2(siny, cosy);
}

Eigen::Quaterniond euler_to_quaternion(double roll, double pitch, double yaw) {
  // Calculate half angles
  double roll_half  = roll * 0.5;
  double pitch_half = pitch * 0.5;
  double yaw_half   = yaw * 0.5;

  // Calculate the sine and cosine of the half angles
  double sr = sin(roll_half);
  double cr = cos(roll_half);
  double sp = sin(pitch_half);
  double cp = cos(pitch_half);
  double sy = sin(yaw_half);
  double cy = cos(yaw_half);

  // Calculate the quaternion components
  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  // Create the Quaternion object
  return Eigen::Quaterniond(w, x, y, z).normalized();
}

std::array<double, 4> compute_path_facing(const Eigen::Vector3d velocity) {
  double yaw = atan2(velocity.y(), velocity.x());
  double pitch, roll = 0.0;

  Eigen::Quaterniond q = euler_to_quaternion(roll, pitch, yaw);
  return {q.w(), q.x(), q.y(), q.z()};
}

struct YamlData {
  double sim_time;
  double dt;
  double trajectory_generator_max_speed;
  std::vector<Eigen::Vector3d> waypoints;
  bool path_facing;
  double floor_height;
  multirotor::SimulatorParams<double, 4> simulator_params;
};

void read_yaml_params(const std::string& file_path, YamlData& data) {
  // Check if file exists
  std::ifstream f(file_path.c_str());
  if (!f.good()) {
    std::string absolute_simulation_config_path = std::filesystem::absolute(file_path).string();
    std::cout << "File " << absolute_simulation_config_path << " does not exist." << std::endl;
    f.close();
    throw std::invalid_argument("File does not exist");
  }
  f.close();
  YAML::Node yaml_config_file = YAML::LoadFile(file_path);

  // Simulation parameters
  data.floor_height = yaml_config_file["sim_config"]["floor_height"].as<double>();

  // Trajectory generator
  data.sim_time = yaml_config_file["sim_config"]["sim_time"].as<double>();
  data.dt       = yaml_config_file["sim_config"]["dt"].as<double>();
  data.trajectory_generator_max_speed =
      yaml_config_file["sim_config"]["trajectory_generator_max_speed"].as<double>();

  for (auto waypoint : yaml_config_file["sim_config"]["trajectory_generator_waypoints"]) {
    data.waypoints.push_back(Eigen::Vector3d(waypoint[0].as<double>(), waypoint[1].as<double>(),
                                             waypoint[2].as<double>()));
  }

  // Initialize simulator params
  multirotor::SimulatorParams<double, 4> p;

  // Dynamics state
  auto yml = yaml_config_file["dynamics"]["state"];

  p.dynamics_params.state.kinematics.position =
      Eigen::Vector3d(yml["position"][0].as<double>(), yml["position"][1].as<double>(),
                      yml["position"][2].as<double>());
  p.dynamics_params.state.kinematics.orientation =
      Eigen::Quaterniond(yml["orientation"][0].as<double>(), yml["orientation"][1].as<double>(),
                         yml["orientation"][2].as<double>(), yml["orientation"][3].as<double>());
  p.dynamics_params.state.kinematics.linear_velocity = Eigen::Vector3d(
      yml["linear_velocity"][0].as<double>(), yml["linear_velocity"][1].as<double>(),
      yml["linear_velocity"][2].as<double>());
  p.dynamics_params.state.kinematics.angular_velocity = Eigen::Vector3d(
      yml["angular_velocity"][0].as<double>(), yml["angular_velocity"][1].as<double>(),
      yml["angular_velocity"][2].as<double>());
  p.dynamics_params.state.kinematics.linear_acceleration = Eigen::Vector3d(
      yml["linear_acceleration"][0].as<double>(), yml["linear_acceleration"][1].as<double>(),
      yml["linear_acceleration"][2].as<double>());
  p.dynamics_params.state.kinematics.angular_acceleration = Eigen::Vector3d(
      yml["angular_acceleration"][0].as<double>(), yml["angular_acceleration"][1].as<double>(),
      yml["angular_acceleration"][2].as<double>());

  // Dynamics model
  yml = yaml_config_file["dynamics"]["model"];

  p.dynamics_params.model_params.gravity =
      Eigen::Vector3d(yml["gravity"][0].as<double>(), yml["gravity"][1].as<double>(),
                      yml["gravity"][2].as<double>());
  p.dynamics_params.model_params.vehicle_mass = yml["vehicle_mass"].as<double>();
  p.dynamics_params.model_params.vehicle_inertia =
      Eigen::Vector3d(yml["vehicle_inertia"][0].as<double>(),
                      yml["vehicle_inertia"][1].as<double>(),
                      yml["vehicle_inertia"][2].as<double>())
          .asDiagonal();
  p.dynamics_params.model_params.vehicle_drag_coefficient =
      yml["vehicle_drag_coefficient"].as<double>();
  p.dynamics_params.model_params.vehicle_aero_moment_coefficient =
      Eigen::Vector3d(yml["vehicle_aero_moment_coefficient"][0].as<double>(),
                      yml["vehicle_aero_moment_coefficient"][1].as<double>(),
                      yml["vehicle_aero_moment_coefficient"][2].as<double>())
          .asDiagonal();
  p.dynamics_params.model_params.force_process_noise_auto_correlation =
      yml["force_process_noise_auto_correlation"].as<double>();
  p.dynamics_params.model_params.moment_process_noise_auto_correlation =
      yml["moment_process_noise_auto_correlation"].as<double>();

  // Motors parameters
  yml = yaml_config_file["dynamics"]["model"]["motors_params"];

  double thrust_coefficient = yml["thrust_coefficient"].as<double>();
  double torque_coefficient = yml["torque_coefficient"].as<double>();
  double x_dist             = yml["x_dist"].as<double>();
  double y_dist             = yml["y_dist"].as<double>();
  double min_speed          = yml["min_speed"].as<double>();
  double max_speed          = yml["max_speed"].as<double>();
  double time_constant      = yml["time_constant"].as<double>();
  double rotational_inertia = yml["rotational_inertia"].as<double>();
  p.dynamics_params.model_params.motors_params =
      multirotor::model::Model<double, 4>::create_quadrotor_x_config(
          thrust_coefficient, torque_coefficient, x_dist, y_dist, min_speed, max_speed,
          time_constant, rotational_inertia);

  // Controller params Indi
  yml = yaml_config_file["controller"]["indi"];

  p.controller_params.indi_controller_params.inertia =
      p.dynamics_params.model_params.vehicle_inertia;
  auto mixing_matrix_6D_4rotors = multirotor::model::Model<double, 4>::compute_mixer_matrix<6>(
      p.dynamics_params.model_params.motors_params);
  p.controller_params.indi_controller_params.mixer_matrix_inverse =
      indi_controller::compute_quadrotor_mixer_matrix_inverse(mixing_matrix_6D_4rotors);
  p.controller_params.indi_controller_params.pid_params.Kp_gains = Eigen::Vector3d(
      yml["Kp"][0].as<double>(), yml["Kp"][1].as<double>(), yml["Kp"][2].as<double>());
  p.controller_params.indi_controller_params.pid_params.Ki_gains = Eigen::Vector3d(
      yml["Ki"][0].as<double>(), yml["Ki"][1].as<double>(), yml["Ki"][2].as<double>());
  p.controller_params.indi_controller_params.pid_params.Kd_gains = Eigen::Vector3d(
      yml["Kd"][0].as<double>(), yml["Kd"][1].as<double>(), yml["Kd"][2].as<double>());
  p.controller_params.indi_controller_params.pid_params.alpha = Eigen::Vector3d(
      yml["alpha"][0].as<double>(), yml["alpha"][1].as<double>(), yml["alpha"][2].as<double>());
  p.controller_params.indi_controller_params.pid_params.antiwindup_cte =
      Eigen::Vector3d(yml["antiwindup_cte"][0].as<double>(), yml["antiwindup_cte"][1].as<double>(),
                      yml["antiwindup_cte"][2].as<double>());
  Eigen::Vector3d angular_acceleration_limit =
      Eigen::Vector3d(yml["angular_acceleration_limit"][0].as<double>(),
                      yml["angular_acceleration_limit"][1].as<double>(),
                      yml["angular_acceleration_limit"][2].as<double>());
  p.controller_params.indi_controller_params.pid_params.lower_output_saturation =
      -1.0 * angular_acceleration_limit;
  p.controller_params.indi_controller_params.pid_params.upper_output_saturation =
      angular_acceleration_limit;
  p.controller_params.indi_controller_params.pid_params.proportional_saturation_flag = true;

  // Imu params
  yml = yaml_config_file["imu"];

  p.imu_params.gyro_noise_var                 = yml["gyro_noise_var"].as<double>();
  p.imu_params.accel_noise_var                = yml["accel_noise_var"].as<double>();
  p.imu_params.gyro_bias_noise_autocorr_time  = yml["gyro_bias_noise_autocorr_time"].as<double>();
  p.imu_params.accel_bias_noise_autocorr_time = yml["accel_bias_noise_autocorr_time"].as<double>();

  // Inertial odometry params
  yml = yaml_config_file["inertial_odometry"];

  p.inertial_odometry_params.alpha = yml["alpha"].as<double>();
  p.inertial_odometry_params.initial_world_orientation =
      p.dynamics_params.state.kinematics.orientation;

  data.simulator_params = p;
  return;
}

DynamicWaypoint::Vector eigen_vector_to_dynamic_waypoint_vector(
    const std::vector<Eigen::Vector3d>& vector_waypoints) {
  DynamicWaypoint::Vector vector_dynamic_waypoints;
  for (auto waypoint : vector_waypoints) {
    DynamicWaypoint dynamic_waypoint;
    dynamic_waypoint.resetWaypoint(waypoint);
    vector_dynamic_waypoints.push_back(dynamic_waypoint);
  }
  return vector_dynamic_waypoints;
}

std::unique_ptr<DynamicTrajectory> get_trajectory_generator(
    const Eigen::Vector3d initial_position,
    const std::vector<Eigen::Vector3d>& waypoints,
    const double speed) {
  // Initialize dynamic trajectory generator
  std::unique_ptr<DynamicTrajectory> trajectory_generator = std::make_unique<DynamicTrajectory>();

  trajectory_generator->updateVehiclePosition(initial_position);
  trajectory_generator->setSpeed(speed);

  // Set waypoints
  DynamicWaypoint::Vector waypoints_to_set = eigen_vector_to_dynamic_waypoint_vector(waypoints);

  // Generate trajectory
  trajectory_generator->setWaypoints(waypoints_to_set);
  double max_time = trajectory_generator->getMaxTime();  // Block until trajectory is generated

  std::cout << "Trajectory generated with max time: " << max_time << std::endl;

  return trajectory_generator;
}

class CsvLogger {
public:
  explicit CsvLogger(const std::string& file_name) : file_name_(file_name) {
    std::cout << "Saving to file: " << file_name << std::endl;
    file_ = std::ofstream(file_name, std::ofstream::out | std::ofstream::trunc);
    file_ << "time,x,y,z,qw,qx,qy,qz,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,ax,ay,az,dwx,dwy,dwz,"
             "fx,fy,fz,tx,ty,tz,mw1,mw2,mw3,mw4,mdw1,mdw2,mdw3,mdw4,x_ref,y_ref,"
             "z_ref,yaw_ref,vx_ref,vy_ref,vz_ref,wx_ref,wy_ref,"
             "wz_ref,ax_ref,ay_ref,az_ref,dwx_ref,dwy_ref,dwz_ref,fx_ref,fy_ref,fz_ref,"
             "tx_ref,ty_ref,tz_ref,mw1_ref,mw2_ref,mw3_ref,mw4_ref,"
             "x_io,y_io,z_io,roll_io,pitch_io,yaw_io,vx_io,vy_io,vz_io,wx_io,wy_io,"
             "wz_io,ax_io,ay_io,az_io"
          << std::endl;
  }

  ~CsvLogger() { file_.close(); }

  void add_double(const double data) {
    // Check if data is nan
    if (std::isnan(data)) {
      // Throw exception
      std::invalid_argument("Data is nan");
      return;
    }
    file_ << data << ",";
  }

  void add_string(const std::string& data, const bool add_final_comma = true) {
    file_ << data;
    if (add_final_comma) file_ << ",";
  }

  void add_vector_row(const Eigen::Vector3d& data, const bool add_final_comma = true) {
    for (size_t i = 0; i < data.size(); ++i) {
      // Check if data is nan
      if (std::isnan(data[i])) {
        // Throw exception
        std::invalid_argument("Data is nan");
        return;
      }
      file_ << data[i];
      if (i < data.size() - 1) {
        file_ << ",";
      } else if (add_final_comma) {
        file_ << ",";
      }
    }
  }

  void add_vector_row(const Eigen::Vector4d& data, const bool add_final_comma = true) {
    for (size_t i = 0; i < data.size(); ++i) {
      // Check if data is nan
      if (std::isnan(data[i])) {
        // Throw exception
        std::invalid_argument("Data is nan");
        return;
      }
      file_ << data[i];
      if (i < data.size() - 1) {
        file_ << ",";
      } else if (add_final_comma) {
        file_ << ",";
      }
    }
  }

  void save(const double time, const multirotor::Simulator<double, 4>& simulator) {
    add_double(time);

    // State ground truth
    const multirotor::state::State<double, 4> state = simulator.get_dynamics_const().get_state();
    std::ostringstream state_stream;
    state_stream << state;
    std::string state_string = state_stream.str();
    add_string(state_string);

    // References
    // Reference trajectory generator
    add_vector_row(simulator.get_reference_position());  // Position
    add_double(simulator.get_reference_yaw());           // Yaw
    add_vector_row(simulator.get_reference_velocity());  // Velocity
    // Reference multirotor_controller
    add_vector_row(
        simulator.get_controller_const().get_desired_angular_velocity());  // Angular velocity
    add_vector_row(simulator.get_reference_acceleration());                // Linear acceleration
    add_vector_row(simulator.get_controller_const()
                       .get_indi_controller_const()
                       .get_desired_angular_acceleration());  // Angular acceleration
    // Force reference compensated with the gravity and in earth frame
    Eigen::Vector3d force_ref =
        simulator.get_dynamics_const().get_state().kinematics.orientation *
            simulator.get_controller_const().get_indi_controller_const().get_desired_thrust() +
        simulator.get_dynamics_const().get_model_const().get_gravity() *
            simulator.get_dynamics_const().get_model_const().get_mass();
    add_vector_row(force_ref);  // Force
    add_vector_row(simulator.get_controller_const()
                       .get_indi_controller_const()
                       .get_desired_torque());  // Torque

    add_vector_row(simulator.get_actuation_motors_angular_velocity());  // Motor angular velocity

    // State inertial odometry
    std::ostringstream io_stream;
    io_stream << simulator.get_inertial_odometry_const();
    std::string io_string = io_stream.str();
    add_string(io_string, false);

    // End line
    file_ << std::endl;
  }

  void close() { file_.close(); }

private:
  std::string file_name_;
  std::ofstream file_;
};

}  // namespace acados_mpc_examples

#endif  // EXAMPLE_UTILS_HPP_
