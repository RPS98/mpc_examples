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
 * @file logger_pybind.cpp
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "mav_flight_mcap_pybind/logger_pybind.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mav_flight_mcap/config.hpp"
#include "mav_flight_mcap/mcap_logger.hpp"
#include "mav_flight_mcap/trajectory_point.hpp"

namespace mav_flight_mcap {
namespace pybindings {

namespace py = pybind11;
using py::literals::operator""_a;

void bindLogger(py::module_& m) {
  // --- TimeMode -------------------------------------------------------------
  py::enum_<TimeMode>(m, "TimeMode",
                      "Time source used when resolving save() timestamps.")
      .value("SIMULATION", TimeMode::SIMULATION,
             "First save() call defines t = 0; timestamps are relative.")
      .value("GLOBAL", TimeMode::GLOBAL,
             "The user-provided time is the absolute POSIX time in seconds.");

  // --- LoggerConfig ---------------------------------------------------------
  py::class_<LoggerConfig>(m, "LoggerConfig",
                           "Configuration for MCAPLogger.")
      .def(py::init<>())
      .def_readwrite("file_path",               &LoggerConfig::file_path)
      .def_readwrite("time_mode",               &LoggerConfig::time_mode)
      .def_readwrite("frame_earth",             &LoggerConfig::frame_earth)
      .def_readwrite("frame_body",              &LoggerConfig::frame_body)
      .def_readwrite("pose_reference_topic",
                     &LoggerConfig::pose_reference_topic)
      .def_readwrite("twist_reference_topic",
                     &LoggerConfig::twist_reference_topic)
      .def_readwrite("trajectory_reference_topic",
                     &LoggerConfig::trajectory_reference_topic)
      .def_readwrite("thrust_command_topic",
                     &LoggerConfig::thrust_command_topic)
      .def_readwrite("twist_command_topic",
                     &LoggerConfig::twist_command_topic)
      .def_readwrite("pose_state_topic",   &LoggerConfig::pose_state_topic)
      .def_readwrite("twist_state_topic",  &LoggerConfig::twist_state_topic)
      .def_readwrite("odom_state_topic",   &LoggerConfig::odom_state_topic)
      .def_readwrite("clock_topic",        &LoggerConfig::clock_topic)
      .def_readwrite("emit_clock_on_every_save",
                     &LoggerConfig::emit_clock_on_every_save)
      .def_readwrite("clock_min_period_s",
                     &LoggerConfig::clock_min_period_s)
      .def_readwrite("compress_zstd",      &LoggerConfig::compress_zstd);

  // --- TrajectoryPoint ------------------------------------------------------
  py::class_<TrajectoryPoint>(m, "TrajectoryPoint",
                              "Single aerostack2 trajectory setpoint.")
      .def(py::init<>())
      .def_readwrite("id",           &TrajectoryPoint::id)
      .def_readwrite("position",     &TrajectoryPoint::position)
      .def_readwrite("twist",        &TrajectoryPoint::twist)
      .def_readwrite("acceleration", &TrajectoryPoint::acceleration)
      .def_readwrite("yaw_angle",    &TrajectoryPoint::yaw_angle);

  // --- MCAPLogger -----------------------------------------------------------
  py::class_<MCAPLogger>(m, "MCAPLogger",
                         "ROS 2 Humble-compatible MCAP logger.")
      .def(py::init<const LoggerConfig&>(), "cfg"_a)

      // Topic setters
      .def("set_pose_reference_topic",       &MCAPLogger::set_pose_reference_topic, "topic"_a)
      .def("set_twist_reference_topic",      &MCAPLogger::set_twist_reference_topic, "topic"_a)
      .def("set_trajectory_reference_topic", &MCAPLogger::set_trajectory_reference_topic, "topic"_a)
      .def("set_thrust_command_topic",       &MCAPLogger::set_thrust_command_topic, "topic"_a)
      .def("set_twist_command_topic",        &MCAPLogger::set_twist_command_topic, "topic"_a)
      .def("set_pose_state_topic",           &MCAPLogger::set_pose_state_topic, "topic"_a)
      .def("set_twist_state_topic",          &MCAPLogger::set_twist_state_topic, "topic"_a)
      .def("set_odom_state_topic",           &MCAPLogger::set_odom_state_topic, "topic"_a)

      // Extras registration
      .def("add_int32_topic",              &MCAPLogger::add_int32_topic, "topic"_a)
      .def("add_string_topic",             &MCAPLogger::add_string_topic, "topic"_a)
      .def("add_float64_topic",            &MCAPLogger::add_float64_topic, "topic"_a)
      .def("add_float64_multi_array_topic",
           &MCAPLogger::add_float64_multi_array_topic, "topic"_a)
      .def("add_vector3_topic",            &MCAPLogger::add_vector3_topic, "topic"_a)

      // Lifecycle
      .def("start",       &MCAPLogger::start)
      .def("close",       &MCAPLogger::close)
      .def("is_running",  &MCAPLogger::isRunning)

      // Single-topic saves
      .def("save_pose_reference",      &MCAPLogger::save_pose_reference,
           "t"_a, "pos"_a, "quat_wxyz"_a)
      .def("save_twist_reference",     &MCAPLogger::save_twist_reference,
           "t"_a, "linear"_a)
      .def("save_trajectory_reference",
           &MCAPLogger::save_trajectory_reference,
           "t"_a, "points"_a)
      .def("save_thrust_command",      &MCAPLogger::save_thrust_command,
           "t"_a, "thrust"_a)
      .def("save_twist_command",       &MCAPLogger::save_twist_command,
           "t"_a, "angular"_a)
      .def("save_pose_state",          &MCAPLogger::save_pose_state,
           "t"_a, "pos"_a, "quat_wxyz"_a)
      .def("save_twist_state",         &MCAPLogger::save_twist_state,
           "t"_a, "linear"_a, "angular"_a)
      .def("save_odom_state",          &MCAPLogger::save_odom_state,
           "t"_a, "pos_earth"_a, "quat_wxyz"_a,
           "linear_body"_a, "angular_body"_a)

      // Aggregate saves
      .def("save_state",               &MCAPLogger::save_state,
           "t"_a, "pos_earth"_a, "quat_wxyz"_a,
           "linear_earth"_a, "angular_body"_a)
      .def("save_position_reference",  &MCAPLogger::save_position_reference,
           "t"_a, "pos_ref"_a, "quat_wxyz_ref"_a, "max_linear_speed"_a)
      .def("save_trajectory_reference_full",
           &MCAPLogger::save_trajectory_reference_full,
           "t"_a, "points"_a, "also_emit_pose_and_twist"_a = false)
      .def("save_actuation",           &MCAPLogger::save_actuation,
           "t"_a, "thrust"_a, "angular_command_body"_a)

      // Extras
      .def("save_int32",
           &MCAPLogger::save_int32, "topic"_a, "t"_a, "value"_a)
      .def("save_string",
           &MCAPLogger::save_string, "topic"_a, "t"_a, "value"_a)
      .def("save_float64",
           &MCAPLogger::save_float64, "topic"_a, "t"_a, "value"_a)
      .def("save_vector3",
           &MCAPLogger::save_vector3, "topic"_a, "t"_a, "value"_a)
      .def("save_float64_multi_array",
           &MCAPLogger::save_float64_multi_array,
           "topic"_a, "t"_a, "value"_a,
           "dims"_a = std::vector<uint32_t>{});
}

}  // namespace pybindings
}  // namespace mav_flight_mcap
