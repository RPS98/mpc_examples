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
 * @file waypoints_simulator.hpp
 *
 * Orchestrator that ties a controller, a reference generator and the MAV
 * simulator together into a single waypoint-tracking example.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_WAYPOINTS_SIMULATOR_HPP_
#define MPC_EXAMPLES_FRAMEWORK_WAYPOINTS_SIMULATOR_HPP_

#include "framework/controller_base.hpp"
#include "framework/trajectory_generator_base.hpp"
#include "mav_simulator/mav_simulator.hpp"
#include "utils/example_config_utils.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace mpc_examples::framework {

/**
 * @brief Aggregated timing statistics produced by a single run().
 *
 * Times are per-step averages in microseconds. Steps are counted at their
 * natural rate (controller at controlPeriod(), INDI/IMU/model at controller_dt).
 */
struct BenchmarkStats {
  double simulated_time_s   = 0.0;
  double real_time_s        = 0.0;
  double sim_speedup        = 0.0;   //!< simulated_time_s / real_time_s
  double controller_mean_us = 0.0;   //!< mean computeCommand() time per outer step
  double indi_mean_us       = 0.0;   //!< mean simulator.updateController() time per INDI step
  double imu_mean_us        = 0.0;   //!< mean simulator.updateImu() time per INDI step
  double model_mean_us      = 0.0;   //!< mean simulator.updateModel() time per INDI step
  std::size_t controller_steps = 0;
  std::size_t indi_steps       = 0;
};

/**
 * @brief Runs a full waypoint-tracking mission for a given controller and
 *        reference generator.
 *
 * Responsibilities:
 *   - Owns the mav_simulator::Simulator (set to RATES mode).
 *   - Calls initialize() on controller and generator at t=0.
 *   - Emits a warning if the generator does not produce every reference field
 *     required by the controller (missing fields stay at zero).
 *   - Executes the triple-loop cadence: outer at controller->controlPeriod(),
 *     INDI/IMU at example_cfg.controller_dt, physics at example_cfg.model_dt.
 *   - Applies zero-order hold on the controller output across INDI sub-steps.
 *   - Detects mission completion via traj_gen->isFinished() and hovers for
 *     example_cfg.hover_time before stopping.
 *   - Logs to CSV at the INDI rate (disabled when example_cfg.benchmark is true).
 *
 * Thread-safety: single-threaded; no concurrency guarantees.
 */
class WaypointsSimulator {
public:
  /**
   * @brief Build a new simulation run.
   *
   * @param controller       Owned controller adapter.
   * @param traj_gen         Owned reference generator adapter.
   * @param example_cfg      Shared example configuration (waypoints, dts, ...).
   * @param simulator_params mav_simulator parameters (model + IMU + controllers).
   * @param output_csv       Output CSV path (absolute or relative to simulator_logs/).
   *
   * @throws std::invalid_argument if @p controller or @p traj_gen is null or
   *         if @p example_cfg has no waypoints.
   */
  WaypointsSimulator(std::unique_ptr<IController> controller,
                     std::unique_ptr<ITrajectoryGenerator> traj_gen,
                     const ExampleConfig& example_cfg,
                     const mav_simulator::SimulatorParameters& simulator_params,
                     const std::string& output_csv);

  /**
   * @brief Execute the mission until the hover phase elapses or sim_time is reached.
   *
   * Populates benchmarkStats() on return.
   */
  void run();

  /** @brief Print the timing statistics collected by the last run(). */
  void printBenchmark() const;

  /** @return Timing statistics from the last run(). Zero-initialised before run(). */
  const BenchmarkStats& benchmarkStats() const { return stats_; }

  /** @return Reference to the underlying mav_simulator. Only valid after construction. */
  const mav_simulator::Simulator& simulator() const { return sim_; }

private:
  void checkCompatibility_() const;

  std::unique_ptr<IController> controller_;
  std::unique_ptr<ITrajectoryGenerator> traj_gen_;
  ExampleConfig example_cfg_;
  std::string output_csv_;
  mav_simulator::Simulator sim_;
  BenchmarkStats stats_;
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_WAYPOINTS_SIMULATOR_HPP_
