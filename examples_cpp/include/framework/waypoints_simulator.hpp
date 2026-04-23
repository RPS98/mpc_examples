// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file waypoints_simulator.hpp
 *
 * Orchestrator that ties a controller, a reference generator and the MAV
 * simulator together into a single waypoint-tracking example.
 *
 * Architecture:
 *   - Waypoint advancement is driven by WaypointScheduler on a time basis so
 *     every controller × generator combination receives identical transitions
 *     at identical simulator times.
 *   - Controller and generator compute times are modelled as latency through
 *     two DelayBuffer queues. The physics loop keeps advancing while the
 *     outer stage has no fresh output; logging records the delay that was
 *     actually applied to each tick.
 *   - Logging uses the ROS 2-compatible MCAP backend (UnifiedMcapLogger).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_WAYPOINTS_SIMULATOR_HPP_
#define MPC_EXAMPLES_FRAMEWORK_WAYPOINTS_SIMULATOR_HPP_

#include "framework/controller_base.hpp"
#include "framework/trajectory_generator_base.hpp"
// TODO: remove once MCAP pipeline validated.
// #include "framework/unified_csv_logger.hpp"
#include "framework/unified_mcap_logger.hpp"
#include "mav_simulator/mav_simulator.hpp"
#include "utils/example_config_utils.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace mpc_examples::framework {

/**
 * @brief Aggregated timing statistics produced by a single run().
 *
 * All mean times are expressed in microseconds. INDI / IMU / model means are
 * per inner (controller_dt) step; controller / generator means are per outer
 * step.
 */
struct BenchmarkStats {
  double simulated_time_s       = 0.0;
  double real_time_s            = 0.0;
  double sim_speedup            = 0.0;
  double controller_mean_us     = 0.0;
  double generator_update_mean_us = 0.0;
  double generator_eval_mean_us   = 0.0;
  double indi_mean_us           = 0.0;
  double imu_mean_us            = 0.0;
  double model_mean_us          = 0.0;
  double tracking_rmse_m        = 0.0;
  std::size_t controller_steps  = 0;
  std::size_t indi_steps        = 0;
};

/**
 * @brief Runs a single controller × generator mission.
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
   * @param output_csv       Output log path (MCAP; empty string disables logging).
   * @param metadata         Run metadata (controller/generator names, run id).
   *
   * @throws std::invalid_argument if @p controller or @p traj_gen is null or
   *         if @p example_cfg has no waypoints.
   */
  WaypointsSimulator(std::unique_ptr<IController> controller,
                     std::unique_ptr<ITrajectoryGenerator> traj_gen,
                     const ExampleConfig& example_cfg,
                     const mav_simulator::SimulatorParameters& simulator_params,
                     const std::string& output_csv,
                     const RunMetadata& metadata);

  /**
   * @brief Execute the mission until the hover phase elapses.
   *
   * Populates benchmarkStats() on return.
   */
  void run();

  /** @brief Print timing statistics to stdout (long form). */
  void printBenchmark() const;

  /** @return Timing statistics from the last run(). Zero-initialised before run(). */
  const BenchmarkStats& benchmarkStats() const { return stats_; }

  /** @return Reference to the underlying mav_simulator. Only valid after construction. */
  const mav_simulator::Simulator& simulator() const { return sim_; }

  /** @return Canonical CSV path used by this run (empty when logging disabled). */
  const std::string& outputCsv() const { return output_csv_; }

private:
  void checkCompatibility_() const;

  std::unique_ptr<IController> controller_;
  std::unique_ptr<ITrajectoryGenerator> traj_gen_;
  ExampleConfig example_cfg_;
  std::string output_csv_;
  RunMetadata metadata_;
  mav_simulator::Simulator sim_;
  BenchmarkStats stats_;
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_WAYPOINTS_SIMULATOR_HPP_
