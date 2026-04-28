// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file waypoints_simulator.cpp
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "framework/waypoints_simulator.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/delay_buffer.hpp"
#include "framework/stdout_progress.hpp"
#include "framework/types.hpp"
#include "framework/waypoint_scheduler.hpp"
#include "utils/utils.hpp"

namespace mpc_examples::framework {

namespace {

using Clock = std::chrono::high_resolution_clock;

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

double meanUs(const std::vector<double>& seconds) {
  if (seconds.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (double x : seconds) {
    sum += x;
  }
  return sum / static_cast<double>(seconds.size()) * 1e6;
}

double resolveDelay(const DelayMode mode, const double measured_s, const double fixed_s) {
  return mode == DelayMode::kMeasured ? std::max(measured_s, 0.0) : std::max(fixed_s, 0.0);
}

struct TimedCommand {
  ControlCommand cmd;
  double compute_time_us  = 0.0;
  double delay_applied_us = 0.0;
};

struct TimedReference {
  ReferenceSample sample;
  double update_time_us   = 0.0;
  double eval_time_us     = 0.0;
  double delay_applied_us = 0.0;
};

}  // namespace

WaypointsSimulator::WaypointsSimulator(std::unique_ptr<IController> controller,
                                       std::unique_ptr<ITrajectoryGenerator> traj_gen,
                                       const ExampleConfig& example_cfg,
                                       const mav_simulator::SimulatorParameters& simulator_params,
                                       const std::string& output_csv,
                                       const RunMetadata& metadata)
    : controller_(std::move(controller)), traj_gen_(std::move(traj_gen)), example_cfg_(example_cfg),
      output_csv_(output_csv), metadata_(metadata), sim_(simulator_params) {
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

  const mav_model::State initial_state = sim_.getState();
  controller_->initialize(initial_state, example_cfg_);
  traj_gen_->initialize(initial_state, example_cfg_);

  // --- Scheduler ------------------------------------------------------------
  WaypointScheduler scheduler;
  scheduler.initialize(example_cfg_.waypoints, initial_state.getPositionVector(),
                       example_cfg_.max_speed, example_cfg_.settle_margin_s);

  // The first waypoint becomes active at t=0.
  traj_gen_->onWaypointChanged(example_cfg_.waypoints.front(), initial_state, 0.0);

  // --- Timing parameters ----------------------------------------------------
  const double model_dt      = example_cfg_.model_dt;
  const double controller_dt = example_cfg_.controller_dt;
  const double outer_dt      = controller_->controlPeriod();
  const double hover_time    = example_cfg_.hover_time;
  const double mission_end_t = scheduler.finalTime();
  const double max_sim_time  = mission_end_t + hover_time;
  const bool benchmark       = example_cfg_.benchmark;
  const bool silent          = example_cfg_.silent;
  const double max_speed     = example_cfg_.max_speed;

  const int N_samples = controller_->referenceHorizonSize();
  const double dt_h   = controller_->referenceHorizonDt();
  std::vector<ReferenceSample> refs(static_cast<std::size_t>(N_samples));

  // --- Logging --------------------------------------------------------------
  // TODO: remove once MCAP pipeline validated.
  // std::unique_ptr<UnifiedCsvLogger> logger;
  // if (!benchmark && !output_csv_.empty()) {
  //   logger = std::make_unique<UnifiedCsvLogger>(output_csv_, metadata_);
  // }
  std::unique_ptr<UnifiedMcapLogger> logger;
  if (!benchmark && !output_csv_.empty()) {
    logger = std::make_unique<UnifiedMcapLogger>(output_csv_, metadata_);
  }

  // --- Delay buffers --------------------------------------------------------
  DelayBuffer<TimedCommand> cmd_buffer;
  DelayBuffer<TimedReference> ref_buffer;

  // --- Benchmark accumulators ----------------------------------------------
  const auto est_outer_steps = static_cast<std::size_t>(max_sim_time / outer_dt) + 1U;
  const auto est_indi_steps  = static_cast<std::size_t>(max_sim_time / controller_dt) + 1U;
  std::vector<double> controller_times;
  std::vector<double> generator_update_times;
  std::vector<double> generator_eval_times;
  std::vector<double> indi_times;
  std::vector<double> imu_times;
  std::vector<double> model_times;
  controller_times.reserve(est_outer_steps);
  generator_update_times.reserve(est_outer_steps);
  generator_eval_times.reserve(est_outer_steps);
  indi_times.reserve(est_indi_steps);
  imu_times.reserve(est_indi_steps);
  model_times.reserve(est_indi_steps);

  double tracking_sq_sum       = 0.0;
  std::size_t tracking_samples = 0;

  // --- Initial synchronous command (warm-start the buffers) -----------------
  // We evaluate the horizon once at t=0 with zero delay so the physics loop
  // has a sensible command from tick zero onwards.
  {
    traj_gen_->update(0.0, initial_state);
    for (int k = 0; k < N_samples; ++k) {
      refs[static_cast<std::size_t>(k)] = traj_gen_->evaluate(k * dt_h);
    }
    const ControlCommand warm_cmd = controller_->computeCommand(initial_state, refs);
    TimedCommand warm{};
    warm.cmd = warm_cmd;
    TimedReference warm_ref{};
    warm_ref.sample = refs.front();
    cmd_buffer.push(warm, 0.0);
    ref_buffer.push(warm_ref, 0.0);
  }

  // --- Main loop ------------------------------------------------------------
  ControlCommand current_cmd    = ControlCommand{};
  ReferenceSample current_ref   = refs.front();
  double current_cmd_compute_us = 0.0;
  double current_cmd_delay_us   = 0.0;
  double current_ref_update_us  = 0.0;
  double current_ref_eval_us    = 0.0;
  double current_ref_delay_us   = 0.0;

  bool hover_active     = false;
  double hover_end_time = max_sim_time;
  int active_index      = scheduler.activeIndex();

  const auto wall_start = Clock::now();
  double t              = 0.0;

  while (t < max_sim_time + 1e-9) {
    if (hover_active && t >= hover_end_time - 1e-9) {
      break;
    }

    // Read ground-truth state
    const mav_model::State state          = sim_.getState();
    const Eigen::Vector3d position        = state.getPositionVector();
    const Eigen::Quaterniond orientation  = state.getOrientationVector();
    const Eigen::Vector3d linear_velocity = state.getLinearVelocityVector();

    if (!isFiniteVector3(position) || !isFiniteQuaternion(orientation) ||
        !isFiniteVector3(linear_velocity)) {
      std::cerr << "\n[WaypointsSimulator] Non-finite state at t=" << t << " s\n";
      break;
    }

    // Scheduler tick: may trigger a replan in the generator.
    const WaypointScheduler::TickResult tick = scheduler.tick(t);
    if (tick.waypoint_changed) {
      traj_gen_->onWaypointChanged(scheduler.waypoint(static_cast<std::size_t>(tick.active_index)),
                                   state, t);
    }
    active_index = tick.active_index;

    // --- Generator step (measure wall-clock) -------------------------------
    const auto gen_t0 = Clock::now();
    traj_gen_->update(t, state);
    const auto gen_t1 = Clock::now();
    for (int k = 0; k < N_samples; ++k) {
      refs[static_cast<std::size_t>(k)] = traj_gen_->evaluate(t + k * dt_h);
    }
    const auto gen_t2         = Clock::now();
    const double gen_update_s = std::chrono::duration<double>(gen_t1 - gen_t0).count();
    const double gen_eval_s   = std::chrono::duration<double>(gen_t2 - gen_t1).count();
    generator_update_times.push_back(gen_update_s);
    generator_eval_times.push_back(gen_eval_s);
    const double gen_delay_s =
        resolveDelay(example_cfg_.generator_delay_mode, gen_update_s + gen_eval_s,
                     example_cfg_.generator_delay_fixed_s);

    TimedReference ref_payload;
    ref_payload.sample           = refs.front();
    ref_payload.update_time_us   = gen_update_s * 1e6;
    ref_payload.eval_time_us     = gen_eval_s * 1e6;
    ref_payload.delay_applied_us = gen_delay_s * 1e6;
    ref_buffer.push(ref_payload, t + gen_delay_s);

    // --- Controller step (measure wall-clock) ------------------------------
    const auto ctrl_t0        = Clock::now();
    const ControlCommand cmd  = controller_->computeCommand(state, refs);
    const auto ctrl_t1        = Clock::now();
    const double ctrl_solve_s = std::chrono::duration<double>(ctrl_t1 - ctrl_t0).count();
    controller_times.push_back(ctrl_solve_s);

    const double ctrl_delay_s = resolveDelay(example_cfg_.controller_delay_mode, ctrl_solve_s,
                                             example_cfg_.controller_delay_fixed_s);

    TimedCommand cmd_payload;
    cmd_payload.cmd              = cmd;
    cmd_payload.compute_time_us  = ctrl_solve_s * 1e6;
    cmd_payload.delay_applied_us = ctrl_delay_s * 1e6;
    cmd_buffer.push(cmd_payload, t + gen_delay_s + ctrl_delay_s);

    // --- Hover detection ---------------------------------------------------
    if (!hover_active && tick.finished && t >= mission_end_t - 1e-9) {
      hover_active   = true;
      hover_end_time = t + hover_time;
      if (!silent) {
        std::cout << "\n  mission finished @ t=" << t << "s · hovering for " << hover_time << "s\n";
      }
    }

    // --- Inner loop: INDI + physics at controller_dt -----------------------
    double t_inner = t;
    while (t_inner < t + outer_dt - 1e-9) {
      const double t_sub = t_inner + controller_dt;

      // Query buffers for the latest sample visible at t_sub.
      if (auto new_cmd = cmd_buffer.latestAvailable(t_sub)) {
        current_cmd            = new_cmd->cmd;
        current_cmd_compute_us = new_cmd->compute_time_us;
        current_cmd_delay_us   = new_cmd->delay_applied_us;
      }
      if (auto new_ref = ref_buffer.latestAvailable(t_sub)) {
        current_ref           = new_ref->sample;
        current_ref_update_us = new_ref->update_time_us;
        current_ref_eval_us   = new_ref->eval_time_us;
        current_ref_delay_us  = new_ref->delay_applied_us;
      }

      sim_.setReferenceRates(current_cmd.thrust_n, current_cmd.angular_rate);

      const auto indi_t0 = Clock::now();
      sim_.updateController(controller_dt);
      const auto indi_t1 = Clock::now();
      sim_.updateImu(controller_dt);
      const auto indi_t2 = Clock::now();

      double t_model = t_inner;
      while (t_model < t_sub - 1e-9) {
        sim_.updateModel(model_dt);
        t_model += model_dt;
      }
      const auto indi_t3 = Clock::now();

      indi_times.push_back(std::chrono::duration<double>(indi_t1 - indi_t0).count());
      imu_times.push_back(std::chrono::duration<double>(indi_t2 - indi_t1).count());
      model_times.push_back(std::chrono::duration<double>(indi_t3 - indi_t2).count());

      t_inner = t_sub;

      // Log + accumulate tracking RMSE after this INDI sub-step.
      const mav_model::State s           = sim_.getState();
      const Eigen::Vector3d tracking_err = s.getPositionVector() - current_ref.position;
      tracking_sq_sum += tracking_err.squaredNorm();
      ++tracking_samples;

      if (logger) {
        LogRow row;
        row.time                       = t_sub;
        row.position                   = s.getPositionVector();
        row.orientation                = s.getOrientationVector();
        row.linear_velocity            = s.getLinearVelocityVector();
        row.angular_velocity           = s.getAngularVelocityVector();
        row.reference_position         = scheduler.waypoint(static_cast<std::size_t>(active_index));
        row.trajectory_position        = current_ref.position;
        row.trajectory_velocity        = current_ref.velocity;
        row.trajectory_orientation     = eulerToQuaternion(0.0, 0.0, current_ref.yaw);
        row.thrust_n                   = current_cmd.thrust_n;
        row.command_angular_velocity   = current_cmd.angular_rate;
        row.motor_w                    = s.getMotorAngularVelocityVector();
        row.controller_compute_time_us = current_cmd_compute_us;
        row.generator_update_time_us   = current_ref_update_us;
        row.generator_eval_time_us     = current_ref_eval_us;
        row.controller_delay_applied_us = current_cmd_delay_us;
        row.generator_delay_applied_us  = current_ref_delay_us;
        row.waypoint_index              = active_index;
        row.hover_active                = hover_active;
        row.max_speed                   = max_speed;
        logger->logRow(row);
      }
    }

    t += outer_dt;

    if (!silent) {
      const Eigen::Vector3d state_pos = sim_.getState().getPositionVector();
      const double err_now            = (state_pos - current_ref.position).norm();
      printStatus(t, max_sim_time, active_index, scheduler.size(), err_now, current_cmd_compute_us);
    }
  }

  const auto wall_end      = Clock::now();
  const double real_time_s = std::chrono::duration<double>(wall_end - wall_start).count();
  const double rmse_m      = tracking_samples > 0
                                 ? std::sqrt(tracking_sq_sum / static_cast<double>(tracking_samples))
                                 : 0.0;

  stats_ = BenchmarkStats{
      .simulated_time_s         = t,
      .real_time_s              = real_time_s,
      .sim_speedup              = (real_time_s > 0.0) ? (t / real_time_s) : 0.0,
      .controller_mean_us       = meanUs(controller_times),
      .generator_update_mean_us = meanUs(generator_update_times),
      .generator_eval_mean_us   = meanUs(generator_eval_times),
      .indi_mean_us             = meanUs(indi_times),
      .imu_mean_us              = meanUs(imu_times),
      .model_mean_us            = meanUs(model_times),
      .tracking_rmse_m          = rmse_m,
      .controller_steps         = controller_times.size(),
      .indi_steps               = indi_times.size(),
  };

  if (!silent) {
    printCaseSummary(stats_.real_time_s, stats_.tracking_rmse_m, stats_.controller_mean_us,
                     stats_.generator_update_mean_us + stats_.generator_eval_mean_us);
  }
}

void WaypointsSimulator::printBenchmark() const {
  std::cout << "Simulated time      : " << stats_.simulated_time_s << " s\n"
            << "Real time           : " << stats_.real_time_s << " s\n"
            << "Sim speedup         : " << stats_.sim_speedup << "x\n"
            << "Controller compute  : " << stats_.controller_mean_us << " µs (avg, "
            << stats_.controller_steps << " steps)\n"
            << "Generator update    : " << stats_.generator_update_mean_us << " µs (avg)\n"
            << "Generator eval      : " << stats_.generator_eval_mean_us << " µs (avg)\n"
            << "INDI controller     : " << stats_.indi_mean_us << " µs (avg)\n"
            << "IMU update          : " << stats_.imu_mean_us << " µs (avg)\n"
            << "Physics model       : " << stats_.model_mean_us << " µs (avg, per INDI step)\n"
            << "Tracking RMSE       : " << stats_.tracking_rmse_m << " m\n";
}

}  // namespace mpc_examples::framework
