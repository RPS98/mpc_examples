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

#include "framework/degenerate_hold.hpp"
#include "framework/delay_buffer.hpp"
#include "framework/stdout_progress.hpp"
#include "framework/types.hpp"
#include "framework/waypoint_scheduler.hpp"
#include "gcopter_lib/trajectory_generator.hpp"
#include "gcopter_lib/types.hpp"
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
  // Optional override of the simulator's initial pose. Applied BEFORE
  // arm() so the HOVER reference (seeded from getState() inside arm)
  // matches the new start pose. Mirrors `vehicle_initial_pose` used by
  // the standalone acados examples.
  if (example_cfg_.has_initial_state) {
    mav_model::State s = sim_.getModel().getState();
    s.setPositionVector(Eigen::Vector3d(example_cfg_.initial_position[0],
                                        example_cfg_.initial_position[1],
                                        example_cfg_.initial_position[2]));
    s.setOrientationVector(eulerToQuaternion(example_cfg_.initial_rpy[0],
                                             example_cfg_.initial_rpy[1],
                                             example_cfg_.initial_rpy[2]));
    sim_.getModel().setState(s);
  }
  sim_.arm();
  sim_.setControlMode(mav_simulator::ControlMode::RATES);

  const mav_model::State initial_state = sim_.getState();
  controller_->initialize(initial_state, example_cfg_);
  traj_gen_->initialize(initial_state, example_cfg_);

  // --- Effective waypoint list (with optional takeoff / land) ---------------
  // Mirrors aerostack2's 3-phase mission (takeoff_behavior → goto loop →
  // land_behavior): prepend a synthetic takeoff wp at
  // (initial.x, initial.y, takeoff_altitude_m) and append a synthetic
  // land wp at (last.x, last.y, 0). Both phases are still flown by the
  // current controller/generator pair so the mcap records the climb and
  // descent transients with the same dynamics. The first/last indices are
  // tracked so the `experiment_active` gating (Phase 6) can skip them.
  std::vector<Eigen::Vector3d> effective_waypoints = example_cfg_.waypoints;
  // `mission_first_idx` / `mission_last_idx` are 0-based inclusive bounds
  // around the user-supplied waypoints inside the effective list. The
  // `experiment_active` mask is True only inside that window so the
  // takeoff and landing transitions stay out of the paper metrics.
  // `mission_first_idx` and `mission_last_idx` are 0-based INCLUSIVE bounds
  // around the user-supplied waypoints inside `effective_waypoints`.
  // `experiment_active` is True only inside that closed interval, so the
  // synthetic takeoff (idx < mission_first_idx) and the synthetic land
  // (idx > mission_last_idx) stay out of the paper metrics — mirroring
  // aerostack2's behavior, where takeoff_behavior and land_behavior keep
  // `experiment_active=False` outside the mission window.
  const std::size_t n_project_wps = effective_waypoints.size();
  std::size_t mission_first_idx = 0U;
  if (example_cfg_.takeoff_altitude_m > 0.0 && n_project_wps > 0U) {
    Eigen::Vector3d takeoff_wp = initial_state.getPositionVector();
    takeoff_wp.z()             = example_cfg_.takeoff_altitude_m;
    effective_waypoints.insert(effective_waypoints.begin(), takeoff_wp);
    mission_first_idx = 1U;
  }
  // Inclusive last index of the user-supplied waypoint segment.
  std::size_t mission_last_idx = (n_project_wps == 0U)
      ? 0U
      : mission_first_idx + (n_project_wps - 1U);
  if (example_cfg_.land_at_end && n_project_wps > 0U) {
    Eigen::Vector3d land_wp = effective_waypoints.back();
    land_wp.z()             = 0.0;
    effective_waypoints.push_back(land_wp);
    // mission_last_idx already points to the user's last waypoint
    // (one-before-land); leave it untouched so the gating excludes the
    // landing transient.
  }

  // --- Scheduler ------------------------------------------------------------
  // In `continuous` mission_mode, override the scheduler tunables so the
  // drone flows through every waypoint with no idle hold between hops.
  // This mirrors aerostack2's `mission_moving_path.py` (single
  // follow_reference goal that lets the gcopter trajectory advance the
  // pose target continuously through the entire waypoint list).
  const bool continuous_mode = (example_cfg_.mission_mode == "continuous");
  const double scheduler_settle_margin_s =
      continuous_mode ? 0.0 : example_cfg_.settle_margin_s;
  const double scheduler_speed_factor =
      continuous_mode ? 1.0 : example_cfg_.scheduler_speed_factor;

  WaypointScheduler scheduler;
  scheduler.initialize(effective_waypoints, initial_state.getPositionVector(),
                       example_cfg_.max_speed, scheduler_settle_margin_s,
                       scheduler_speed_factor);

  // The first waypoint becomes active at t=0.
  traj_gen_->onWaypointChanged(effective_waypoints.front(), initial_state, 0.0);

  // --- follow_reference emulation (continuous_mode only) --------------------
  // Mirror aerostack2's moving_path chain:
  //   capa 1: mission generator pre-fits a smooth polynomial between
  //           every pair of consecutive waypoints (gcopter p2p; the
  //           drone briefly rests at each waypoint of the mission
  //           plan, mirroring `mission_moving_path.py`'s sampled TF
  //           trace). The full mission target plan is the
  //           concatenation of those segments in time.
  //   capa 2: broadcaster samples the active segment at outer-loop
  //           rate and publishes the TF target (`target_now`).
  //   capa 3 (mpc_trajectory only): `follow_reference_plugin_trajectory`
  //           re-emits modify_waypoint when target_now has drifted
  //           more than `modify_threshold` from the last published
  //           one, at most at `modify_frequency` Hz.
  //   capa 4 (mpc_trajectory only): the local generator (gcopter /
  //           jerk_limited) re-plans between drone pose+velocity and
  //           the modify_waypoint sample.
  //   capa 5: the controller consumes the reference.
  //
  // PID / Pos-MPC bypass capas 3-4 — aerostack2's
  // `follow_reference_plugin_position` publishes the TF target sample
  // directly as `motion_reference/pose`. The mav equivalent: the local
  // `WaypointReferenceGenerator` is updated *every outer tick* (no
  // rate limit), so its `evaluate()` returns `target_now` verbatim.
  //
  // `MissionTargetPlan` is the capa-1+2 sampler. Implementation:
  // gcopter is pre-generated point-to-point for each segment
  // (wp[i], wp[i+1]) — a model that L-BFGS solves reliably (verified
  // empirically: the multi-waypoint generate() diverges; the p2p call
  // converges for every segment in the paper geometry). When a
  // segment's gcopter pre-fit fails, that segment falls back to a
  // linear interpolation at `max_speed`.
  struct MissionTargetPlan {
    std::vector<Eigen::Vector3d> wps;
    std::vector<double> t_at_wp;  // arrival time at wps[i].
    // One pre-fitted gcopter trajectory per (wps[i], wps[i+1]) hop.
    // Empty entry → fall back to piecewise-linear for that segment.
    std::vector<std::unique_ptr<gcopter_lib::TrajectoryGenerator>> seg_plans;
    double duration() const { return t_at_wp.empty() ? 0.0 : t_at_wp.back(); }
    Eigen::Vector3d position(double t) const {
      if (wps.empty()) {
        return Eigen::Vector3d::Zero();
      }
      if (t <= 0.0 || t_at_wp.size() == 1) {
        return wps.front();
      }
      if (t >= t_at_wp.back()) {
        return wps.back();
      }
      // Locate the active segment (wps[i] → wps[i+1]).
      std::size_t i = 0U;
      while (i + 1U < t_at_wp.size() && t > t_at_wp[i + 1U]) {
        ++i;
      }
      const double t_local = t - t_at_wp[i];
      // Use the smooth gcopter p2p plan when available; otherwise
      // linear interpolation between wps[i] and wps[i+1].
      if (i < seg_plans.size() && seg_plans[i] && seg_plans[i]->isValid()) {
        const double t_clamped = std::clamp(
            t_local, 0.0, seg_plans[i]->duration());
        return seg_plans[i]->position(t_clamped);
      }
      const double seg_dt = t_at_wp[i + 1U] - t_at_wp[i];
      const double alpha = (seg_dt > 0.0) ? (t_local / seg_dt) : 0.0;
      return (1.0 - alpha) * wps[i] + alpha * wps[i + 1U];
    }
  };

  MissionTargetPlan target_plan;
  Eigen::Vector3d target_last_published = initial_state.getPositionVector();
  double target_last_modify_t = -1.0e9;
  // Degenerate-hold state. Mirrors aerostack2's `degenerate_hold_`,
  // `degenerate_target_` and `init_yaw_angle_` in
  // generate_polynomial_trajectory_behavior. Active only inside the
  // follow_reference emulator (target_plan_active branch below).
  bool degenerate_hold_active   = false;
  Eigen::Vector3d degenerate_target = initial_state.getPositionVector();
  double degenerate_yaw         = quaternionToEuler(initial_state.getOrientationVector()).z();
  // The follow_reference emulator is now active for every controller
  // (PID, Pos-MPC and mpc_trajectory) in continuous_mode. The rate
  // limit applied to capa 3 differs by controller scope: trajectory
  // controllers rate-limit at the configured 10 Hz / 5 cm to protect
  // the acados QP from too-frequent local replans; position-scope
  // controllers (PID, Pos-MPC) bypass capa 3-4 entirely by updating
  // the `waypoints` adapter every outer tick — see the
  // `controller_uses_local_generator` branch below.
  const bool target_plan_active = continuous_mode &&
                                  effective_waypoints.size() >= 2U;
  const bool controller_uses_local_generator =
      (metadata_.controller_name == "mpc_trajectory");
  const double effective_modify_period_s =
      controller_uses_local_generator ? example_cfg_.target_modify_period_s : 0.0;
  const double effective_modify_threshold_m =
      controller_uses_local_generator ? example_cfg_.target_modify_threshold_m : 0.0;
  if (target_plan_active) {
    target_plan.wps     = effective_waypoints;
    target_plan.t_at_wp.assign(target_plan.wps.size(), 0.0);
    target_plan.seg_plans.resize(target_plan.wps.size() > 0
                                     ? target_plan.wps.size() - 1U
                                     : 0U);

    // capa 1: pre-generate one gcopter p2p trajectory per segment
    // (wps[i] → wps[i+1]). Each plan starts and ends at rest, so the
    // concatenation is C0-continuous (positions match) but not C1 —
    // the drone briefly rests at each waypoint, which is the
    // physically-stable behaviour. Mirror the gcopter parameters of
    // the local generator (config_gcopter.yaml) so the smoothing model
    // matches end-to-end.
    gcopter_lib::GeneratorConfig seg_cfg;
    seg_cfg.params.mass                = 1.0;
    seg_cfg.params.gravity             = 9.81;
    seg_cfg.params.horizontal_drag     = 0.1;
    seg_cfg.params.vertical_drag       = 0.1;
    seg_cfg.params.parasitic_drag      = 0.01;
    seg_cfg.params.speed_smooth_factor = 0.01;
    seg_cfg.limits.max_velocity        = example_cfg_.max_speed;
    seg_cfg.limits.max_body_rate       = 6.0;
    seg_cfg.limits.max_tilt_angle      = 0.785;
    seg_cfg.limits.min_thrust          = 0.1;
    seg_cfg.limits.max_thrust          = 30.0;

    std::size_t failed_segments = 0U;
    for (std::size_t i = 0U; i + 1U < target_plan.wps.size(); ++i) {
      const Eigen::Vector3d& p0 = target_plan.wps[i];
      const Eigen::Vector3d& p1 = target_plan.wps[i + 1U];
      std::vector<gcopter_lib::Waypoint> seg_wps(2);
      seg_wps[0].position = p0;
      seg_wps[0].velocity = Eigen::Vector3d::Zero();
      seg_wps[1]          = gcopter_lib::EndWaypoint(p1);
      auto seg_gen = std::make_unique<gcopter_lib::TrajectoryGenerator>(seg_cfg);
      const bool ok = seg_gen->generate(seg_wps, example_cfg_.max_speed);
      double seg_dt;
      if (ok && seg_gen->isValid()) {
        seg_dt                   = seg_gen->duration();
        target_plan.seg_plans[i] = std::move(seg_gen);
      } else {
        // Linear-interpolation fallback: traverse the segment at
        // max_speed so the cumulative timeline keeps a sensible total.
        const double dist = (p1 - p0).norm();
        seg_dt            = dist / std::max(example_cfg_.max_speed, 1e-9);
        target_plan.seg_plans[i].reset();
        ++failed_segments;
      }
      target_plan.t_at_wp[i + 1U] = target_plan.t_at_wp[i] + seg_dt;
    }

    if (!example_cfg_.silent) {
      std::cout << "  follow_reference emulation: gcopter p2p mission plan, "
                << "duration=" << target_plan.duration() << " s ("
                << target_plan.seg_plans.size() - failed_segments << "/"
                << target_plan.seg_plans.size() << " segments smooth)";
      if (controller_uses_local_generator) {
        std::cout << ", modify_period=" << effective_modify_period_s
                  << " s, threshold=" << effective_modify_threshold_m << " m";
      } else {
        std::cout << ", controller-scope=position → modify every tick";
      }
      std::cout << "\n";
    }
  }
  const double target_plan_duration = target_plan.duration();

  // --- Timing parameters ----------------------------------------------------
  const double model_dt      = example_cfg_.model_dt;
  const double controller_dt = example_cfg_.controller_dt;
  const double outer_dt      = controller_->controlPeriod();
  const double hover_time    = example_cfg_.hover_time;
  // When follow_reference emulation is active, the target_plan duration
  // drives the mission end (plus start_delay + a small settle margin so
  // the drone can catch up the moving target after the last sample).
  // Otherwise fall back to the scheduler's per-segment time budget.
  const double mission_end_t =
      target_plan_active ?
          (example_cfg_.target_start_delay_s + target_plan_duration) :
          scheduler.finalTime();
  // sim_config.sim_time is a hard cap on the wall of simulated time: the run
  // ends at min(mission_end + hover, sim_time), even if that truncates the
  // hover phase or the mission itself.
  const double max_sim_time = std::min(mission_end_t + hover_time, example_cfg_.sim_time);
  const bool benchmark      = example_cfg_.benchmark;
  const bool silent         = example_cfg_.silent;
  const double max_speed    = example_cfg_.max_speed;

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

  // Detect whether the active controller consumes a velocity / acceleration
  // horizon. Trajectory-scope controllers (mpc_trajectory) do; position-only
  // ones (pid, mpc_position) don't. Mirrors aerostack2's behaviour: the
  // trajectory_generation behavior only emits `motion_reference/trajectory`
  // when it is actually planning a trajectory (i.e. the trj scope).
  const ReferenceFieldMask required_fields = controller_->requiredReferenceFields();
  const bool emit_trajectory_horizon =
      hasField(required_fields, ReferenceField::kVelocity) ||
      hasField(required_fields, ReferenceField::kAcceleration);

  // Per-topic emission state so the MCAP mirrors aerostack2's mission
  // publish pattern: rate-limited pose reference + latched semantics
  // (one publication per value change) on the four mission-side topics.
  // `t_last_mission_pose_ref_pub_` is initialised to -inf so the first
  // tick inside the mission window emits the topic. The `std::optional`
  // sentinels for the latched topics guarantee the first row emits the
  // initial value once (so the reviewer's ZOH resampler has data to
  // hold from).
  const double mission_pose_ref_period_s =
      (example_cfg_.mission_pose_ref_freq > 0.0)
          ? 1.0 / example_cfg_.mission_pose_ref_freq
          : 0.0;
  double t_last_mission_pose_ref_pub = -1.0e30;
  std::optional<int> last_waypoint_index;
  std::optional<double> last_max_speed;
  std::optional<bool> last_experiment_active;
  std::optional<bool> last_hover_active;

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

    // --- follow_reference emulation tick (continuous_mode only) ------------
    // Sample the moving-target polynomial at the current simulator time and,
    // when it has shifted enough from the last published target and the
    // modify period has elapsed, request a replan of the local generator.
    // This reproduces aerostack2's follow_reference_plugin_trajectory →
    // generate_polynomial_trajectory_behavior modify_waypoint chain.
    //
    // The degenerate-hold gate runs *before* the modify check: if the
    // current target is within `degenerate_distance_m` of the drone, we
    // skip the generator entirely and publish a static reference horizon
    // (target, v=0, a=0, yaw=latched). Mirrors as2 tryEnterDegenerateHold.
    bool skip_generator_for_hold = false;
    if (target_plan_active) {
      const double t_motion = std::max(0.0, t - example_cfg_.target_start_delay_s);
      const double t_target = std::min(t_motion, target_plan_duration);
      const Eigen::Vector3d target_now = target_plan.position(t_target);
      const bool target_now_degenerate = isDegenerateTarget(
          target_now, position, example_cfg_.degenerate_distance_m);
      if (target_now_degenerate) {
        if (!degenerate_hold_active && !silent) {
          std::cerr << "\n[WaypointsSimulator] Target within "
                    << example_cfg_.degenerate_distance_m
                    << " m of vehicle at t=" << t
                    << " s: degenerate-hold engaged.\n";
        }
        if (!degenerate_hold_active) {
          // Latch the yaw on entry, like as2 anchors `init_yaw_angle_`.
          degenerate_yaw = quaternionToEuler(orientation).z();
        }
        degenerate_hold_active    = true;
        degenerate_target         = target_now;
        skip_generator_for_hold   = true;
        target_last_published     = target_now;
        target_last_modify_t      = t;
      } else {
        if (degenerate_hold_active) {
          if (!silent) {
            std::cerr << "\n[WaypointsSimulator] Degenerate-hold released at t="
                      << t << " s: regenerating trajectory.\n";
          }
          // Force a replan on the first tick after the hold so the local
          // generator picks up the new target from the live drone state.
          traj_gen_->onWaypointChanged(target_now, state, t);
          target_last_published = target_now;
          target_last_modify_t  = t;
          degenerate_hold_active = false;
        } else {
          const double dt_since_modify = t - target_last_modify_t;
          const double dist            = (target_now - target_last_published).norm();
          if (dt_since_modify >= effective_modify_period_s &&
              dist >= effective_modify_threshold_m) {
            traj_gen_->onWaypointChanged(target_now, state, t);
            target_last_published = target_now;
            target_last_modify_t  = t;
          }
        }
      }
    }

    // --- Generator step (measure wall-clock) -------------------------------
    const auto gen_t0 = Clock::now();
    if (skip_generator_for_hold) {
      // Static horizon latched to (degenerate_target, degenerate_yaw, v=0,
      // a=0). Skip traj_gen_->update entirely.
      fillStaticHorizon(degenerate_target, degenerate_yaw, refs,
                        static_cast<std::size_t>(N_samples));
    } else {
      traj_gen_->update(t, state);
    }
    const auto gen_t1 = Clock::now();
    if (!skip_generator_for_hold) {
      for (int k = 0; k < N_samples; ++k) {
        refs[static_cast<std::size_t>(k)] = traj_gen_->evaluate(t + k * dt_h);
      }
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

    // --- Controller step (measure wall-clock) ------------------------------
    const auto ctrl_t0        = Clock::now();
    const ControlCommand cmd  = controller_->computeCommand(state, refs);
    const auto ctrl_t1        = Clock::now();
    const double ctrl_solve_s = std::chrono::duration<double>(ctrl_t1 - ctrl_t0).count();
    controller_times.push_back(ctrl_solve_s);

    const double ctrl_delay_s = resolveDelay(example_cfg_.controller_delay_mode, ctrl_solve_s,
                                             example_cfg_.controller_delay_fixed_s);

    ref_buffer.push(ref_payload, t + gen_delay_s);

    TimedCommand cmd_payload;
    cmd_payload.cmd              = cmd;
    cmd_payload.compute_time_us  = ctrl_solve_s * 1e6;
    cmd_payload.delay_applied_us = ctrl_delay_s * 1e6;
    cmd_buffer.push(cmd_payload, t + gen_delay_s + ctrl_delay_s);

    // --- Hover detection ---------------------------------------------------
    const bool mission_done = target_plan_active ?
        (t >= mission_end_t - 1e-9) :
        (tick.finished && t >= mission_end_t - 1e-9);
    if (!hover_active && mission_done) {
      hover_active   = true;
      hover_end_time = t + hover_time;
      if (!silent) {
        std::cout << "\n  mission finished @ t=" << t << "s · hovering for " << hover_time << "s\n";
      }
    }

    // Build the trajectory horizon (TrajectorySetpoints payload) once per
    // outer tick when the controller consumes a velocity/acceleration
    // horizon. Emitted on the first inner sub-step below so the topic
    // cadence matches the outer-loop rate (100 Hz), comparable to the
    // aerostack2 trajectory_generation_behavior publish cadence.
    std::vector<mav_flight_review::TrajectoryPoint> trajectory_horizon;
    if (emit_trajectory_horizon) {
      trajectory_horizon.reserve(static_cast<std::size_t>(N_samples));
      for (int k = 0; k < N_samples; ++k) {
        const ReferenceSample& s = refs[static_cast<std::size_t>(k)];
        mav_flight_review::TrajectoryPoint p;
        p.position     = s.position;
        p.twist        = s.velocity;
        p.acceleration = s.acceleration;
        p.yaw_angle    = static_cast<float>(s.yaw);
        trajectory_horizon.push_back(p);
      }
    }
    bool first_inner_sub = true;

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
        row.hover_active                = hover_active;
        // experiment_active / waypoint_index alignment (Phase 6).
        //
        // The `mav_flight_review.metrics` resolver picks the
        // `tracking_rmse_m` mask by signal presence:
        //   1. `waypoint_index` with multiple segments → post_settling
        //   2. `experiment_active` Bool present       → experiment_active
        //   3. fallback                               → all
        //
        // For both backends to land on the SAME branch per combo:
        //   * triangle (stepwise): publish waypoint_index = active_index
        //     (0..N-1), experiment_active = True between the takeoff
        //     hop and the pre-land hover. Resolver picks `post_settling`.
        //   * moving_path (continuous): publish waypoint_index = 0
        //     constant (no per-waypoint segments) and experiment_active
        //     = True while the moving target is in motion. Resolver
        //     skips `post_settling` (only one segment) and picks
        //     `experiment_active`.
        if (target_plan_active) {
          row.waypoint_index = 0;
          // Restrict the mission window to the user-supplied waypoints so
          // takeoff (target_plan.t_at_wp[..mission_first_idx]) and landing
          // (target_plan.t_at_wp[mission_last_idx..]) stay outside.
          double t_mission_start = example_cfg_.target_start_delay_s;
          double t_mission_end   = mission_end_t;
          if (mission_first_idx < target_plan.t_at_wp.size()) {
            t_mission_start = example_cfg_.target_start_delay_s +
                              target_plan.t_at_wp[mission_first_idx];
          }
          if (mission_last_idx < target_plan.t_at_wp.size()) {
            t_mission_end = example_cfg_.target_start_delay_s +
                            target_plan.t_at_wp[mission_last_idx];
          }
          row.experiment_active = !hover_active &&
                                  (t >= t_mission_start) &&
                                  (t <= t_mission_end + 1e-9);
        } else {
          row.waypoint_index = active_index;
          // experiment_active is True only while the scheduler is tracking
          // a user-supplied waypoint (mission_first_idx..mission_last_idx
          // inclusive). Takeoff (idx < mission_first_idx) and landing
          // (idx > mission_last_idx) stay out of the paper metrics, just
          // like aerostack2 keeps experiment_active=False outside the
          // mission window.
          const std::size_t active_uidx = static_cast<std::size_t>(active_index);
          row.experiment_active = !hover_active &&
              (active_uidx >= mission_first_idx) &&
              (active_uidx <= mission_last_idx);
        }
        row.max_speed = max_speed;
        // Mission topics are silent during the synthetic takeoff/landing
        // phases so that mav_flight_review's segment detector restricts
        // analysis to the user-supplied waypoint window.
        row.publish_mission_signals = row.experiment_active;

        // `debug/mission/reference/pose` payload: aerostack2 publishes
        // the active waypoint (stepwise) for the triangle mission and
        // the live moving-TF sample for moving_path. Mirror that split
        // here so the reviewer's `pose_ref` curve carries the same
        // semantics on both backends.
        if (target_plan_active) {
          const double t_motion = std::max(0.0, t - example_cfg_.target_start_delay_s);
          const double t_target = std::min(t_motion, target_plan_duration);
          row.mission_pose_ref_position = target_plan.position(t_target);
        } else {
          row.mission_pose_ref_position = row.reference_position;
        }

        row.publishes_desired_velocity = controller_->providesDesiredVelocity();
        if (row.publishes_desired_velocity) {
          row.desired_velocity = controller_->lastDesiredVelocity();
        }

        // Mission-topic emission gates. aerostack2 publishes
        // `debug/mission/reference/pose` at a fixed rate inside the goto /
        // follow_reference loop and the three latched mission topics only
        // on value change. Mirror that here:
        //   * pose_ref: emit at most once per mission_pose_ref_period_s
        //     and only inside the mission window (`publish_mission_signals`).
        //   * waypoint_index, max_speed: latched semantics — emit only on
        //     value change. Gated by the mission window so the synthetic
        //     takeoff / land phases stay silent (matching as2).
        //   * experiment_active, hover_active: latched semantics — emit on
        //     value change unconditionally so the reviewer sees the False→
        //     True and True→False transitions even at takeoff/land.
        if (row.publish_mission_signals && mission_pose_ref_period_s > 0.0 &&
            (t_sub - t_last_mission_pose_ref_pub) >= mission_pose_ref_period_s - 1e-9) {
          row.publish_mission_pose_ref = true;
          t_last_mission_pose_ref_pub  = t_sub;
        }
        if (row.publish_mission_signals) {
          if (!last_waypoint_index || *last_waypoint_index != row.waypoint_index) {
            row.publish_waypoint_index_change = true;
            last_waypoint_index               = row.waypoint_index;
          }
          if (!last_max_speed || *last_max_speed != row.max_speed) {
            row.publish_max_speed_change = true;
            last_max_speed               = row.max_speed;
          }
        }
        if (!last_experiment_active || *last_experiment_active != row.experiment_active) {
          row.publish_experiment_active_change = true;
          last_experiment_active               = row.experiment_active;
        }
        if (!last_hover_active || *last_hover_active != row.hover_active) {
          row.publish_hover_active_change = true;
          last_hover_active               = row.hover_active;
        }

        // `motion_reference/trajectory` mirrors aerostack2's
        // trajectory_generation_behavior output: emit one TrajectorySetpoints
        // per outer tick (gated to the mission window), only when the
        // controller actually consumes a velocity/acceleration horizon.
        if (emit_trajectory_horizon && first_inner_sub && row.publish_mission_signals) {
          row.trajectory_horizon         = trajectory_horizon;
          row.publish_trajectory_horizon = true;
        }
        first_inner_sub = false;

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
