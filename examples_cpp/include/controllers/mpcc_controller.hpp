// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpcc_controller.hpp
 *
 * Acados MPCC (Model Predictive Contouring Control) wrapped behind the
 * IController interface. Self-contained: the controller loads the
 * mission YAML inside :meth:`initialize` and builds its own
 * arc-length-reparametrised spline. ``computeCommand()`` ignores the
 * ``references`` argument and instead sets the solver's online
 * parameters from a sliding window over the internal spline.
 *
 * Mirrors examples_py/examples_py/controllers/mpcc_controller.py.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_ADAPTERS_MPCC_CONTROLLER_HPP_
#define MPC_EXAMPLES_ADAPTERS_MPCC_CONTROLLER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "framework/controller_base.hpp"
#include "mpcc_acados/acados_mpc.hpp"
#include "spline_trajectory_generator/arc_length_reparametrization.hpp"
#include "spline_trajectory_generator/trajectory_generator.hpp"

namespace mpc_examples::adapters {

/**
 * @brief MPCC adapter (ignores the framework's per-stage references).
 *
 * Horizon size advertised: 1 (the framework still queries one sample
 * per tick so the generator hook fires; the controller discards it).
 * Required reference fields: kPosition (cheapest mask that satisfies
 * WaypointsSimulator's compatibility check against any generator).
 */
class MpccController : public framework::IController {
public:
  struct Config {
    std::string mpc_yaml_path;        //!< Path to config_mpcc.yaml (top-level key `mpcc:`).
    std::string ocp_json_file_path;   //!< Optional; auto-resolved if empty.
    std::string mission_yaml;         //!< Empty → bundled demo (mission_loader::defaultPaths()).
    std::string gates_yaml;
    double desired_speed         = 4.0;
    double origin_offset_m       = 0.0;
    double closing_exit_margin_m = 3.0;
    double target_segment_length = 1.0;
    int    samples_per_segment   = 400;
  };

  explicit MpccController(const Config& cfg);

  /**
   * @brief Read config_mpcc.yaml.
   *
   * Expected structure (mirrors config_mpcc.yaml):
   *   controller.ocp_json_file_path  (optional) — acados OCP json path.
   *   mpcc.parameters / mpcc.constraints — solver weights and bounds.
   *     The `mpcc:` key (NOT `mpc:`) is required; the adapter rewrites
   *     the block to `mpc:` at runtime when handing it to upstream
   *     ``configureMpcFromYaml`` (which is hardcoded against `mpc:`).
   *   circuit (optional) — mission anchors mirroring CircuitGenerator.
   */
  static Config loadConfigFromYaml(const std::string& path);

  void initialize(const mav_model::State& initial_state,
                  const ExampleConfig& example_cfg) override;

  int referenceHorizonSize() const override { return 1; }
  double referenceHorizonDt() const override { return control_period_; }
  double controlPeriod() const override { return control_period_; }

  framework::ControlCommand computeCommand(
      const mav_model::State& state,
      const std::vector<framework::ReferenceSample>& references) override;

  framework::ReferenceFieldMask requiredReferenceFields() const override {
    return framework::makeMask({framework::ReferenceField::kPosition});
  }

  const std::string& name() const override { return name_; }
  double lastSolveTimeMicros() const override { return last_solve_us_; }
  Eigen::Vector3d lastDesiredVelocity() const override { return last_desired_velocity_; }
  bool providesDesiredVelocity() const override { return true; }

private:
  /// Rewrite @p source_path with the top-level `mpcc:` block renamed
  /// to `mpc:` so the upstream ``configureMpcFromYaml`` (hardcoded
  /// against `mpc:`) can consume it. Returns the temp file path.
  static std::string rekeyMpccToMpc(const std::string& source_path);

  Config cfg_;
  std::unique_ptr<acados_mpc::MPC> mpc_;
  std::unique_ptr<spline::TrajectoryGenerator> traj_;

  double control_period_ = 0.04;
  double last_solve_us_  = 0.0;
  Eigen::Vector3d last_desired_velocity_ = Eigen::Vector3d::Zero();
  double s_eval_         = 0.0;
  std::string name_      = "MpccController";
};

}  // namespace mpc_examples::adapters

#endif  // MPC_EXAMPLES_ADAPTERS_MPCC_CONTROLLER_HPP_
