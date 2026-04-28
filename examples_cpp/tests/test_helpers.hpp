// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MPC_EXAMPLES_TESTS_TEST_HELPERS_HPP_
#define MPC_EXAMPLES_TESTS_TEST_HELPERS_HPP_

#include <Eigen/Dense>

#include <string>

#include "framework/types.hpp"
#include "mav_model/datatypes/state.hpp"
#include "utils/example_config_utils.hpp"

namespace mpc_examples::testing {

#ifndef MPC_EXAMPLES_REPO_ROOT
#  error "MPC_EXAMPLES_REPO_ROOT must be injected via target_compile_definitions."
#endif

/// Absolute path to a repo-relative file (configs/..., libs/..., etc.).
inline std::string repoPath(const std::string& relative) {
  return std::string(MPC_EXAMPLES_REPO_ROOT) + "/" + relative;
}

/// Build a deterministic hover state at @p position (0 velocity, identity orientation).
inline mav_model::State hoverStateAt(const Eigen::Vector3d& position) {
  mav_model::State s;
  s.setPositionVector(position);
  s.setOrientationVector(Eigen::Quaterniond::Identity());
  s.setLinearVelocityVector(Eigen::Vector3d::Zero());
  return s;
}

/**
 * @brief Load the production ``config_example.yaml`` and trim it in memory so
 *        the test-simulation loop stays deterministic and fast.
 *
 * Reading the real config from ``configs/simulation/`` means a regression in
 * that YAML also fails the adapter tests. The trimming only touches fields
 * that do not influence the semantic correctness of the adapter under test.
 */
inline ExampleConfig loadTestSimConfig() {
  ExampleConfig cfg         = loadExampleConfig(repoPath("configs/simulation/config_example.yaml"));
  cfg.sim_time              = 0.5;
  cfg.silent                = true;
  cfg.benchmark             = false;
  cfg.controller_delay_mode = DelayMode::kFixed;
  cfg.controller_delay_fixed_s = 0.0;
  cfg.generator_delay_mode     = DelayMode::kFixed;
  cfg.generator_delay_fixed_s  = 0.0;
  return cfg;
}

/// Build a small reference vector covering a controller's horizon.
inline std::vector<framework::ReferenceSample> horizonAtPosition(const Eigen::Vector3d& target,
                                                                 int n_samples) {
  framework::ReferenceSample s;
  s.position     = target;
  s.velocity     = Eigen::Vector3d::Zero();
  s.acceleration = Eigen::Vector3d::Zero();
  s.yaw          = 0.0;
  return std::vector<framework::ReferenceSample>(n_samples, s);
}

}  // namespace mpc_examples::testing

#endif  // MPC_EXAMPLES_TESTS_TEST_HELPERS_HPP_
