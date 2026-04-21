// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file run_example.cpp
 * @brief Unified example: PidGeometricController + JerkLimitedGenerator.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "adapters/controllers/pid_geometric_controller.hpp"
#include "adapters/trajectory_generators/jerk_limited_generator.hpp"
#include "framework/waypoints_simulator.hpp"
#include "mav_simulator/simulator_yaml.hpp"
#include "utils/example_config_utils.hpp"

namespace {

struct Args {
  std::string example_config_path    = "configs/simulation/config_example.yaml";
  std::string simulator_config_path  = "configs/simulation/config_simulator.yaml";
  std::string controller_config_path = "configs/controllers/config_pid.yaml";
  std::string trajectory_config_path = "configs/generators/config_jerk_limited.yaml";
  std::string output_file            = "simulator_logs/pid_jerk_limited_log.csv";
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if ((a == "-c" || a == "--example_config") && i + 1 < argc) {
      args.example_config_path = argv[++i];
    } else if ((a == "-s" || a == "--simulator_config") && i + 1 < argc) {
      args.simulator_config_path = argv[++i];
    } else if ((a == "-k" || a == "--controller_config") && i + 1 < argc) {
      args.controller_config_path = argv[++i];
    } else if ((a == "-t" || a == "--trajectory_config") && i + 1 < argc) {
      args.trajectory_config_path = argv[++i];
    } else if ((a == "-f" || a == "--output_file") && i + 1 < argc) {
      args.output_file = argv[++i];
    } else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0] << "\n"
                << "  -c, --example_config     <yaml> (default: " << args.example_config_path << ")\n"
                << "  -s, --simulator_config   <yaml> (default: " << args.simulator_config_path << ")\n"
                << "  -k, --controller_config  <yaml> (default: " << args.controller_config_path << ")\n"
                << "  -t, --trajectory_config  <yaml> (default: " << args.trajectory_config_path << ")\n"
                << "  -f, --output_file        <csv>  (default: " << args.output_file << ")\n";
      std::exit(0);
    }
  }
  args.output_file = mpc_examples::detail::normalizeOutputPath(args.output_file);
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace mpc_examples;

  const Args args                 = parseArgs(argc, argv);
  const ExampleConfig example_cfg = loadExampleConfig(args.example_config_path);
  const auto sim_params =
      mav_simulator::loadSimulatorParametersFromYaml(args.simulator_config_path);

  const auto controller_cfg =
      adapters::PidGeometricController::loadConfigFromYaml(args.controller_config_path);
  const auto traj_cfg =
      adapters::JerkLimitedGenerator::loadConfigFromYaml(args.trajectory_config_path);

  auto controller = std::make_unique<adapters::PidGeometricController>(controller_cfg);
  auto traj_gen   = std::make_unique<adapters::JerkLimitedGenerator>(traj_cfg);

  framework::WaypointsSimulator simulator(std::move(controller), std::move(traj_gen),
                                          example_cfg, sim_params, args.output_file);
  simulator.run();
  simulator.printBenchmark();
  return 0;
}
