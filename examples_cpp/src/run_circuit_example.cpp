// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file run_circuit_example.cpp
 *
 * Closed-loop circuit-mission entry point. Iterates ``sim_config.runs[]``
 * from ``configs/simulation/config_circuit_example.yaml`` and executes
 * only the entries whose generator is ``circuit``. Two cases pre-wired
 * by default:
 *
 *   - mpcc           + circuit  (MPCC controller, self-contained spline)
 *   - mpc_trajectory + circuit  (trajectory MPC over generator samples)
 *
 * Links against ``mpc_examples_factories_circuit``, which bundles the
 * MPCC adapter, the trajectory MPC adapter (reusable in this scope) and
 * the closed-loop circuit generator. MPC-Position is NOT included to
 * avoid an ODR collision on ``acados_mpc::MPC``.
 *
 * Extra CLI flags consumed here and exported as environment variables
 * before the dispatch:
 *
 *   --mission-yaml PATH   →  CIRCUIT_MISSION_YAML
 *   --gates-yaml   PATH   →  CIRCUIT_GATES_YAML
 *
 * Both :class:`CircuitGenerator` and :class:`MpccController` honour
 * those env vars (env > config YAML > package demo defaults), so a
 * higher-level launcher can inject its own mission without editing
 * any YAML inside this repo.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/factories.hpp"
#include "framework/parallel_runner.hpp"
#include "framework/stdout_progress.hpp"
#include "framework/unified_mcap_logger.hpp"
#include "framework/waypoints_simulator.hpp"
#include "mav_simulator/simulator_yaml.hpp"
#include "utils/example_config_utils.hpp"

namespace {

struct Args {
  std::string example_config_path   = "configs/simulation/config_circuit_example.yaml";
  std::string simulator_config_path = "configs/simulation/config_simulator.yaml";
  std::string output_dir;
  std::string only_controller;
  std::string only_generator;
  std::string mission_yaml;
  std::string gates_yaml;
  bool parallel_override = false;
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if ((a == "-c" || a == "--example_config") && i + 1 < argc) {
      args.example_config_path = argv[++i];
    } else if ((a == "-s" || a == "--simulator_config") && i + 1 < argc) {
      args.simulator_config_path = argv[++i];
    } else if ((a == "--output-dir" || a == "--output_dir") && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else if (a == "--only-controller" && i + 1 < argc) {
      args.only_controller = argv[++i];
    } else if (a == "--only-generator" && i + 1 < argc) {
      args.only_generator = argv[++i];
    } else if ((a == "--mission-yaml" || a == "--mission_yaml") && i + 1 < argc) {
      args.mission_yaml = argv[++i];
    } else if ((a == "--gates-yaml" || a == "--gates_yaml") && i + 1 < argc) {
      args.gates_yaml = argv[++i];
    } else if (a == "--parallel") {
      args.parallel_override = true;
    } else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0] << "\n"
                << "  -c, --example_config    <yaml>  (default: " << args.example_config_path << ")\n"
                << "  -s, --simulator_config  <yaml>  (default: " << args.simulator_config_path << ")\n"
                << "      --output-dir        <dir>   (default: simulator_logs/<run_id>)\n"
                << "      --only-controller   <name>  run only the entry matching this controller\n"
                << "      --only-generator    <name>  run only the entry matching this generator\n"
                << "      --mission-yaml      <yaml>  override mission via CIRCUIT_MISSION_YAML env\n"
                << "      --gates-yaml        <yaml>  override gates via CIRCUIT_GATES_YAML env\n"
                << "      --parallel                  override sim_config.parallel = true\n";
      std::exit(0);
    }
  }
  return args;
}

std::string makeRunId() {
  using namespace std::chrono;
  const auto now = system_clock::to_time_t(system_clock::now());
  std::tm tm_local{};
  localtime_r(&now, &tm_local);
  std::ostringstream oss;
  oss << std::put_time(&tm_local, "%Y%m%d_%H%M%S");
  return oss.str();
}

struct CaseResult {
  std::string controller;
  std::string generator;
  bool succeeded = false;
  std::string csv_path;
  mpc_examples::framework::BenchmarkStats stats{};
  std::string error;
};

bool isCircuitRun(const mpc_examples::RunSpec& spec) {
  return spec.generator == mpc_examples::framework::GeneratorKeys::kCircuit;
}

void runCase(const mpc_examples::RunSpec& spec,
             const mpc_examples::ExampleConfig& example_cfg,
             const mav_simulator::SimulatorParameters& sim_params,
             const std::filesystem::path& cpp_dir,
             const std::string& run_id,
             std::size_t index,
             std::size_t total,
             CaseResult& out,
             bool print_banner = true) {
  using namespace mpc_examples;

  out.controller = spec.controller;
  out.generator  = spec.generator;

  const std::string ext      = (example_cfg.output_format == "csv") ? ".csv" : ".mcap";
  const std::string csv_name = spec.controller + "_" + spec.generator + ext;
  out.csv_path               = (cpp_dir / csv_name).string();

  if (print_banner) {
    framework::printCaseBanner(index, total, spec.controller, spec.generator, run_id);
  }
  try {
    // is_trajectory_scope = true: PID falls back to the parallel
    // trajectory variant if ever wired in; MPCC and mpc_trajectory
    // ignore the flag entirely.
    auto controller = framework::makeController(spec.controller, spec.controller_config,
                                                /*is_trajectory_scope=*/true);
    auto generator  = framework::makeGenerator(spec.generator, spec.generator_config);

    framework::RunMetadata meta;
    meta.controller_name = spec.controller;
    meta.generator_name  = spec.generator;
    meta.run_id          = run_id;
    meta.language        = "cpp";

    framework::WaypointsSimulator sim(std::move(controller), std::move(generator), example_cfg,
                                      sim_params, out.csv_path, meta);
    sim.run();
    out.stats     = sim.benchmarkStats();
    out.succeeded = true;
  } catch (const std::exception& ex) {
    out.error = ex.what();
    if (print_banner) {
      std::cerr << "\n  [FAILED] " << ex.what() << "\n";
    }
  }
}

void printSummaryTable(const std::vector<CaseResult>& results) {
  mpc_examples::framework::printRule();
  std::cout << "Final summary (" << results.size() << " circuit cases)\n";
  mpc_examples::framework::printRule('-');
  std::cout << std::left << std::setw(20) << "controller" << std::setw(18) << "generator"
            << std::right << std::setw(10) << "rmse[m]" << std::setw(14) << "ctrl_us"
            << std::setw(14) << "gen_us" << std::setw(10) << "real[s]"
            << "\n";
  mpc_examples::framework::printRule('-');
  for (const auto& r : results) {
    std::cout << std::left << std::setw(20) << r.controller << std::setw(18) << r.generator;
    if (!r.succeeded) {
      std::cout << "  FAILED: " << r.error << "\n";
      continue;
    }
    std::cout << std::right << std::fixed << std::setprecision(3) << std::setw(10)
              << r.stats.tracking_rmse_m << std::setprecision(0) << std::setw(14)
              << r.stats.controller_mean_us << std::setw(14)
              << (r.stats.generator_update_mean_us + r.stats.generator_eval_mean_us)
              << std::setprecision(2) << std::setw(10) << r.stats.real_time_s << "\n";
  }
  mpc_examples::framework::printRule();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace mpc_examples;

  const Args args = parseArgs(argc, argv);

  // Export mission/gates overrides as env vars before any adapter
  // initialises (CircuitGenerator / MpccController read them inside
  // their own initialize()).
  if (!args.mission_yaml.empty()) {
    if (!std::filesystem::exists(args.mission_yaml)) {
      std::cerr << "--mission-yaml: file not found at '" << args.mission_yaml << "'\n";
      return 1;
    }
    setenv("CIRCUIT_MISSION_YAML",
           std::filesystem::absolute(args.mission_yaml).c_str(), /*overwrite=*/1);
  }
  if (!args.gates_yaml.empty()) {
    if (!std::filesystem::exists(args.gates_yaml)) {
      std::cerr << "--gates-yaml: file not found at '" << args.gates_yaml << "'\n";
      return 1;
    }
    setenv("CIRCUIT_GATES_YAML",
           std::filesystem::absolute(args.gates_yaml).c_str(), /*overwrite=*/1);
  }

  const ExampleConfig example_cfg = loadExampleConfig(args.example_config_path);
  const auto sim_params =
      mav_simulator::loadSimulatorParametersFromYaml(args.simulator_config_path);

  if (example_cfg.runs.empty()) {
    std::cerr << "No runs defined in 'sim_config.runs' — nothing to do.\n";
    return 1;
  }

  const std::string run_id            = makeRunId();
  std::filesystem::path out_root      = args.output_dir.empty()
                                            ? (std::filesystem::path("simulator_logs") / run_id)
                                            : std::filesystem::path(args.output_dir);
  const std::filesystem::path cpp_dir = out_root / "cpp";
  std::error_code ec;
  std::filesystem::create_directories(cpp_dir, ec);
  if (ec) {
    std::cerr << "Could not create output directory '" << cpp_dir << "': " << ec.message() << "\n";
    return 1;
  }

  const bool parallel = args.parallel_override || example_cfg.parallel;

  std::cout << "circuit_examples · run_id=" << run_id << " · output_dir=" << out_root.string()
            << (parallel ? " · mode=parallel" : "") << "\n";

  const auto caseSelected = [&](const RunSpec& spec) {
    if (!args.only_controller.empty() && spec.controller != args.only_controller) return false;
    if (!args.only_generator.empty() && spec.generator != args.only_generator) return false;
    return true;
  };
  const bool explicit_combo = !args.only_controller.empty() && !args.only_generator.empty();

  std::vector<RunSpec> scoped;
  scoped.reserve(example_cfg.runs.size());
  bool explicit_combo_found = false;
  for (const auto& spec : example_cfg.runs) {
    const bool matches_filters = caseSelected(spec);
    if (explicit_combo && matches_filters) explicit_combo_found = true;
    if (!spec.enabled && !(explicit_combo && matches_filters)) continue;
    if (!isCircuitRun(spec)) {
      std::cout << "[skipped] " << spec.controller << " + " << spec.generator
                << " (not in circuit_examples scope)\n";
      continue;
    }
    if (!matches_filters) {
      std::cout << "[skipped] " << spec.controller << " + " << spec.generator
                << " (filtered out by --only-*)\n";
      continue;
    }
    scoped.push_back(spec);
  }

  if (scoped.empty()) {
    if (explicit_combo && !explicit_combo_found) {
      std::cerr << "No entry matching --only-controller='" << args.only_controller
                << "' --only-generator='" << args.only_generator
                << "' found in sim_config.runs[]. Add it to "
                   "configs/simulation/config_circuit_example.yaml.\n";
    } else {
      std::cerr << "No enabled runs match circuit_examples' scope "
                   "(generator == 'circuit'";
      if (!args.only_controller.empty() || !args.only_generator.empty()) {
        std::cerr << ", --only-controller='" << args.only_controller << "', --only-generator='"
                  << args.only_generator << "'";
      }
      std::cerr << "). Nothing to do.\n";
    }
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(0);
  }

  std::vector<CaseResult> results(scoped.size());

  if (!parallel) {
    for (std::size_t i = 0; i < scoped.size(); ++i) {
      runCase(scoped[i], example_cfg, sim_params, cpp_dir, run_id, i, scoped.size(), results[i]);
    }
  } else {
    framework::warnIfOvercommit(scoped.size());
    framework::warnIfMeasuredDelay(example_cfg);

    ExampleConfig parallel_cfg = example_cfg;
    parallel_cfg.silent        = true;

    framework::StdoutMutex stdout_mtx;
    framework::runScopedCasesParallel(scoped.size(), [&](std::size_t i) {
      const auto& spec  = scoped[i];
      const auto prefix = framework::casePrefix(i, scoped.size(), spec.controller, spec.generator);
      stdout_mtx.with([&] { std::cout << prefix << "start (run_id=" << run_id << ")\n"; });

      runCase(spec, parallel_cfg, sim_params, cpp_dir, run_id, i, scoped.size(), results[i],
              /*print_banner=*/false);

      stdout_mtx.with([&] {
        if (results[i].succeeded) {
          std::cout << prefix << std::fixed << "done in " << std::setprecision(2)
                    << results[i].stats.real_time_s << "s"
                    << " · rmse=" << std::setprecision(3) << results[i].stats.tracking_rmse_m << "m"
                    << " · ctrl_mean=" << std::setprecision(0)
                    << results[i].stats.controller_mean_us << "µs"
                    << " · gen_mean="
                    << (results[i].stats.generator_update_mean_us +
                        results[i].stats.generator_eval_mean_us)
                    << "µs\n";
        } else {
          std::cerr << prefix << "[FAILED] " << results[i].error << "\n";
        }
      });
    });
  }

  printSummaryTable(results);
  std::cout << "Done · run_id=" << run_id << " · output_dir=" << out_root.string() << "\n";

  // Bypass global destructors: acados + spline + simulator shutdown can
  // race during exit-time destruction (same issue as trajectory_examples).
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(0);
}
