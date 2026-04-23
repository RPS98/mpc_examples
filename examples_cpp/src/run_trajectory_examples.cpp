// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file run_trajectory_examples.cpp
 *
 * Trajectory-examples entry point. Iterates ``sim_config.runs[]`` from
 * ``configs/simulation/config_example.yaml`` and executes only the entries
 * whose generator produces a full trajectory (``gcopter`` or
 * ``jerk_limited``) — i.e. the four showcase cases:
 *
 *   - pid            + gcopter
 *   - pid            + jerk_limited
 *   - mpc_trajectory + gcopter
 *   - mpc_trajectory + jerk_limited
 *
 * Links against ``mpc_examples_factories_trajectory`` which bundles the
 * PID and MPC-Trajectory adapters (plus the four generators, for
 * coherence). MPC-Position is excluded to avoid an ODR collision on
 * ``acados_mpc::MPC``.
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
#include <unordered_set>
#include <vector>

#include "framework/factories.hpp"
#include "framework/parallel_runner.hpp"
#include "framework/stdout_progress.hpp"
// TODO: remove once MCAP pipeline validated.
// #include "framework/unified_csv_logger.hpp"
#include "framework/unified_mcap_logger.hpp"
#include "framework/waypoints_simulator.hpp"
#include "mav_simulator/simulator_yaml.hpp"
#include "utils/example_config_utils.hpp"

namespace {

struct Args {
  std::string example_config_path   = "configs/simulation/config_example.yaml";
  std::string simulator_config_path = "configs/simulation/config_simulator.yaml";
  std::string output_dir;        //!< Empty → simulator_logs/<run_id>.
  std::string only_controller;   //!< Empty → no filter.
  std::string only_generator;    //!< Empty → no filter.
  bool parallel_override = false; //!< CLI-forced parallel run; YAML otherwise governs.
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
    } else if (a == "--parallel") {
      args.parallel_override = true;
    } else if (a == "-h" || a == "--help") {
      std::cout << "Usage: " << argv[0] << "\n"
                << "  -c, --example_config    <yaml>  (default: " << args.example_config_path << ")\n"
                << "  -s, --simulator_config  <yaml>  (default: " << args.simulator_config_path << ")\n"
                << "      --output-dir        <dir>   (default: simulator_logs/<run_id>)\n"
                << "      --only-controller   <name>  run only the entry matching this controller\n"
                << "      --only-generator    <name>  run only the entry matching this generator\n"
                << "      --parallel                  override sim_config.parallel = true\n"
                << "                                  (one std::thread per enabled case)\n";
      std::exit(0);
    }
  }
  return args;
}

std::string makeRunId() {
  using namespace std::chrono;
  const auto now = system_clock::to_time_t(system_clock::now());
  std::tm tm_local{};
#if defined(_WIN32)
  localtime_s(&tm_local, &now);
#else
  localtime_r(&now, &tm_local);
#endif
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

// Only full-trajectory generators are valid for trajectory_examples.
bool isTrajectoryRun(const mpc_examples::RunSpec& spec) {
  return spec.generator == mpc_examples::framework::GeneratorKeys::kGcopter ||
         spec.generator == mpc_examples::framework::GeneratorKeys::kJerkLimited;
}

// Executes one (controller, generator) case. Set `print_banner` to false when
// the caller (e.g. the parallel dispatcher) handles stdout decoration itself.
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

  // TODO: remove once MCAP pipeline validated.
  // const std::string csv_name = spec.controller + "_" + spec.generator + ".csv";
  const std::string csv_name = spec.controller + "_" + spec.generator + ".mcap";
  out.csv_path               = (cpp_dir / csv_name).string();

  if (print_banner) {
    framework::printCaseBanner(index, total, spec.controller, spec.generator, run_id);
  }
  try {
    auto controller = framework::makeController(spec.controller, spec.controller_config);
    auto generator  = framework::makeGenerator(spec.generator, spec.generator_config);

    framework::RunMetadata meta;
    meta.controller_name = spec.controller;
    meta.generator_name  = spec.generator;
    meta.run_id          = run_id;
    meta.language        = "cpp";

    framework::WaypointsSimulator sim(std::move(controller), std::move(generator),
                                      example_cfg, sim_params, out.csv_path, meta);
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
  std::cout << "Final summary (" << results.size() << " trajectory cases)\n";
  mpc_examples::framework::printRule('-');
  std::cout << std::left
            << std::setw(20) << "controller"
            << std::setw(18) << "generator"
            << std::right
            << std::setw(10) << "rmse[m]"
            << std::setw(14) << "ctrl_us"
            << std::setw(14) << "gen_us"
            << std::setw(10) << "real[s]"
            << "\n";
  mpc_examples::framework::printRule('-');
  for (const auto& r : results) {
    std::cout << std::left << std::setw(20) << r.controller << std::setw(18) << r.generator;
    if (!r.succeeded) {
      std::cout << "  FAILED: " << r.error << "\n";
      continue;
    }
    std::cout << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << r.stats.tracking_rmse_m
              << std::setprecision(0)
              << std::setw(14) << r.stats.controller_mean_us
              << std::setw(14)
              << (r.stats.generator_update_mean_us + r.stats.generator_eval_mean_us)
              << std::setprecision(2)
              << std::setw(10) << r.stats.real_time_s << "\n";
  }
  mpc_examples::framework::printRule();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace mpc_examples;

  const Args args                 = parseArgs(argc, argv);
  const ExampleConfig example_cfg = loadExampleConfig(args.example_config_path);
  const auto sim_params =
      mav_simulator::loadSimulatorParametersFromYaml(args.simulator_config_path);

  if (example_cfg.runs.empty()) {
    std::cerr << "No runs defined in 'sim_config.runs' — nothing to do.\n";
    return 1;
  }

  const std::string run_id = makeRunId();
  std::filesystem::path out_root =
      args.output_dir.empty()
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

  std::cout << "trajectory_examples · run_id=" << run_id
            << " · output_dir=" << out_root.string()
            << (parallel ? " · mode=parallel" : "") << "\n";

  const auto caseSelected = [&](const RunSpec& spec) {
    if (!args.only_controller.empty() && spec.controller != args.only_controller) return false;
    if (!args.only_generator.empty()  && spec.generator  != args.only_generator)  return false;
    return true;
  };

  // First pass: announce skips and collect the in-scope specs (preserving the
  // YAML order so the [i/N] indexing matches between sequential and parallel).
  std::vector<RunSpec> scoped;
  scoped.reserve(example_cfg.runs.size());
  for (const auto& spec : example_cfg.runs) {
    if (!spec.enabled) continue;
    if (!isTrajectoryRun(spec)) {
      std::cout << "[skipped] " << spec.controller << " + " << spec.generator
                << " (not in trajectory_examples scope)\n";
      continue;
    }
    if (!caseSelected(spec)) {
      std::cout << "[skipped] " << spec.controller << " + " << spec.generator
                << " (filtered out by --only-*)\n";
      continue;
    }
    scoped.push_back(spec);
  }

  if (scoped.empty()) {
    std::cerr << "No enabled runs match trajectory_examples' scope "
                 "(generator ∈ {gcopter, jerk_limited}";
    if (!args.only_controller.empty() || !args.only_generator.empty()) {
      std::cerr << ", --only-controller='" << args.only_controller
                << "', --only-generator='" << args.only_generator << "'";
    }
    std::cerr << "). Nothing to do.\n";
    return 0;
  }

  std::vector<CaseResult> results(scoped.size());

  if (!parallel) {
    for (std::size_t i = 0; i < scoped.size(); ++i) {
      runCase(scoped[i], example_cfg, sim_params, cpp_dir, run_id, i, scoped.size(),
              results[i]);
    }
  } else {
    framework::warnIfOvercommit(scoped.size());
    framework::warnIfMeasuredDelay(example_cfg);

    // Each worker reads from `parallel_cfg`, which silences the in-loop
    // progress bar that would otherwise interleave \r writes between threads.
    ExampleConfig parallel_cfg = example_cfg;
    parallel_cfg.silent = true;

    framework::StdoutMutex stdout_mtx;
    framework::runScopedCasesParallel(scoped.size(), [&](std::size_t i) {
      const auto& spec  = scoped[i];
      const auto prefix = framework::casePrefix(i, scoped.size(), spec.controller, spec.generator);
      stdout_mtx.with([&] { std::cout << prefix << "start (run_id=" << run_id << ")\n"; });

      runCase(spec, parallel_cfg, sim_params, cpp_dir, run_id, i, scoped.size(),
              results[i], /*print_banner=*/false);

      stdout_mtx.with([&] {
        if (results[i].succeeded) {
          std::cout << prefix << std::fixed
                    << "done in " << std::setprecision(2) << results[i].stats.real_time_s << "s"
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
  return 0;
}
