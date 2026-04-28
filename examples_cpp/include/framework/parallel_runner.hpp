// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file parallel_runner.hpp
 *
 * Helpers shared by the position/trajectory entry-point binaries to dispatch
 * a set of pre-filtered cases concurrently — one std::thread per case. The
 * helpers cover only the boilerplate that is otherwise duplicated:
 *
 *   - StdoutMutex        : RAII serialisation of std::cout / std::cerr.
 *   - casePrefix         : human-readable identifier shown in front of every
 *                          parallel-mode line, e.g. "[3/8 mpc_position×waypoints] ".
 *   - warnIfOvercommit   : informs when scoped runs > hardware threads.
 *   - warnIfMeasuredDelay: warns about wall-clock contamination of *_us metrics.
 *   - runScopedCasesParallel: launches and joins one std::thread per index.
 *
 * Header-only; depends on the framework runtime that the entry-points already
 * include.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_PARALLEL_RUNNER_HPP_
#define MPC_EXAMPLES_FRAMEWORK_PARALLEL_RUNNER_HPP_

#include <cstddef>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "utils/example_config_utils.hpp"

namespace mpc_examples::framework {

/**
 * @brief Coarse-grained mutex used to serialise stdout/stderr writes when
 * several worker threads emit progress lines concurrently.
 *
 * Use the with(callable) helper to print under the lock; the callable is
 * expected to write to std::cout / std::cerr directly.
 */
class StdoutMutex {
public:
  template <typename Fn>
  void with(Fn&& fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::forward<Fn>(fn)();
    std::cout.flush();
  }

private:
  std::mutex mutex_;
};

/**
 * @brief Human-readable prefix shown in front of each parallel-mode line.
 *
 * Example: "[3/8 mpc_position×waypoints] ".
 */
inline std::string casePrefix(std::size_t index,
                              std::size_t total,
                              const std::string& controller,
                              const std::string& generator) {
  std::ostringstream oss;
  oss << "[" << (index + 1) << "/" << total << " " << controller << "×" << generator << "] ";
  return oss.str();
}

/**
 * @brief Emit a warning to stderr if the requested number of parallel runs
 * exceeds the number of hardware threads reported by the platform.
 *
 * Hardware-threads count of 0 (unknown) is treated as "no overcommit warning".
 */
inline void warnIfOvercommit(std::size_t n_runs) {
  const unsigned cores = std::thread::hardware_concurrency();
  if (cores > 0 && n_runs > cores) {
    std::cerr << "[warning] " << n_runs << " parallel runs vs " << cores
              << " hardware threads — wall-clock metrics may inflate due to "
                 "CPU contention.\n";
  }
}

/**
 * @brief Emit a warning if any of the configured delay modes is `measured`,
 * since CPU contention between concurrent workers will inflate the per-tick
 * compute times that drive the delay buffer.
 */
inline void warnIfMeasuredDelay(const ExampleConfig& cfg) {
  if (cfg.controller_delay_mode == DelayMode::kMeasured ||
      cfg.generator_delay_mode == DelayMode::kMeasured) {
    std::cerr << "[warning] parallel mode + 'measured' delay → CPU contention "
                 "contaminates *_compute_time_us logged in CSVs. Use "
                 "controller_delay_mode/generator_delay_mode: fixed for "
                 "reproducible timing.\n";
  }
}

/**
 * @brief Launch one std::thread per index in [0, n_runs) and join them all.
 *
 * @param n_runs   Number of workers to spawn (typically the number of cases
 *                 enabled in the binary's scope).
 * @param case_fn  Callable invoked as `case_fn(std::size_t index)` from each
 *                 worker. Must be safe to call concurrently — i.e. write only
 *                 to per-index slots and synchronise any shared I/O.
 */
template <typename CaseFn>
void runScopedCasesParallel(std::size_t n_runs, CaseFn case_fn) {
  std::vector<std::thread> workers;
  workers.reserve(n_runs);
  for (std::size_t i = 0; i < n_runs; ++i) {
    workers.emplace_back([i, case_fn]() mutable { case_fn(i); });
  }
  for (auto& t : workers) {
    t.join();
  }
}

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_PARALLEL_RUNNER_HPP_
