// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file stdout_progress.hpp
 *
 * Lightweight stdout progress reporter shared by the unified runner so every
 * example prints structured, uniform output (banner, progress ticks, summary).
 *
 * Header-only; zero external dependencies beyond STL.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_STDOUT_PROGRESS_HPP_
#define MPC_EXAMPLES_FRAMEWORK_STDOUT_PROGRESS_HPP_

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace mpc_examples::framework {

/** @brief Print a separator line of dashes. */
inline void printRule(char ch = '=', int width = 72) {
  std::cout << '\n' << std::string(static_cast<std::size_t>(width), ch) << '\n';
}

/**
 * @brief Print the banner for a single controller × generator case.
 *
 * Example:
 *   [ 3/12] mpc_position × gcopter  (run_id=20260421_153012)
 */
inline void printCaseBanner(std::size_t index,
                            std::size_t total,
                            const std::string& controller,
                            const std::string& generator,
                            const std::string& run_id) {
  std::cout << "\n[" << std::setw(2) << (index + 1) << "/" << total << "] " << controller << " × "
            << generator << "  (run_id=" << run_id << ")" << std::endl;
}

/**
 * @brief One-line status update during a run.
 *
 * Overwrites the current line using `\r` (no newline). Caller should emit a
 * newline after the run finishes.
 */
inline void printStatus(double t,
                        double t_total,
                        int waypoint_index,
                        std::size_t n_waypoints,
                        double last_err_m,
                        double ctrl_time_us) {
  const double progress   = (t_total > 0.0) ? std::clamp(t / t_total, 0.0, 1.0) : 0.0;
  constexpr int kBarWidth = 30;
  const int pos           = static_cast<int>(kBarWidth * progress);

  std::ostringstream bar;
  bar << '[';
  for (int i = 0; i < kBarWidth; ++i) {
    bar << (i < pos ? '=' : (i == pos ? '>' : ' '));
  }
  bar << ']';

  std::cout << "\r  " << bar.str() << "  t=" << std::fixed << std::setprecision(2) << t << "/"
            << t_total << "s  wp=" << (waypoint_index + 1) << "/" << n_waypoints
            << "  err=" << std::setprecision(3) << last_err_m << "m"
            << "  ctrl=" << std::setprecision(0) << ctrl_time_us << "µs      " << std::flush;
}

/** @brief Summary line emitted after each run finishes. */
inline void printCaseSummary(double wall_time_s,
                             double rmse_m,
                             double ctrl_mean_us,
                             double gen_mean_us) {
  std::cout << "\n  done in " << std::fixed << std::setprecision(2) << wall_time_s
            << "s wall · rmse=" << std::setprecision(3) << rmse_m << "m"
            << " · ctrl_mean=" << std::setprecision(0) << ctrl_mean_us << "µs"
            << " · gen_mean=" << gen_mean_us << "µs\n";
}

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_STDOUT_PROGRESS_HPP_
