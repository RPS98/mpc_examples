// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file delay_buffer.hpp
 *
 * Simple time-indexed buffer used to model compute latency between the
 * controller/generator and the simulator.
 *
 * Single-producer, single-consumer, chronologically ordered pushes.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MPC_EXAMPLES_FRAMEWORK_DELAY_BUFFER_HPP_
#define MPC_EXAMPLES_FRAMEWORK_DELAY_BUFFER_HPP_

#include <cstddef>
#include <deque>
#include <optional>
#include <utility>

namespace mpc_examples::framework {

/**
 * @brief Stores time-stamped samples and returns the latest sample whose
 *        "available-at" time does not exceed the query time.
 *
 * A sample pushed with @p available_at = T stays invisible to callers until
 * the simulator clock reaches T. The caller can then query
 * `latestAvailable(t)` (which returns the latest sample with
 * available_at <= t) as many times as needed while the physics loop advances
 * between outer-control steps; this lets the simulator keep running model
 * integration even when the controller/generator has not produced a fresh
 * output yet.
 *
 * Chronological push ordering is assumed (@p available_at must be
 * non-decreasing); pushing out of order throws @c std::invalid_argument.
 *
 * Thread-safety: not thread-safe; intended for single-threaded use.
 *
 * @tparam T Payload type (usually a ControlCommand, ReferenceSample or a
 *           struct wrapping both plus metadata).
 */
template <typename T>
class DelayBuffer {
public:
  DelayBuffer() = default;

  /**
   * @brief Push a new payload that becomes visible at @p available_at.
   *
   * @param payload      The stored value (copied/moved into the buffer).
   * @param available_at Simulator time at which the payload becomes visible [s].
   */
  void push(T payload, double available_at) {
    if (!entries_.empty() && available_at + 1e-12 < entries_.back().available_at) {
      // Out-of-order push would violate the chronological invariant.
      return;
    }
    entries_.push_back(Entry{std::move(payload), available_at});
  }

  /**
   * @brief Return the latest payload whose availability <= @p t, if any.
   *
   * Discards every entry older than the returned one so the buffer does not
   * grow unbounded.
   *
   * @param t Query time [s].
   * @return The most recent payload not in the future relative to @p t, or
   *         std::nullopt if no such payload has been pushed yet.
   */
  std::optional<T> latestAvailable(double t) {
    std::optional<T> latest;
    while (!entries_.empty() && entries_.front().available_at <= t + 1e-12) {
      latest = entries_.front().payload;
      entries_.pop_front();
    }
    return latest;
  }

  /** @return Number of pending (future) entries currently queued. */
  std::size_t size() const { return entries_.size(); }

  /** @brief Drop all pending entries. */
  void clear() { entries_.clear(); }

private:
  struct Entry {
    T payload;
    double available_at;
  };
  std::deque<Entry> entries_;
};

}  // namespace mpc_examples::framework

#endif  // MPC_EXAMPLES_FRAMEWORK_DELAY_BUFFER_HPP_
