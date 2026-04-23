// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file time_source.hpp
 *
 * Time resolution policy for the MCAP logger.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_MCAP__TIME_SOURCE_HPP_
#define MAV_FLIGHT_MCAP__TIME_SOURCE_HPP_

#include <cstdint>

#include "mav_flight_mcap/config.hpp"

namespace mav_flight_mcap {

/**
 * @brief Converts user-supplied timestamps into nanosecond epochs.
 *
 * In SIMULATION mode the first resolve() call establishes t0; subsequent
 * resolves return (t - t0) in nanoseconds.
 *
 * In GLOBAL mode resolve() returns (t * 1e9) as-is (user-supplied time is
 * interpreted as an absolute POSIX epoch expressed in seconds).
 */
class TimeSource {
 public:
  /** @brief Construct a TimeSource with the given mode. */
  explicit TimeSource(TimeMode mode);

  /**
   * @brief Resolve a user-supplied timestamp (seconds) to nanoseconds.
   *
   * @param t_user User timestamp (seconds).
   * @return Timestamp in nanoseconds according to the active mode.
   */
  uint64_t resolve(double t_user);

  /** @brief Reset the SIMULATION origin; no-op under GLOBAL. */
  void reset();

  /** @brief Current mode. */
  inline TimeMode mode() const { return mode_; }

 private:
  TimeMode mode_;
  bool     has_origin_ = false;
  double   t_origin_s_ = 0.0;
};

}  // namespace mav_flight_mcap

#endif  // MAV_FLIGHT_MCAP__TIME_SOURCE_HPP_
