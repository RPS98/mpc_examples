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
 * @file time_source.cpp
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "mav_flight_mcap/time_source.hpp"

#include <cmath>
#include <cstdint>

namespace mav_flight_mcap {

namespace {
constexpr double kNanosPerSecond = 1.0e9;
}  // namespace

TimeSource::TimeSource(TimeMode mode) : mode_(mode) {}

uint64_t TimeSource::resolve(double t_user) {
  double t_effective = t_user;
  if (mode_ == TimeMode::SIMULATION) {
    if (!has_origin_) {
      t_origin_s_  = t_user;
      has_origin_  = true;
    }
    t_effective = t_user - t_origin_s_;
  }
  if (!(t_effective >= 0.0)) {
    // Clamp to 0 to avoid wraparound in uint64_t; tolerates small fp noise.
    t_effective = 0.0;
  }
  const double ns_d = std::round(t_effective * kNanosPerSecond);
  return static_cast<uint64_t>(ns_d);
}

void TimeSource::reset() {
  has_origin_ = false;
  t_origin_s_ = 0.0;
}

}  // namespace mav_flight_mcap
