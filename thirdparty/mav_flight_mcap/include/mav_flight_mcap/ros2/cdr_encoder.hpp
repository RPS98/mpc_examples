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
 * @file cdr_encoder.hpp
 *
 * Declares a facade encode<T>(Buffer&, const T&) over eProsima Fast-CDR,
 * emitting CDR little-endian (DDS_CDR variant), matching the payload that
 * rosbag2 Humble stores in MCAP (message_encoding="cdr").
 *
 * Implementations live in src/cdr_impl.cpp; only the listed types are
 * instantiated explicitly.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_MCAP__ROS2__CDR_ENCODER_HPP_
#define MAV_FLIGHT_MCAP__ROS2__CDR_ENCODER_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mav_flight_mcap/ros2/ros2_types.hpp"

namespace mav_flight_mcap {
namespace ros2 {
namespace cdr {

/** @brief CDR payload buffer (raw bytes, starting with the 4B encapsulation header). */
using Buffer = std::vector<std::byte>;

/**
 * @brief Serialize a ROS 2 message value to CDR (little-endian, DDS-CDR).
 *
 * The output buffer contains a valid CDR payload: 4-byte encapsulation header
 * (0x00 0x01 0x00 0x00), followed by field data with proper alignment. The
 * resulting bytes are byte-compatible with rosbag2 Humble on x86_64.
 *
 * @tparam T One of the supported ROS 2 message POD types in ros2_types.hpp.
 * @param[out] out  Destination buffer (resized to fit payload).
 * @param[in]  value Message instance to encode.
 */
template <class T>
void encode(Buffer& out, const T& value);

}  // namespace cdr
}  // namespace ros2
}  // namespace mav_flight_mcap

#endif  // MAV_FLIGHT_MCAP__ROS2__CDR_ENCODER_HPP_
