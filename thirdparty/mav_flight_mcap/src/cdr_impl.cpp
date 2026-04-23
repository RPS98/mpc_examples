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
 * @file cdr_impl.cpp
 *
 * Explicit instantiations of encode<T>() using eProsima Fast-CDR 1.0.29.
 * Emits CDR little-endian (DDS-CDR encapsulation), byte-compatible with
 * rosbag2 Humble on x86_64.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "mav_flight_mcap/ros2/cdr_encoder.hpp"

#include <cstring>
#include <stdexcept>
#include <type_traits>

#include <fastcdr/Cdr.h>
#include <fastcdr/FastBuffer.h>
#include <fastcdr/exceptions/NotEnoughMemoryException.h>

namespace mav_flight_mcap {
namespace ros2 {
namespace cdr {

namespace {

// --- Tunables -----------------------------------------------------------------

constexpr size_t kInitialBufferBytes = 4096;
constexpr size_t kMaxBufferBytes     = 32 * 1024 * 1024;  // 32 MiB hard cap.

// --- Low-level Fast-CDR field serialization ----------------------------------

void serializeTime(eprosima::fastcdr::Cdr& ser, const Time& t) {
  ser << t.sec << t.nanosec;
}

void serializeHeader(eprosima::fastcdr::Cdr& ser, const Header& h) {
  serializeTime(ser, h.stamp);
  ser << h.frame_id;
}

void serializeVector3(eprosima::fastcdr::Cdr& ser, const Vector3& v) {
  ser << v.x << v.y << v.z;
}

void serializePoint(eprosima::fastcdr::Cdr& ser, const Point& p) {
  ser << p.x << p.y << p.z;
}

void serializeQuaternion(eprosima::fastcdr::Cdr& ser, const Quaternion& q) {
  ser << q.x << q.y << q.z << q.w;
}

void serializePose(eprosima::fastcdr::Cdr& ser, const Pose& p) {
  serializePoint(ser, p.position);
  serializeQuaternion(ser, p.orientation);
}

void serializeTwist(eprosima::fastcdr::Cdr& ser, const Twist& t) {
  serializeVector3(ser, t.linear);
  serializeVector3(ser, t.angular);
}

void serializeCovariance36(eprosima::fastcdr::Cdr& ser,
                           const std::array<double, 36>& cov) {
  // Fixed-size array: no length prefix, just the 36 doubles.
  for (double v : cov) {
    ser << v;
  }
}

void serializePoseWithCovariance(eprosima::fastcdr::Cdr& ser,
                                 const PoseWithCovariance& p) {
  serializePose(ser, p.pose);
  serializeCovariance36(ser, p.covariance);
}

void serializeTwistWithCovariance(eprosima::fastcdr::Cdr& ser,
                                  const TwistWithCovariance& t) {
  serializeTwist(ser, t.twist);
  serializeCovariance36(ser, t.covariance);
}

void serializeTrajectoryPointMsg(eprosima::fastcdr::Cdr& ser,
                                 const TrajectoryPointMsg& p) {
  ser << p.id;
  serializeVector3(ser, p.position);
  serializeVector3(ser, p.twist);
  serializeVector3(ser, p.acceleration);
  ser << p.yaw_angle;
}

void serializeMultiArrayDimension(eprosima::fastcdr::Cdr& ser,
                                  const MultiArrayDimension& d) {
  ser << d.label << d.size << d.stride;
}

void serializeMultiArrayLayout(eprosima::fastcdr::Cdr& ser,
                               const MultiArrayLayout& l) {
  // sequence<MultiArrayDimension>: length, then elements.
  const uint32_t n = static_cast<uint32_t>(l.dim.size());
  ser << n;
  for (const auto& d : l.dim) {
    serializeMultiArrayDimension(ser, d);
  }
  ser << l.data_offset;
}

// --- Message roots ------------------------------------------------------------

void serializeRoot(eprosima::fastcdr::Cdr& ser, const PoseStamped& m) {
  serializeHeader(ser, m.header);
  serializePose(ser, m.pose);
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const TwistStamped& m) {
  serializeHeader(ser, m.header);
  serializeTwist(ser, m.twist);
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const Vector3& m) {
  serializeVector3(ser, m);
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const Odometry& m) {
  serializeHeader(ser, m.header);
  ser << m.child_frame_id;
  serializePoseWithCovariance(ser, m.pose);
  serializeTwistWithCovariance(ser, m.twist);
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const Thrust& m) {
  serializeHeader(ser, m.header);
  ser << m.thrust << m.thrust_normalized;
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const TrajectorySetpoints& m) {
  serializeHeader(ser, m.header);
  const uint32_t n = static_cast<uint32_t>(m.setpoints.size());
  ser << n;
  for (const auto& p : m.setpoints) {
    serializeTrajectoryPointMsg(ser, p);
  }
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const Clock& m) {
  serializeTime(ser, m.clock);
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const Int32& m) {
  ser << m.data;
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const String& m) {
  ser << m.data;
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const Float64& m) {
  ser << m.data;
}

void serializeRoot(eprosima::fastcdr::Cdr& ser, const Float64MultiArray& m) {
  serializeMultiArrayLayout(ser, m.layout);
  const uint32_t n = static_cast<uint32_t>(m.data.size());
  ser << n;
  for (double v : m.data) {
    ser << v;
  }
}

// --- Grow-on-overflow driver --------------------------------------------------

template <class T>
void encodeImpl(Buffer& out, const T& value) {
  size_t cap = kInitialBufferBytes;
  while (true) {
    eprosima::fastcdr::FastBuffer fb;
    if (!fb.reserve(cap)) {
      throw std::runtime_error("mav_flight_mcap: FastBuffer reserve failed");
    }
    eprosima::fastcdr::Cdr ser(fb,
                               eprosima::fastcdr::Cdr::LITTLE_ENDIANNESS,
                               eprosima::fastcdr::Cdr::DDS_CDR);
    try {
      ser.serialize_encapsulation();
      serializeRoot(ser, value);
      const size_t len = ser.getSerializedDataLength();
      const std::byte* src = reinterpret_cast<const std::byte*>(fb.getBuffer());
      out.assign(src, src + len);
      return;
    } catch (const eprosima::fastcdr::exception::NotEnoughMemoryException&) {
      if (cap >= kMaxBufferBytes) {
        throw;
      }
      cap = std::min(cap * 2, kMaxBufferBytes);
    }
  }
}

}  // namespace

// --- Explicit instantiations of encode<T> ------------------------------------

template <>
void encode<PoseStamped>(Buffer& out, const PoseStamped& v) {
  encodeImpl(out, v);
}
template <>
void encode<TwistStamped>(Buffer& out, const TwistStamped& v) {
  encodeImpl(out, v);
}
template <>
void encode<Vector3>(Buffer& out, const Vector3& v) {
  encodeImpl(out, v);
}
template <>
void encode<Odometry>(Buffer& out, const Odometry& v) {
  encodeImpl(out, v);
}
template <>
void encode<Thrust>(Buffer& out, const Thrust& v) {
  encodeImpl(out, v);
}
template <>
void encode<TrajectorySetpoints>(Buffer& out, const TrajectorySetpoints& v) {
  encodeImpl(out, v);
}
template <>
void encode<Clock>(Buffer& out, const Clock& v) {
  encodeImpl(out, v);
}
template <>
void encode<Int32>(Buffer& out, const Int32& v) {
  encodeImpl(out, v);
}
template <>
void encode<String>(Buffer& out, const String& v) {
  encodeImpl(out, v);
}
template <>
void encode<Float64>(Buffer& out, const Float64& v) {
  encodeImpl(out, v);
}
template <>
void encode<Float64MultiArray>(Buffer& out, const Float64MultiArray& v) {
  encodeImpl(out, v);
}

}  // namespace cdr
}  // namespace ros2
}  // namespace mav_flight_mcap
