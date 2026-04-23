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
 * @file test_cdr_roundtrip.cpp
 *
 * Encode with the logger's encode<T>(), decode back with Fast-CDR, and
 * compare field-by-field. Covers the tricky alignment cases: sequences,
 * strings, and fixed-size arrays (36-double covariance).
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <fastcdr/Cdr.h>
#include <fastcdr/FastBuffer.h>

#include "mav_flight_mcap/ros2/cdr_encoder.hpp"
#include "mav_flight_mcap/ros2/ros2_types.hpp"

namespace mav_flight_mcap::ros2 {

namespace {

// Build a FastCdr reader over an already-encoded CDR payload (incl. header).
class CdrReader {
 public:
  explicit CdrReader(const cdr::Buffer& buf)
      : fb_(const_cast<char*>(reinterpret_cast<const char*>(buf.data())),
            buf.size()),
        ser_(fb_,
             eprosima::fastcdr::Cdr::LITTLE_ENDIANNESS,
             eprosima::fastcdr::Cdr::DDS_CDR) {
    ser_.read_encapsulation();
  }

  eprosima::fastcdr::Cdr& cdr() { return ser_; }

 private:
  eprosima::fastcdr::FastBuffer fb_;
  eprosima::fastcdr::Cdr        ser_;
};

void readTime(eprosima::fastcdr::Cdr& s, Time& t) {
  s >> t.sec >> t.nanosec;
}
void readHeader(eprosima::fastcdr::Cdr& s, Header& h) {
  readTime(s, h.stamp);
  s >> h.frame_id;
}
void readVec3(eprosima::fastcdr::Cdr& s, Vector3& v) {
  s >> v.x >> v.y >> v.z;
}
void readPoint(eprosima::fastcdr::Cdr& s, Point& p) {
  s >> p.x >> p.y >> p.z;
}
void readQuat(eprosima::fastcdr::Cdr& s, Quaternion& q) {
  s >> q.x >> q.y >> q.z >> q.w;
}
void readPose(eprosima::fastcdr::Cdr& s, Pose& p) {
  readPoint(s, p.position);
  readQuat(s, p.orientation);
}
void readTwist(eprosima::fastcdr::Cdr& s, Twist& t) {
  readVec3(s, t.linear);
  readVec3(s, t.angular);
}
void readCov36(eprosima::fastcdr::Cdr& s, std::array<double, 36>& cov) {
  for (auto& v : cov) s >> v;
}

}  // namespace

TEST(CdrRoundtrip, PoseStamped) {
  PoseStamped m;
  m.header.stamp.sec     = 123;
  m.header.stamp.nanosec = 456789U;
  m.header.frame_id      = "earth";
  m.pose.position        = Point{1.5, -2.25, 3.125};
  m.pose.orientation     = Quaternion{0.1, 0.2, 0.3, 0.927362};

  cdr::Buffer buf;
  cdr::encode(buf, m);
  ASSERT_GE(buf.size(), 4U);

  CdrReader reader(buf);
  PoseStamped out;
  readHeader(reader.cdr(), out.header);
  readPose(reader.cdr(), out.pose);

  EXPECT_EQ(out.header.stamp.sec,       m.header.stamp.sec);
  EXPECT_EQ(out.header.stamp.nanosec,   m.header.stamp.nanosec);
  EXPECT_EQ(out.header.frame_id,        m.header.frame_id);
  EXPECT_DOUBLE_EQ(out.pose.position.x, m.pose.position.x);
  EXPECT_DOUBLE_EQ(out.pose.position.y, m.pose.position.y);
  EXPECT_DOUBLE_EQ(out.pose.position.z, m.pose.position.z);
  EXPECT_DOUBLE_EQ(out.pose.orientation.w, m.pose.orientation.w);
  EXPECT_DOUBLE_EQ(out.pose.orientation.x, m.pose.orientation.x);
  EXPECT_DOUBLE_EQ(out.pose.orientation.y, m.pose.orientation.y);
  EXPECT_DOUBLE_EQ(out.pose.orientation.z, m.pose.orientation.z);
}

TEST(CdrRoundtrip, OdometryWithCovariance) {
  Odometry m;
  m.header.stamp.sec     = 5;
  m.header.stamp.nanosec = 0;
  m.header.frame_id      = "earth";
  m.child_frame_id       = "drone0/base_link";
  m.pose.pose.position   = Point{0, 0, 10};
  m.pose.pose.orientation = Quaternion{0, 0, 0, 1};
  for (size_t i = 0; i < 36; ++i) {
    m.pose.covariance[i]  = static_cast<double>(i);
    m.twist.covariance[i] = -static_cast<double>(i);
  }
  m.twist.twist.linear  = Vector3{1, 2, 3};
  m.twist.twist.angular = Vector3{0.1, 0.2, 0.3};

  cdr::Buffer buf;
  cdr::encode(buf, m);

  CdrReader reader(buf);
  Odometry out;
  readHeader(reader.cdr(), out.header);
  reader.cdr() >> out.child_frame_id;
  readPose(reader.cdr(), out.pose.pose);
  readCov36(reader.cdr(), out.pose.covariance);
  readTwist(reader.cdr(), out.twist.twist);
  readCov36(reader.cdr(), out.twist.covariance);

  EXPECT_EQ(out.child_frame_id, m.child_frame_id);
  for (size_t i = 0; i < 36; ++i) {
    EXPECT_DOUBLE_EQ(out.pose.covariance[i],  m.pose.covariance[i])  << i;
    EXPECT_DOUBLE_EQ(out.twist.covariance[i], m.twist.covariance[i]) << i;
  }
  EXPECT_DOUBLE_EQ(out.twist.twist.linear.x, 1.0);
  EXPECT_DOUBLE_EQ(out.twist.twist.angular.z, 0.3);
}

TEST(CdrRoundtrip, TrajectorySetpointsThreePoints) {
  TrajectorySetpoints m;
  m.header.frame_id = "earth";
  for (int i = 0; i < 3; ++i) {
    TrajectoryPointMsg p;
    p.id        = "wp" + std::to_string(i);
    p.position  = Vector3{static_cast<double>(i),   1.0 * i, 10.0};
    p.twist     = Vector3{0.5 * i, 0.0, 0.0};
    p.acceleration = Vector3{};
    p.yaw_angle = 0.5f * static_cast<float>(i);
    m.setpoints.push_back(p);
  }

  cdr::Buffer buf;
  cdr::encode(buf, m);

  CdrReader reader(buf);
  Header h;
  readHeader(reader.cdr(), h);
  uint32_t n = 0;
  reader.cdr() >> n;
  ASSERT_EQ(n, 3U);
  for (uint32_t i = 0; i < n; ++i) {
    TrajectoryPointMsg out;
    reader.cdr() >> out.id;
    readVec3(reader.cdr(), out.position);
    readVec3(reader.cdr(), out.twist);
    readVec3(reader.cdr(), out.acceleration);
    reader.cdr() >> out.yaw_angle;
    EXPECT_EQ(out.id,          m.setpoints[i].id);
    EXPECT_DOUBLE_EQ(out.position.x, m.setpoints[i].position.x);
    EXPECT_FLOAT_EQ(out.yaw_angle,   m.setpoints[i].yaw_angle);
  }
}

TEST(CdrRoundtrip, Float64MultiArrayWithLayout) {
  Float64MultiArray m;
  MultiArrayDimension d;
  d.label  = "data";
  d.size   = 4;
  d.stride = 4;
  m.layout.dim.push_back(d);
  m.layout.data_offset = 0;
  m.data = {1.0, 2.0, 3.0, 4.0};

  cdr::Buffer buf;
  cdr::encode(buf, m);

  CdrReader reader(buf);
  uint32_t n_dim = 0;
  reader.cdr() >> n_dim;
  ASSERT_EQ(n_dim, 1U);
  MultiArrayDimension out_d;
  reader.cdr() >> out_d.label >> out_d.size >> out_d.stride;
  EXPECT_EQ(out_d.label,  "data");
  EXPECT_EQ(out_d.size,   4U);
  EXPECT_EQ(out_d.stride, 4U);
  uint32_t data_offset = 0;
  reader.cdr() >> data_offset;
  EXPECT_EQ(data_offset, 0U);
  uint32_t n_data = 0;
  reader.cdr() >> n_data;
  ASSERT_EQ(n_data, 4U);
  for (uint32_t i = 0; i < n_data; ++i) {
    double v = 0;
    reader.cdr() >> v;
    EXPECT_DOUBLE_EQ(v, static_cast<double>(i + 1));
  }
}

TEST(CdrRoundtrip, EncapsulationHeaderIsLittleEndianDDS) {
  Int32 m{42};
  cdr::Buffer buf;
  cdr::encode(buf, m);
  ASSERT_GE(buf.size(), 4U);
  // Encapsulation header: 0x00, 0x01, 0x00, 0x00 (CDR_LE, options=0).
  EXPECT_EQ(static_cast<uint8_t>(buf[0]), 0x00);
  EXPECT_EQ(static_cast<uint8_t>(buf[1]), 0x01);
  EXPECT_EQ(static_cast<uint8_t>(buf[2]), 0x00);
  EXPECT_EQ(static_cast<uint8_t>(buf[3]), 0x00);
}

}  // namespace mav_flight_mcap::ros2
