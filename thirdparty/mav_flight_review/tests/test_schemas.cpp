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
 * @file test_schemas.cpp
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "mav_flight_mcap/ros2/schemas.hpp"

namespace mav_flight_mcap::ros2::schemas {

// Every type name must use the "<pkg>/msg/<Type>" form used by rosbag2 Humble.
TEST(SchemasTest, TypeNamesFollowMsgForm) {
  const std::vector<std::string_view> type_names = {
      kTypePoseStamped, kTypeTwistStamped, kTypeVector3, kTypeOdometry,
      kTypeThrust, kTypeTrajectorySetpoints, kTypeClock, kTypeInt32,
      kTypeString, kTypeFloat64, kTypeFloat64MultiArray,
  };
  for (const auto& name : type_names) {
    const std::string s(name);
    EXPECT_NE(s.find("/msg/"), std::string::npos) << "type: " << s;
  }
}

// Composite schemas must contain a canonical separator at least once.
TEST(SchemasTest, CompositeSchemasHaveSeparator) {
  const std::vector<std::string_view> composite = {
      kSchemaPoseStamped, kSchemaTwistStamped, kSchemaOdometry,
      kSchemaThrust, kSchemaTrajectorySetpoints, kSchemaClock,
      kSchemaFloat64MultiArray,
  };
  for (const auto& text : composite) {
    const std::string t(text);
    EXPECT_NE(t.find(kSeparator), std::string::npos);
  }
}

// PoseStamped transitively needs Header, Time, Pose, Point, Quaternion.
TEST(SchemasTest, PoseStampedContainsAllDeps) {
  const std::string s(kSchemaPoseStamped);
  EXPECT_NE(s.find("MSG: std_msgs/Header"),           std::string::npos);
  EXPECT_NE(s.find("MSG: builtin_interfaces/Time"),   std::string::npos);
  EXPECT_NE(s.find("MSG: geometry_msgs/Pose"),        std::string::npos);
  EXPECT_NE(s.find("MSG: geometry_msgs/Point"),       std::string::npos);
  EXPECT_NE(s.find("MSG: geometry_msgs/Quaternion"),  std::string::npos);
}

// Odometry transitively needs *PoseWithCovariance, *TwistWithCovariance, etc.
TEST(SchemasTest, OdometryContainsAllDeps) {
  const std::string s(kSchemaOdometry);
  EXPECT_NE(s.find("MSG: geometry_msgs/PoseWithCovariance"),
            std::string::npos);
  EXPECT_NE(s.find("MSG: geometry_msgs/TwistWithCovariance"),
            std::string::npos);
  EXPECT_NE(s.find("MSG: geometry_msgs/Twist"),    std::string::npos);
  EXPECT_NE(s.find("MSG: geometry_msgs/Vector3"),  std::string::npos);
  EXPECT_NE(s.find("float64[36] covariance"),      std::string::npos);
}

// TrajectorySetpoints must mention as2_msgs/TrajectoryPoint + Vector3.
TEST(SchemasTest, TrajectorySetpointsDeps) {
  const std::string s(kSchemaTrajectorySetpoints);
  EXPECT_NE(s.find("MSG: as2_msgs/TrajectoryPoint"), std::string::npos);
  EXPECT_NE(s.find("MSG: geometry_msgs/Vector3"),    std::string::npos);
  EXPECT_NE(s.find("float32 yaw_angle"),             std::string::npos);
}

// Standalone schemas must NOT contain a dependency separator.
TEST(SchemasTest, StandaloneSchemasHaveNoSeparator) {
  const std::vector<std::string_view> standalone = {
      kSchemaInt32, kSchemaString, kSchemaFloat64, kSchemaVector3,
  };
  for (const auto& text : standalone) {
    const std::string t(text);
    EXPECT_EQ(t.find(kSeparator), std::string::npos);
  }
}

}  // namespace mav_flight_mcap::ros2::schemas
