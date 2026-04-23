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
 * @file test_mcap_format.cpp
 *
 * Writes a small MCAP using MCAPLogger, re-reads it with mcap::McapReader,
 * and validates schemas, channels, and payloads.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <mcap/reader.hpp>

#include "mav_flight_mcap/mcap_logger.hpp"

namespace mav_flight_mcap {

namespace {

std::filesystem::path tempPath(const std::string& name) {
  auto dir = std::filesystem::temp_directory_path() / "mav_flight_mcap_tests";
  std::filesystem::create_directories(dir);
  return dir / name;
}

}  // namespace

class McapFormatTest : public ::testing::Test {
 protected:
  std::filesystem::path path_;
  void SetUp() override {
    path_ = tempPath("format.mcap");
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
};

TEST_F(McapFormatTest, WritesRos2SchemaAndCdrEncoding) {
  {
    LoggerConfig cfg;
    cfg.file_path = path_.string();
    cfg.time_mode = TimeMode::SIMULATION;
    cfg.compress_zstd = false;  // easier to inspect.
    cfg.emit_clock_on_every_save = true;

    MCAPLogger logger(cfg);
    logger.add_float64_topic("/drone0/debug/solve_time");
    logger.start();

    const Eigen::Vector3d pos(1, 2, 3);
    const Eigen::Vector4d q(1, 0, 0, 0);
    logger.save_pose_state(0.0, pos, q);
    logger.save_twist_state(0.0, Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 0, 0.5));
    logger.save_odom_state(0.0, pos, q,
                           Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 0, 0.5));
    logger.save_pose_reference(0.0, pos, q);
    logger.save_twist_reference(0.0, Eigen::Vector3d(2, 2, 2));
    logger.save_thrust_command(0.0, 10.0);
    logger.save_twist_command(0.0, Eigen::Vector3d(0.1, 0.1, 0.1));
    logger.save_float64("/drone0/debug/solve_time", 0.0, 1.4e-3);
    logger.close();
  }

  mcap::McapReader reader;
  const auto status = reader.open(path_.string());
  ASSERT_TRUE(status.ok()) << status.message;

  auto messages_view = reader.readMessages();
  std::unordered_map<std::string, std::string> topic_to_schema;  // topic -> type
  std::unordered_set<std::string>               channel_encodings;
  std::unordered_set<std::string>               schema_encodings;

  size_t count = 0;
  for (const auto& msg_view : messages_view) {
    ++count;
    channel_encodings.insert(msg_view.channel->messageEncoding);
    if (msg_view.schema) {
      schema_encodings.insert(msg_view.schema->encoding);
      topic_to_schema[msg_view.channel->topic] = msg_view.schema->name;
    }
  }

  EXPECT_GT(count, 0U);
  EXPECT_EQ(channel_encodings.size(), 1U);
  EXPECT_TRUE(channel_encodings.count("cdr"));
  EXPECT_EQ(schema_encodings.size(), 1U);
  EXPECT_TRUE(schema_encodings.count("ros2msg"));

  EXPECT_EQ(topic_to_schema["/drone0/self_localization/pose"],
            "geometry_msgs/msg/PoseStamped");
  EXPECT_EQ(topic_to_schema["/drone0/self_localization/twist"],
            "geometry_msgs/msg/TwistStamped");
  EXPECT_EQ(topic_to_schema["/drone0/sensor_measurements/odom"],
            "nav_msgs/msg/Odometry");
  EXPECT_EQ(topic_to_schema["/drone0/motion_reference/pose"],
            "geometry_msgs/msg/PoseStamped");
  EXPECT_EQ(topic_to_schema["/drone0/actuator_command/thrust"],
            "as2_msgs/msg/Thrust");
  EXPECT_EQ(topic_to_schema["/drone0/debug/solve_time"],
            "std_msgs/msg/Float64");
  EXPECT_EQ(topic_to_schema["/clock"],
            "rosgraph_msgs/msg/Clock");
}

TEST_F(McapFormatTest, StartLocksTopicSetters) {
  LoggerConfig cfg;
  cfg.file_path = path_.string();
  MCAPLogger logger(cfg);
  logger.start();
  EXPECT_THROW(logger.set_pose_state_topic("/nope"), std::logic_error);
  EXPECT_THROW(logger.add_int32_topic("/nope"), std::logic_error);
  logger.close();
}

TEST_F(McapFormatTest, SaveWithoutStartThrows) {
  LoggerConfig cfg;
  cfg.file_path = path_.string();
  MCAPLogger logger(cfg);
  EXPECT_THROW(
    logger.save_pose_state(0.0, Eigen::Vector3d::Zero(),
                           Eigen::Vector4d(1, 0, 0, 0)),
    std::logic_error);
}

TEST_F(McapFormatTest, SimulationModeMakesFirstStampZero) {
  {
    LoggerConfig cfg;
    cfg.file_path = path_.string();
    cfg.time_mode = TimeMode::SIMULATION;
    cfg.compress_zstd = false;
    MCAPLogger logger(cfg);
    logger.start();
    logger.save_pose_state(100.5, Eigen::Vector3d::Zero(),
                           Eigen::Vector4d(1, 0, 0, 0));
    logger.save_pose_state(100.6, Eigen::Vector3d::Zero(),
                           Eigen::Vector4d(1, 0, 0, 0));
    logger.close();
  }
  mcap::McapReader reader;
  ASSERT_TRUE(reader.open(path_.string()).ok());
  uint64_t first_stamp = UINT64_MAX;
  auto view = reader.readMessages();
  for (const auto& m : view) {
    if (m.channel->topic == "/drone0/self_localization/pose") {
      first_stamp = std::min(first_stamp, m.message.logTime);
    }
  }
  EXPECT_EQ(first_stamp, 0U);
}

}  // namespace mav_flight_mcap
