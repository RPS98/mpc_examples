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
 * @file mcap_logger.cpp
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "mav_flight_mcap/mcap_logger.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mcap/writer.hpp>

#include "mav_flight_mcap/ros2/cdr_encoder.hpp"
#include "mav_flight_mcap/ros2/channel_registry.hpp"
#include "mav_flight_mcap/ros2/ros2_types.hpp"
#include "mav_flight_mcap/ros2/schemas.hpp"
#include "mav_flight_mcap/time_source.hpp"

namespace mav_flight_mcap {

namespace {

// Extra-topic registry: topic name -> ROS type name.
enum class ExtraKind {
  INT32,
  STRING,
  FLOAT64,
  FLOAT64_MULTI_ARRAY,
  VECTOR3,
};

ros2::Time splitNs(uint64_t ns) {
  ros2::Time t;
  t.sec     = static_cast<int32_t>(ns / 1'000'000'000ULL);
  t.nanosec = static_cast<uint32_t>(ns % 1'000'000'000ULL);
  return t;
}

ros2::Quaternion toQuat(const Eigen::Vector4d& q_wxyz) {
  ros2::Quaternion q;
  q.w = q_wxyz(0);
  q.x = q_wxyz(1);
  q.y = q_wxyz(2);
  q.z = q_wxyz(3);
  return q;
}

ros2::Point toPoint(const Eigen::Vector3d& v) {
  ros2::Point p;
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
  return p;
}

ros2::Vector3 toVec3(const Eigen::Vector3d& v) {
  ros2::Vector3 r;
  r.x = v.x();
  r.y = v.y();
  r.z = v.z();
  return r;
}

Eigen::Vector3d rotateEarthToBody(const Eigen::Vector4d& q_wxyz,
                                  const Eigen::Vector3d& v_earth) {
  Eigen::Quaterniond q(q_wxyz(0), q_wxyz(1), q_wxyz(2), q_wxyz(3));
  q.normalize();
  return q.conjugate() * v_earth;
}

}  // namespace

// --- Impl ---------------------------------------------------------------------

struct MCAPLogger::Impl {
  LoggerConfig cfg;
  TimeSource   time_src;

  mcap::McapWriter writer;
  std::optional<ros2::ChannelRegistry> registry;

  bool running        = false;
  bool topics_locked  = false;

  // Schema ids for the fixed types.
  mcap::SchemaId schema_pose_stamped_id        = 0;
  mcap::SchemaId schema_twist_stamped_id       = 0;
  mcap::SchemaId schema_odometry_id            = 0;
  mcap::SchemaId schema_thrust_id              = 0;
  mcap::SchemaId schema_trajectory_id          = 0;
  mcap::SchemaId schema_clock_id               = 0;
  mcap::SchemaId schema_int32_id               = 0;
  mcap::SchemaId schema_string_id              = 0;
  mcap::SchemaId schema_float64_id             = 0;
  mcap::SchemaId schema_float64_multi_array_id = 0;
  mcap::SchemaId schema_vector3_id             = 0;

  // Channel ids for fixed topics.
  mcap::ChannelId ch_pose_reference = 0;
  mcap::ChannelId ch_twist_reference = 0;
  mcap::ChannelId ch_trajectory_reference = 0;
  mcap::ChannelId ch_thrust_command = 0;
  mcap::ChannelId ch_twist_command = 0;
  mcap::ChannelId ch_pose_state = 0;
  mcap::ChannelId ch_twist_state = 0;
  mcap::ChannelId ch_odom_state = 0;
  mcap::ChannelId ch_clock = 0;

  // Extras: topic -> kind, plus channel id.
  std::unordered_map<std::string, ExtraKind>        extra_kind;
  std::unordered_map<std::string, mcap::ChannelId>  extra_channel;

  // Message sequence counter per channel (optional but nice to have).
  std::unordered_map<mcap::ChannelId, uint32_t> seq;

  // Clock throttle.
  bool     clock_has_last = false;
  uint64_t clock_last_ns  = 0;

  explicit Impl(const LoggerConfig& c) : cfg(c), time_src(c.time_mode) {}

  void requireNotLocked() const {
    if (topics_locked) {
      throw std::logic_error(
        "mav_flight_mcap: cannot modify topics after start()");
    }
  }

  void requireRunning() const {
    if (!running) {
      throw std::logic_error(
        "mav_flight_mcap: logger is not running; call start() first");
    }
  }

  uint32_t nextSeq(mcap::ChannelId ch) {
    return ++seq[ch];
  }

  void writeMessage(mcap::ChannelId ch, uint64_t t_ns,
                    const ros2::cdr::Buffer& payload) {
    mcap::Message msg;
    msg.channelId    = ch;
    msg.sequence     = nextSeq(ch);
    msg.logTime      = t_ns;
    msg.publishTime  = t_ns;
    msg.data         = reinterpret_cast<const std::byte*>(payload.data());
    msg.dataSize     = payload.size();
    const auto status = writer.write(msg);
    if (!status.ok()) {
      throw std::runtime_error(
        "mav_flight_mcap: MCAP write failed: " + status.message);
    }
  }

  void maybeWriteClock(uint64_t t_ns) {
    if (!cfg.emit_clock_on_every_save) {
      return;
    }
    if (clock_has_last && cfg.clock_min_period_s > 0.0) {
      const uint64_t min_delta_ns = static_cast<uint64_t>(
          cfg.clock_min_period_s * 1e9);
      if (t_ns < clock_last_ns + min_delta_ns) {
        return;
      }
    }
    ros2::Clock clk;
    clk.clock = splitNs(t_ns);
    ros2::cdr::Buffer buf;
    ros2::cdr::encode(buf, clk);
    writeMessage(ch_clock, t_ns, buf);
    clock_has_last = true;
    clock_last_ns  = t_ns;
  }

  ros2::Header makeHeader(uint64_t t_ns, const std::string& frame_id) {
    ros2::Header h;
    h.stamp    = splitNs(t_ns);
    h.frame_id = frame_id;
    return h;
  }
};

// --- Ctor/dtor ---------------------------------------------------------------

MCAPLogger::MCAPLogger(const LoggerConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

MCAPLogger::~MCAPLogger() {
  try {
    close();
  } catch (...) {
    // Swallow in destructor; caller can explicitly close() to see errors.
  }
}

// --- Setters ------------------------------------------------------------------

void MCAPLogger::set_pose_reference_topic(const std::string& t) {
  impl_->requireNotLocked();
  impl_->cfg.pose_reference_topic = t;
}
void MCAPLogger::set_twist_reference_topic(const std::string& t) {
  impl_->requireNotLocked();
  impl_->cfg.twist_reference_topic = t;
}
void MCAPLogger::set_trajectory_reference_topic(const std::string& t) {
  impl_->requireNotLocked();
  impl_->cfg.trajectory_reference_topic = t;
}
void MCAPLogger::set_thrust_command_topic(const std::string& t) {
  impl_->requireNotLocked();
  impl_->cfg.thrust_command_topic = t;
}
void MCAPLogger::set_twist_command_topic(const std::string& t) {
  impl_->requireNotLocked();
  impl_->cfg.twist_command_topic = t;
}
void MCAPLogger::set_pose_state_topic(const std::string& t) {
  impl_->requireNotLocked();
  impl_->cfg.pose_state_topic = t;
}
void MCAPLogger::set_twist_state_topic(const std::string& t) {
  impl_->requireNotLocked();
  impl_->cfg.twist_state_topic = t;
}
void MCAPLogger::set_odom_state_topic(const std::string& t) {
  impl_->requireNotLocked();
  impl_->cfg.odom_state_topic = t;
}

// --- Extras registration ------------------------------------------------------

void MCAPLogger::add_int32_topic(const std::string& topic) {
  impl_->requireNotLocked();
  impl_->extra_kind[topic] = ExtraKind::INT32;
}
void MCAPLogger::add_string_topic(const std::string& topic) {
  impl_->requireNotLocked();
  impl_->extra_kind[topic] = ExtraKind::STRING;
}
void MCAPLogger::add_float64_topic(const std::string& topic) {
  impl_->requireNotLocked();
  impl_->extra_kind[topic] = ExtraKind::FLOAT64;
}
void MCAPLogger::add_float64_multi_array_topic(const std::string& topic) {
  impl_->requireNotLocked();
  impl_->extra_kind[topic] = ExtraKind::FLOAT64_MULTI_ARRAY;
}
void MCAPLogger::add_vector3_topic(const std::string& topic) {
  impl_->requireNotLocked();
  impl_->extra_kind[topic] = ExtraKind::VECTOR3;
}

// --- Lifecycle ----------------------------------------------------------------

void MCAPLogger::start() {
  if (impl_->running) {
    throw std::logic_error("mav_flight_mcap: start() called twice");
  }
  if (impl_->cfg.file_path.empty()) {
    throw std::invalid_argument(
      "mav_flight_mcap: LoggerConfig.file_path is empty");
  }

  mcap::McapWriterOptions opts("ros2");
  opts.compression =
      impl_->cfg.compress_zstd ? mcap::Compression::Zstd : mcap::Compression::None;
  opts.compressionLevel = mcap::CompressionLevel::Default;

  const auto status = impl_->writer.open(impl_->cfg.file_path, opts);
  if (!status.ok()) {
    throw std::runtime_error(
      "mav_flight_mcap: cannot open '" + impl_->cfg.file_path +
      "': " + status.message);
  }

  impl_->registry.emplace(impl_->writer);
  auto& reg = *impl_->registry;

  // Register the fixed schemas (only if their channels are used; we always
  // register all known ones for simplicity — costs are negligible).
  impl_->schema_pose_stamped_id = reg.registerSchema(
      ros2::schemas::kTypePoseStamped,  ros2::schemas::kSchemaPoseStamped);
  impl_->schema_twist_stamped_id = reg.registerSchema(
      ros2::schemas::kTypeTwistStamped, ros2::schemas::kSchemaTwistStamped);
  impl_->schema_odometry_id = reg.registerSchema(
      ros2::schemas::kTypeOdometry,     ros2::schemas::kSchemaOdometry);
  impl_->schema_thrust_id = reg.registerSchema(
      ros2::schemas::kTypeThrust,       ros2::schemas::kSchemaThrust);
  impl_->schema_trajectory_id = reg.registerSchema(
      ros2::schemas::kTypeTrajectorySetpoints,
      ros2::schemas::kSchemaTrajectorySetpoints);
  impl_->schema_clock_id = reg.registerSchema(
      ros2::schemas::kTypeClock,        ros2::schemas::kSchemaClock);

  // Fixed channels.
  impl_->ch_pose_reference       = reg.registerChannel(
      impl_->cfg.pose_reference_topic, impl_->schema_pose_stamped_id);
  impl_->ch_twist_reference      = reg.registerChannel(
      impl_->cfg.twist_reference_topic, impl_->schema_twist_stamped_id);
  impl_->ch_trajectory_reference = reg.registerChannel(
      impl_->cfg.trajectory_reference_topic, impl_->schema_trajectory_id);
  impl_->ch_thrust_command       = reg.registerChannel(
      impl_->cfg.thrust_command_topic, impl_->schema_thrust_id);
  impl_->ch_twist_command        = reg.registerChannel(
      impl_->cfg.twist_command_topic, impl_->schema_twist_stamped_id);
  impl_->ch_pose_state           = reg.registerChannel(
      impl_->cfg.pose_state_topic, impl_->schema_pose_stamped_id);
  impl_->ch_twist_state          = reg.registerChannel(
      impl_->cfg.twist_state_topic, impl_->schema_twist_stamped_id);
  impl_->ch_odom_state           = reg.registerChannel(
      impl_->cfg.odom_state_topic, impl_->schema_odometry_id);
  impl_->ch_clock                = reg.registerChannel(
      impl_->cfg.clock_topic, impl_->schema_clock_id);

  // Extra channels (register schemas on-demand so unused ones don't pollute).
  for (const auto& [topic, kind] : impl_->extra_kind) {
    mcap::SchemaId sid = 0;
    switch (kind) {
      case ExtraKind::INT32:
        if (impl_->schema_int32_id == 0) {
          impl_->schema_int32_id = reg.registerSchema(
              ros2::schemas::kTypeInt32, ros2::schemas::kSchemaInt32);
        }
        sid = impl_->schema_int32_id;
        break;
      case ExtraKind::STRING:
        if (impl_->schema_string_id == 0) {
          impl_->schema_string_id = reg.registerSchema(
              ros2::schemas::kTypeString, ros2::schemas::kSchemaString);
        }
        sid = impl_->schema_string_id;
        break;
      case ExtraKind::FLOAT64:
        if (impl_->schema_float64_id == 0) {
          impl_->schema_float64_id = reg.registerSchema(
              ros2::schemas::kTypeFloat64, ros2::schemas::kSchemaFloat64);
        }
        sid = impl_->schema_float64_id;
        break;
      case ExtraKind::FLOAT64_MULTI_ARRAY:
        if (impl_->schema_float64_multi_array_id == 0) {
          impl_->schema_float64_multi_array_id = reg.registerSchema(
              ros2::schemas::kTypeFloat64MultiArray,
              ros2::schemas::kSchemaFloat64MultiArray);
        }
        sid = impl_->schema_float64_multi_array_id;
        break;
      case ExtraKind::VECTOR3:
        if (impl_->schema_vector3_id == 0) {
          impl_->schema_vector3_id = reg.registerSchema(
              ros2::schemas::kTypeVector3, ros2::schemas::kSchemaVector3);
        }
        sid = impl_->schema_vector3_id;
        break;
    }
    impl_->extra_channel[topic] = reg.registerChannel(topic, sid);
  }

  impl_->running       = true;
  impl_->topics_locked = true;
}

void MCAPLogger::close() {
  if (!impl_->running) {
    return;
  }
  impl_->writer.close();
  impl_->running = false;
}

bool MCAPLogger::isRunning() const { return impl_->running; }

// --- Save helpers -------------------------------------------------------------

namespace {

// Private helper that returns the resolved ns AND writes /clock, used by every
// public save_*.
struct SaveCtx {
  uint64_t t_ns;
};

}  // namespace

// --- Single-topic saves -------------------------------------------------------

void MCAPLogger::save_pose_reference(double t,
                                     const Eigen::Vector3d& pos,
                                     const Eigen::Vector4d& q_wxyz) {
  impl_->requireRunning();
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::PoseStamped m;
  m.header        = impl_->makeHeader(t_ns, impl_->cfg.frame_earth);
  m.pose.position = toPoint(pos);
  m.pose.orientation = toQuat(q_wxyz);
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(impl_->ch_pose_reference, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_twist_reference(double t, const Eigen::Vector3d& linear) {
  impl_->requireRunning();
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::TwistStamped m;
  m.header       = impl_->makeHeader(t_ns, impl_->cfg.frame_earth);
  m.twist.linear = toVec3(linear);
  // angular left at zero (spec: only linear module is meaningful).
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(impl_->ch_twist_reference, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_trajectory_reference(
    double t, const std::vector<TrajectoryPoint>& points) {
  impl_->requireRunning();
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::TrajectorySetpoints m;
  m.header = impl_->makeHeader(t_ns, impl_->cfg.frame_earth);
  m.setpoints.reserve(points.size());
  for (const auto& p : points) {
    ros2::TrajectoryPointMsg tp;
    tp.id           = p.id;
    tp.position     = toVec3(p.position);
    tp.twist        = toVec3(p.twist);
    tp.acceleration = toVec3(p.acceleration);
    tp.yaw_angle    = p.yaw_angle;
    m.setpoints.push_back(std::move(tp));
  }
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(impl_->ch_trajectory_reference, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_thrust_command(double t, double thrust) {
  impl_->requireRunning();
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::Thrust m;
  m.header            = impl_->makeHeader(t_ns, impl_->cfg.frame_body);
  m.thrust            = static_cast<float>(thrust);
  m.thrust_normalized = 0.0f;
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(impl_->ch_thrust_command, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_twist_command(double t, const Eigen::Vector3d& angular) {
  impl_->requireRunning();
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::TwistStamped m;
  m.header        = impl_->makeHeader(t_ns, impl_->cfg.frame_body);
  m.twist.angular = toVec3(angular);
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(impl_->ch_twist_command, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_pose_state(double t,
                                 const Eigen::Vector3d& pos,
                                 const Eigen::Vector4d& q_wxyz) {
  impl_->requireRunning();
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::PoseStamped m;
  m.header           = impl_->makeHeader(t_ns, impl_->cfg.frame_earth);
  m.pose.position    = toPoint(pos);
  m.pose.orientation = toQuat(q_wxyz);
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(impl_->ch_pose_state, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_twist_state(double t,
                                  const Eigen::Vector3d& linear,
                                  const Eigen::Vector3d& angular) {
  impl_->requireRunning();
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::TwistStamped m;
  m.header        = impl_->makeHeader(t_ns, impl_->cfg.frame_body);
  m.twist.linear  = toVec3(linear);
  m.twist.angular = toVec3(angular);
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(impl_->ch_twist_state, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_odom_state(double t,
                                 const Eigen::Vector3d& pos_earth,
                                 const Eigen::Vector4d& q_wxyz,
                                 const Eigen::Vector3d& linear_body,
                                 const Eigen::Vector3d& angular_body) {
  impl_->requireRunning();
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::Odometry m;
  m.header            = impl_->makeHeader(t_ns, impl_->cfg.frame_earth);
  m.child_frame_id    = impl_->cfg.frame_body;
  m.pose.pose.position    = toPoint(pos_earth);
  m.pose.pose.orientation = toQuat(q_wxyz);
  // covariance stays zero (array<double,36>).
  m.twist.twist.linear  = toVec3(linear_body);
  m.twist.twist.angular = toVec3(angular_body);
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(impl_->ch_odom_state, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

// --- Aggregate saves ----------------------------------------------------------

void MCAPLogger::save_state(double t,
                            const Eigen::Vector3d& pos_earth,
                            const Eigen::Vector4d& q_wxyz,
                            const Eigen::Vector3d& linear_earth,
                            const Eigen::Vector3d& angular_body) {
  const Eigen::Vector3d linear_body = rotateEarthToBody(q_wxyz, linear_earth);
  save_pose_state(t, pos_earth, q_wxyz);
  save_twist_state(t, linear_earth, angular_body);
  save_odom_state(t, pos_earth, q_wxyz, linear_body, angular_body);
}

void MCAPLogger::save_position_reference(double t,
                                         const Eigen::Vector3d& pos_ref,
                                         const Eigen::Vector4d& q_wxyz_ref,
                                         const Eigen::Vector3d& max_linear_speed) {
  save_pose_reference(t, pos_ref, q_wxyz_ref);
  save_twist_reference(t, max_linear_speed);
}

void MCAPLogger::save_trajectory_reference_full(
    double t, const std::vector<TrajectoryPoint>& points,
    bool also_emit_pose_and_twist) {
  save_trajectory_reference(t, points);
  if (also_emit_pose_and_twist && !points.empty()) {
    const auto& first = points.front();
    Eigen::Vector4d q_wxyz(
        std::cos(first.yaw_angle * 0.5),
        0.0, 0.0, std::sin(first.yaw_angle * 0.5));
    save_pose_reference(t, first.position, q_wxyz);
    save_twist_reference(t, first.twist);
  }
}

void MCAPLogger::save_actuation(double t,
                                double thrust,
                                const Eigen::Vector3d& angular_cmd_body) {
  save_thrust_command(t, thrust);
  save_twist_command(t, angular_cmd_body);
}

// --- Extras -------------------------------------------------------------------

namespace {
mcap::ChannelId extraChannelOrThrow(
    const std::unordered_map<std::string, mcap::ChannelId>& extra_channel,
    const std::unordered_map<std::string, ExtraKind>& extra_kind,
    const std::string& topic, ExtraKind expected) {
  auto it = extra_channel.find(topic);
  if (it == extra_channel.end()) {
    throw std::logic_error(
      "mav_flight_mcap: extra topic '" + topic +
      "' was not registered before start()");
  }
  auto it_kind = extra_kind.find(topic);
  if (it_kind == extra_kind.end() || it_kind->second != expected) {
    throw std::logic_error(
      "mav_flight_mcap: extra topic '" + topic +
      "' registered with a different type");
  }
  return it->second;
}
}  // namespace

void MCAPLogger::save_int32(const std::string& topic, double t, int32_t value) {
  impl_->requireRunning();
  const mcap::ChannelId ch = extraChannelOrThrow(
      impl_->extra_channel, impl_->extra_kind, topic, ExtraKind::INT32);
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::Int32 m;
  m.data = value;
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(ch, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_string(const std::string& topic, double t,
                             const std::string& value) {
  impl_->requireRunning();
  const mcap::ChannelId ch = extraChannelOrThrow(
      impl_->extra_channel, impl_->extra_kind, topic, ExtraKind::STRING);
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::String m;
  m.data = value;
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(ch, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_float64(const std::string& topic, double t, double value) {
  impl_->requireRunning();
  const mcap::ChannelId ch = extraChannelOrThrow(
      impl_->extra_channel, impl_->extra_kind, topic, ExtraKind::FLOAT64);
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::Float64 m;
  m.data = value;
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(ch, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_vector3(const std::string& topic, double t,
                              const Eigen::Vector3d& value) {
  impl_->requireRunning();
  const mcap::ChannelId ch = extraChannelOrThrow(
      impl_->extra_channel, impl_->extra_kind, topic, ExtraKind::VECTOR3);
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::Vector3 m = toVec3(value);
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(ch, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

void MCAPLogger::save_float64_multi_array(const std::string& topic, double t,
                                          const std::vector<double>& value,
                                          const std::vector<uint32_t>& dims) {
  impl_->requireRunning();
  const mcap::ChannelId ch = extraChannelOrThrow(
      impl_->extra_channel, impl_->extra_kind, topic,
      ExtraKind::FLOAT64_MULTI_ARRAY);
  const uint64_t t_ns = impl_->time_src.resolve(t);
  ros2::Float64MultiArray m;
  if (dims.empty()) {
    ros2::MultiArrayDimension d;
    d.label  = "data";
    d.size   = static_cast<uint32_t>(value.size());
    d.stride = static_cast<uint32_t>(value.size());
    m.layout.dim.push_back(std::move(d));
  } else {
    m.layout.dim.reserve(dims.size());
    uint32_t stride = 1;
    for (auto it = dims.rbegin(); it != dims.rend(); ++it) stride *= *it;
    for (size_t i = 0; i < dims.size(); ++i) {
      ros2::MultiArrayDimension d;
      d.label  = "dim" + std::to_string(i);
      d.size   = dims[i];
      d.stride = stride;
      stride /= dims[i];
      m.layout.dim.push_back(std::move(d));
    }
  }
  m.layout.data_offset = 0;
  m.data               = value;
  ros2::cdr::Buffer buf;
  ros2::cdr::encode(buf, m);
  impl_->writeMessage(ch, t_ns, buf);
  impl_->maybeWriteClock(t_ns);
}

}  // namespace mav_flight_mcap
