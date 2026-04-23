// mav_flight_mcap — header-only quadrotor MCAP recorder (C++/Python parity).
//
// Writes a `.mcap` file with protobuf-encoded messages on four channel groups:
//   - /drone/state       -> mav_flight_mcap::State
//   - /drone/reference   -> mav_flight_mcap::Reference
//   - /drone/actuation   -> mav_flight_mcap::Actuation
//   - /drone/extras/<n>  -> mav_flight_mcap::Scalar (one per declared extra field)
//
// SI units throughout. Quaternions are (w, x, y, z), body -> world.
// Angular velocity in body frame; linear velocity in world frame.
//
// Link against: Eigen3::Eigen, protobuf::libprotobuf, mcap_vendor::mcap,
//               and the generated protobuf library (mav_flight_mcap_proto).
#pragma once

#include <Eigen/Dense>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>

#include <mcap/writer.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "mav_flight.pb.h"

namespace mav_flight_mcap {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace detail {

inline std::string serialize_file_descriptor_set(
    const google::protobuf::Descriptor* desc) {
  google::protobuf::FileDescriptorSet set;
  std::unordered_set<std::string> seen;

  std::function<void(const google::protobuf::FileDescriptor*)> add_file =
      [&](const google::protobuf::FileDescriptor* file) {
        if (!seen.insert(file->name()).second) return;
        for (int i = 0; i < file->dependency_count(); ++i) {
          add_file(file->dependency(i));
        }
        google::protobuf::FileDescriptorProto* proto = set.add_file();
        file->CopyTo(proto);
      };
  add_file(desc->file());

  std::string out;
  set.SerializeToString(&out);
  return out;
}

inline void set_vec3(::mav_flight_mcap::Vec3& m, const Eigen::Vector3d& v) {
  m.set_x(v.x());
  m.set_y(v.y());
  m.set_z(v.z());
}

inline void set_quat(::mav_flight_mcap::Quaternion& m,
                     const Eigen::Quaterniond& q) {
  m.set_w(q.w());
  m.set_x(q.x());
  m.set_y(q.y());
  m.set_z(q.z());
}

}  // namespace detail

// ---------------------------------------------------------------------------
// MCAPRecorder — API parity with the Python class of the same name.
// ---------------------------------------------------------------------------

class MCAPRecorder {
 public:
  static constexpr const char* kTopicState       = "/drone/state";
  static constexpr const char* kTopicReference   = "/drone/reference";
  static constexpr const char* kTopicActuation   = "/drone/actuation";
  static constexpr const char* kTopicExtraPrefix = "/drone/extras/";

  explicit MCAPRecorder(const std::string& file_path,
                        int n_motors = 4,
                        const std::vector<std::string>& extra_fields = {})
      : n_motors_(n_motors), extra_fields_(extra_fields) {
    if (n_motors_ < 0) {
      throw std::invalid_argument("n_motors must be >= 0");
    }
    for (size_t i = 0; i < extra_fields_.size(); ++i) {
      for (size_t j = i + 1; j < extra_fields_.size(); ++j) {
        if (extra_fields_[i] == extra_fields_[j]) {
          throw std::invalid_argument("extra_fields must be unique");
        }
      }
    }

    mcap::McapWriterOptions options("mav_flight_mcap");
    const auto status = writer_.open(file_path, options);
    if (!status.ok()) {
      throw std::runtime_error("Failed to open MCAP '" + file_path +
                               "': " + status.message);
    }
    open_ = true;

    state_chan_ = register_channel(kTopicState, State::descriptor());
    reference_chan_ =
        register_channel(kTopicReference, Reference::descriptor());
    actuation_chan_ =
        register_channel(kTopicActuation, Actuation::descriptor());
    extra_chans_.reserve(extra_fields_.size());
    for (const auto& name : extra_fields_) {
      extra_chans_.push_back(register_channel(
          std::string(kTopicExtraPrefix) + name, Scalar::descriptor()));
    }
  }

  ~MCAPRecorder() { close(); }

  MCAPRecorder(const MCAPRecorder&)            = delete;
  MCAPRecorder& operator=(const MCAPRecorder&) = delete;
  MCAPRecorder(MCAPRecorder&&)                 = delete;
  MCAPRecorder& operator=(MCAPRecorder&&)      = delete;

  void close() {
    if (!open_) return;
    writer_.close();
    open_ = false;
  }

  void save(double time,
            const Eigen::Vector3d& position,
            const Eigen::Quaterniond& orientation,
            const Eigen::Vector3d& linear_velocity,
            const Eigen::Vector3d& angular_velocity,
            const Eigen::Vector3d& reference_position,
            const Eigen::Vector3d& reference_velocity,
            const Eigen::Quaterniond& reference_orientation,
            const Eigen::Vector3d& reference_angular_velocity,
            double thrust,
            const Eigen::Vector3d& command_angular_velocity,
            const Eigen::VectorXd& motor_angular_velocity,
            const std::vector<double>& extras = {}) {
    if (!open_) {
      throw std::runtime_error("MCAPRecorder is closed");
    }
    if (motor_angular_velocity.size() != n_motors_) {
      throw std::invalid_argument(
          "motor_angular_velocity size does not match n_motors");
    }
    if (!extras.empty() && extras.size() != extra_fields_.size()) {
      throw std::invalid_argument("extras size does not match extra_fields");
    }

    const uint64_t t_ns =
        static_cast<uint64_t>(std::llround(time * 1e9));

    // --- /drone/state ---
    State state;
    state.set_time(time);
    detail::set_vec3(*state.mutable_position(), position);
    detail::set_quat(*state.mutable_orientation(), orientation);
    detail::set_vec3(*state.mutable_linear_velocity(), linear_velocity);
    detail::set_vec3(*state.mutable_angular_velocity(), angular_velocity);
    publish(state_chan_, state, t_ns);

    // --- /drone/reference ---
    Reference reference;
    reference.set_time(time);
    detail::set_vec3(*reference.mutable_position(), reference_position);
    detail::set_vec3(*reference.mutable_linear_velocity(), reference_velocity);
    detail::set_quat(*reference.mutable_orientation(), reference_orientation);
    detail::set_vec3(*reference.mutable_angular_velocity(),
                     reference_angular_velocity);
    publish(reference_chan_, reference, t_ns);

    // --- /drone/actuation ---
    Actuation actuation;
    actuation.set_time(time);
    actuation.set_thrust(thrust);
    detail::set_vec3(*actuation.mutable_command_angular_velocity(),
                     command_angular_velocity);
    actuation.mutable_motor_angular_velocity()->Reserve(
        static_cast<int>(motor_angular_velocity.size()));
    for (Eigen::Index i = 0; i < motor_angular_velocity.size(); ++i) {
      actuation.add_motor_angular_velocity(motor_angular_velocity(i));
    }
    publish(actuation_chan_, actuation, t_ns);

    // --- /drone/extras/<name> ---
    for (size_t i = 0; i < extra_fields_.size(); ++i) {
      Scalar msg;
      msg.set_time(time);
      msg.set_value(extras.empty() ? std::nan("") : extras[i]);
      publish(extra_chans_[i], msg, t_ns);
    }
  }

 private:
  mcap::ChannelId register_channel(const std::string& topic,
                                   const google::protobuf::Descriptor* desc) {
    const std::string fds = detail::serialize_file_descriptor_set(desc);
    mcap::Schema schema(desc->full_name(), "protobuf", fds);
    writer_.addSchema(schema);
    mcap::Channel channel(topic, "protobuf", schema.id);
    writer_.addChannel(channel);
    return channel.id;
  }

  template <typename MsgT>
  void publish(mcap::ChannelId channel_id, const MsgT& msg, uint64_t t_ns) {
    std::string data;
    msg.SerializeToString(&data);
    mcap::Message out;
    out.channelId   = channel_id;
    out.sequence    = sequence_++;
    out.logTime     = t_ns;
    out.publishTime = t_ns;
    out.dataSize    = data.size();
    out.data        = reinterpret_cast<const std::byte*>(data.data());
    const auto status = writer_.write(out);
    if (!status.ok()) {
      throw std::runtime_error("MCAP write failed: " + status.message);
    }
  }

  int n_motors_;
  std::vector<std::string> extra_fields_;
  mcap::McapWriter writer_;
  bool open_ = false;
  mcap::ChannelId state_chan_     = 0;
  mcap::ChannelId reference_chan_ = 0;
  mcap::ChannelId actuation_chan_ = 0;
  std::vector<mcap::ChannelId> extra_chans_;
  uint32_t sequence_ = 0;
};

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

inline Eigen::Quaterniond eulerToQuaternion(double roll,
                                            double pitch,
                                            double yaw) {
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  return Eigen::Quaterniond(
      cr * cp * cy + sr * sp * sy,  // w
      sr * cp * cy - cr * sp * sy,  // x
      cr * sp * cy + sr * cp * sy,  // y
      cr * cp * sy - sr * sp * cy   // z
  );
}

inline std::tuple<double, double, double> quaternionToEuler(
    const Eigen::Quaterniond& q) {
  const double w = q.w();
  const double x = q.x();
  const double y = q.y();
  const double z = q.z();
  const double roll  = std::atan2(2.0 * (w * x + y * z),
                                  1.0 - 2.0 * (x * x + y * y));
  double sinp        = 2.0 * (w * y - z * x);
  if (sinp > 1.0) sinp = 1.0;
  if (sinp < -1.0) sinp = -1.0;
  const double pitch = std::asin(sinp);
  const double yaw   = std::atan2(2.0 * (w * z + x * y),
                                  1.0 - 2.0 * (y * y + z * z));
  return {roll, pitch, yaw};
}

inline void printProgress(double fraction) {
  if (fraction < 0.0) fraction = 0.0;
  if (fraction > 1.0) fraction = 1.0;
  constexpr int kWidth = 40;
  const int filled     = static_cast<int>(fraction * kWidth);
  std::string bar(filled, '=');
  bar.append(kWidth - filled, ' ');
  std::printf("\r[%s] %3d%%", bar.c_str(),
              static_cast<int>(fraction * 100.0));
  std::fflush(stdout);
}

}  // namespace mav_flight_mcap
