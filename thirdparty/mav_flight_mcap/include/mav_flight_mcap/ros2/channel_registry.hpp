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
 * @file channel_registry.hpp
 *
 * Helper that caches MCAP schema/channel ids by ROS 2 type name and topic,
 * so schemas are registered at most once regardless of how many topics share
 * the same type.
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef MAV_FLIGHT_MCAP__ROS2__CHANNEL_REGISTRY_HPP_
#define MAV_FLIGHT_MCAP__ROS2__CHANNEL_REGISTRY_HPP_

#include <string>
#include <string_view>
#include <unordered_map>

#include <mcap/types.hpp>

// Forward declare MCAP writer to avoid pulling writer.hpp into public headers.
namespace mcap {
class McapWriter;
}  // namespace mcap

namespace mav_flight_mcap {
namespace ros2 {

/**
 * @brief Registers (and caches) MCAP Schemas and Channels for ros2msg/CDR.
 *
 * Schemas are keyed by ROS type name (e.g. "geometry_msgs/msg/PoseStamped")
 * and registered at most once. Channels are keyed by topic name and must be
 * registered explicitly with registerChannel().
 */
class ChannelRegistry {
 public:
  /** @brief Bind the registry to a writer. The writer must outlive the registry. */
  explicit ChannelRegistry(mcap::McapWriter& writer);

  /**
   * @brief Register a schema by ROS type name, or return the cached id.
   *
   * @param ros_type   ROS 2 canonical name, e.g. "geometry_msgs/msg/PoseStamped".
   * @param schema_text Multi-part ros2msg schema text (see schemas.hpp).
   * @return           MCAP schema id (stable for the life of the writer).
   */
  mcap::SchemaId registerSchema(std::string_view ros_type,
                                std::string_view schema_text);

  /**
   * @brief Register a channel for a topic tied to a previously registered schema.
   *
   * @param topic      Topic name, e.g. "/drone0/self_localization/pose".
   * @param schema_id  Id returned by registerSchema.
   * @return           MCAP channel id.
   */
  mcap::ChannelId registerChannel(std::string_view topic,
                                  mcap::SchemaId schema_id);

  /**
   * @brief Look up a previously registered channel by topic.
   *
   * @param topic Topic name.
   * @return      Channel id or 0 if not registered.
   */
  mcap::ChannelId channelIdFor(std::string_view topic) const;

 private:
  mcap::McapWriter* writer_;
  std::unordered_map<std::string, mcap::SchemaId>  schema_by_type_;
  std::unordered_map<std::string, mcap::ChannelId> channel_by_topic_;
};

}  // namespace ros2
}  // namespace mav_flight_mcap

#endif  // MAV_FLIGHT_MCAP__ROS2__CHANNEL_REGISTRY_HPP_
