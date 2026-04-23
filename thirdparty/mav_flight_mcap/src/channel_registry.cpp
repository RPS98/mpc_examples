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
 * @file channel_registry.cpp
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "mav_flight_mcap/ros2/channel_registry.hpp"

#include <mcap/writer.hpp>

namespace mav_flight_mcap {
namespace ros2 {

namespace {
// ros2msg / CDR encoding strings expected by rosbag2 Humble.
constexpr const char* kSchemaEncoding  = "ros2msg";
constexpr const char* kMessageEncoding = "cdr";
}  // namespace

ChannelRegistry::ChannelRegistry(mcap::McapWriter& writer) : writer_(&writer) {}

mcap::SchemaId ChannelRegistry::registerSchema(std::string_view ros_type,
                                               std::string_view schema_text) {
  const std::string type_key(ros_type);
  auto it = schema_by_type_.find(type_key);
  if (it != schema_by_type_.end()) {
    return it->second;
  }
  mcap::Schema schema(type_key, kSchemaEncoding, std::string(schema_text));
  writer_->addSchema(schema);
  schema_by_type_.emplace(type_key, schema.id);
  return schema.id;
}

mcap::ChannelId ChannelRegistry::registerChannel(std::string_view topic,
                                                 mcap::SchemaId schema_id) {
  const std::string topic_key(topic);
  auto it = channel_by_topic_.find(topic_key);
  if (it != channel_by_topic_.end()) {
    return it->second;
  }
  mcap::Channel channel(topic_key, kMessageEncoding, schema_id);
  writer_->addChannel(channel);
  channel_by_topic_.emplace(topic_key, channel.id);
  return channel.id;
}

mcap::ChannelId ChannelRegistry::channelIdFor(std::string_view topic) const {
  const std::string topic_key(topic);
  auto it = channel_by_topic_.find(topic_key);
  return (it == channel_by_topic_.end()) ? 0 : it->second;
}

}  // namespace ros2
}  // namespace mav_flight_mcap
