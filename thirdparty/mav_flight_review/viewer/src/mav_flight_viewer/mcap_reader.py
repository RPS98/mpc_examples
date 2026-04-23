# Copyright 2025 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""High-level iterator over a ROS 2-compatible MCAP file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from mcap.reader import make_reader

from .ros2_schemas import decode


@dataclass(frozen=True)
class DecodedMsg:
    """A single MCAP message, decoded into a nested dict of primitives."""

    topic: str
    schema_name: str
    log_time_ns: int
    publish_time_ns: int
    payload: Dict[str, Any]


class FlightBag:
    """Open a ROS 2-compatible MCAP file and iterate decoded messages."""

    def __init__(self, path: Path | str) -> None:
        """Store the path; the file is opened lazily on iterate()."""
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Path of the backing MCAP file."""
        return self._path

    def topics(self) -> Dict[str, str]:
        """Return a mapping from topic name to ROS 2 schema name."""
        with open(self._path, 'rb') as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            return {
                c.topic: summary.schemas[c.schema_id].name
                for c in summary.channels.values()
            }

    def iter_messages(self, topic: Optional[str] = None) -> Iterator[DecodedMsg]:
        """Yield DecodedMsg for every message (optionally filtered by topic)."""
        with open(self._path, 'rb') as f:
            reader = make_reader(f)
            topics = [topic] if topic else None
            for schema, channel, message in reader.iter_messages(topics=topics):
                if schema is None:
                    continue
                payload = decode(schema.name, bytes(message.data))
                yield DecodedMsg(
                    topic=channel.topic,
                    schema_name=schema.name,
                    log_time_ns=message.log_time,
                    publish_time_ns=message.publish_time,
                    payload=payload,
                )
