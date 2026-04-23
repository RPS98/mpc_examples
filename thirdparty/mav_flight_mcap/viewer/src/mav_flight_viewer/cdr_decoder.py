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

"""Minimal pure-Python CDR reader for ROS 2 Humble MCAP payloads."""

from __future__ import annotations

import struct
from typing import Callable, List, TypeVar

T = TypeVar('T')


class CdrReader:
    """Read OMG CDR little/big-endian payloads, including the 4B encapsulation header.

    Alignment is counted relative to the start of the payload *after* the
    4-byte encapsulation header, matching the behaviour of Fast-CDR and
    rosbag2 Humble. Concretely: the reader keeps an absolute position `pos`,
    and alignment uses `(pos - 4) % n == 0`.
    """

    def __init__(self, data: bytes) -> None:
        """Parse the encapsulation header and position the cursor after it."""
        if len(data) < 4:
            raise ValueError('CDR payload too short (missing encapsulation header).')
        self._data = data
        if data[0] != 0 or data[2] != 0 or data[3] != 0:
            raise ValueError(f'Unexpected CDR encapsulation header {data[:4].hex()}.')
        self._endian = '<' if data[1] == 1 else '>'
        self._pos = 4  # payload starts here; alignment counter resets (pos - 4).

    # --- Primitives -----------------------------------------------------------

    def read_bool(self) -> bool:
        """Read a boolean (1 byte)."""
        self._align(1)
        v = self._data[self._pos]
        self._pos += 1
        return bool(v)

    def read_i8(self) -> int:
        """Read a signed 8-bit integer."""
        self._align(1)
        (v,) = struct.unpack_from(self._endian + 'b', self._data, self._pos)
        self._pos += 1
        return v

    def read_u8(self) -> int:
        """Read an unsigned 8-bit integer."""
        self._align(1)
        v = self._data[self._pos]
        self._pos += 1
        return v

    def read_i16(self) -> int:
        """Read a signed 16-bit integer."""
        self._align(2)
        (v,) = struct.unpack_from(self._endian + 'h', self._data, self._pos)
        self._pos += 2
        return v

    def read_u16(self) -> int:
        """Read an unsigned 16-bit integer."""
        self._align(2)
        (v,) = struct.unpack_from(self._endian + 'H', self._data, self._pos)
        self._pos += 2
        return v

    def read_i32(self) -> int:
        """Read a signed 32-bit integer."""
        self._align(4)
        (v,) = struct.unpack_from(self._endian + 'i', self._data, self._pos)
        self._pos += 4
        return v

    def read_u32(self) -> int:
        """Read an unsigned 32-bit integer."""
        self._align(4)
        (v,) = struct.unpack_from(self._endian + 'I', self._data, self._pos)
        self._pos += 4
        return v

    def read_i64(self) -> int:
        """Read a signed 64-bit integer."""
        self._align(8)
        (v,) = struct.unpack_from(self._endian + 'q', self._data, self._pos)
        self._pos += 8
        return v

    def read_u64(self) -> int:
        """Read an unsigned 64-bit integer."""
        self._align(8)
        (v,) = struct.unpack_from(self._endian + 'Q', self._data, self._pos)
        self._pos += 8
        return v

    def read_f32(self) -> float:
        """Read a 32-bit float."""
        self._align(4)
        (v,) = struct.unpack_from(self._endian + 'f', self._data, self._pos)
        self._pos += 4
        return v

    def read_f64(self) -> float:
        """Read a 64-bit float."""
        self._align(8)
        (v,) = struct.unpack_from(self._endian + 'd', self._data, self._pos)
        self._pos += 8
        return v

    def read_string(self) -> str:
        """Read a ROS 2 CDR string (uint32 length including NUL, then UTF-8 bytes)."""
        n = self.read_u32()
        if n == 0:
            return ''
        raw = self._data[self._pos:self._pos + n]
        self._pos += n
        # The CDR string terminator is included in the length; drop it.
        if raw.endswith(b'\x00'):
            raw = raw[:-1]
        return raw.decode('utf-8')

    # --- Composites -----------------------------------------------------------

    def read_sequence(self, elem: Callable[['CdrReader'], T]) -> List[T]:
        """Read a variable-length sequence (uint32 length + elements)."""
        n = self.read_u32()
        return [elem(self) for _ in range(n)]

    def read_array(self, n: int, elem: Callable[['CdrReader'], T]) -> List[T]:
        """Read a fixed-length array (no length prefix)."""
        return [elem(self) for _ in range(n)]

    # --- Alignment ------------------------------------------------------------

    def _align(self, n: int) -> None:
        """Advance pos until (pos - 4) is a multiple of `n` (payload-relative)."""
        base = self._pos - 4
        pad = (-base) % n
        self._pos += pad
