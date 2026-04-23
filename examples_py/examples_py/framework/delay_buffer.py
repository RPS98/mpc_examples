#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Time-indexed buffer used to model compute latency.

Python mirror of ``examples/framework/include/framework/delay_buffer.hpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional


_EPS = 1e-12


@dataclass
class _Entry:
    payload: Any
    available_at: float


class DelayBuffer:
    """Chronologically-ordered queue of timestamped payloads.

    A sample pushed with ``available_at = T`` stays invisible to callers until
    the simulator clock reaches T. Single-threaded, single-producer use.
    """

    def __init__(self) -> None:
        self._entries: Deque[_Entry] = deque()

    def push(self, payload: Any, available_at: float) -> None:
        """Push a new payload that becomes visible at ``available_at``."""
        if self._entries and available_at + _EPS < self._entries[-1].available_at:
            # Out-of-order push would violate the chronological invariant.
            return
        self._entries.append(_Entry(payload=payload, available_at=available_at))

    def latest_available(self, t: float) -> Optional[Any]:
        """Return the latest payload whose availability <= @p t, if any.

        Discards every entry older than the returned one.
        """
        latest: Optional[Any] = None
        while self._entries and self._entries[0].available_at <= t + _EPS:
            latest = self._entries[0].payload
            self._entries.popleft()
        return latest

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
