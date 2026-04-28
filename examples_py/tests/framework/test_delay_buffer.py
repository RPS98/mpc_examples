#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`examples_py.framework.DelayBuffer`."""

import unittest

from examples_py.framework.delay_buffer import DelayBuffer


class DelayBufferTest(unittest.TestCase):

    def test_empty_returns_none(self) -> None:
        buf: DelayBuffer = DelayBuffer()
        self.assertIsNone(buf.latest_available(0.0))
        self.assertEqual(len(buf), 0)

    def test_delivers_payload_once_its_available_at_elapses(self) -> None:
        buf: DelayBuffer = DelayBuffer()
        buf.push(42, 1.0)

        self.assertIsNone(buf.latest_available(0.5))
        self.assertEqual(len(buf), 1)

        v = buf.latest_available(1.0)
        self.assertEqual(v, 42)
        self.assertEqual(len(buf), 0)
        self.assertIsNone(buf.latest_available(1.0))

    def test_returns_latest_available_and_discards_older_entries(self) -> None:
        buf: DelayBuffer = DelayBuffer()
        buf.push(1, 0.1)
        buf.push(2, 0.2)
        buf.push(3, 0.3)

        v = buf.latest_available(0.25)
        self.assertEqual(v, 2)
        self.assertEqual(len(buf), 1)

        self.assertIsNone(buf.latest_available(0.29))
        v3 = buf.latest_available(0.30)
        self.assertEqual(v3, 3)

    def test_out_of_order_push_is_ignored(self) -> None:
        buf: DelayBuffer = DelayBuffer()
        buf.push(1, 1.0)
        buf.push(2, 0.5)  # out of order

        self.assertEqual(len(buf), 1)
        v = buf.latest_available(1.0)
        self.assertEqual(v, 1)

    def test_clear_drops_pending_entries(self) -> None:
        buf: DelayBuffer = DelayBuffer()
        buf.push(1, 1.0)
        buf.push(2, 2.0)
        self.assertEqual(len(buf), 2)
        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertIsNone(buf.latest_available(5.0))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
