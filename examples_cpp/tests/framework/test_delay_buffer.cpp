// Copyright 2025 Universidad Politécnica de Madrid
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "framework/delay_buffer.hpp"

using mpc_examples::framework::DelayBuffer;

TEST(DelayBufferTest, EmptyReturnsNullopt) {
  DelayBuffer<int> buf;
  EXPECT_FALSE(buf.latestAvailable(0.0).has_value());
  EXPECT_EQ(buf.size(), 0u);
}

TEST(DelayBufferTest, DeliversPayloadOnceItsAvailableAtElapses) {
  DelayBuffer<int> buf;
  buf.push(42, 1.0);

  // Before the available_at, the payload is invisible.
  EXPECT_FALSE(buf.latestAvailable(0.5).has_value());
  EXPECT_EQ(buf.size(), 1u);

  // At available_at it becomes visible (and is consumed).
  auto v = buf.latestAvailable(1.0);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v.value(), 42);
  EXPECT_EQ(buf.size(), 0u);

  // Subsequent queries at the same time return nothing (buffer drained).
  EXPECT_FALSE(buf.latestAvailable(1.0).has_value());
}

TEST(DelayBufferTest, ReturnsLatestAvailableAndDiscardsOlderEntries) {
  DelayBuffer<int> buf;
  buf.push(1, 0.1);
  buf.push(2, 0.2);
  buf.push(3, 0.3);

  // At t = 0.25 the buffer should hand back payload "2" (the latest with
  // available_at <= t) and drop "1" + "2".
  auto v = buf.latestAvailable(0.25);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v.value(), 2);
  EXPECT_EQ(buf.size(), 1u);

  // The remaining entry (3) becomes visible later.
  EXPECT_FALSE(buf.latestAvailable(0.29).has_value());
  auto v3 = buf.latestAvailable(0.30);
  ASSERT_TRUE(v3.has_value());
  EXPECT_EQ(v3.value(), 3);
}

TEST(DelayBufferTest, OutOfOrderPushIsIgnored) {
  DelayBuffer<int> buf;
  buf.push(1, 1.0);
  buf.push(2, 0.5);  // out of order: must be a no-op

  EXPECT_EQ(buf.size(), 1u);
  auto v = buf.latestAvailable(1.0);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v.value(), 1);
}

TEST(DelayBufferTest, ClearDropsPendingEntries) {
  DelayBuffer<int> buf;
  buf.push(1, 1.0);
  buf.push(2, 2.0);
  EXPECT_EQ(buf.size(), 2u);

  buf.clear();
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_FALSE(buf.latestAvailable(5.0).has_value());
}
