#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract reference generator interface consumed by :class:`WaypointsSimulator`.

Python mirror of
``examples/framework/include/framework/trajectory_generator_base.hpp``.

Point-to-point contract: the scheduler owns the waypoint list. Generators only
see the next waypoint through :meth:`on_waypoint_changed`, which is called
exactly once per waypoint transition. Between transitions, :meth:`update` and
:meth:`evaluate` query the already-computed trajectory without replanning.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from abc import ABC, abstractmethod

import numpy as np
from mavpy.model import State

from .example_config import ExampleConfig
from .types import ReferenceField, ReferenceSample


class ITrajectoryGenerator(ABC):
    """Abstract base for point-to-point reference generators."""

    @abstractmethod
    def initialize(
        self,
        initial_state: State,
        example_cfg: ExampleConfig,
    ) -> None:
        """Configure the generator once at the start of the mission."""

    @abstractmethod
    def on_waypoint_changed(
        self,
        next_waypoint: np.ndarray,
        state: State,
        t_start: float,
    ) -> None:
        """Plan a trajectory from the current state to ``next_waypoint``.

        Called once per waypoint transition (including the first one). This is
        where batch generators should run their solvers; simple setpoint-style
        generators may just record the new target.
        """

    @abstractmethod
    def update(self, t: float, state: State) -> None:
        """Lightweight per-step hook. Must not replan."""

    @abstractmethod
    def evaluate(self, t: float) -> ReferenceSample:
        """Evaluate the currently-planned trajectory at absolute time @p t."""

    @abstractmethod
    def provided_reference_fields(self) -> ReferenceField:
        """Bitmask of :class:`ReferenceField` values produced by this generator."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable generator name."""
