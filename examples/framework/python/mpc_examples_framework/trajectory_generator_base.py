#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract reference generator interface consumed by :class:`WaypointsSimulator`.

Python mirror of
``examples/framework/include/framework/trajectory_generator_base.hpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from mavpy.model import State

from .example_config import ExampleConfig
from .types import ReferenceField, ReferenceSample


class ITrajectoryGenerator(ABC):
    """Abstract base for reference generators (waypoints, jerk-limited, ...).

    Adapts a waypoint list into a time-parameterised reference signal that the
    controller can consume. :meth:`evaluate` is called multiple times per outer
    step to populate the controller horizon.

    Implementations should leave unfilled fields at zero and advertise their
    capabilities via :meth:`provided_reference_fields`.
    """

    @abstractmethod
    def initialize(
        self,
        waypoints: List[np.ndarray],
        initial_state: State,
        example_cfg: ExampleConfig,
    ) -> None:
        """Configure the generator before the simulation starts."""

    @abstractmethod
    def update(self, t: float, state: State) -> None:
        """Notify the generator of the current time and state.

        Called once per outer control step, before :meth:`evaluate`.
        """

    @abstractmethod
    def evaluate(self, t: float) -> ReferenceSample:
        """Evaluate the reference at absolute time @p t."""

    @abstractmethod
    def is_finished(self, t: float) -> bool:
        """Whether the mission is finished at time @p t."""

    @abstractmethod
    def current_waypoint_index(self) -> int:
        """Zero-based index of the waypoint currently being tracked."""

    @abstractmethod
    def provided_reference_fields(self) -> ReferenceField:
        """Bitmask of :class:`ReferenceField` values produced by this generator."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable generator name."""
