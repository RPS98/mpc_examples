#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract controller interface consumed by :class:`WaypointsSimulator`.

Python mirror of ``examples/framework/include/framework/controller_base.hpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from mavpy.model import State

from .example_config import ExampleConfig
from .types import ControlCommand, ReferenceField, ReferenceSample


class IController(ABC):
    """Abstract base for outer-loop controllers (PID cascade, MPC, ...).

    The :class:`WaypointsSimulator` queries the horizon shape (N+1 samples at
    :meth:`reference_horizon_dt`) and calls :meth:`compute_command` once per
    control period. Ground-truth state comes from the simulator; references are
    sampled from an :class:`ITrajectoryGenerator` at times
    ``[t, t+dt, ..., t+N*dt]``.

    Timing model:
      - Outer control loop runs at :meth:`control_period`.
      - MPC controllers return ``N_horizon + 1`` samples; PID returns 1.

    Units:
      - Position [m], linear velocity [m/s], acceleration [m/s^2], yaw [rad].
      - Thrust [N] (collective), body rates [rad/s] (body frame).
    """

    @abstractmethod
    def initialize(self, initial_state: State, example_cfg: ExampleConfig) -> None:
        """Configure the controller before the simulation starts."""

    @abstractmethod
    def reference_horizon_size(self) -> int:
        """Number of reference samples required per :meth:`compute_command`."""

    @abstractmethod
    def reference_horizon_dt(self) -> float:
        """Time step between consecutive reference samples [s]."""

    @abstractmethod
    def control_period(self) -> float:
        """Outer control period [s] (e.g. ``mpc_dt`` or ``pid_dt``)."""

    @abstractmethod
    def compute_command(
        self,
        state: State,
        references: List[ReferenceSample],
    ) -> ControlCommand:
        """Compute the control command for the current step."""

    @abstractmethod
    def required_reference_fields(self) -> ReferenceField:
        """Bitmask of :class:`ReferenceField` values this controller requires."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable controller name."""

    @abstractmethod
    def last_solve_time_micros(self) -> float:
        """Wall-clock time of the last :meth:`compute_command` call [us]."""

    def last_velocity_command(self) -> np.ndarray:
        """Saturated velocity setpoint produced by the controller's outer
        loop, if it computes one (e.g. cascaded position PID). Default:
        zero vector.

        Used by :class:`WaypointsSimulator` to override the generator's
        velocity reference in the log when the controller's own intermediate
        setpoint is what the cascaded inner loop is actually tracking.
        """
        return np.zeros(3, dtype=float)

    def provides_velocity_command(self) -> bool:
        """True iff :meth:`last_velocity_command` returns a meaningful value
        that should override the generator's velocity reference in the log.
        Default: False (generator's velocity is logged as-is).
        """
        return False
