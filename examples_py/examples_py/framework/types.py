#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Common data types used across the integrated examples framework.

Python mirror of ``examples/framework/include/framework/types.hpp``.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Iterable

import numpy as np


class ReferenceField(IntFlag):
    """Fields that a reference sample may carry.

    Trajectory generators advertise which fields they produce via
    :meth:`ITrajectoryGenerator.provided_reference_fields`; controllers advertise
    which fields they need via :meth:`IController.required_reference_fields`.
    The :class:`WaypointsSimulator` compares both masks at construction and
    warns if the generator does not cover all required fields.
    """

    NONE = 0
    POSITION = 1 << 0
    VELOCITY = 1 << 1
    ACCELERATION = 1 << 2


def make_mask(fields: Iterable[ReferenceField]) -> ReferenceField:
    """Build a combined :class:`ReferenceField` mask from an iterable."""
    mask = ReferenceField.NONE
    for f in fields:
        mask |= f
    return mask


def has_field(mask: ReferenceField, field_value: ReferenceField) -> bool:
    """Check whether @p mask contains @p field_value."""
    return bool(mask & field_value)


def _zero_vec3() -> np.ndarray:
    return np.zeros(3, dtype=float)


@dataclass
class ReferenceSample:
    """Reference sample at a single point in time along the prediction horizon.

    All fields are expressed in the world frame. Position [m], velocity [m/s]
    (0 when the generator only produces positions), acceleration [m/s^2]
    (0 when the generator only produces positions), yaw [rad] (absolute; path-
    facing is handled by the generator).
    """

    position: np.ndarray = field(default_factory=_zero_vec3)
    velocity: np.ndarray = field(default_factory=_zero_vec3)
    acceleration: np.ndarray = field(default_factory=_zero_vec3)
    yaw: float = 0.0


@dataclass
class ControlCommand:
    """Control command produced by a controller at the outer loop rate.

    ``thrust_n``: collective thrust [N].
    ``angular_rate``: body rates [rad/s] in the body frame.
    """

    thrust_n: float = 0.0
    angular_rate: np.ndarray = field(default_factory=_zero_vec3)
