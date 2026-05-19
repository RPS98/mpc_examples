#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers that derive the MPC reference speed from the YAML's soft ‖v‖²
upper bound (``constraints.uh[0]``) and apply it back to the solver.

Mirrors :file:`examples_cpp/include/controllers/mpc_speed_utils.hpp`. The
YAML ``uh`` encodes ``max_speed²`` in m²/s²; ``max_vel_percentage ∈ (0, 1]``
is the safety knob that caps ``v_ref`` below that bound — matches the
aerostack2 ``as2_position_mpc_plugin`` convention.
"""

__authors__ = 'Rafael Perez-Segui'
__license__ = 'BSD-3-Clause'

import math

import numpy as np


def read_uh_default(mpc, who: str) -> float:
    """Read ``constraints.uh[0]`` from a YAML-configured MPC.

    Returns 0.0 when the OCP has no ``‖v‖²`` constraint compiled in
    (``nh_size == 0``) — callers can use 0 as a sentinel for "skip the
    v_ref derivation". Raises ``ValueError`` when ``nh_size > 0`` but the
    bound is non-positive (the convention is that the YAML encodes
    ``max_speed²`` there, not a placeholder).
    """
    nb = mpc.get_nonlinear_constraint_bounds()
    if int(nb.nh_size) == 0:
        return 0.0
    uh = nb.get_uh()
    uh0 = float(uh[0])
    if uh0 <= 0.0:
        raise ValueError(
            f'{who}: constraints.uh[0] must be > 0 in the YAML '
            f'(it encodes max_speed²). Got uh={uh0}.')
    return uh0


def derive_v_ref(uh_default: float, max_vel_percentage: float, who: str) -> float:
    """Return ``sqrt(uh_default) * max_vel_percentage``.

    Returns 0.0 when ``uh_default`` is 0 (no constraint available). Raises
    ``ValueError`` if ``max_vel_percentage`` is outside ``(0, 1]``.
    """
    if max_vel_percentage <= 0.0 or max_vel_percentage > 1.0:
        raise ValueError(f'{who}: max_vel_percentage must be in (0, 1].')
    return math.sqrt(uh_default) * max_vel_percentage


def update_speed_constraint(mpc, v_ref: float) -> None:
    """Set the solver's runtime ``uh = v_ref²`` so the soft penalty matches
    the reference speed used by the ramp / velocity feed-forward. No-op when
    the OCP has no ``‖v‖²`` constraint."""
    nb = mpc.get_nonlinear_constraint_bounds()
    if int(nb.nh_size) <= 0:
        return
    nb.set_uh(np.array([v_ref * v_ref], dtype=float))
    mpc.update_nonlinear_constraint_bounds()
