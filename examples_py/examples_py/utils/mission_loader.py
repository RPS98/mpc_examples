"""Load a gate-based mission as a list of absolute waypoints.

This module is the single source of truth for any race-circuit mission
that follows the per-gate frame schema (a closed loop of gate centres
plus per-row modifiers that resolve to absolute world poses).

The two input files are:

  - gates_config.yaml: per-gate world pose, ``gateN: [x, y, z, yaw]``.
  - mission.yaml: ordered list of waypoints expressed in a per-gate
    frame plus a string of modifiers. Matches the schema parsed by
    upstream race-pilot nodes (see e.g.
    ``aerostack2/thirdparties/as2_race_pilot/src/controller_client.cpp``,
    ``parseWaypointsSection`` and ``convertWaypoints``).

Modifiers handled:

  - wo : adds ``params.wo`` to z (vertical offset, usually 0.0).
  - vt : tangent velocity, vector (1, 0, 0) in the gate frame, rotated to
         world. If the row has its own velocity vector, that one is used
         and ``vt`` is ignored.
  - vn : velocity points from this waypoint to the next (computed AFTER
         all positions are in world frame).
  - ny : no-yaw flag, kept verbatim on the output Waypoint for the
         downstream consumer (face-reference); does not affect xyz.
  - sy : custom face point; the row[4] vector (in gate frame) is rotated
         to world.
  - hs/av/ls : speed selectors, scale the unit velocity by params.hs /
               params.av / params.ls respectively (default: av).
  - gp, rt : not transformed (carried through in the modifiers string for
             downstream consumers that need them).

The output is a list of ``Waypoint`` objects with positions, velocities
and face_points expressed in the world frame. Order and length match
the ``fly_waypoints`` section of the mission.yaml.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import yaml


@dataclass
class GatePose:
    """Pose of a gate in world frame: position + yaw (rad)."""

    x: float
    y: float
    z: float
    yaw: float


@dataclass
class MissionParams:
    """`params:` block of mission.yaml."""

    num_laps: int = 1
    ls: float = 1.0
    av: float = 1.0
    hs: float = 1.0
    wo: float = 0.0


@dataclass
class Waypoint:
    """One waypoint, position/velocity/face_point in WORLD frame."""

    frame_id: str
    modifiers: str
    position: np.ndarray  # shape (3,)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    face_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    has_velocity: bool = False
    has_face_point: bool = False


@dataclass
class MissionData:
    """All data parsed from gates_config.yaml + mission.yaml."""

    gates: dict[str, GatePose]
    params: MissionParams
    takeoff: list[Waypoint]
    fly: list[Waypoint]


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _parse_gates(gates_yaml: dict) -> dict[str, GatePose]:
    raw = gates_yaml.get('gates_poses', {})
    out: dict[str, GatePose] = {}
    for name, vals in raw.items():
        if len(vals) != 4:
            raise ValueError(
                f"gate '{name}' must be [x, y, z, yaw], got {vals}"
            )
        out[name] = GatePose(x=float(vals[0]), y=float(vals[1]),
                             z=float(vals[2]), yaw=float(vals[3]))
    return out


def _parse_params(params_node: dict | None) -> MissionParams:
    if not params_node:
        return MissionParams()
    return MissionParams(
        num_laps=int(params_node.get('num_laps', 1)),
        ls=float(params_node.get('ls', 1.0)),
        av=float(params_node.get('av', 1.0)),
        hs=float(params_node.get('hs', 1.0)),
        wo=float(params_node.get('wo', 0.0)),
    )


def _parse_waypoints(section_node: list, section_name: str) -> list[Waypoint]:
    if not isinstance(section_node, list):
        raise TypeError(f"'{section_name}' must be a sequence (list)")

    out: list[Waypoint] = []
    for i, row in enumerate(section_node):
        if not isinstance(row, list) or len(row) < 3:
            raise ValueError(
                f"{section_name}[{i}] must be "
                "[frame_id, modifiers, [x,y,z], (opt [vx,vy,vz]), (opt [fpx,fpy,fpz])], "
                f"got {row}"
            )
        frame_id = str(row[0])
        modifiers = str(row[1])
        pos = row[2]
        if not isinstance(pos, list) or len(pos) != 3:
            raise ValueError(f"{section_name}[{i}]: position must be [x,y,z]")
        position = np.array([float(pos[0]), float(pos[1]), float(pos[2])])

        velocity = np.zeros(3)
        has_velocity = False
        if len(row) >= 4 and row[3] and isinstance(row[3], list) and len(row[3]) == 3:
            velocity = np.array([float(v) for v in row[3]])
            has_velocity = True

        face_point = np.zeros(3)
        has_face_point = False
        if len(row) >= 5 and row[4] and isinstance(row[4], list) and len(row[4]) == 3:
            face_point = np.array([float(v) for v in row[4]])
            has_face_point = True

        out.append(Waypoint(frame_id=frame_id, modifiers=modifiers,
                            position=position, velocity=velocity,
                            face_point=face_point,
                            has_velocity=has_velocity,
                            has_face_point=has_face_point))
    return out


def _gate_to_world(gate: GatePose, vec_local: np.ndarray, translate: bool) -> np.ndarray:
    """Apply a gate-frame → world-frame transform to a 3-vector.

    The gate frame is the world frame translated by (x, y, z) and rotated
    by `yaw` around the world Z axis (no roll/pitch). If `translate` is
    False the translation part is omitted (use for velocities).
    """
    c, s = math.cos(gate.yaw), math.sin(gate.yaw)
    x, y, z = vec_local
    rx = c * x - s * y
    ry = s * x + c * y
    rz = z
    if translate:
        return np.array([gate.x + rx, gate.y + ry, gate.z + rz])
    return np.array([rx, ry, rz])


def _resolve_to_world(wp: Waypoint, gates: dict[str, GatePose]) -> Waypoint:
    """Express a single waypoint in world coordinates (in-place copy)."""
    if wp.frame_id == 'drone0/map' or wp.frame_id == 'map':
        # already in world (a.k.a. the drone map frame)
        return Waypoint(
            frame_id='map', modifiers=wp.modifiers,
            position=wp.position.copy(),
            velocity=wp.velocity.copy(),
            face_point=wp.face_point.copy(),
            has_velocity=wp.has_velocity, has_face_point=wp.has_face_point,
        )

    if wp.frame_id not in gates:
        raise KeyError(
            f"waypoint frame_id '{wp.frame_id}' not present in gates_config.yaml"
        )

    gate = gates[wp.frame_id]
    out = Waypoint(
        frame_id='map', modifiers=wp.modifiers,
        position=_gate_to_world(gate, wp.position, translate=True),
        velocity=_gate_to_world(gate, wp.velocity, translate=False) if wp.has_velocity else np.zeros(3),
        face_point=_gate_to_world(gate, wp.face_point, translate=True) if wp.has_face_point else np.zeros(3),
        has_velocity=wp.has_velocity, has_face_point=wp.has_face_point,
    )
    return out


def _apply_velocity_modifiers(
    waypoints: list[Waypoint],
    params: MissionParams,
    gates: dict[str, GatePose],
    raw_section: list[Waypoint],
) -> None:
    """Resolve `vt` / `vn` modifiers and speed scaling in-place.

    Must be called after positions have been converted to world frame.
    """
    n = len(waypoints)
    for i, wp in enumerate(waypoints):
        if not wp.has_velocity:
            if 'vt' in wp.modifiers:
                # Unit vector (1,0,0) in the original gate frame → world
                raw_frame = raw_section[i].frame_id
                if raw_frame in gates:
                    vt = _gate_to_world(gates[raw_frame], np.array([1.0, 0.0, 0.0]),
                                        translate=False)
                else:
                    vt = np.array([1.0, 0.0, 0.0])
                wp.velocity = vt
            elif 'vn' in wp.modifiers and i < n - 1:
                direction = waypoints[i + 1].position - wp.position
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    raise ValueError(
                        f"waypoint {i} and next waypoint share the same position; "
                        "cannot resolve `vn` modifier"
                    )
                wp.velocity = direction

        # Speed scaling
        speed = params.av
        if 'hs' in wp.modifiers:
            speed = params.hs
        elif 'ls' in wp.modifiers:
            speed = params.ls

        nrm = np.linalg.norm(wp.velocity)
        if nrm > 1e-6:
            wp.velocity = wp.velocity / nrm * speed


def load(mission_yaml: str, gates_yaml: str) -> MissionData:
    """Public entry point: load gates + mission and return absolute waypoints.

    All paths must be absolute or relative to the current working
    directory. The output `fly` and `takeoff` waypoint lists have their
    position, velocity and face_point already expressed in world frame.
    """
    gates_raw = _load_yaml(gates_yaml)
    mission_raw = _load_yaml(mission_yaml)

    gates = _parse_gates(gates_raw)
    params = _parse_params(mission_raw.get('params'))
    takeoff_raw = _parse_waypoints(mission_raw.get('takeoff_waypoints', []),
                                   'takeoff_waypoints')
    fly_raw = _parse_waypoints(mission_raw.get('fly_waypoints', []),
                               'fly_waypoints')

    takeoff_world = [_resolve_to_world(wp, gates) for wp in takeoff_raw]
    fly_world = [_resolve_to_world(wp, gates) for wp in fly_raw]

    # 'wo' vertical offset
    for wp in takeoff_world + fly_world:
        if 'wo' in wp.modifiers:
            wp.position[2] += params.wo

    _apply_velocity_modifiers(takeoff_world, params, gates, takeoff_raw)
    _apply_velocity_modifiers(fly_world, params, gates, fly_raw)

    # Replicate the fly loop `num_laps` times. Consecutive duplicates
    # (typically the closing waypoint of lap N coinciding with the
    # opening waypoint of lap N+1) would break downstream polynomial
    # trajectory generators that assert segment_time > 0; deduplicate
    # them after the expansion so the multi-lap path is monotonic.
    fly_world = _expand_laps(fly_world, params.num_laps)

    return MissionData(gates=gates, params=params,
                       takeoff=takeoff_world, fly=fly_world)


def _waypoints_equal(a: Waypoint, b: Waypoint, tol: float = 1e-3) -> bool:
    """L-infinity check on XYZ; modifiers/velocities are ignored."""
    return bool(np.max(np.abs(a.position - b.position)) <= tol)


def _expand_laps(fly: list[Waypoint], num_laps: int) -> list[Waypoint]:
    """Concatenate ``fly`` ``num_laps`` times, dropping consecutive duplicates.

    ``num_laps <= 1`` returns a deduplicated copy of the single lap (some
    missions also feature back-to-back identical anchors within a single
    lap; dropping them is harmless).
    """
    if not fly:
        return []
    laps = max(int(num_laps), 1)
    expanded: list[Waypoint] = []
    for _ in range(laps):
        for wp in fly:
            if expanded and _waypoints_equal(expanded[-1], wp):
                continue
            expanded.append(wp)
    return expanded


def fly_positions(mission: MissionData) -> np.ndarray:
    """Convenience: stack `fly` positions into an (N, 3) ndarray."""
    return np.stack([wp.position for wp in mission.fly], axis=0)


def fly_velocities(mission: MissionData) -> np.ndarray:
    """Stack `fly` velocity vectors into an (N, 3) ndarray."""
    return np.stack([wp.velocity for wp in mission.fly], axis=0)


def fly_face_points(mission: MissionData) -> np.ndarray:
    """Stack `fly` face points into an (N, 3) ndarray (zeros if not set)."""
    return np.stack([wp.face_point for wp in mission.fly], axis=0)


def _default_paths() -> tuple[str, str]:
    """Resolve the demo mission YAMLs shipped with this package.

    The defaults point at ``configs/missions/demo_circuit.yaml`` and
    ``configs/missions/demo_gates.yaml`` inside the repo root so the
    package is standalone. Downstream consumers (e.g. a higher-level
    project that wraps this one) pass their own canonical paths via
    :func:`load`.
    """
    from pathlib import Path
    # mission_loader.py lives at
    #   examples_py/examples_py/utils/mission_loader.py
    # so parents[3] resolves to the mav_examples repo root.
    base = Path(__file__).resolve().parents[3] / 'configs' / 'missions'
    return (str(base / 'demo_circuit.yaml'),
            str(base / 'demo_gates.yaml'))


def load_default() -> MissionData:
    """Load using the canonical paths shipped with the repo."""
    mission_path, gates_path = _default_paths()
    return load(mission_path, gates_path)


__all__ = [
    'GatePose', 'MissionParams', 'Waypoint', 'MissionData',
    'load', 'load_default',
    'fly_positions', 'fly_velocities', 'fly_face_points',
]
