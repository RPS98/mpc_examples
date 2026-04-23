"""Default rerun blueprint for a flight log.

Produces a two-column layout:

* Left column: the 3D world view (drone + reference + trajectory + waypoints).
* Right column: a grid of time-series views (position, velocity, angular
  velocity, orientation RPY, actuation and tracking error), plus an extras
  view that auto-populates from any scalar under ``scalars/extras/``.
"""

from __future__ import annotations

import rerun.blueprint as rrb

__all__ = ["default_blueprint"]


def default_blueprint() -> rrb.Blueprint:
    """Return the default rerun blueprint for the flight dashboard."""
    time_views = rrb.Grid(
        rrb.TimeSeriesView(
            name="Position [m]",
            origin="/scalars/position",
            contents=["+ $origin/**"],
        ),
        rrb.TimeSeriesView(
            name="Velocity [m/s]",
            origin="/scalars/velocity",
            contents=["+ $origin/**"],
        ),
        rrb.TimeSeriesView(
            name="Angular velocity [rad/s]",
            origin="/scalars/angular_velocity",
            contents=["+ $origin/**"],
        ),
        rrb.TimeSeriesView(
            name="Orientation RPY [rad]",
            origin="/scalars/rpy",
            contents=["+ $origin/**"],
        ),
        rrb.TimeSeriesView(
            name="Actuation",
            origin="/scalars/actuation",
            contents=["+ $origin/**"],
        ),
        rrb.TimeSeriesView(
            name="Tracking error",
            origin="/scalars/error",
            contents=["+ $origin/**"],
        ),
        rrb.TimeSeriesView(
            name="Motors [rad/s]",
            origin="/scalars/motors",
            contents=["+ $origin/**"],
        ),
        rrb.TimeSeriesView(
            name="Extras",
            origin="/scalars/extras",
            contents=["+ $origin/**"],
        ),
        grid_columns=2,
    )

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D world", origin="/world"),
            time_views,
            column_shares=[1.0, 1.4],
        ),
        collapse_panels=False,
    )
