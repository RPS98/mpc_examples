"""mav_flight_viewer — rerun.io dashboard for mav_flight_mcap logs."""

from __future__ import annotations

from .dashboard import default_blueprint
from .viewer import MCAPViewer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MCAPViewer",
    "default_blueprint",
]
