# Copyright 2025 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Matplotlib dashboard for ROS 2-compatible flight MCAPs."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._cursor import SyncedCursor
from .data_model import FlightData, Series, TopicMap


class FlightPlotter:
    """Render evolution-of-time subplots over the data extracted from an MCAP."""

    def __init__(
            self,
            mcap_path: Path | str,
            label: str | None = None,
            mapping: TopicMap | None = None) -> None:
        """Load the MCAP and decode every message for plotting."""
        self.mcap_path = Path(mcap_path)
        self.label = label or self.mcap_path.stem
        self.data = FlightData.load(self.mcap_path, mapping)
        self._t0 = self.data.time_origin_s
        self._cursor: SyncedCursor | None = None

    # --- Public API -----------------------------------------------------------

    def build_figure(self) -> Figure:
        """Build the full dashboard and return the Figure."""
        n_extras = len(self.data.extras)
        rows = 4 + (1 if n_extras else 0)
        fig = plt.figure(figsize=(15, 2.4 * rows), constrained_layout=True)
        fig.suptitle(self.label, fontsize=11)
        gs = fig.add_gridspec(rows, 3)

        ax_pos   = fig.add_subplot(gs[0, 0])
        ax_vel   = fig.add_subplot(gs[0, 1])
        ax_vmod  = fig.add_subplot(gs[0, 2])
        ax_ang   = fig.add_subplot(gs[1, 0])
        ax_cmd   = fig.add_subplot(gs[1, 1])
        ax_thr   = fig.add_subplot(gs[1, 2])
        ax_ref_p = fig.add_subplot(gs[2, 0])
        ax_ref_v = fig.add_subplot(gs[2, 1])
        ax_err   = fig.add_subplot(gs[2, 2])
        ax_top   = fig.add_subplot(gs[3, :])

        self.plot_position(ax_pos)
        self.plot_velocity(ax_vel)
        self.plot_velocity_norm(ax_vmod)
        self.plot_angular_velocity(ax_ang)
        self.plot_command_angular_velocity(ax_cmd)
        self.plot_thrust(ax_thr)
        self.plot_reference_position(ax_ref_p)
        self.plot_reference_speed(ax_ref_v)
        self.plot_position_error(ax_err)
        self.plot_topics_overview(ax_top)

        time_axes: List[Axes] = [
            ax_pos, ax_vel, ax_vmod, ax_ang, ax_cmd, ax_thr,
            ax_ref_p, ax_ref_v, ax_err,
        ]
        if n_extras:
            ax_extras = fig.add_subplot(gs[4, :])
            self.plot_extras(ax_extras)
            time_axes.append(ax_extras)

        self._cursor = SyncedCursor(fig, time_axes)
        return fig

    def show(self) -> None:
        """Build and show the figure blockingly."""
        self.build_figure()
        plt.show()

    def save_figure(self, path: Path | str, dpi: int = 150) -> None:
        """Build and save the figure to `path`."""
        fig = self.build_figure()
        fig.savefig(path, dpi=dpi, bbox_inches='tight')

    # --- Subplots -------------------------------------------------------------

    def _plot_role(self, ax: Axes, roles: List[str], title: str, ylabel: str) -> None:
        """Plot every series listed in `roles` onto `ax`."""
        drawn = False
        for role in roles:
            for s in self.data.series.get(role, []):
                ax.plot(s.t - self._t0, s.y, label=s.label, linewidth=1.2)
                drawn = True
        ax.set_title(title)
        ax.set_xlabel('time [s]')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if drawn:
            ax.legend(loc='best', fontsize=8)

    def plot_position(self, ax: Axes) -> None:
        """State position (x, y, z)."""
        self._plot_role(ax, ['pose_state_x', 'pose_state_y', 'pose_state_z'],
                        'Position (state)', 'position [m]')

    def plot_velocity(self, ax: Axes) -> None:
        """State linear velocity components (vx, vy, vz)."""
        self._plot_role(ax, ['twist_state_vx', 'twist_state_vy', 'twist_state_vz'],
                        'Linear velocity (state)', 'velocity [m/s]')

    def plot_velocity_norm(self, ax: Axes) -> None:
        """Modulus of linear velocity (|v|)."""
        self._plot_role(ax, ['twist_state_vnorm'],
                        'Linear speed modulus', 'speed [m/s]')

    def plot_angular_velocity(self, ax: Axes) -> None:
        """State angular velocity in body frame."""
        self._plot_role(ax, ['twist_state_wx', 'twist_state_wy', 'twist_state_wz'],
                        'Angular velocity (body)', r'$\omega$ [rad/s]')

    def plot_command_angular_velocity(self, ax: Axes) -> None:
        """Commanded body angular velocity."""
        self._plot_role(ax, ['twist_cmd_wx', 'twist_cmd_wy', 'twist_cmd_wz'],
                        'Commanded angular velocity', r'$\omega_{cmd}$ [rad/s]')

    def plot_thrust(self, ax: Axes) -> None:
        """Thrust command."""
        self._plot_role(ax, ['thrust'], 'Thrust command', 'thrust [N]')

    def plot_reference_position(self, ax: Axes) -> None:
        """Position reference (x_ref, y_ref, z_ref)."""
        self._plot_role(ax, ['pose_ref_x', 'pose_ref_y', 'pose_ref_z'],
                        'Position reference', 'position [m]')

    def plot_reference_speed(self, ax: Axes) -> None:
        """Reference maximum speed modulus."""
        self._plot_role(ax, ['twist_ref_vmax'], 'Reference max speed', '|v|_max [m/s]')

    def plot_position_error(self, ax: Axes) -> None:
        """Norm of the position tracking error ||p - p_ref|| (nearest-time alignment)."""
        ps_x = self.data.series.get('pose_state_x', [])
        pr_x = self.data.series.get('pose_ref_x', [])
        if not ps_x or not pr_x:
            ax.set_title('Position error (n/a)')
            ax.grid(True, alpha=0.3)
            return
        sx = ps_x[0]
        sy = self.data.series['pose_state_y'][0]
        sz = self.data.series['pose_state_z'][0]
        rx = pr_x[0]
        ry = self.data.series['pose_ref_y'][0]
        rz = self.data.series['pose_ref_z'][0]

        # Align reference samples to state samples by nearest-time index.
        import numpy as np
        idx = np.searchsorted(rx.t, sx.t).clip(0, len(rx.t) - 1)
        ex = sx.y - rx.y[idx]
        ey = sy.y - ry.y[idx]
        ez = sz.y - rz.y[idx]
        err = np.sqrt(ex * ex + ey * ey + ez * ez)
        ax.plot(sx.t - self._t0, err, label='||p - p_ref||', color='black')
        ax.set_title('Position tracking error')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('error [m]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    def plot_topics_overview(self, ax: Axes) -> None:
        """Text panel summarising every topic and schema in the MCAP."""
        ax.axis('off')
        lines = [f'{self.mcap_path.name}', '']
        for topic, schema in sorted(self.data.topics.items()):
            lines.append(f'  {topic}  ->  {schema}')
        ax.text(0.01, 0.99, '\n'.join(lines),
                transform=ax.transAxes, va='top', ha='left',
                family='monospace', fontsize=8)

    def plot_extras(self, ax: Axes) -> None:
        """Plot every decoded extra scalar time-series on a shared axis."""
        drawn = False
        for topic, s in self.data.extras.items():
            ax.plot(s.t - self._t0, s.y, label=topic, linewidth=1.0)
            drawn = True
        ax.set_title('Extras')
        ax.set_xlabel('time [s]')
        ax.grid(True, alpha=0.3)
        if drawn:
            ax.legend(loc='best', fontsize=7, ncol=2)
