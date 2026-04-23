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

"""Entry point for the `mav-view` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data_model import TopicMap
from .plotter import FlightPlotter


def main(argv: list[str] | None = None) -> int:
    """Open an MCAP in matplotlib and return 0 on success."""
    parser = argparse.ArgumentParser(prog='mav-view', description=__doc__)
    parser.add_argument('mcap', type=Path, help='Path to the MCAP file.')
    parser.add_argument('--save', type=Path, default=None,
                        help='Save the figure instead of showing it.')
    parser.add_argument('--label', default=None,
                        help='Label for the figure title.')
    parser.add_argument('--pose-state', default=None)
    parser.add_argument('--twist-state', default=None)
    parser.add_argument('--odom-state', default=None)
    parser.add_argument('--pose-reference', default=None)
    parser.add_argument('--twist-reference', default=None)
    parser.add_argument('--thrust-command', default=None)
    parser.add_argument('--twist-command', default=None)
    args = parser.parse_args(argv)

    if not args.mcap.exists():
        print(f'error: file not found: {args.mcap}', file=sys.stderr)
        return 2

    mapping = TopicMap()
    for attr in ('pose_state', 'twist_state', 'odom_state',
                 'pose_reference', 'twist_reference',
                 'thrust_command', 'twist_command'):
        override = getattr(args, attr)
        if override:
            setattr(mapping, attr, override)

    plotter = FlightPlotter(args.mcap, label=args.label, mapping=mapping)
    if args.save is not None:
        plotter.save_figure(args.save)
        print(f'Saved figure to {args.save}')
    else:
        plotter.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
