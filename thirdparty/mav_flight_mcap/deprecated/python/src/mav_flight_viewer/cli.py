"""``mav-view`` CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .viewer import MCAPViewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mav-view",
        description="Stream an mav_flight_mcap .mcap log into the rerun.io viewer.",
    )
    parser.add_argument("mcap", type=Path, help="Path to a .mcap file")
    parser.add_argument(
        "--label",
        default=None,
        help="Application id shown in the rerun viewer (defaults to file stem)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the recording to this .rrd file instead of opening the viewer",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn the viewer GUI; connect to an already-running rerun",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    viewer = MCAPViewer(args.mcap, label=args.label)
    if args.save is not None:
        out = viewer.save(args.save)
        print(f"Saved {out}")
    else:
        viewer.show(spawn=not args.no_spawn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
