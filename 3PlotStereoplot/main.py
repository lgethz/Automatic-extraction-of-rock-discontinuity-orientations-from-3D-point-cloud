"""Load orientation results and open their stereoplot."""

import argparse
from pathlib import Path

from plotstereoplot import load_measurements, plot_stereonet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dip and dip-direction measurements on a stereonet."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="orientation results file (dip, dip direction, and optional plane ID)",
    )
    parser.add_argument(
        "--title",
        help="plot title (defaults to the input file's parent folder)",
    )
    args = parser.parse_args()

    input_file = args.input_file.resolve()
    measurements = load_measurements(input_file)
    plot_stereonet(
        measurements,
        input_file=input_file,
        title=args.title or input_file.parent.name,
    )


if __name__ == "__main__":
    main()