"""Plot dip and dip-direction measurements on a lower-hemisphere stereonet."""

import csv
import math
import re
import webbrowser
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle


Measurement = Tuple[float, float]
Vector3 = Tuple[float, float, float]
GRID_SIZE = 200
BANDWIDTH_DEGREES = 10.0
GRID_EXTENT = 1.04
DENSITY_COLORS = (
    "#073bff",
    "#00aaff",
    "#00d6a0",
    "#a8e600",
    "#ffe000",
    "#ff8100",
    "#df1d24",
)


def load_measurements(file_path: Path) -> List[Measurement]:
    """Read dip and dip direction from the first two CSV columns."""
    measurements = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{file_path}:{reader.line_num}: expected dip and dip direction"
                )

            try:
                dip = float(row[0].strip())
                dip_direction = float(row[1].strip())
            except ValueError:
                if not measurements and (
                    any(character.isalpha() for character in row[0])
                    or any(character.isalpha() for character in row[1])
                ):
                    continue
                raise ValueError(
                    f"{file_path}:{reader.line_num}: dip and dip direction must be numeric"
                ) from None

            if not math.isfinite(dip) or not 0.0 <= dip <= 90.0:
                raise ValueError(
                    f"{file_path}:{reader.line_num}: dip must be between 0 and 90 degrees"
                )
            if not math.isfinite(dip_direction):
                if dip == 0.0:
                    dip_direction = 0.0
                else:
                    raise ValueError(
                        f"{file_path}:{reader.line_num}: dip direction must be finite "
                        "for a non-horizontal plane"
                    )
            if not 0.0 <= dip_direction <= 360.0:
                raise ValueError(
                    f"{file_path}:{reader.line_num}: dip direction must be between 0 and 360 degrees"
                )
            measurements.append((dip, dip_direction % 360.0))

    if not measurements:
        raise ValueError(f"No valid orientation rows found in {file_path}")
    return measurements


def plane_pole(dip: float, dip_direction: float) -> Vector3:
    """Return a lower-hemisphere pole for a plane orientation."""
    if dip == 0.0:
        return (0.0, 0.0, 1.0)

    plunge = math.radians(90.0 - dip)
    trend = math.radians((dip_direction + 180.0) % 360.0)
    horizontal = math.cos(plunge)
    return (
        horizontal * math.sin(trend),
        horizontal * math.cos(trend),
        math.sin(plunge),
    )


def project_pole(pole: Vector3) -> Tuple[float, float]:
    """Project a lower-hemisphere pole to an equal-area unit disk."""
    east, north, down = pole
    scale = math.sqrt(1.0 + down)
    return east / scale, north / scale


def estimate_density(
    poles: np.ndarray,
    bandwidth_degrees: float = BANDWIDTH_DEGREES,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Estimate angular pole density on an equal-area stereonet grid."""
    coordinates = np.linspace(-GRID_EXTENT, GRID_EXTENT, grid_size)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)

    radius = np.hypot(grid_x, grid_y)
    scale_to_disk = np.maximum(radius, 1.0)
    disk_x = grid_x / scale_to_disk
    disk_y = grid_y / scale_to_disk
    down = 1.0 - disk_x**2 - disk_y**2
    horizontal_scale = np.sqrt(1.0 + down)
    grid_vectors = np.column_stack(
        (
            (disk_x * horizontal_scale).ravel(),
            (disk_y * horizontal_scale).ravel(),
            down.ravel(),
        )
    )

    bandwidth = math.radians(bandwidth_degrees)
    density = np.empty(len(grid_vectors), dtype=float)
    for start in range(0, len(grid_vectors), 256):
        stop = min(start + 256, len(grid_vectors))
        dots = grid_vectors[start:stop] @ poles.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        density[start:stop] = np.exp(
            -0.5 * (angles / bandwidth) ** 2
        ).mean(axis=1)

    return density.reshape(grid_size, grid_size)


def _output_path(input_file: Path, title: str) -> Path:
    safe_title = re.sub(r"[^\w-]+", "_", title, flags=re.UNICODE).strip("_")
    suffix = f"_{safe_title}" if safe_title else ""
    return input_file.with_name(f"{input_file.stem}{suffix}_stereoplot.png")


def plot_stereonet(
    measurements: Sequence[Measurement],
    input_file: Path,
    title: str,
) -> Path:
    """Render a pole-density stereonet and open the resulting image."""
    poles = np.asarray(
        [plane_pole(dip, direction) for dip, direction in measurements],
        dtype=float,
    )
    pole_points = np.asarray([project_pole(tuple(pole)) for pole in poles])
    density = estimate_density(poles)

    figure, axes = plt.subplots(figsize=(7.4, 7.4), facecolor="#f4f6fa")
    axes.set_facecolor("#f4f6fa")
    axes.imshow(
        density,
        origin="lower",
        extent=(-GRID_EXTENT, GRID_EXTENT, -GRID_EXTENT, GRID_EXTENT),
        interpolation="bilinear",
        cmap=LinearSegmentedColormap.from_list(
            "stereonet_density", DENSITY_COLORS, N=256
        ),
        zorder=1,
    )
    axes.scatter(
        pole_points[:, 0],
        pole_points[:, 1],
        s=10,
        color="white",
        edgecolors="#132b46",
        linewidths=0.35,
        alpha=0.65,
        zorder=2,
        rasterized=True,
    )

    outline = Circle(
        (0.0, 0.0),
        1.0,
        facecolor="none",
        edgecolor="#172b42",
        linewidth=1.6,
        zorder=3,
    )
    axes.add_patch(outline)
    for artist in axes.images + list(axes.collections):
        artist.set_clip_path(outline)

    for azimuth in range(0, 360, 10):
        angle = math.radians(azimuth)
        inner_radius = 1.0
        outer_radius = 1.035 if azimuth % 30 == 0 else 1.018
        axes.plot(
            [inner_radius * math.sin(angle), outer_radius * math.sin(angle)],
            [inner_radius * math.cos(angle), outer_radius * math.cos(angle)],
            color="#263b50",
            linewidth=0.8 if azimuth % 30 == 0 else 0.55,
            zorder=4,
            clip_on=False,
        )

    for label, azimuth in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        angle = math.radians(azimuth)
        axes.text(
            1.12 * math.sin(angle),
            1.12 * math.cos(angle),
            label,
            ha="center",
            va="center",
            color="#172b42",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

    axes.set_xlim(-1.2, 1.2)
    axes.set_ylim(-1.2, 1.2)
    axes.set_aspect("equal")
    axes.axis("off")
    figure.suptitle(title, fontsize=15, fontweight="semibold", color="#172b42", y=0.97)
    figure.text(
        0.5,
        0.925,
        f"Lower-hemisphere equal-area projection  |  {len(measurements)} planes",
        ha="center",
        fontsize=9.5,
        color="#53657a",
    )
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.04, top=0.89)

    output_file = _output_path(input_file, title)
    figure.savefig(output_file, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)

    if not webbrowser.open(output_file.resolve().as_uri(), new=2):
        raise RuntimeError(f"Could not open stereoplot image: {output_file}")
    print(f"Stereoplot saved to {output_file}")
    return output_file
