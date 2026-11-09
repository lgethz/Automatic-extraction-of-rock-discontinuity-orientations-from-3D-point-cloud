import argparse
import math
from pathlib import Path

import numpy as np


def octree_downsample(
    points: np.ndarray,
    *,
    voxel_size: float | None = None,
    max_depth: int | None = None,
) -> np.ndarray:
    """Return one centroid for each occupied leaf of a uniform-depth octree."""
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must be a non-empty array with shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")
    if voxel_size is None and max_depth is None:
        raise ValueError("Set voxel_size, max_depth, or both.")

    if voxel_size is not None and (
        not math.isfinite(voxel_size) or voxel_size <= 0
    ):
        raise ValueError("voxel_size must be a finite positive number.")

    if max_depth is not None:
        if isinstance(max_depth, bool) or not isinstance(max_depth, int):
            raise ValueError("max_depth must be a non-negative integer.")
        if max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")

    minimum = points.min(axis=0)
    extent = points.max(axis=0) - minimum
    root_width = float(extent.max())

    # If all points coincide, there is only one meaningful occupied leaf.
    if root_width == 0:
        return points[:1].copy()

    target_depth = 0
    if voxel_size is not None:
        target_depth = max(
            0,
            math.ceil(math.log2(root_width) - math.log2(voxel_size)),
        )

    if max_depth is not None:
        depth = min(target_depth, max_depth) if voxel_size is not None else max_depth
    else:
        depth = target_depth

    if depth > 30:
        raise ValueError(
            "Effective octree depth exceeds 30; increase voxel_size or "
            "set a lower max_depth."
        )

    cells_per_axis = 1 << depth
    leaf_width = root_width / cells_per_axis

    # The root is a cube anchored at the cloud's minimum XYZ coordinate.
    cell_indices = np.floor((points - minimum) / leaf_width).astype(np.int64)
    np.clip(cell_indices, 0, cells_per_axis - 1, out=cell_indices)

    _, inverse = np.unique(cell_indices, axis=0, return_inverse=True)
    leaf_count = int(inverse.max()) + 1

    sums = np.zeros((leaf_count, 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse, minlength=leaf_count)

    return sums / counts[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downsample an XYZ point cloud using uniform octree leaves."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--voxel-size",
        type=float,
        help="Target maximum leaf width, in the point cloud's coordinate units.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum subdivision depth; 0 puts the whole cloud in one leaf.",
    )
    args = parser.parse_args()

    if args.voxel_size is None and args.max_depth is None:
        parser.error("set --voxel-size, --max-depth, or both")

    source = np.loadtxt(args.input_file, dtype=np.float64)
    if source.ndim == 1:
        source = source.reshape(1, -1)
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError("Input file must contain at least three XYZ columns.")

    downsampled = octree_downsample(
        source[:, :3],
        voxel_size=args.voxel_size,
        max_depth=args.max_depth,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_file, downsampled, fmt="%.8f")
    print(f"Points: {len(source)} -> {len(downsampled)}")
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()