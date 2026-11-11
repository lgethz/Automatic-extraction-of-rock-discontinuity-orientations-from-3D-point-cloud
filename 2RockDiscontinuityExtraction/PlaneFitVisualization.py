"""Visualize fitted plane segments and report their point-to-plane residuals."""

import colorsys
import numpy as np
import open3d as o3d

from utils import create_coordinate_frame


def distinct_colors(count):
    """Generate a deterministic, non-repeating color for every plane."""
    colors = []
    used = set()
    hue_step = 0.618033988749895

    for index in range(count):
        hue = (index * hue_step) % 1.0
        color = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
        while color in used:
            hue = (hue + 0.001) % 1.0
            color = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
        colors.append(color)
        used.add(color)

    return colors


def create_plane_grid(points, plane_model, divisions=4):
    """Create an in-plane wire grid spanning the projected segment points."""
    points = np.asarray(points, dtype=float)
    plane_model = np.asarray(plane_model, dtype=float)
    normal = plane_model[:3]
    normal_length = np.linalg.norm(normal)
    if normal_length <= np.finfo(float).eps:
        raise ValueError("Cannot visualize a plane with a zero-length normal.")

    unit_normal = normal / normal_length
    reference = np.eye(3)[np.argmin(np.abs(unit_normal))]
    axis_u = np.cross(unit_normal, reference)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(unit_normal, axis_u)

    center = points.mean(axis=0)
    center -= (np.dot(normal, center) + plane_model[3]) / np.dot(normal, normal) * normal
    offsets = points - center
    coordinates_u = offsets @ axis_u
    coordinates_v = offsets @ axis_v
    minimum_u, maximum_u = coordinates_u.min(), coordinates_u.max()
    minimum_v, maximum_v = coordinates_v.min(), coordinates_v.max()

    margin = 0.04 * max(maximum_u - minimum_u, maximum_v - minimum_v)
    minimum_u -= margin
    maximum_u += margin
    minimum_v -= margin
    maximum_v += margin

    vertices = []
    lines = []

    def add_line(start, end):
        first = len(vertices)
        vertices.extend((start, end))
        lines.append((first, first + 1))

    for fraction in np.linspace(0.0, 1.0, divisions + 1):
        coordinate_u = minimum_u + fraction * (maximum_u - minimum_u)
        coordinate_v = minimum_v + fraction * (maximum_v - minimum_v)
        add_line(
            center + coordinate_u * axis_u + minimum_v * axis_v,
            center + coordinate_u * axis_u + maximum_v * axis_v,
        )
        add_line(
            center + minimum_u * axis_u + coordinate_v * axis_v,
            center + maximum_u * axis_u + coordinate_v * axis_v,
        )

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(vertices))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    return grid


def show_fitted_planes(planes, parameters, logger):
    """Show each fitted plane grid with its segment points in Open3D."""
    planes = np.asarray(planes, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    if planes.ndim != 2 or planes.shape[1] < 7:
        raise ValueError("Expected plane points with segment IDs in the last column.")

    segment_ids = np.unique(planes[:, -1])
    colors = distinct_colors(len(segment_ids))
    geometries = []
    point_count = 0

    for parameter in parameters:
        normal_a, normal_b, normal_c, offset, plane_index = parameter
        segment_index = int(plane_index)
        if not 0 <= segment_index < len(segment_ids):
            raise ValueError(
                f"Fitted plane references an invalid segment index: {segment_index}"
            )

        segment_id = segment_ids[segment_index]
        segment = planes[planes[:, -1] == segment_id, :3]
        if not len(segment):
            raise ValueError(f"No points found for fitted segment {segment_id}.")

        color = colors[segment_index]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(segment)
        point_cloud.paint_uniform_color(color)

        plane_grid = create_plane_grid(
            segment,
            (normal_a, normal_b, normal_c, offset),
        )
        plane_grid.paint_uniform_color(color)
        geometries.extend((point_cloud, plane_grid))
        point_count += len(segment)

        distances = np.abs(segment @ np.asarray((normal_a, normal_b, normal_c)) + offset)
        distances /= np.linalg.norm((normal_a, normal_b, normal_c))
        logger.info(
            "Fitted plane %d (segment %s): %d points; point-to-plane "
            "residual RMS=%.6g, 95th percentile=%.6g; grid color RGB=%s.",
            segment_index,
            segment_id,
            len(segment),
            float(np.sqrt(np.mean(distances**2))),
            float(np.percentile(distances, 95)),
            tuple(round(channel, 2) for channel in color),
        )

    if not geometries:
        logger.warning("No fitted planes are available to visualize.")
        return

    all_points = planes[:, :3]
    geometries.append(create_coordinate_frame(all_points))
    logger.info(
        "Showing %d fitted planes covering %d points. Close the Open3D window "
        "to continue to the stereoplot.",
        len(geometries) // 2,
        point_count,
    )
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Fitted planes and segment points",
        width=1280,
        height=800,
    )
