'''
@Author: Mingming Ren 任铭铭 
@Date: 2024-11-26 18:49:12 
@Last Modified by:   Mingming Ren ren 
@Last Modified time: 2025-07-06 18:49:12 
'''
import open3d as o3d
import numpy as np

def fit_plane_ransac(cloud, distance_threshold=None):
    point_count = len(cloud.points)
    if point_count < 3:
        raise ValueError("At least 3 points are required to fit a plane.")

    if distance_threshold is None:
        points = np.asarray(cloud.points)
        distance_threshold = float(np.max(np.ptp(points, axis=0))) * 0.01
        if distance_threshold <= 0:
            raise ValueError("Cannot fit a plane to a zero-extent point cloud.")

    ransac_n = min(100, point_count)
    plane_model, inliers = cloud.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=1000)
    return inliers, plane_model

def calculate_dip(a, b, c):
    dip_rad = np.arctan(np.sqrt(a**2 + b**2) / abs(c))
    dip_deg = np.degrees(dip_rad)
    return dip_deg

def calculate_dip_direction(a, b, c):
    normal = np.asarray([a, b, c], dtype=float)
    normal_length = np.linalg.norm(normal)
    if normal_length == 0:
        raise ValueError("Cannot calculate dip direction from a zero normal.")

    a, b, c = normal / normal_length
    tolerance = 1e-12
    if np.hypot(a, b) <= tolerance:
        return float("nan")

    if abs(c) <= tolerance:
        if b < -tolerance or (abs(b) <= tolerance and a < 0):
            a, b = -a, -b
    elif c < 0:
        a, b = -a, -b

    dip_direction = np.degrees(np.arctan2(a, b))
    return float(dip_direction % 360)

def fit_plane(planes, logger):
    idxs = np.unique(planes[:,-1])
    parameters = []
    orientations = []
    for i in range(len(idxs)):
        item = idxs[i]
        cond = planes[:,-1]==item
        points = planes[cond]
        points = points[:, 0:3]
        pcl_tmp = o3d.geometry.PointCloud()
        pcl_tmp.points = o3d.utility.Vector3dVector(points)
        distance_threshold = float(np.max(np.ptp(points, axis=0))) * 0.01
        logger.info(
            "Fitting plane %s from %d points with RANSAC distance threshold %.6g "
            "(1%% of maximum axis span; ransac_n=%d).",
            item,
            len(points),
            distance_threshold,
            min(100, len(points)),
        )
        _, plane_model = fit_plane_ransac(pcl_tmp, distance_threshold)
        plane_model = np.asarray(plane_model, dtype=float)
        if (
            plane_model.shape != (4,)
            or not np.isfinite(plane_model).all()
            or np.linalg.norm(plane_model[:3]) <= np.finfo(float).eps
        ):
            logger.warning(
                "Skipping plane %s: RANSAC returned an invalid plane model.",
                item,
            )
            continue

        a, b, c, d = plane_model
        parameters.append([a,b,c,d,i])
        dip_angle = calculate_dip(a, b, c)
        dip_direction = calculate_dip_direction(a, b, c)
        orientations.append([dip_angle, dip_direction, i])
    return parameters, orientations
