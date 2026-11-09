'''
@Author: Mingming Ren 任铭铭 
@Date: 2024-11-26 18:49:12 
@Last Modified by:   Mingming Ren ren 
@Last Modified time: 2025-07-06 18:49:12 
'''
import os, sys
import argparse
import time
import numpy as np
import torch
from utils import *
import open3d as o3d
import random

from ClusteringAnalysis import iukm
from DBSCAN import idbscan
from FitPlane import calculate_dip, calculate_dip_direction, fit_plane

MAXPOINTS = 70000

BATCH_SIZE = 256

def parase_arguments():
    parser = argparse.ArgumentParser()
    cur_path = os.getcwd()

    ## Parameters
    parser.add_argument('--dataset_root', type=str, default=os.path.join(cur_path, 'dataset'))
    parser.add_argument('--data_set', type=str, default='Ouray')
    parser.add_argument('--res_root', type=str, default=os.path.join(cur_path, 'results'))
    parser.add_argument(
        '--log_root',
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'),
    )
    parser.add_argument('--data_suffix', type=str, default='.xyz')
    parser.add_argument('--file_suffix', type=str, default='.txt')


    parser.add_argument('--angle', type=float, default=38)
    parser.add_argument('--iter', type=int, default=5)
    parser.add_argument('--min_clusters', type=int, default=100)

    parser.add_argument('--show_cluster_res', type=eval, default=False, choices=[True, False])
    parser.add_argument('--save_cluster_res', type=eval, default=True, choices=[True, False])
    parser.add_argument('--show_planes_res', type=eval, default=False, choices=[True, False])
    parser.add_argument('--save_planes_res', type=eval, default=True, choices=[True, False])
    parser.add_argument('--save_orientation_res', type=eval, default=True, choices=[True, False])
    

    args = parser.parse_args()
    return args

def clustering_analysis(logger):
    # load data
    points, normals, n = load_data(os.path.join(args.dataset_root, args.data_set,
                                                 args.data_set)+args.data_suffix)
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_rows = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & np.isfinite(normal_lengths)
        & (normal_lengths > np.finfo(float).eps)
    )
    invalid_count = int((~valid_rows).sum())
    if invalid_count:
        logger.warning(
            "Excluding %d points with invalid coordinates or zero/invalid normals.",
            invalid_count,
        )
    points = points[valid_rows]
    normals = normals[valid_rows]
    n = len(points)
    if n == 0:
        raise ValueError(
            "No points with valid coordinates and normals remain for extraction."
        )
    normals = normal_unitilize(normals)
    logger.info("Loaded %d points and normals for clustering.", n)
    idxs = np.arange(0, n)
    alpha = 0.7
    size_n = int(n * alpha)
    while size_n > MAXPOINTS:
        alpha -= 0.03
        size_n = int(n*alpha)
    idxs_random = np.random.choice(idxs, size=size_n, replace=False)
    selected_points = points[idxs_random, :]
    selected_points_normals = normals[idxs_random, :]
    logger.info(
        "Using a random clustering sample of %d/%d points (%.1f%%; maximum %d).",
        size_n,
        n,
        100.0 * size_n / n if n else 0.0,
        MAXPOINTS,
    )
    s_time = time.time()
    clazz = iukm(selected_points, selected_points_normals, idxs_random, logger, args.angle, args.iter)
    while len(clazz) <= 1:
        clazz = iukm(selected_points, selected_points_normals, idxs_random, logger, args.angle, args.iter)
        print(len(clazz))
    list_of_class= [[] for _ in range(len(clazz))]
    result = torch.cat(clazz, dim=0)
    tensor_points = torch.from_numpy(points)
    normals = torch.from_numpy(normals)
    start = 0
    end = min(start + BATCH_SIZE, len(normals))
    while start < len(normals):
        batch = normals[start:end, :]
        normal_similarity = torch.sum(
            batch.unsqueeze(1) * result.unsqueeze(0),
            dim=2,
        )
        rr = torch.arccos(torch.clamp(torch.abs(normal_similarity), max=1.0))
        rrr = torch.argmin(rr, dim=1)
        for i in range(rrr.shape[0]):
            list_of_class[rrr[i]].append(i+start)
        start = end
        end = min(start + BATCH_SIZE, len(normals))
    for cluster_idx, point_indices in enumerate(list_of_class):
        logger.info(
            "Orientation group %d assigned %d points.",
            cluster_idx,
            len(point_indices),
        )
    if args.save_orientation_res:
        group_normals = result.detach().cpu().numpy().reshape(-1, 3)
        if len(group_normals) != len(list_of_class):
            raise ValueError(
                "The number of orientation-group normals does not match "
                "the number of assigned groups."
            )
        orientation_groups = []
        point_normals = normals.cpu().numpy()
        for group_idx, point_indices in enumerate(list_of_class):
            if not point_indices:
                raise ValueError(
                    "Orientation group %d has no assigned points; "
                    "cannot calculate its mean orientation." % group_idx
                )
            group_point_normals = point_normals[point_indices]
            orientation_scatter = group_point_normals.T @ group_point_normals
            _, eigenvectors = np.linalg.eigh(orientation_scatter)
            a, b, c = eigenvectors[:, -1]
            orientation_groups.append([
                calculate_dip(a, b, c),
                calculate_dip_direction(a, b, c),
                group_idx,
                len(point_indices),
            ])
        res_dir = os.path.join(args.res_root, args.data_set, str(timestamp))
        os.makedirs(res_dir, exist_ok=True)
        file_path = os.path.join(
            res_dir,
            'orientation_groups' + args.file_suffix,
        )
        np.savetxt(
            file_path,
            orientation_groups,
            delimiter=',',
            fmt='%.3f',
            header=(
                'group_dip_degrees, group_dip_direction_degrees, '
                'orientation_group_id, point_count'
            ),
        )
        logger.info(
            'The orientation-group summary is saved to %s.',
            file_path,
        )
    end_time = time.time()
    logger.info(
        "Clustering analysis  is completed! Elasped time is {}s, and the number of cluster is {}".format(end_time-s_time, len(clazz)))

    # 整合结果
    total_res = []
    for i in range(len(list_of_class)):
        cla = list_of_class[i]
        points_ = tensor_points[cla]
        normals_ = normals[cla]
        label_ = torch.full((len(cla), 1), i)
        tmp = torch.cat((points_, normals_, label_), dim=1)
        total_res.append(tmp)

    if args.save_cluster_res:
        res_dir = os.path.join(args.res_root, args.data_set, str(timestamp))
        os.makedirs(res_dir, exist_ok=True)
        file_name = str(len(clazz))+'_' + 'clustering_analysis' + args.file_suffix
        file_path = os.path.join(res_dir, file_name)
        save_result(
            file_path,
            total_res,
            'x, y, z, nx, ny, nz, orientation_group_id',
        )
        logger.info('The result is saved to {}'.format(file_path))

    if args.show_cluster_res:
        colors = []
        for k,v in cnames.items():
            rgb = hex_to_rgb(v)
            colors.append(rgb)
            random.shuffle(colors)
        pcls = []
        for j in range(len(list_of_class)):
            seg = list_of_class[j]
            pcl_tmp = o3d.geometry.PointCloud()
            pcl_tmp.points = o3d.utility.Vector3dVector(points[seg,:])
            pcl_tmp.paint_uniform_color(colors[j])
            pcls.append(pcl_tmp)
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for pcd in pcls:
            vis.add_geometry(pcd)
        vis.add_geometry(create_coordinate_frame(points))
        render_opt = vis.get_render_option()
        render_opt.point_size = 2  # 设置点的大小
        set_plot_view(vis, points)
        vis.run()
    return total_res

def plane_segmentation(clusters, logger):
    logger.info("Starting plane segmentation...")
    start_time = time.time()
    all_planes = idbscan(clusters, args.min_clusters, logger)
    end_time = time.time()
    logger.info("The eclapsed time for plane segmentation is {}s".format(end_time-start_time))
    all_planes = np.array(all_planes)

    if args.save_planes_res:
        res_dir = os.path.join(args.res_root, args.data_set, str(timestamp))
        os.makedirs(res_dir, exist_ok=True)
        file_name = 'all_planes' + args.file_suffix
        file_path = os.path.join(res_dir, file_name)
        np.savetxt(
            file_path,
            all_planes,
            delimiter=',',
            fmt='%.3f',
            header=(
                'x, y, z, nx, ny, nz, orientation_group_id, '
                'plane_segment_id'
            ),
        )
        logger.info('The result is saved to {}'.format(file_path))
    
    if args.show_planes_res:
        joints = np.unique(all_planes[:, -1])
        colors = [hex_to_rgb(value) for value in cnames.values()]
        orientation_groups = np.unique(all_planes[:, -2])
        group_colors = {
            group_id: colors[index % len(colors)]
            for index, group_id in enumerate(orientation_groups)
        }
        pcls = []
        for item in joints:
            condition = all_planes[:, -1] == item
            J_cur = all_planes[condition]
            X = J_cur[:, 0:3]
            group_id = J_cur[0, -2]
            color = group_colors[group_id]
            pcl_tmp = o3d.geometry.PointCloud()
            pcl_tmp.points = o3d.utility.Vector3dVector(X)
            pcl_tmp.paint_uniform_color(color)
            pcls.append(pcl_tmp)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for pcd in pcls:
            vis.add_geometry(pcd)

        vis.add_geometry(create_coordinate_frame(all_planes[:, 0:3]))
        render_opt = vis.get_render_option()
        render_opt.point_size = 2
        set_plot_view(vis, all_planes[:, 0:3])
        vis.run()
        res_dir = os.path.join(args.res_root, args.data_set, str(timestamp))
        os.makedirs(res_dir, exist_ok=True)
        plot_path = os.path.join(res_dir, 'segmented_planes.png')
        vis.capture_screen_image(plot_path, do_render=True)
        if not os.path.isfile(plot_path) or os.path.getsize(plot_path) == 0:
            vis.destroy_window()
            raise RuntimeError(
                'Failed to save the segmented-plane plot to %s.' % plot_path
            )
        logger.info('Segmented-plane plot saved to %s.', plot_path)
        vis.destroy_window()
    return all_planes

def orientation_extraction(planes, logger):
    logger.info("Starting orientation extraction...")
    start_time = time.time()
    params, orientations = fit_plane(planes, logger)
    end_time = time.time()
    logger.info("The eclapsed time for orientation extraction is {}s".format(end_time-start_time))

    if args.save_orientation_res:
        res_dir = os.path.join(args.res_root, args.data_set, str(timestamp))
        os.makedirs(res_dir, exist_ok=True)
        file_name = 'parameters' + args.file_suffix
        file_path = os.path.join(res_dir, file_name)
        np.savetxt(
            file_path,
            params,
            delimiter=',',
            fmt='%.3f',
            header='normal_a, normal_b, normal_c, plane_offset_d, plane_id',
        )
        file_name_ = 'orientations' + args.file_suffix
        file_path_ = os.path.join(res_dir, file_name_)
        np.savetxt(
            file_path_,
            orientations,
            delimiter=',',
            fmt='%.3f',
            header='dip_degrees, dip_direction_degrees, plane_id',
        )
        logger.info('The result of parameters is saved to {}'.format(file_path))
        logger.info('The result og orientations is saved to {}'.format(file_path_))


args = parase_arguments()
timestamp = time.localtime()
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", timestamp)

if __name__ == '__main__':
    # initialize logger
    log_dir = os.path.join(args.log_root, args.data_set)
    os.makedirs(log_dir, exist_ok=True)
    logger_name = 'log'+str(timestamp)
    logger = creat_logger('logger(%d)' % (os.getpid()), log_dir, logger_name)
    logger.info('Command: {}'.format(' '.join(sys.argv)))
    # clustering analysis
    clusters = clustering_analysis(logger)
    c = np.array(clusters[0])
    for i in range(1, len(clusters)):
        c = np.concatenate((c, clusters[i]), axis=0)
    # plane segmentation
    planes = plane_segmentation(c, logger)
    # orientation extraction
    orientation_extraction(planes, logger)
