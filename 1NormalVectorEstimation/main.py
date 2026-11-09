import os, sys
import argparse
import time
import math
import numpy as np
from dataset import BaseDataset
from utils import get_logger, creat_logger, get_log_dir, ParamManager, feature_threshold_selection, display_point_cloud_with_normals, display_downsampling_comparison, RMS, create_coordinate_frame
from dppca import DualProjectionNormalEstimation
from octreebaseddownsampling import octree_downsample
import open3d as o3d
import copy


def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False

    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    cur_path = os.getcwd()
    ## Dataset
    parser.add_argument('--dataset_root', type=str, default=os.path.join(cur_path, 'dataset'))
    parser.add_argument('--data_set', type=str, default='')
    parser.add_argument(
        '--log_root',
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'),
    )
    parser.add_argument('--evaluate', type=eval, default=False, choices=[True, False])
    parser.add_argument('--data_list', type=str, default='')
    parser.add_argument(
        '--downsample',
        type=parse_bool,
        default=False,
        help='Downsample each point cloud with the octree before estimating normals.',
    )
    parser.add_argument(
        '--downsample_voxel_size',
        type=float,
        default=None,
        help='Target octree leaf width in point-cloud coordinate units.',
    )
    parser.add_argument(
        '--downsample_max_depth',
        type=int,
        default=8,
        help='Maximum octree subdivision depth (default: 8).',
    )
    parser.add_argument(
        '--downsample_plot',
        type=parse_bool,
        default=False,
        help='Display the downsampled point cloud before normal estimation.',
    )
    
    ## Parameters
    parser.add_argument('--k_pca', type=int, default=50)
    parser.add_argument('--k_dppca', type=int, default=250)
    parser.add_argument('--k_recomputation', type=int, default=50)
    parser.add_argument('--threshold_dummy', type=float, default=0.7)
    parser.add_argument('--rate_radius', type=list, default=[0.35, 0.65, 0.85])
    parser.add_argument('--num_bin', type=int, default=150)
    parser.add_argument('--n_jobs', type=int, default=4)

    ## Test
    parser.add_argument('--show_res', type=eval, default=False, choices=[True, False])
    parser.add_argument('--save_normal_xyz', type=eval, default=False, choices=[True, False])
    parser.add_argument('--show_dummy', type=eval, default=False, choices=[True, False])
    parser.add_argument('--sigma_threshold', type=float, default=0.05)
    parser.add_argument('--debug', type=eval, default=False, choices=[True, False])


    args = parser.parse_args()
    if args.downsample and args.evaluate:
        parser.error(
            '--evaluate cannot be used with --downsample true because '
            'ground-truth normals are still at the original point locations. '
            'Use --downsample false to evaluate at full resolution.'
        )
    if args.downsample_max_depth < 0:
        parser.error('--downsample_max_depth must be non-negative.')
    if args.downsample_voxel_size is not None and args.downsample_voxel_size <= 0:
        parser.error('--downsample_voxel_size must be positive.')
    if args.downsample_plot and not args.downsample:
        parser.error('--downsample_plot requires --downsample true.')
    for parameter in ('k_pca', 'k_dppca', 'k_recomputation'):
        if getattr(args, parameter) < 3:
            parser.error('--%s must be at least 3.' % parameter)
    return args

args = parse_arguments()

def run(data_list):
    # Log
    PID = os.getpid()
    time_stmp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    output_dir = os.path.join(args.log_root, args.data_set, args.data_list, 'test_%s_%s_%s_%s_%s' % (args.k_dppca, args.k_pca, args.k_recomputation, args.threshold_dummy, time_stmp))
    os.makedirs(output_dir, exist_ok=False)
    logger = creat_logger('test(%d)(%s-%s)' % (PID, args.data_set, args.data_list), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    # dataset
    data_set = BaseDataset(root=args.dataset_root,
                           data_set=args.data_set,
                           data_list=data_list,
                           evaluate=args.evaluate)
    
    for shape_idx, shape_name in enumerate(data_set.cur_sets):
        # Normal estimation
        pcl_raw, nor_gt = data_set.get_data(shape_name)
        logger.info(
            'Loaded point cloud %s with %d points.',
            shape_name,
            len(pcl_raw),
        )
        original_points = pcl_raw
        if args.downsample:
            original_point_count = len(pcl_raw)
            logger.info(
                'Starting octree downsampling for %s '
                '(target voxel size=%s, maximum depth=%d).',
                shape_name,
                args.downsample_voxel_size,
                args.downsample_max_depth,
            )
            pcl_raw = octree_downsample(
                pcl_raw,
                voxel_size=args.downsample_voxel_size,
                max_depth=args.downsample_max_depth,
            ).astype(np.float32)
            logger.info(
                'Octree downsampling for %s: %d -> %d points '
                '(voxel_size=%s, max_depth=%d)',
                shape_name,
                original_point_count,
                len(pcl_raw),
                args.downsample_voxel_size,
                args.downsample_max_depth,
            )
            downsampled_dir = os.path.join(output_dir, 'downsampled')
            os.makedirs(downsampled_dir, exist_ok=True)
            np.savetxt(
                os.path.join(downsampled_dir, shape_name + '.xyz'),
                pcl_raw,
                fmt='%.6f',
            )
            logger.info(
                'Saved downsampled point cloud to %s.',
                os.path.join(downsampled_dir, shape_name + '.xyz'),
            )
            if args.downsample_plot:
                display_downsampling_comparison(
                    original_points,
                    pcl_raw,
                    shape_name,
                )

        # processing
        if len(pcl_raw) < 3:
            raise ValueError(
                'Point cloud %s has fewer than 3 points after preprocessing; '
                'normal estimation requires at least 3.' % shape_name
            )
        params_args = copy.copy(args)
        for parameter in ('k_pca', 'k_dppca', 'k_recomputation'):
            requested_neighbors = getattr(args, parameter)
            effective_neighbors = min(requested_neighbors, len(pcl_raw))
            if effective_neighbors != requested_neighbors:
                logger.warning(
                    '%s=%d exceeds the %d available points for %s; using %d.',
                    parameter,
                    requested_neighbors,
                    len(pcl_raw),
                    shape_name,
                    effective_neighbors,
                )
                setattr(params_args, parameter, effective_neighbors)
        params = ParamManager(params_args)
        logger.info(
            'Starting normal-vector estimation for %s (%d points; '
            'k_pca=%d, k_dppca=%d, k_recomputation=%d, workers=%d).',
            shape_name,
            len(pcl_raw),
            params.k_pca,
            params.k_dppca,
            params.k_recomputation,
            args.n_jobs,
        )
        dppca = DualProjectionNormalEstimation(params=params, pcl=pcl_raw, nor_gt=nor_gt, logger=logger)
        dppca.load_pcd_data()
        all_normals = [None] * len(pcl_raw)
        sigmas, pca_normal = dppca.get_sigma(all_normals)
        # all_normals = copy.deepcopy(pca_normal)
        sigmas = np.asarray(sigmas)
        TP = {
                "debug_taproot": args.debug,
                "sigma_threshold": args.sigma_threshold
            }
        feat, _ = feature_threshold_selection(sigmas, TP, args.num_bin)
        logger.info("Feature threshold: %f", feat)
        # logger.info("Max sigma bar:", max_sigm_bar)

        sharp_idx, threshold = dppca.classfication(sigmas, all_normals, feat)
        sharp_idx = np.asarray(list(sharp_idx), dtype=int)
        logger.info("The number of sharp points: %d", sharp_idx.shape[0])
        # 建立索引映射
        index_table= np.full(len(pcl_raw), -1, dtype=int)
        for i in range(len(sharp_idx)):
            index_table[sharp_idx[i]] = i
        # print(all_normals.shape)
        start_time = time.time()

        # 计算所有尖锐特征点的近邻点信息，全局保存
        if sharp_idx.size:
            dppca.init_neighbors(sharp_idx)
            dppca.batch_execute(
                sharp_idx,
                all_normals,
                threshold,
                pca_normal,
                index_table,
                args.n_jobs,
            )
        else:
            logger.info('No sharp points detected; retaining PCA normals.')
        end_time = time.time()
        x_mid = (np.max(pcl_raw[:, 0]) + np.min(pcl_raw[:, 0])) / 2
        y_mid = (np.max(pcl_raw[:, 1]) + np.min(pcl_raw[:, 1])) / 2
        z_mid = (np.max(pcl_raw[:, 2]) + np.min(pcl_raw[:, 2])) / 2
        view_p = np.array([x_mid, y_mid, z_mid])
        # view_p = np.array([1.956, 14.910, -0.0965])
        all_normals = [[0,0,0] if n is None else n for n in all_normals]
        all_normals = np.asarray(all_normals)
        for i in range(len(pcl_raw)):
            A = all_normals[i]
            B = view_p - pcl_raw[i]
            dot = np.dot(A, B)
            if dot > 0:
                all_normals[i, 0] = -1 * all_normals[i, 0]
                all_normals[i, 1] = -1 * all_normals[i, 1]
                all_normals[i, 2] = -1 * all_normals[i, 2]
        predict_normal = all_normals
        if args.save_normal_xyz:
            normal_dir = os.path.join(output_dir, 'pred_normal')
            os.makedirs(normal_dir, exist_ok=True)
            path_save = os.path.join(normal_dir, shape_name)

            pc_nor = np.concatenate([pcl_raw, predict_normal], -1)
            # pc_curve = np.concatenate([pcl_raw, curvs], -1)
            np.savetxt(path_save + '.xyz', pc_nor, fmt='%.6f')
            np.savetxt(path_save + '.sig', sigmas, fmt='%.6f')
            logger.info('Saved estimated normals to %s.xyz.', path_save)
        if args.evaluate:
            rms, dummy, d_index = RMS(nor_gt, predict_normal)
            logger.info('[rms:{}, NBP:{}]'.format(rms, dummy))
            if args.show_dummy:
                dummys = pcl_raw[d_index]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(dummys)
                center = (dummys.min(axis=0) + dummys.max(axis=0)) / 2
                o3d.visualization.draw_geometries(
                    [pcd, create_coordinate_frame(dummys)],
                    front=[0.0, -1.0, 0.0],
                    lookat=center.tolist(),
                    up=[0.0, 0.0, 1.0],
                    zoom=0.7,
                )
        if args.show_res:
            display_point_cloud_with_normals(pcl_raw, predict_normal)

    logger.info('elasped time is {}'.format((end_time-start_time)))
    return 1


if __name__  == '__main__':
    run(args.data_list)