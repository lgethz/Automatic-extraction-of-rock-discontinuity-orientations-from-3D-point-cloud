import os, sys
import time
import logging
import numpy as np
import open3d as o3d
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

def get_log_dir(root, prefix='', postfix=''):
    log_name = prefix + time.strftime('%y$m$d_%H%M%S', time.localtime()) + postfix
    log_dir = os.path.join(root, log_name, 'log')
    os.makedirs(log_dir)
    return log_dir, log_name

def creat_logger(log_name, log_dir=None, file_name='log'):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, file_name+'.txt'), mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('Output and logs will be saved to: {}'.format(log_dir))
    return logger

def get_logger(args, log_dir, log_name, file_name, model=None):
    logger = creat_logger(log_name=log_name, log_dir=log_dir, file_name=file_name)
    logger.info('Command: {}'.format(' '.join(sys.argv)))
    arg_str = '\n'.join(['  {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
    logger.info('Arguments:\n' + arg_str)
    if model is not None:
        logger.info(repr(model))
    return logger

def get_kdtree(pcd):
    return o3d.geometry.KDTreeFlann(pcd)

# 使用最小二乘法拟合直线
def fit_line(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# 计算两条直线的交点
def find_intersection(m1, c1, m2, c2):
    if m1 == m2:
        return None  # 平行或重合
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

# 分割点集
def point_side(point, m, c):
    return point[1] - (m * point[0] + c)

def RMS(ground_truth, normals, tau=10):
    """
    计算法向量与真值的误差
    @param ground_truth: 法向量真值
    @param normals: 计算得到的法向量
    @param tau: 阈值，默认值为10度
    @return: (均方根误差, 超过阈值的数量)
    """
    N = len(ground_truth)
    print(N)
    rms = 0
    dummy = 0
    threshold = np.deg2rad(tau)  # 将阈值转换为弧度
    d_index = []
    for i in range(N):
        g = ground_truth[i]
        n = normals[i]
        product = np.dot(g, n)
        product = np.clip(product, -1.0, 1.0)  # 防止浮点数误差导致的arccos值超出范围
        angle = np.arccos(product)
        
        if angle <= threshold:
            rms += (angle)**2
        else:
            rms += (np.pi/2)**2
            dummy += 1
            d_index.append(i)
    
    rms = np.sqrt(rms / N)
    
    return rms, dummy, d_index

def display_point_cloud_with_normals(points, normals):
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3).')
    if normals.shape != points.shape:
        raise ValueError('normals must have the same shape as points.')

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    max_displayed_vectors = 50000
    stride = max(1, int(np.ceil(len(points) / max_displayed_vectors)))
    displayed_indices = np.arange(0, len(points), stride)
    cloud_extent = float(np.max(np.ptp(points, axis=0))) if len(points) else 0.0
    vector_length = max(0.1, cloud_extent * 0.01)

    # 创建 LineSet 对象
    line_set = o3d.geometry.LineSet()
    displayed_points = points[displayed_indices]
    displayed_normals = normals[displayed_indices]
    line_set.points = o3d.utility.Vector3dVector(
        np.vstack((
            displayed_points,
            displayed_points + displayed_normals * vector_length,
        ))
    )
    line_set.lines = o3d.utility.Vector2iVector([
        [i, i + len(displayed_indices)]
        for i in range(len(displayed_indices))
    ])
    line_set.colors = o3d.utility.Vector3dVector(
        [[0, 1, 0] for _ in range(len(displayed_indices))]
    )

    # 可视化点云和法向量
    coordinate_frame = create_coordinate_frame(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    o3d.visualization.draw_geometries(
        [pcd, line_set, coordinate_frame],
        front=[0.0, -1.0, 0.0],
        lookat=center.tolist(),
        up=[0.0, 0.0, 1.0],
        zoom=0.7,
    )


def create_coordinate_frame(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3).')

    if len(points):
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        origin = (minimum + maximum) / 2
        size = float(np.max(maximum - minimum)) * 0.1
    else:
        origin = np.zeros(3)
        size = 1.0
    if size <= 0:
        size = 1.0

    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size,
        origin=origin.tolist(),
    )


def display_downsampling_comparison(original_points, downsampled_points, shape_name):
    original_points = np.asarray(original_points, dtype=np.float64)
    downsampled_points = np.asarray(downsampled_points, dtype=np.float64)
    if original_points.ndim != 2 or original_points.shape[1] != 3:
        raise ValueError('original_points must have shape (N, 3).')
    if downsampled_points.ndim != 2 or downsampled_points.shape[1] != 3:
        raise ValueError('downsampled_points must have shape (N, 3).')
    if len(original_points) == 0 or len(downsampled_points) == 0:
        raise ValueError('Both point clouds must contain at least one point.')

    original_center = (original_points.min(axis=0) + original_points.max(axis=0)) / 2
    downsampled_center = (
        downsampled_points.min(axis=0) + downsampled_points.max(axis=0)
    ) / 2
    combined_extent = max(
        float(np.max(np.ptp(original_points, axis=0))),
        float(np.max(np.ptp(downsampled_points, axis=0))),
    )
    separation = max(combined_extent * 1.5, 1.0)
    left_center = np.array([-separation / 2, 0.0, 0.0])
    right_center = np.array([separation / 2, 0.0, 0.0])
    original_display = original_points - original_center + left_center
    downsampled_display = downsampled_points - downsampled_center + right_center

    original_cloud = o3d.geometry.PointCloud()
    original_cloud.points = o3d.utility.Vector3dVector(original_display)
    original_cloud.paint_uniform_color([0.65, 0.65, 0.65])
    downsampled_cloud = o3d.geometry.PointCloud()
    downsampled_cloud.points = o3d.utility.Vector3dVector(downsampled_display)
    downsampled_cloud.paint_uniform_color([0.95, 0.35, 0.1])

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=(
            '%s - Original (left: %d points) | Octree downsampled '
            '(right: %d points)' % (
                shape_name,
                len(original_points),
                len(downsampled_points),
            )
        )
    )
    visualizer.add_geometry(original_cloud)
    visualizer.add_geometry(downsampled_cloud)
    visualizer.add_geometry(create_coordinate_frame(original_display))
    visualizer.add_geometry(create_coordinate_frame(downsampled_display))
    render_options = visualizer.get_render_option()
    render_options.point_size = 2
    view_control = visualizer.get_view_control()
    view_control.set_front([0.0, -1.0, 0.0])
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_lookat([0.0, 0.0, 0.0])
    view_control.set_zoom(0.7)
    visualizer.run()
    visualizer.destroy_window()


class ParamManager():
    def __init__(self, args):
        self.k_pca = args.k_pca
        self.k_dppca = args.k_dppca
        self.k_recomputation = args.k_recomputation
        self.t_dummy = args.threshold_dummy
        self.rate_radius = args.rate_radius

    def toString(self):
        s = "[c_t:{}, k_pca:{}, k_dppca:{}, k_recompute:{}, t_dummy:{}, rate_radius:{}]".format(self.c_t, self.k_pca,
                                                                                                self.k_dppca, self.k_recomputation,
                                                                                                self.t_dummy, self.rate_radius)
        return s
    
def feature_threshold_selection(sigms, TP, bin):
    DEBUG = TP.get("debug_taproot", False)
    length_taproot = bin

    taproot = np.zeros(length_taproot)
    for i in range(1, length_taproot + 1):
        low = (i - 1) / length_taproot
        high = i / length_taproot
        taproot[i - 1] = np.sum(sigms >= low) - np.sum(sigms >= high)

    m_id = np.argmax(taproot)  
    m_value = taproot[m_id]
    max_sigm_bar = (m_id + 1) / length_taproot 

   
    uni_taproot = taproot / m_value
    y = uni_taproot.reshape(-1, 1)
    n = len(y)


    e = np.ones(n)
    D = spdiags([e, -e], [0, 1], n - 1, n).toarray()


    x1 = cp.Variable((n, 1))

    lambda_val = 0.5
    objective = 0.5 * cp.sum_squares(y - x1) + lambda_val * cp.norm1(D @ x1)
    prob = cp.Problem(cp.Minimize(objective))
    prob.solve(solver=cp.SCS)

    x1_val = np.array(x1.value).flatten()

    feat = None
    for i in range(m_id + 1, length_taproot - 1):
        diff_taproot = x1_val[i - 1] - x1_val[i + 1]
        if (diff_taproot < TP["sigma_threshold"]) and \
           (x1_val[i] < (x1_val[m_id] - TP["sigma_threshold"])):
            feat = (i + 1) / length_taproot 
            break

    if DEBUG:
        x = np.arange(1, length_taproot + 1) / length_taproot
        plt.figure()
        plt.plot(x, taproot, linewidth=3.0, label="taproot")
        plt.plot(x, x1_val * m_value, "m-", linewidth=3.0, label="smoothed")
        if feat is not None:
            plt.axvline(x=feat, color="r", linestyle="--", label="feature threshold")
        plt.legend()
        plt.show()

    return feat, max_sigm_bar
