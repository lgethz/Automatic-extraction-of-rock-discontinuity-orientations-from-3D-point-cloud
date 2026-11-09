# -*- coding: utf-8 -*-
import time
import copy
import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import multiprocessing as mp

from skeleton_improved import get_h0, myCenters, get_density_weights, get_local_points
from utils import fit_line, find_intersection

POINTS = None
NEIG_DPPCA = None
NEIG_RECOMP = None
NORMAL_RECORD = None
PCA_NORMAL = None
PARAMS = None

def pca_local(neigh):
    if neigh.shape[0] < 3:
        return None, 0, 0
    X = neigh - neigh.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[-1], Vt, S**2

def projection_transform_local(target_point, points, vh, sequ):
    n_v = vh[sequ[2]]
    V_prim = points + (target_point - points) @ (n_v.reshape(-1, 1) * n_v.reshape(1, -1))
    M = vh.T[:, sequ]
    M_inv = np.linalg.inv(M)
    V_new = (M_inv @ (V_prim.T - target_point.reshape(-1, 1))).T
    return V_new[:, 0:2]

def find_feature_points_improved_local(y, dist_sq, radius_min, radius_max):
    min_y_index, max_y_index = -1, -1
    min_y_val, max_y_val = np.inf, -np.inf
    for i in range(len(y)):
        if dist_sq[i] <= radius_min and y[i] < min_y_val:
            min_y_val, min_y_index = y[i], i
        if dist_sq[i] <= radius_max and y[i] > max_y_val:
            max_y_val, max_y_index = y[i], i
    return max_y_index, min_y_index

def feature_selection_improved_local(points2d_normal, rates):
    y_vals = points2d_normal[:, 1]
    min_y_index = np.argmin(y_vals)
    max_y_index = np.argmax(y_vals)
    abs_min_y = np.abs(y_vals[min_y_index])
    abs_max_y = np.abs(y_vals[max_y_index])
    center = points2d_normal[0]
    x = points2d_normal[:, 0]
    y = y_vals
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
    feature_points_max, feature_points_min = [], []
    for rate in rates:
        radius_min = (abs_min_y * rate) ** 2
        radius_max = (abs_max_y * rate) ** 2
        max_idx, min_idx = find_feature_points_improved_local(y, dist_sq, radius_min, radius_max)
        feature_points_max.append(max_idx)
        feature_points_min.append(min_idx)
    return feature_points_max, feature_points_min

def skeleton_extraction_fast_local(points, f_max, f_min, iterations, t_error):
    h0 = get_h0(points)
    h = h0 * 1.5
    features_index = f_max + f_min
    centers = points[features_index, :]
    myCenters_ = myCenters(centers, h, maxPoints=50)
    density_weights = get_density_weights(points, h0)
    error = 100
    for j in range(iterations):
        if error < t_error:
            break
        local_indices = get_local_points(points, centers, h)
        error = myCenters_.contract(points, local_indices, h, density_weights)
        myCenters_.update_properties()
    return myCenters_.centers

def fit_segmentation_and_return_idx(points1, points2, points):
    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = points2[:, 0], points2[:, 1]
    m1, c1 = fit_line(x1, y1)
    m2, c2 = fit_line(x2, y2)
    intersection = find_intersection(m1, c1, m2, c2)
    if intersection is None:
        return -1
    A = np.array([points1[2, 0], m1 * points1[2, 0] + c1])
    B = intersection
    C = np.array([points2[2, 0], m2 * points2[2, 0] + c2])
    vec_AB = A - B
    vec_BC = C - B
    bisector_vec = (vec_AB / np.linalg.norm(vec_AB) + vec_BC / np.linalg.norm(vec_BC))
    bisector_vec /= np.linalg.norm(bisector_vec)
    m_bisector = bisector_vec[1] / bisector_vec[0]
    c_bisector = B[1] - m_bisector * B[0]
    y_pred = m_bisector * points[:, 0] + c_bisector
    mask = points[:, 1] >= y_pred
    left_idx = np.where(mask)[0]
    right_idx = np.where(~mask)[0]
    return left_idx if 0 in left_idx else right_idx

def _init_worker(points_arr, neig_dppca_arr, neig_recomp_arr, normal_record_arr, pca_normal_arr, params_obj):
    global POINTS, NEIG_DPPCA, NEIG_RECOMP, NORMAL_RECORD, PCA_NORMAL, PARAMS
    POINTS = points_arr
    NEIG_DPPCA = neig_dppca_arr
    NEIG_RECOMP = neig_recomp_arr
    NORMAL_RECORD = normal_record_arr
    PCA_NORMAL = pca_normal_arr
    PARAMS = params_obj

def _worker_execute_single(args):
    index, threshold, index_table, sharp_set = args

    idxs = NEIG_DPPCA[index_table[index]]
    neighbor_points = POINTS[idxs]
    target_point = POINTS[index]

    normal_vector, vh, sigma = pca_local(neighbor_points)
    if normal_vector is None:
        return (index, None, False, {})

    projection1 = projection_transform_local(target_point, neighbor_points, vh, [0, 1, 2])
    f_max, f_min = feature_selection_improved_local(projection1, PARAMS.rate_radius)

    projection2 = projection_transform_local(target_point, neighbor_points, vh, [1, 2, 0])

    try:
        skeleton = skeleton_extraction_fast_local(projection2, f_max, f_min, 3, 0.0001)
    except Exception:
        normal_vector, vh, sigma = pca_local(neighbor_points)
        if normal_vector is None:
            return (index, None, False, {})
        return (index, normal_vector, False, {})

    idx = fit_segmentation_and_return_idx(skeleton[0:3, :], skeleton[3:6, :], projection2,)

    if isinstance(idx, int) and idx == -1:
        normal_vector, vh, sigma = pca_local(neighbor_points)
        if normal_vector is None:
            return (index, None, False, {})
        return (index, normal_vector, False, {})
    
    try:
        cand_idx_local = f_max[2] if (len(f_max) > 2 and f_max[2] in idx) else (f_min[2] if (len(f_min) > 2 and f_min[2] in idx) else None)
    except Exception:
        cand_idx_local = None

    if cand_idx_local is None:
        normal_vector, vh, sigma = pca_local(neighbor_points)
        if normal_vector is None:
            return (index, None, False, {})
        return (index, normal_vector, False, {})

    new_id = idxs[cand_idx_local]

    if new_id not in sharp_set:
        normal_vector = NORMAL_RECORD[new_id]
        normal_vector = np.array(normal_vector)
    else:
        neigh_idxs = NEIG_RECOMP[index_table[new_id]]
        neighbor_points_ = POINTS[neigh_idxs]
        normal_vector, vh, sigma = pca_local(neighbor_points_)
        if normal_vector is None:
            return (index, None, False, {})

    try:
        neig = neighbor_points[idx]
    except Exception:
        normal_vector, vh, sigma = pca_local(neighbor_points)
        if normal_vector is None:
            return (index, None, False, {})
        return (index, normal_vector, False, {})

    normal_vec, vh2, sigma2 = pca_local(neig)
    if normal_vec is None:
        return (index, None, False, {})

    sig = np.array(sigma2, dtype=float)
    if sig.size < 3 or (sig.sum() == 0):
        s = 0.0
    else:
        if sig.size == 3:
            s = sig[-1] / sig.sum()
        else:
            s = sig[-1] / sig.sum()

    is_dummy = False
    dummys_local = {}

    if s > PARAMS.t_dummy * threshold:
        is_dummy = True
        dummys_local[str(index)] = idxs 

    return (index, np.array(normal_vector), is_dummy, dummys_local)

class DualProjectionNormalEstimation:
    def __init__(self, params, pcl, nor_gt, logger):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        self.pcd = pcd
        self.points = np.asarray(self.pcd.points)
        self.kdtree = None
        self.o_kdtree = None
        self.nor_gt = nor_gt
        self.params = params
        self.logger = logger
        # placeholders that should be initialized
        self.neig_dppca = None
        self.neig_recomp = None
        self.normal_record = None
        self.pca_normal = None
        self.sharp_idxs = None

    def load_pcd_data(self):
        from sklearn.neighbors import KDTree
        self.kdtree = KDTree(self.points)
        self.logger.debug('Loading PCD and constructing kdtree')

    def init_neighbors(self, sharp_idxs):
        self.sharp_idxs = sharp_idxs
        self.neig_dppca = self.kdtree.query(self.points[sharp_idxs], k=self.params.k_dppca, return_distance=False)
        self.neig_recomp = self.kdtree.query(self.points[sharp_idxs], k=self.params.k_recomputation, return_distance=False)

    def get_sigma(self, all_normals):
        N = self.points.shape[0]
        curvs = np.zeros(N, dtype=np.float32)
        idxs = self.kdtree.query(self.points, k=self.params.k_pca, return_distance=False)

        for i in range(N):
            neigh = self.points[idxs[i]]
            X = neigh - neigh.mean(axis=0, keepdims=True)
            cov = X.T @ X
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            curv = eigvals[0] / eigvals.sum()

            all_normals[i] = normal.tolist() 
            curvs[i] = curv
        pca_normals = copy.deepcopy(all_normals)
        self.normal_record = copy.deepcopy(all_normals)
        return curvs, pca_normals
    
    def classfication(self, curvs, all_normals, ct):
        curvs = np.array(curvs)
        sharp = np.where(curvs[:] >= ct)
        sharp_set = set(sharp[0]) 
        all_normals[:] = [None if i in sharp_set else all_normals[i] for i in range(len(all_normals))]
        total_sigma = 0
        total_sigma += sum(curvs[i] for i in range(len(curvs)) if i not in sharp_set)
        return sharp_set, total_sigma/(len(all_normals)-len(sharp))

    def batch_execute(self, indexs, all_normals, threshold, pca_normal, index_table, n_jobs=4):
        if self.kdtree is None:
            self.load_pcd_data()

        # ensure neighbor arrays and records are prepared
        if self.neig_dppca is None or self.neig_recomp is None:
            raise RuntimeError("Please call init_neighbors(sharp_idxs) before batch_execute")

        points_arr = self.points 
        neig_dppca_arr = np.asarray(self.neig_dppca)
        neig_recomp_arr = np.asarray(self.neig_recomp)
        normal_record_arr = np.asarray(self.normal_record) 
        pca_normal_arr = np.asarray(pca_normal)
        params_obj = self.params

        sharp_set = set(self.sharp_idxs) if self.sharp_idxs is not None else set()

        self.logger.debug('Starting multiprocessing batch_execute with n_jobs={}'.format(n_jobs))
        start_time = time.time()

        tasks = [(int(idx), float(threshold), index_table, sharp_set) for idx in indexs]

        with mp.Pool(processes=n_jobs, initializer=_init_worker,
                     initargs=(points_arr, neig_dppca_arr, neig_recomp_arr, normal_record_arr, pca_normal_arr, params_obj)) as pool:
            results = pool.map(_worker_execute_single, tasks)

        dummys = {}
        all_dummy = []
        for idx, nv, flag, d_local in results:
            if nv is None:
                all_normals[idx] = [0, 0, 0]
                continue
            if flag:
                all_dummy.append(idx)
                dummys.update(d_local)
                continue
            all_normals[idx] = nv.tolist() if isinstance(nv, np.ndarray) else nv

        end_time = time.time()
        self.logger.info('Multiprocessing batch_execute finished, time: {:.3f}s, dummys count: {}'.format(end_time - start_time, len(dummys)))

        threshold_angle_deg = 10
        threshold_angle_rad = np.radians(threshold_angle_deg)

        for key, value in dummys.items():
            n_pca = np.array(pca_normal[int(key)])
            n_dppca_list = [all_normals[v] for v in value if all_normals[v] is not None]
            if len(n_dppca_list) == 0:
                continue
            n_dppca = np.array(n_dppca_list)
            cos_thresh = np.cos(threshold_angle_rad)
            sim_matrix = np.matmul(n_dppca, n_dppca.T)
            adj = sim_matrix > cos_thresh
            graph = csr_matrix(adj)
            n_components, labels = connected_components(csgraph=graph, directed=False)

            clazz = []
            for c in range(n_components):
                members = n_dppca[labels == c]
                if len(members) > 7:
                    clazz.append(members[0])
            clazz = np.array(clazz)
            if clazz.shape[0] > 0:
                cos_vals = np.clip(np.abs(np.dot(n_pca, clazz.T)), -1.0, 1.0)
                re__ = np.arccos(cos_vals)
                all_normals[int(key)] = clazz[np.argmin(re__)].tolist()

        return all_normals