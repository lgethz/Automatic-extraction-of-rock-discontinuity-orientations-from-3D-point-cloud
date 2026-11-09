import numpy as np
from scipy.spatial import distance

h = 0

class myCenter:
    def __init__(self, center, h, index):
        self.center = center
        self.h = h
        self.index = index
        self.closest_neighbours = np.array([])
        self.eigen_vectors = np.zeros((3,3))
        self.sigma = 0.5

    def set_closest_neighbours(self, closest_neighbours):
        self.closest_neighbours = closest_neighbours

    def set_h(self,h):
        self.h = h

    def set_eigen_vectors(self,eigen_vectors):
        self.eigen_vectors = eigen_vectors

    def set_sigma(self,sigma):
        self.sigma = sigma 

    def set_center(self,center):
        self.center = center

class myCenters:
    def __init__(self, centers, h0, maxPoints):
        self.centers = centers
        self.myCenters = [myCenter(center, h, index) for index, center in enumerate(centers)]
        self.skeleton = {}
        self.closest = []
        self.sigmas = np.array([None] * len(centers))
        self.h0 = h0
        self.h = h0
        self.eigen_vectors = [None] * len(centers)
        self.branch_points = [None] * len(centers)
        self.maxPoints = maxPoints
        self.get_nearest_neighbours()
        self.Nremoved = 0

    def get_nearest_neighbours(self):
        distances = distance.squareform(distance.pdist(self.centers))
        self.closest = np.argsort(distances, axis=1)
        for center in self.myCenters:
            closest = self.closest[center.index, :].copy()
            sorted_local_distances = distances[center.index, closest] ** 2
            in_neighboorhood = np.argmax(sorted_local_distances >= (center.h) ** 2)
            if in_neighboorhood == 0:
                in_neighboorhood = -1
            center.set_closest_neighbours(closest[1:in_neighboorhood])

    def contract(self, points, local_indices, h, density_weights, mu=0.35):
        self.h = h
        error_center = 0
        N = 0
        for myCenter in self.myCenters:
            centers_indices = myCenter.closest_neighbours
            # centers_in = np.array(self.centers[centers_indices])
            centers_in = np.take(self.centers, centers_indices, axis=0)
            my_local_indices = local_indices[myCenter.index]
            local_points = points[my_local_indices]
            if local_points.shape[0] > 2 and len(centers_in) > 1:
                density_weights_points = density_weights[my_local_indices]
                term1 = get_term1(myCenter.center, local_points, h, density_weights_points)
                term2 = get_term2(myCenter.center, centers_in, h)
                if term1.any() and term2.any():
                    sigma, vecs = get_sigma(myCenter.center, centers_in, h)
                    new_center = term1 + mu * sigma * term2
                    error_center += np.linalg.norm(myCenter.center - new_center)
                    N += 1
                    myCenter.set_center(new_center)
                    myCenter.set_eigen_vectors(vecs)
                    myCenter.set_sigma(sigma)
                    myCenter.set_h(h)
        return error_center
    
    def update_properties(self):
        for center in self.myCenters:
            index = center.index
            self.centers[index] = center.center
            self.eigen_vectors[index] = center.eigen_vectors
            self.sigmas[index] = center.sigma
        self.get_nearest_neighbours()
        self.update_sigmas()

    def update_sigmas(self):
        k = 5
        new_sigmas = []
        for center in self.myCenters:
            indices = np.array(self.closest[center.index, :k]).astype(int)
            sigma_nearest_k_neighbours = self.sigmas[indices]
            mean_sigma = np.mean(sigma_nearest_k_neighbours)
            new_sigmas.append(mean_sigma)
        for index, center in enumerate(self.myCenters):
            center.set_sigma(new_sigmas[index])
            self.sigmas[center.index] = new_sigmas[index]

def get_h0(points):
    x_max, x_min = points[:,0].max(), points[:,0].min()
    y_max, y_min = points[:,1].max(), points[:,1].min()
    diagonal = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    Npoints = len(points)
    return diagonal / (Npoints ** (1. / 3))

def get_density_weights(points, h0, for_center=False, center=[0,0,0]):
    if for_center:
        r = points - center
        r2 = np.einsum('ij,ij->i', r, r)
        density_weights = np.sum(np.exp(-r2 / ((h0 / 4) ** 2))) + 1
    else:
        r = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', r, r)
        density_weights = np.sum(np.exp(-r2 / ((h0 / 4) ** 2)), axis=1) + 1
    return density_weights

def get_local_points(points, centers, h, maxLocalPoints=5000):
    local_indices = []
    for center in centers:
        x, y = center
        where_square = ((points[:,0] >= (x - h)) & (points[:, 0] <= (x + h)) & (points[:,1] >= (y - h)) & 
                        (points[:, 1] <= (y + h)))
        square = points[where_square]
        indices_square = np.where(where_square == True)[0]
        square_squared = np.square(square - [x,y])
        where_sphere = np.sum(square_squared, axis=1) <= h**2
        local_sphere_indices = indices_square[where_sphere]
        local_indices.append(local_sphere_indices)
    return local_indices

def get_term1(center, points, h, density_weights):
    r = points - center
    r2 = np.einsum('ij,ij->i', r, r)
    thetas = np.exp(-r2 / ((h/2) ** 2))
    alphas = thetas / density_weights
    # denom = np.einsum('i->', alphas)
    denom = np.sum(alphas)  # 计算加权总和
    if denom > 1e-20:
        # term1 = np.einsum('j,jk->k', alphas, points) / denom
        term1 = np.dot(alphas, points) / denom
    else: 
        term1 = np.array(False)
    return term1

def get_term2(center, centers, h):
    x = center - centers
    r2 = np.einsum('ij,ij->i', x, x)
    r = 1 / (np.sqrt(r2) + 1e-7)
    thetas = np.exp(-r2 / ((h/2) ** 2))
    # betas = np.einsum('i,i->i', thetas, r)
    betas = thetas * r
    denom = np.sum(betas)
    if denom > 1e-20:
        num = np.dot(betas, x)
        term2 = num / denom
    else:
        term2 = np.array(False)
    return term2

def get_sigma(center, centers, h):
    r = centers - center
    r2 = np.einsum('ij,ij->i', r, r)
    thetas = np.exp(-r2 / ((h/2) ** 2))
    cov = np.einsum('j,jk,jl->kl', thetas, r, r)
    values, vectors = np.linalg.eig(cov)
    if np.iscomplex(values).any():
        values = np.real(values)
        vectors = np.real(vectors)
    sorted_indices = np.argsort(-values)
    values_sorted = values[sorted_indices]
    vectors_sorted = vectors[:,sorted_indices]
    sigma = values_sorted[0] / np.sum(values_sorted)
    return sigma, vectors_sorted
