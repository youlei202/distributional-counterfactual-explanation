import ot
import numpy as np
from typing import Optional
import torch


class WassersteinDivergence:
    def __init__(self, reg=1):
        self.nu = None
        self.reg = reg

    def distance(self, y_s: np.array, y_t: np.array):

        proj_y_s_dist_mass = torch.ones(len(y_s)) / len(y_s)
        proj_y_t_dist_mass = torch.ones(len(y_t)) / len(y_t)

        M_y = ot.dist(
            y_s.reshape(y_s.shape[0], 1),
            y_t.reshape(y_t.shape[0], 1),
            metric="sqeuclidean",
        ).to('cpu')

        # self.nu = ot.emd(proj_y_s_dist_mass, proj_y_t_dist_mass, M_y)
        # self.nu = ot.bregman.sinkhorn(
        #     proj_y_s_dist_mass, proj_y_t_dist_mass, M_y, reg=self.reg
        # )
        self.nu = torch.diag(torch.ones(len(y_s)))
        dist = torch.sum(self.nu * M_y)

        return dist


class SlicedWassersteinDivergence:
    def __init__(self, dim: int, n_proj: int, reg=1):
        self.dim = dim
        self.n_proj = n_proj
        self.thetas = np.random.randn(n_proj, dim)
        self.thetas /= np.linalg.norm(self.thetas, axis=1)[:, None]

        self.reg = reg

        self.mu_list = []

    def distance(self, X_s: np.array, X_t: np.array):
        """
        Compute the sliced Wasserstein distance between X_s and X_t

        Parameters:
        X_s : np.ndarray, shape (n_samples_a, dim)
            samples in the source domain
        X_t : np.ndarray, shape (n_samples_b, dim)
            samples in the target domain
        metric : str, optional
            metric to be used for Wasserstein-1 distance computation

        Returns:
        swd : float
            Sliced Wasserstein Distance between X_s and X_t
        """

        proj_X_s_dist_mass = torch.ones(len(X_s)) / len(X_s)
        proj_X_t_dist_mass = torch.ones(len(X_t)) / len(X_t)

        self.mu_list = []
        dist = 0
        for theta in self.thetas:
            
            # Project data onto the vector theta
            theta = torch.from_numpy(theta).float()
            proj_X_s = X_s.to('cpu') @ theta
            proj_X_t = X_t.to('cpu') @ theta

            M_x = ot.dist(
                proj_X_s.reshape(proj_X_s.shape[0], 1),
                proj_X_t.reshape(proj_X_t.shape[0], 1),
                metric="sqeuclidean",
            )

            # Compute 1D Wasserstein distance and accumulate
            mu = ot.emd(proj_X_s_dist_mass, proj_X_t_dist_mass, M_x)
            # mu = ot.bregman.sinkhorn(
            #     proj_X_s_dist_mass, proj_X_t_dist_mass, M_x, reg=self.reg
            # )
            self.mu_list.append(mu)

            dist += torch.sum(mu * M_x)

        return dist / self.n_proj
