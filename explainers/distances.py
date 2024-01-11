import ot
import numpy as np
from typing import Optional
import torch


class WassersteinDivergence:
    def __init__(self, reg=1):
        self.nu = None
        self.reg = reg

    def distance(self, y_s: torch.tensor, y_t: torch.tensor, delta=0):
        # Validate delta
        if delta < 0 or delta > 0.5:
            raise ValueError("Delta should be between 0 and 0.5")

        # Calculate quantiles
        lower_quantile_s = torch.quantile(y_s, delta)
        upper_quantile_s = torch.quantile(y_s, 1 - delta)
        lower_quantile_t = torch.quantile(y_t, delta)
        upper_quantile_t = torch.quantile(y_t, 1 - delta)

        # Filter data points
        y_s_filtered = y_s[(y_s >= lower_quantile_s) & (y_s <= upper_quantile_s)]
        y_t_filtered = y_t[(y_t >= lower_quantile_t) & (y_t <= upper_quantile_t)]

        proj_y_s_dist_mass = torch.ones(len(y_s_filtered)) / len(y_s_filtered)
        proj_y_t_dist_mass = torch.ones(len(y_t_filtered)) / len(y_t_filtered)

        M_y = ot.dist(
            y_s_filtered.reshape(y_s_filtered.shape[0], 1),
            y_t_filtered.reshape(y_t_filtered.shape[0], 1),
            metric="sqeuclidean",
        ).to("cpu")

        self.nu = ot.emd(proj_y_s_dist_mass, proj_y_t_dist_mass, M_y)
        # self.nu = ot.bregman.sinkhorn(
        #     proj_y_s_dist_mass, proj_y_t_dist_mass, M_y, reg=self.reg
        # )
        # self.nu = torch.diag(torch.ones(len(y_s)))
        dist = torch.sum(self.nu * M_y)

        return dist, self.nu


class SlicedWassersteinDivergence:
    def __init__(self, dim: int, n_proj: int, reg=1):
        self.dim = dim
        self.n_proj = n_proj
        self.thetas = np.random.randn(n_proj, dim)
        self.thetas /= np.linalg.norm(self.thetas, axis=1)[:, None]
        self.wd = WassersteinDivergence()

        self.reg = reg

        self.mu_list = []

    def distance(self, X_s: torch.tensor, X_t: torch.tensor, delta=0):
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

        self.mu_list = []
        dist = 0
        for theta in self.thetas:
            # Project data onto the vector theta
            theta = torch.from_numpy(theta).float()
            proj_X_s = X_s.to("cpu") @ theta
            proj_X_t = X_t.to("cpu") @ theta

            dist_wd, mu = self.wd.distance(proj_X_s, proj_X_t, delta)

            self.mu_list.append(mu)

            dist += dist_wd

        return dist / self.n_proj
