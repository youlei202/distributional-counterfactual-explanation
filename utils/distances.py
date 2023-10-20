import ot
import numpy as np
from typing import Optional


class SlicedWassersteinDivergence:
    def __init__(self, dim: int, n_proj: int):
        self.dim = dim
        self.n_proj = n_proj
        self.thetas = np.random.randn(n_proj, dim)
        thetas /= np.linalg.norm(thetas, axis=1)[:, None]

    def distance(self, X_s: np.array, X_t: np.array, p: Optional[int] = 2):
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
        # Generate random projection vectors on the unit sphere

        swd = 0
        for theta in self.thetas:
            # Project data onto the vector theta
            proj_X_s = X_s.dot(theta)
            proj_X_t = X_t.dot(theta)

            # Compute 1D Wasserstein distance and accumulate
            swd += ot.wasserstein_1d(proj_X_s, proj_X_t, None, None, p=p)

        return swd / self.n_proj
