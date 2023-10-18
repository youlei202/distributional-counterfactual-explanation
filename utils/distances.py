import ot


def sliced_wasserstein_distance(X_s, X_t, thetas, a=None, b=None, metric="sqeuclidean"):
    """
    Compute the sliced Wasserstein distance between X_s and X_t

    Parameters:
    X_s : np.ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : np.ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : np.ndarray, shape (n_samples_a,), optional
        weights of each sample of X_s, default is uniform weight
    b : np.ndarray, shape (n_samples_b,), optional
        weights of each sample of X_t, default is uniform weight
    metric : str, optional
        metric to be used for Wasserstein-1 distance computation
    n_projections : int, optional
        number of projections

    Returns:
    swd : float
        Sliced Wasserstein Distance between X_s and X_t
    """
    # Generate random projection vectors on the unit sphere

    swd = 0
    n_projections = len(thetas)
    for theta in thetas:
        # Project data onto the vector theta
        proj_X_s = X_s.dot(theta)
        proj_X_t = X_t.dot(theta)

        # Compute 1D Wasserstein distance and accumulate
        swd += ot.wasserstein_1d(proj_X_s, proj_X_t, a, b, p=2)

    return swd / n_projections
