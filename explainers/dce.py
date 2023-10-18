from abc import ABC, abstractmethod
import torch

class DistributionalCounterfactualExplainer:
    def __init__(self, model):
        """
        Initialize the counterfactual explainer.

        Parameters:
        - model: The predictive model to be explained.
        """
        self.model = model

    def _compute_gradient(self, K, X, X_prime, mu, nu, lambda_val):
        """
        Compute the gradient of Q(X) with respect to each x_i in X.

        Parameters:
        - K: List of theta vectors.
        - X: Tensor of shape (n, d) where n is the number of x_i vectors and d is the dimension of each vector.
        - X_prime: Tensor of shape (n, d) representing the x'_j vectors.
        - mu: Tensor of shape (len(K), n, n) representing mu values for each theta, i, and j.
        - nu: Tensor of shape (n, n) representing nu values for each i and j.
        - lambda_val: Scalar lambda value.
        - b: Blackbox model (assumed to be a PyTorch model).

        Returns:
        - Gradient tensor of shape (n, d).
        """

        X = torch.from_numpy(X).float()
        X_prime = torch.from_numpy(X_prime).float()

        # lambda_val = torch.Tensor(lambda_val).double()

        # Ensure X requires gradient for autograd
        X.requires_grad_(True)

        # Initialize Q value to 0
        Q = torch.tensor(0.0, dtype=torch.float)

        n, m = X.shape[0], X_prime.shape[0]

        for k, theta in enumerate(K):
            theta = torch.from_numpy(theta).float()
            for i in range(n):
                for j in range(m):
                    term1 = (
                        mu[k][i, j]
                        * (torch.dot(theta, X[i]) - torch.dot(theta, X_prime[j])) ** 2
                    )
                    term2 = lambda_val * nu[i, j] * (self.model(X[i]) - self.model(X_prime[j])) ** 2
                    Q += term1 + term2.item()

        # Compute gradient
        Q.backward()

        return X.grad
