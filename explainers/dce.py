from abc import ABC, abstractmethod
import torch
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from typing import Optional
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DistributionalCounterfactualExplainer:
    def __init__(self, model, X, epsilon=0.1, lambda_val=0.5, n_proj=50):
        """
        Initialize the counterfactual explainer.

        Parameters:
        - model: The predictive model to be explained.
        """
        self.model = model

        self.X = torch.from_numpy(X).float()
        self.X_prime = self.X.clone()

        self.X.requires_grad_(True).retain_grad()
        self.best_X = None

        self.y = self.model(self.X)
        self.y_prime = self.y.clone()

        self.best_y = None

        self.swd = SlicedWassersteinDivergence(X.shape[1], n_proj=n_proj)
        self.wd = WassersteinDivergence()

        # Introduce an optimizer for self.X

        self.Q = torch.tensor(torch.inf, dtype=torch.float)
        self.best_Q = self.Q

        self.epsilon = epsilon

        self.lambda_val = lambda_val

    def _compute_gradient_for_X(self, mu_list, nu):
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

        n, m = self.X.shape[0], self.X_prime.shape[0]

        # Compute the first term
        self.term1 = torch.tensor(0.0, dtype=torch.float)
        for k, theta in enumerate(self.swd.thetas):
            mu = mu_list[k]
            theta = torch.from_numpy(theta).float()
            for i in range(n):
                for j in range(m):
                    self.term1 += (
                        mu[i, j]
                        * (
                            torch.dot(theta, self.X[i])
                            - torch.dot(theta, self.X_prime[j])
                        )
                        ** 2
                    )
        self.term1 /= self.swd.n_proj

        # Compute the second term
        term = torch.tensor(0.0, dtype=torch.float)
        for i in range(n):
            for j in range(m):
                term += (
                    nu[i, j]
                    * (self.model(self.X[i]) - self.model(self.X_prime[j])) ** 2
                ).item()
        self.term2 = self.lambda_val * (self.epsilon - term)

        # Compute the objective function Q
        self.Q = self.term1 + self.term2

        # Compute gradient
        self.Q.backward()

        return self.X.grad

    def optimize(
        self,
        initial_lr: Optional[float] = 0.1,
        max_iter: Optional[int] = 100,
        tol=1e-6,
        weight_decay=0.0,
    ):

        optimizer = optim.Adam([self.X], lr=initial_lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=10, factor=0.5, verbose=True
        )

        past_Qs = [float("inf")] * 5  # Store the last 5 Q values for moving average
        for i in range(max_iter):
            optimizer.zero_grad()
            self.swd.distance(X_s=self.X, X_t=self.X_prime)
            self.wd.distance(y_s=self.y, y_t=self.y_prime)

            self._compute_gradient_for_X(mu_list=self.swd.mu_list, nu=self.wd.nu)
            optimizer.step()

            # scheduler.step(self.Q)

            self.y = self.model(self.X)

            if self.Q < self.best_Q:
                self.best_Q = self.Q.clone().detach()
                self.best_X = self.X.clone().detach()
                self.best_y = self.y.clone().detach()

            # Check for convergence using moving average of past Q changes
            past_Qs.pop(0)
            past_Qs.append(self.Q.item())
            avg_Q_change = (past_Qs[-1] - past_Qs[0]) / 5
            if abs(avg_Q_change) < tol:
                print(f"Converged at iteration {i+1}")
                break

            print(
                f"Iter {i+1}: Q = {self.Q}, term1 = {self.term1}, term2 = {self.term2}"
            )
