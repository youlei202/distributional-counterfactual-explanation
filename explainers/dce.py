from abc import ABC, abstractmethod
import torch
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from typing import Optional
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.logger_config import setup_logger

logger = setup_logger()


class DistributionalCounterfactualExplainer:
    def __init__(
        self, model, X, y_target, epsilon=0.1, lr=0.1, lambda_val=0.5, n_proj=50
    ):
        # Set the device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the appropriate device
        self.model = model.to(self.device)

        # Transfer data to the device
        self.X_prime = torch.from_numpy(X).float().to(self.device)
        noise = torch.randn_like(self.X_prime) * 0.01
        self.X = (self.X_prime + noise).to(self.device)

        self.X.requires_grad_(True).retain_grad()
        self.best_X = None
        self.Qx_grads = None
        self.optimizer = optim.Adam([self.X], lr=lr)

        self.y = self.model(self.X)
        self.y_prime = y_target.clone().to(self.device)
        self.best_y = None

        self.swd = SlicedWassersteinDivergence(X.shape[1], n_proj=n_proj)
        self.wd = WassersteinDivergence()

        self.Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)
        self.best_Q = self.Q

        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=self.device)
        self.lambda_val = torch.tensor(
            lambda_val, dtype=torch.float, device=self.device
        )

    def _update_Q(self, mu_list, nu):
        n, m = self.X.shape[0], self.X_prime.shape[0]

        thetas = [
            torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas
        ]

        # Compute the first term
        self.term1 = torch.tensor(0.0, dtype=torch.float).to(self.device)
        for k, theta in enumerate(thetas):
            mu = mu_list[k]
            mu = mu.to(self.device)
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
        self.term1 /= torch.tensor(
            self.swd.n_proj, dtype=torch.float, device=self.device
        )

        # Compute the second term
        term = torch.tensor(0.0, dtype=torch.float)
        for i in range(n):
            for j in range(m):
                term += (
                    nu[i, j]
                    * (self.model(self.X[i]) - self.model(self.X_prime[j])) ** 2
                ).item()
        self.term2 = self.lambda_val * (self.epsilon - term)

        self.Q = self.term1 + self.term2

    def _update_X_grads(self, mu_list, nu):
        n, m = self.X.shape[0], self.X_prime.shape[0]
        thetas = [
            torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas
        ]
        grads = torch.zeros_like(self.X).to(self.device)

        # Obtain model gradients with a dummy backward pass
        outputs = self.model(self.X)
        loss = outputs.sum()

        # Ensure gradients are zeroed out before backward pass
        self.X.grad = None
        loss.backward()
        model_grads = self.X.grad.clone()  # Store the gradients

        # Compute the first term
        for i in range(n):
            for k, theta in enumerate(thetas):
                mu = mu_list[k]
                for j in range(m):
                    diff1 = (
                        torch.dot(theta, self.X[i]) - torch.dot(theta, self.X_prime[j])
                    ).item()
                    grads[i].add_(
                        mu[i][j].item() * diff1 * theta
                    )  # Using in-place addition

        # Compute the second term
        # No need to loop through i and j. Instead, use broadcasting.
        diff_model = self.model(self.X).unsqueeze(1) - self.model(
            self.X_prime
        ).unsqueeze(0)
        nu = nu.to(self.device)
        grads -= (
            self.lambda_val * nu.unsqueeze(-1) * diff_model * model_grads.unsqueeze(1)
        ).sum(dim=1)

        self.Qx_grads = grads
        self.X.grad = self.Qx_grads

    def optimize(
        self,
        max_iter: Optional[int] = 100,
        tol=1e-6,
    ):
        past_Qs = [float("inf")] * 5  # Store the last 5 Q values for moving average
        for i in range(max_iter):
            self.swd.distance(X_s=self.X, X_t=self.X_prime)
            self.wd.distance(y_s=self.y, y_t=self.y_prime)

            # Reset the gradients
            self.optimizer.zero_grad()

            # Compute the gradients for self.X
            self._update_X_grads(mu_list=self.swd.mu_list, nu=self.wd.nu)

            # Perform an optimization step
            self.optimizer.step()

            # Update the Q value and y by the newly optimized X
            self._update_Q(mu_list=self.swd.mu_list, nu=self.wd.nu)
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
                logging.info(f"Converged at iteration {i+1}")
                break

            logging.info(
                f"Iter {i+1}: Q = {self.Q}, term1 = {self.term1}, term2 = {self.term2}"
            )
