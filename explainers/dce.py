from abc import ABC, abstractmethod
import torch
from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
from typing import Optional
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.logger_config import setup_logger
import math

logger = setup_logger()


class DistributionalCounterfactualExplainer:
    def __init__(
        self,
        model,
        df_X,
        explain_columns,
        y_target,
        lr=0.1,
        init_eta=0.5,
        n_proj=50,
        delta=0.1,
    ):
        self.X = df_X.values
        # Find indices of explain_columns in df_X
        self.explain_indices = [df_X.columns.get_loc(col) for col in explain_columns]

        self.explain_columns = explain_columns

        # Set the device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = torch.from_numpy(self.X).float().to(self.device)

        # Move model to the appropriate device
        self.model = model.to(self.device)

        # Transfer data to the device
        self.X_prime = self.X.clone()

        noise = torch.randn_like(self.X_prime[:, self.explain_indices]) * 0.01
        self.X[:, self.explain_indices] = (
            self.X_prime[:, self.explain_indices] + noise
        ).to(self.device)

        self.X.requires_grad_(True).retain_grad()
        self.best_X = None
        self.Qx_grads = None
        self.optimizer = optim.SGD([self.X], lr=lr)

        self.y = self.model(self.X)
        self.y_prime = y_target.clone().to(self.device)
        self.best_y = None

        self.swd = SlicedWassersteinDivergence(
            self.X_prime[:, self.explain_indices].shape[1], n_proj=n_proj
        )
        self.wd = WassersteinDivergence()

        self.Q = torch.tensor(torch.inf, dtype=torch.float, device=self.device)
        self.best_gap = np.inf

        self.init_eta = torch.tensor(init_eta, dtype=torch.float, device=self.device)

        self.delta = delta
        self.found_feasible_solution = False

    def _update_Q(self, mu_list, nu, eta):
        n, m = (
            self.X[:, self.explain_indices].shape[0],
            self.X_prime[:, self.explain_indices].shape[0],
        )

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
                            torch.dot(theta, self.X[:, self.explain_indices][i])
                            - torch.dot(theta, self.X_prime[:, self.explain_indices][j])
                        )
                        ** 2
                    )
        self.term1 /= torch.tensor(
            self.swd.n_proj, dtype=torch.float, device=self.device
        )

        # Compute the second term
        self.term2 = torch.tensor(0.0, dtype=torch.float)
        for i in range(n):
            for j in range(m):
                self.term2 += (
                    nu[i, j] * (self.model(self.X[i]) - self.y_prime[j]) ** 2
                ).item()

        self.Q = (1 - eta) * self.term1 + eta * self.term2

    def _update_X_grads(self, mu_list, nu, eta, tau):
        n, m = (
            self.X[:, self.explain_indices].shape[0],
            self.X_prime[:, self.explain_indices].shape[0],
        )
        thetas = [
            torch.from_numpy(theta).float().to(self.device) for theta in self.swd.thetas
        ]

        # Obtain model gradients with a dummy backward pass
        outputs = self.model(self.X)
        loss = outputs.sum()

        # Ensure gradients are zeroed out before backward pass
        self.X.grad = None
        loss.backward()
        model_grads = self.X.grad[
            :, self.explain_indices
        ].clone()  # Store the gradients

        # Compute the first term
        X_proj = torch.stack(
            [torch.matmul(self.X[:, self.explain_indices], theta) for theta in thetas],
            dim=1,
        )  # [n, num_thetas]
        X_prime_proj = torch.stack(
            [
                torch.matmul(self.X_prime[:, self.explain_indices], theta)
                for theta in thetas
            ],
            dim=1,
        )  # [m, num_thetas]

        # Use broadcasting to compute differences for all i, j
        differences = (
            X_proj[:, :, None] - X_prime_proj.T[None, :, :]
        )  # Shape [n, num_thetas, m]

        # Multiply by mu and sum over j
        gradient_term1_matrix = torch.stack(
            [mu.to(self.device) * differences[:, k, :] for k, mu in enumerate(mu_list)],
            dim=1,
        )  # [n, num_thetas, m]
        gradient_term1 = torch.sum(
            gradient_term1_matrix, dim=2
        )  # Shape [n, num_thetas]

        # Weight by theta to get the gradient
        gradient_term1 = torch.matmul(
            gradient_term1, torch.stack(thetas)
        )  # Shape [n, d]

        # Compute the second term
        diff_model = self.model(self.X).unsqueeze(1) - self.y_prime.reshape(
            len(self.y_prime), 1
        )
        nu = nu.to(self.device)

        self.nu = nu
        self.diff_model = diff_model
        self.model_grads = model_grads

        gradient_term2 = (nu.unsqueeze(-1) * diff_model * model_grads.unsqueeze(1)).sum(
            dim=1
        )

        self.Qx_grads = (1 - eta) * gradient_term1 + eta * gradient_term2
        # self.Qx_grads = gradient_term2
        self.X.grad.zero_()
        self.X.grad[:, self.explain_indices] = self.Qx_grads * tau

    def __perform_SGD(self, past_Qs, eta, tau):
        # Reset the gradients
        self.optimizer.zero_grad()

        # Compute the gradients for self.X[:, self.explain_indices]
        self._update_X_grads(mu_list=self.swd.mu_list, nu=self.wd.nu, eta=eta, tau=tau)

        # Perform an optimization step
        self.optimizer.step()

        # Update the Q value, X_all, and y by the newly optimized X
        self._update_Q(mu_list=self.swd.mu_list, nu=self.wd.nu, eta=eta)
        self.y = self.model(self.X)

        # logger.info(f"\t  Qx_grads = {self.Qx_grads}")

        # Check for convergence using moving average of past Q changes
        past_Qs.pop(0)
        past_Qs.append(self.Q.item())
        avg_Q_change = (past_Qs[-1] - past_Qs[0]) / 5
        return avg_Q_change

    def optimize_without_chance_constraints(
        self,
        eta=0.9,
        max_iter: Optional[int] = 100,
        tau=10,
        tol=1e-6,
    ):
        logger.info("Optimization (without chance constraints) started")
        past_Qs = [float("inf")] * 5  # Store the last 5 Q values for moving average
        for i in range(max_iter):
            self.swd.distance(
                X_s=self.X[:, self.explain_indices],
                X_t=self.X_prime[:, self.explain_indices],
                delta=self.delta,
            )
            self.wd.distance(y_s=self.y, y_t=self.y_prime, delta=self.delta)

            avg_Q_change = self.__perform_SGD(past_Qs, eta=eta, tau=tau)

            logger.info(
                f"Iter {i+1}: Q = {self.Q}, term1 = {self.term1}, term2 = {self.term2}"
            )

            if abs(avg_Q_change) < tol:
                logger.info(f"Converged at iteration {i+1}")
                break

        self.best_X = self.X.clone().detach()
        self.best_y = self.y.clone().detach()

    def optimize(
        self,
        U_1: float,
        U_2: float,
        alpha=0.05,
        l=0.2,
        r=1,
        kappa=0.05,
        max_iter: Optional[int] = 100,
        tau=10,
        tol=1e-6,
    ):
        self.interval_left = l
        self.interval_right = r

        logger.info("Optimization started")
        past_Qs = [float("inf")] * 5  # Store the last 5 Q values for moving average
        for i in range(max_iter):
            self.swd.distance(
                X_s=self.X[:, self.explain_indices],
                X_t=self.X_prime[:, self.explain_indices],
                delta=self.delta,
            )
            self.wd.distance(y_s=self.y, y_t=self.y_prime, delta=self.delta)
            _, self.Qv_upper = self.wd.distance_interval(
                self.y, self.y_prime, delta=self.delta, alpha=alpha
            )
            _, self.Qu_upper = self.swd.distance_interval(
                self.X[:, self.explain_indices],
                self.X_prime[:, self.explain_indices],
                delta=self.delta,
                alpha=alpha,
            )

            (
                eta,
                self.interval_left,
                self.interval_right,
            ) = self._get_eta_interval_narrowing(
                U_1=U_1,
                U_2=U_2,
                Qu_upper=self.Qu_upper,
                Qv_upper=self.Qv_upper,
                l=self.interval_left,
                r=self.interval_right,
                kappa=kappa,
            )

            logger.info(
                f"U_1-Qu_upper={U_1-self.Qu_upper}, U_2-Qv_upper={U_2-self.Qv_upper}"
            )
            logger.info(f"eta={eta}, l={self.interval_left}, r={self.interval_right}")

            avg_Q_change = self.__perform_SGD(past_Qs, eta=eta, tau=tau)

            if (U_1 - self.Qu_upper) < 0 or (U_2 - self.Qv_upper) < 0:
                gap = np.inf
            else:
                gap = (U_1 - self.Qu_upper) + (U_2 - self.Qv_upper)

            if gap < self.best_gap:
                self.best_gap = gap
                self.best_X = self.X.clone().detach()
                self.best_y = self.y.clone().detach()
                self.found_feasible_solution = True

            logger.info(
                f"Iter {i+1}: Q = {self.Q}, term1 = {self.term1}, term2 = {self.term2}"
            )

            if abs(avg_Q_change) < tol:
                logger.info(f"Converged at iteration {i+1}")
                break

        if not self.found_feasible_solution:
            self.best_gap = gap
            self.best_X = self.X.clone().detach()
            self.best_y = self.y.clone().detach()

    def _get_eta_set_shrinking(self):
        return 0.99

    def _get_eta_interval_narrowing(
        self, U_1, U_2, Qu_upper, Qv_upper, l=0, r=1, kappa=0.05
    ):
        """
        Implements the interval narrowing algorithm.

        Parameters:
        Qv_upper, Qu_upper (float): Upper confidence limits.
        l, r (float): Current lower and upper bounds of the interval.
        kappa (float): Contraction factor for the interval.

        Returns:
        eta (float): The point in the interval [l, r] that maximizes the objective function.
        l, r (float): Updated lower and upper bounds of the interval.
        """

        if not math.isfinite(Qv_upper):
            return l, l, r

        if not math.isfinite(Qu_upper):
            return r, l, r

        eta = self.__choose_eta_within_interval(
            a=U_1 - Qu_upper, b=U_2 - Qv_upper, l=l, r=r
        )

        # Narrow the interval
        if eta > (l + r) / 2:
            l = l + kappa * (r - l)
        else:
            r = r - kappa * (r - l)
        return eta, l, r

    def __choose_eta_within_interval(self, a, b, l, r):
        if (a < 0 and b >= 0) or (a >= 0 and b < 0):
            return l if a < 0 else r
        else:
            # For a, b both negative or both positive
            if a < 0 and b < 0:
                # Both negative: more weight to the more negative
                eta_proportion = b / (a + b)
            else:
                # Both positive: more weight to the less positive
                eta_proportion = a / (a + b)

            # Scale eta to be within the range [l, r]
            return l + eta_proportion * (r - l)
