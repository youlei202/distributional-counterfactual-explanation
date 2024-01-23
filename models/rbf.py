import torch.nn as nn
import torch

class RBFNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RBFNet, self).__init__()
        self.hidden_dim = hidden_dim

        # Parameters for the RBF layer
        self.centers = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.sigmas = nn.Parameter(torch.Tensor(hidden_dim))

        # Output layer
        self.output = nn.Linear(hidden_dim, 1)

        # Initialize parameters
        self.init_params()

    def init_params(self):
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, x):
        # Calculate RBF layer outputs
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)
        out = torch.exp(-dist_sq / (2 * self.sigmas**2))

        # Linear layer
        out = self.output(out)
        return out
