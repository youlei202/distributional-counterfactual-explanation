import torch.nn as nn
import torch

class LinearSVM(nn.Module):
    """Linear SVM Classifier"""

    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


def svm_loss(outputs, labels):
    """Hinge loss for SVM"""
    return torch.mean(torch.clamp(1 - outputs.t() * labels, min=0))