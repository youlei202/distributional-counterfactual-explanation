import torch.nn as nn

class BlackBoxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super(BlackBoxModel, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Second fully connected layer (hidden layer)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))