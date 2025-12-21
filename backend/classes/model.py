import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initializes the DQN model.

        Args:
            input_dim: Dimension of the input state (flattened).
            output_dim: Number of possible actions.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def build_model(input_dim: int, output_dim: int, learning_rate: float = 0.001):
    """
    Builds and returns the DQN model along with its optimizer.

    Args:
        input_dim: Dimension of the input state (flattened).
        output_dim: Number of possible actions.
        learning_rate: Learning rate for the optimizer.

    Returns:
        model: The DQN model.
        optimizer: The optimizer for training the model.
    """
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer
