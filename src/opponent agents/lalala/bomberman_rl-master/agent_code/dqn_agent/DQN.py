import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional

# Hyper parameters
BATCH_SIZE = 32  # Sample size
LR = 0.01  # Learning rate
GAMMA = 0.9  # Reward discount
TARGET_REPLACE_ITER = 100  # Frequency of target network updates
MEMORY_CAPACITY = 5000  # Memory capacity
N_ACTIONS = 6  # Number of actions
N_STATES = 2  # Number of states


# Define Networks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 50)      # First fully connected layer
        self.fc1.weight.data.normal_(0, 0.1)    # Weights initialization (normal distribution with mean 0 and variance 0.1)
        self.out = nn.Linear(50, N_ACTIONS)     # Second fully connected layer
        self.out.weight.data.normal_(0, 0.1)    # Weights initialization (normal distribution with mean 0 and variance 0.1)

    def forward(self, x):   # x is state
        # Connect the input layer to the hidden layer and use the excitation function ReLU to process the values after passing through the hidden layer
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)     # Connect the hidden layer to the output layer to get the final output value
        return actions_value