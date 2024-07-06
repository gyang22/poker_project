import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from environment import PokerEnvironment
import random


class PokerAgent(nn.Module):

    def __init__(self, state_size, action_size):
        super(PokerAgent, self).__init__()
        self.input_layer = nn.Linear(state_size, 128)
        self.h1 = nn.Linear(128, 128)
        self.h2 = nn.Linear(128, 128)
        self.h3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_size+1)

        self.epsilon = 0.1

    def forward(self, x):
        x = F.leaky_relu(self.input_layer(x))
        x = F.leaky_relu(self.h1(x))
        x = F.leaky_relu(self.h2(x))
        x = F.leaky_relu(self.h3(x))
        return F.relu(self.output_layer(x))
    
    def act(self, state):
        if random.random() < self.epsilon:
            action = random.choice([0, 1, 2])
            raise_amount = random.uniform(0, 100)
        else:
            flattened_state = PokerEnvironment.flatten_state(state)
            state_tensor = torch.FloatTensor(flattened_state)

            with torch.no_grad():
                outputs = self.forward(state_tensor)
            action_probs = outputs[:-1]
            raise_amount = outputs[-1].item()

            action = torch.argmax(action_probs).item()
        return action, raise_amount
    
    