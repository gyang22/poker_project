import torch
from torch import nn 
from collections import deque, namedtuple
import random
import numpy as np



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        x = self.hidden(x)
        return x


class DQNPokerAgent:

    def __init__(self, state_size, action_size, lr, gamma, epsilon, epsilon_decay, memory_capacity, batch_size, starting_fortune):
        self.state_size = state_size
        self.action_size = action_size

        self.hand = []
        self.readable_hand = []
        self.current_bet = 0
        self.previous_bet = 0
        self.balance = starting_fortune

        self.model = DQN(state_size, action_size)

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.memory = ReplayMemory(memory_capacity)

        self.batch_size = batch_size

        self.target_model = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.update_target_model()

    def add_to_hand(self, cards):
        self.readable_hand.extend(cards)
        
    def bet(self, amount: float):
        self.current_bet = min(self.balance, amount)
        self.balance -= self.current_bet
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def is_all_in(self):
        if self.balance == 0:
            return True
        return False

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # Get the Q-values for the given state
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.model(state_tensor)
            
            # Determine the range of Q-values
            q_min, q_max = q_vals.min().item(), q_vals.max().item()
            
            # Generate a random action within the range of Q-values
            random_action = q_min + (q_max - q_min) * torch.rand_like(q_vals)
            
            return random_action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_vals = self.model(state_tensor)
        
        return q_vals
    

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)
        done_batch = torch.FloatTensor(batch.done)

        current_q_values = self.model(state_batch).gather(1, action_batch)
        next_q_values = self.target_model(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.functional.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))


    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay