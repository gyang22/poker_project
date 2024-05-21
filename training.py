from PokerAgents import DQN, DQNPokerAgent, ReplayMemory, Transition
import numpy as np
import torch
from torch import nn 
from PokerGameEnvironment import PokerGameEnvironment
from PokerGame import PokerGame
from RandomPlayer import RandomPlayer
import gymnasium as gym



num_episodes = 1000
update_freq = 10

starting_fortune = 1000.0
raise_factor = 1.2

agent = DQNPokerAgent(state_size=373, action_size=4, lr=0.0001, gamma=0.99, epsilon=0.9, 
                      epsilon_decay=0.05, memory_capacity=600, batch_size=64, starting_fortune=starting_fortune)

env = PokerGameEnvironment(PokerGame(1, 2, 
    [RandomPlayer(i, starting_fortune=starting_fortune, raise_factor=raise_factor) for i in range(5)] + [agent]), 5)

button = 0
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    data = env.play(button)
    
    for entry in data: #current_data = [self.flatten_state(), action, reward, None, done]
        state = entry[0]

        agent.remember(state, action, reward, next_state, done)

    #agent.train_step()

    #agent.decay_epsilon()
    
    #if episode % update_freq == 0:
        #agent.update_target_model()

    print(f"Episode {episode} done.")
    button = button + 1 % env.NUM_PLAYERS