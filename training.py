from PokerAgents import DQN, DQNPokerAgent, ReplayMemory, Transition
import numpy as np
import torch
from torch import nn 
from PokerGameEnvironment import PokerGameEnvironment
from PokerGame import PokerGame
from RandomPlayer import RandomPlayer
import gymnasium as gym
import matplotlib.pyplot as plt


def split_reward(reward: float, data: list):
    return [reward / len(data) for _ in range(len(data))]

reward_results = []

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
    
    data, total_reward, winner = env.play(button)

    reward_results.append(total_reward)
    if winner:
        env.game.players[winner].balance += total_reward
    
    rewards = split_reward(total_reward, data)

    shifted_data = data[1:] + [None]
    
    for entry, reward, next_entry in zip(data, rewards, shifted_data): #current_data = [self.flatten_state(), action, reward, None, done]
        state = entry[0]
        action = entry[1]
        if next_entry:
            next_state = next_entry[0]
            agent.remember(state, action, reward, next_state, done)
        else:
            next_state = state
            agent.remember(state, action, reward, next_state, True)

    agent.train_step()

    agent.decay_epsilon()
    
    if episode % update_freq == 0:
        agent.update_target_model()

    print(f"Episode {episode} done.")
    button = button + 1 % env.NUM_PLAYERS




x_values = list(range(len(reward_results)))


plt.plot(x_values, reward_results)
plt.show()