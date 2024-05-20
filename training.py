from PokerAgents import DQN, DQNPokerAgent, ReplayMemory, Transition
import numpy as np
import torch
from torch import nn 
from PokerGameEnvironment import PokerGameEnvironment




num_episodes = 1000
update_freq = 10

env = PokerGameEnvironment()
agent = DQNPokerAgent(state_size=10, action_size=5)

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        state_data = env.step()
        next_state = state_data['state']
        reward = state_data['reward']
        done = state_data['done']
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        if done:
            agent.train_step()

    agent.decay_epsilon()
    
    if episode % update_freq == 0:
        agent.update_target_model()

    print(f"Episode {episode} done.")