from PokerAgents import DQN, DQNPokerAgent, ReplayMemory, Transition
import numpy as np
import torch
from torch import nn 
from PokerGameEnvironment import PokerGameEnvironment
from PokerGame import PokerGame
from RandomPlayer import RandomPlayer
import gymnasium as gym
import matplotlib.pyplot as plt
import time



def split_reward(reward: float, data: list):
    return [reward / len(data) for _ in range(len(data))]


start_time = time.time()


reward_results = []

cumulative_reward = 0
cumulative_rewards = []

action_ratios = [[0] * 4]
era = 0

num_episodes = 10000
update_freq = 10

starting_fortune = 1000.0
raise_factor = 1.2

agent = DQNPokerAgent(state_size=373, action_size=4, lr=0.0001, gamma=0.99, epsilon=0.95, 
                      epsilon_decay=0.999, min_epsilon=0.05, memory_capacity=2000, batch_size=128, starting_fortune=starting_fortune)

actions = [0] * agent.action_size

env = PokerGameEnvironment(PokerGame(1, 2, 
    [RandomPlayer(i, starting_fortune=starting_fortune, raise_factor=raise_factor) for i in range(5)] + [agent]), 5)

button = 0
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    data, total_reward, winner = env.play(button)

    reward_results.append(total_reward)

    cumulative_reward += total_reward
    cumulative_rewards.append(cumulative_reward)

    
    rewards = split_reward(total_reward, data)

    shifted_data = data[1:] + [None]
    
    for entry, reward, next_entry in zip(data, rewards, shifted_data): #current_data = [self.flatten_state(), action, reward, None, done]
        state = entry[0]
        action = entry[1]
        actions[action] += 1
        action_ratios[era][action] += 1
        net_reward = reward + entry[2]
        done = entry[4]
        if next_entry:
            next_state = next_entry[0]
            agent.remember(state, action, net_reward, next_state, done)
        else:
            next_state = state
            done = True
            agent.remember(state, action, net_reward, next_state, done)

    agent.train_step()

    agent.decay_epsilon()
    
    if episode % update_freq == 0:
        agent.update_target_model()
        env.game = PokerGame(1, 2, 
                [RandomPlayer(i, starting_fortune=starting_fortune, raise_factor=raise_factor) for i in range(5)] + [agent])
        env.game.players[env.player_id].balance = starting_fortune
        action_ratios.append([0] * 4)
        era += 1

        

    print(f"Episode {episode} done.")
    print(f"Total reward {total_reward}.")
    button = button + 1 % env.NUM_PLAYERS

end_time = time.time()

print(f"Training took {end_time - start_time} seconds to complete.")


print(actions)



fig, axs  = plt.subplots(2, 2, figsize=(15, 8))

axs[0, 0].plot(reward_results)
axs[0, 0].set_title('Reward By Episode')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Reward')

labels = ["fold", "call", "raise", "check"]
axs[0, 1].bar(labels, actions)
axs[0, 1].set_title('Action Count')
axs[0, 1].set_xlabel('Action')
axs[0, 1].set_ylabel('Number of Times Chosen')

for i, count in enumerate(actions):
    axs[0, 1].text(i, count + 0.5, str(count), ha='center')

axs[1, 0].plot(cumulative_rewards)
axs[1, 0].set_title('Cumulative Reward Over Time')
axs[1, 0].set_xlabel('Episode')
axs[1, 0].set_ylabel('Cumulative Reward')

for action_bar in action_ratios:
    axs[1, 1].bar([0, 1, 2, 3], action_bar)

plt.tight_layout()

plt.show()