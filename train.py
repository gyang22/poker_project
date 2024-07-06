import torch
import torch.nn as nn
import torch.optim as optim
from environment import PokerEnvironment
from agent import PokerAgent
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from replay_buffer import ReplayBuffer



def save_checkpoint(agents, optimizers, generation, file_path="checkpoint.pkl"):
    checkpoint = {
        'agents': [agent.state_dict() for agent in agents],
        'optimizers': [optimizer.state_dict() for optimizer in optimizers],
        'generation': generation,
        'epsilon': agents[0].epsilon 
    }
    with open(file_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(file_path="checkpoint.pkl"):
    with open(file_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint

def train_and_evolve(env, num_generations, num_episodes, agents, optimizers, buffer, batch_size, start_generation=0):
    best_scores = []
    action_distributions = []

    for generation in range(start_generation, num_generations):
        scores = np.zeros(len(agents))
        action_counts = np.zeros(3) 

        for episode in range(num_episodes):
            env.reset()
            episode_rewards = np.zeros(len(agents))
            game_memory = [[] for _ in range(len(agents))]

            done = False
            while not done:
                current_player = env.current_player

                if not env.active_players[current_player]:
                    env.current_player = (env.current_player + 1) % env.num_players
                    continue

                agent = agents[current_player]
                state = env.get_state(current_player)
                flattened_state = PokerEnvironment.flatten_state(state)

                action, raise_amount = agent.act(state)
                action_counts[action] += 1

                next_state, rewards, done = env.step(action, raise_amount)
                reward = rewards[current_player] 
                # if action == 0:
                #     reward -= 1
                flattened_next_state = PokerEnvironment.flatten_state(next_state)
                game_memory[current_player].append((flattened_state, action, reward, flattened_next_state, done))

                episode_rewards[current_player] += reward

            scores += episode_rewards

            for i, agent in enumerate(agents):
                total_reward = episode_rewards[i]
                num_actions = len(game_memory[i])
                distributed_reward = total_reward / num_actions if num_actions > 0 else 0

                for state, action, reward, next_state, done in game_memory[i]:
                    buffer.add(state, action, distributed_reward, next_state, done)
           
    
        if len(buffer) >= batch_size:
            for i, agent in enumerate(agents):
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = agent(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_values = agent(next_states)
                    next_q_values = next_q_values.max(1)[0]
                    expected_q_values = rewards + (1 - dones) * next_q_values

                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()

        best_scores.append(np.max(scores))
        action_distributions.append(action_counts.tolist())  

        print(f"Generation {generation + 1}: Best Score: {np.max(scores)}")
        print("Scores:", scores)
        save_checkpoint(agents, optimizers, generation + 1)

 
        top_indices = np.argsort(scores)[-3:]
        top_agents = [agents[i] for i in top_indices]

       
        new_agents = []
        for i in range(len(top_agents)):
            new_agents.append(top_agents[i])
            new_agents.append(mutate_agent(top_agents[i]))
        agents = new_agents

  
        for agent in agents:
            agent.epsilon = max(agent.epsilon * 0.99, 0.01)

    return best_scores, action_distributions, agents


def plot_results(generations, best_scores, action_distributions):
   
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Best Scores over Generations', 'Action Distribution Ratios'))

    fig.add_trace(go.Scatter(x=generations, y=best_scores, mode='lines+markers', name='Best Scores'), row=1, col=1)
    fig.update_xaxes(title_text='Generation', row=1, col=1)
    fig.update_yaxes(title_text='Best Score', row=1, col=1)

    action_ratios = np.array(action_distributions) / np.sum(action_distributions, axis=1, keepdims=True)
    action_labels = ['Fold', 'Call/Check', 'Raise']

    for i, label in enumerate(action_labels):
        fig.add_trace(go.Scatter(x=generations, y=action_ratios[:, i], mode='lines+markers', name=label), row=1, col=2)

    fig.update_xaxes(title_text='Generation', row=1, col=2)
    fig.update_yaxes(title_text='Action Ratio', row=1, col=2)

    fig.update_layout(title='Training Progress')
    fig.show()



def select_top_agents(agents, scores, top_n=3):
    top_indices = np.argsort(scores)[-top_n:]
    return [agents[i] for i in top_indices]

def mutate_agent(agent):
    state_size = agent.input_layer.in_features
    action_size = agent.output_layer.out_features - 1
    new_agent = PokerAgent(state_size, action_size)
    new_agent.load_state_dict(agent.state_dict())
    new_agent.epsilon = agent.epsilon 
    for param in new_agent.parameters():
        param.data += torch.randn_like(param) * 0.02
    return new_agent

def save_agents(agents, filename):
    with open(filename, 'wb') as f:
        pickle.dump(agents, f)

def load_agents(filename):
    with open(filename, 'rb') as f:
        agents = pickle.load(f)
    return agents

if __name__ == "__main__":
    restart = True
    num_players = 6
    env = PokerEnvironment(num_players=num_players, starting_chips=1000)
    num_generations = 1000
    num_episodes = 10
    batch_size = 32

    state_size = PokerEnvironment.flatten_state(env.get_state(0)).shape[0]
    action_size = 3
    
    checkpoint_file = "checkpoint.pkl"
    start_generation = 0

    if restart and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file)
        agents = [PokerAgent(state_size, action_size) for _ in range(num_players)]
        for agent, state_dict in zip(agents, checkpoint['agents']):
            agent.load_state_dict(state_dict)
        optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in agents]
        for optimizer, state_dict in zip(optimizers, checkpoint['optimizers']):
            optimizer.load_state_dict(state_dict)
        start_generation = checkpoint['generation']
        epsilon = checkpoint['epsilon']
        for agent in agents:
            agent.epsilon = epsilon
        print(f"Resuming from generation {start_generation}")
    else:
        agents = [PokerAgent(state_size, action_size) for _ in range(num_players)]
        optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in agents]
    

    buffer = ReplayBuffer(buffer_size=10000)

    best_scores, action_distributions, agents = train_and_evolve(env, num_generations, num_episodes, agents, optimizers, buffer, batch_size, start_generation)
    generations = list(range(num_generations))
    plot_results(generations, best_scores, action_distributions)

    save_agents(agents, 'trained_agents.pkl')


