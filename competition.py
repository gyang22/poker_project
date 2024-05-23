import copy
import time
from PokerAgents import DQNPokerAgent
from PokerGame import PokerGame
from PokerGameEnvironment import PokerGameEnvironment


starting_fortune = 1000.0

agent = DQNPokerAgent(state_size=373, action_size=4, lr=0.0001, gamma=0.99, epsilon=0.95, 
                      epsilon_decay=0.999, min_epsilon=0.05, memory_capacity=2000, batch_size=128, starting_fortune=starting_fortune)

env = PokerGameEnvironment(PokerGame(1, 2, [agent, copy.deepcopy(agent) for _ in range(5)], [0, 1, 2, 3, 4, 5, 6]), compete=True)





