import gymnasium as gym
from PokerGameEnvironment import PokerGameEnvironment
from PokerAgents import ActorCriticRNN
import tensorflow as tf
import numpy as np
from PokerGame import PokerGame


env = PokerGameEnvironment()
model = ActorCriticRNN(num_actions=4)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def act():
    action_probabilites, value = model.call(tf.convert_to_tensor(env.flatten_state()), dtype=tf.float32)
    action_probabilites = action_probabilites.numpy()[0]

    action = np.random.choice(np.arange(len(action_probabilites)), p=action_probabilites)

    return action, value

def preflop(game: PokerGame, button_position: int):
    active_player = (button_position + 1) % game.NUM_PLAYERS

    for p in game.players:
        p_cards = [game.deck.deal() for _ in range(2)]
        p.add_to_hand(p_cards)

    

button_position = 0

for episode in range(100):
    done = False
    env.reset()
    
    players = []

    game = PokerGame(env.SMALL_BLIND, env.BIG_BLIND, players)
    folded_indexes = set()

    button_position = (button_position + 1) % game.NUM_PLAYERS

    while not done:
        
        preflop(game, button_position)
        

        # check action validity

        # simulate other player moves, how to update model 

        action, value = act()

        next_state, reward, done, _, _ = env.step(action)


