import gymnasium as gym
from PokerGameEnvironment import PokerGameEnvironment
from PokerAgents import ActorCriticRNN
import tensorflow as tf
import numpy as np

env = PokerGameEnvironment()
model = ActorCriticRNN(num_actions=4)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)





for episode in range(100):
    done = False
    env.reset()

    while not done:
        action_probabilites, value = model.call(tf.convert_to_tensor(env.flatten_state()), dtype=tf.float32)
        action_probabilites = action_probabilites.numpy()[0]

        action = np.random.choice(np.arange(len(action_probabilites)), p=action_probabilites)

        # check action validity

        # simulate other player moves, how to update model 

        next_state, reward, done, _, _ = env.step(action)


