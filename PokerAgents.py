import tensorflow as tf

class ActorCriticRNN(tf.keras.Model):

    def __init__(self, num_actions: int, num_units=128):
        super().__init__()

        self.actor = tf.keras.layers.Dense(num_actions)
        self.critic = tf.keras.layers.Dense(1)
        self.common = tf.keras.layers.LSTM(num_units, return_sequences=True, return_state=True)

    def call(self, inputs: tf.Tensor, states=None, return_state=False, training=False):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
        



