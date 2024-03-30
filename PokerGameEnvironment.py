import numpy as np
import tensorflow as tf

class PokerGameEnvironment:
    
    def __init__(self):

        self.state_space = {
            "player_hand": {
                "type": "int",
                "shape": (2, 2),
            }
        }

        self.action_space = {
            "type": "int",
            "num_values": 4
        }