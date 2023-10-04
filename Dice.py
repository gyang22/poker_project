import random

"""
Class representing a standard random, n-sided die (six sides by default).
"""
class Dice:

    def __init__(self, sides=6):
        self.num_sides = sides

    
    def roll(self):
        return random.randint(1, self.num_sides)
