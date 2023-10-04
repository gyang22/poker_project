import random

"""
Class representing a standard random, n-sided die (six sides by default).
"""
class Dice:

    def __init__(self, sides=6):
        self.num_sides = sides

    
    def roll(self):
        return random.randint(1, self.num_sides)


"""
Class representing a specialized die which only rolls in a specific order that is given on instantiation.
"""
class PredeterminedDice(Dice):

    def __init__(self, roll_sequence, sides=6):
        super().__init__(sides)
        self.sequence = roll_sequence

    
    def roll(self):
        while True:
            for roll in self.sequence:
                yield roll
