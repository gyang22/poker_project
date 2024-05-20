import unittest
from Deck import Deck
from Player import Player, BettingPlayer
from Card import Card
from PokerGame import PokerGame
from PokerGameEnvironment import PokerGameEnvironment
import numpy as np


class TestPokerGameEnvironment(unittest.TestCase):

    test_env = PokerGameEnvironment(PokerGame(0, 1, [Player(s) for s in "012345"]), 0)
    
    def test_array_to_cards(self):
        test_hand = np.zeros(shape=(104,))
        test_hand[35] = 1
        test_hand[57] = 1
        test_hand = self.test_env.array_to_cards(test_hand)
        self.assertEqual(test_hand, [Card(1, 9), Card(0, 5)])



if __name__ == "main":
    unittest.main()