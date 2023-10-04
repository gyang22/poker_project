import numpy as np
import random
from Card import Card

"""
Class representing a deck of cards, with 52 distinct cards and associated methods.
"""
class Deck:

    NUM_CARDS = 52
    NUM_SUITS = 4
    NUM_RANKS = 13

    """
    Cards are initialized in predetermined unshuffled order, with 52 default cards.
    """
    def __init__(self):
        self.cards = []
        for suit in range(Deck.NUM_SUITS):
            for rank in range(Deck.NUM_RANKS):
                self.cards.append(Card(suit, rank + 1))
        self.wastepile = []


    """
    Shuffles this deck in place, type of shuffle depends on first argument: str type.
    Type can be "faro", "bridge".
    Returns none.
    """
    def shuffle(self, type):
        if len(self.cards) != Deck.NUM_CARDS:
            raise Exception("Not all cards are in main pile, move cards from waste pile back to main pile.")
        if type == "faro":
            self.__faro_shuffle()
        elif type == "bridge":
            self.__bridge_shuffle()
        elif type == "random":
            self.__random_shuffle()
        else:
            raise Exception("Not an accepted type of shuffle; please use 'faro', 'bridge', 'random'.")

    
    def __faro_shuffle(self):
        new_deck = []
        for index in range(Deck.NUM_CARDS // 2):
            new_deck.append(self.cards[index])
            new_deck.append(self.cards[Deck.NUM_CARDS // 2 + index])
        self.cards = new_deck

    def __bridge_shuffle(self):
        pass

    def __random_shuffle(self):
        new_deck = []
        for i in range(Deck.NUM_CARDS):
            new_deck.append(self.cards.pop(random.randint(0, Deck.NUM_CARDS - i - 1)))
        self.cards = new_deck


    """
    Prints main and waste piles.
    """
    def display_deck(self):
        print("Main Pile:")
        print(self.cards)
        print("Waste Pile:")
        print(self.wastepile)


    """
    Removes and returns the card on the top of the deck.
    """
    def deal(self):
        return self.cards.pop()


    """
    Adds cards(s) to the waste pile. If this card is already in the main pile or waste pile, an exception is raised.
    """
    def add_to_waste(self, *cards):
        for card in cards:
            if card in self.cards or card in self.wastepile:
                raise Exception("Duplicate of card detected, make sure that this is the correct card.")
            else:
                self.wastepile.append(card)


    def return_to_main_pile(self):
        self.cards += self.wastepile
        self.wastepile = []


    def main_pile_size(self):
        return len(self.cards)


    def waste_pile_size(self):
        return len(self.wastepile)


if __name__ == "__main__":
    deck1 = Deck()
    deck1.display_deck()
    deck1.shuffle("random")
    deck1.display_deck()