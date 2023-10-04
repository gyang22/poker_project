
"""
Class representing a single card, with suit and rank and associated methods.
"""
class Card:

    """
    0 = CLUBS
    1 = DIAMONDS
    2 = HEARTS
    3 = SPADES

    1 = ACE
    2 = 2
    ...
    11 = JACK
    12 = QUEEN
    13 = KING
    """
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank


    def name_form(self):
        return f"{self.__num_to_rank(self.rank)} of {self.__num_to_suit(self.suit)}"
    

    def __num_to_suit(self, num):
        if num == 0:
            return "Clubs"
        elif num == 1:
            return "Diamonds"
        elif num == 2:
            return "Hearts"
        elif num == 3:
            return "Spades"
        else:
            raise Exception("Not a valid suit, must be between [0, 3].")
    
    
    def __num_to_rank(self, num):
        if num == 1:
            return "Ace"
        elif num >= 2 and num <= 10:
            return num
        elif num == 11:
            return "Jack"
        elif num == 12:
            return "Queen"
        elif num == 13:
            return "King"
        else:
            raise Exception("Not a valid rank, must be between [1, 13].")


    def __repr__(self):
        return self.name_form()
    

    def __str__(self):
        return self.name_form()