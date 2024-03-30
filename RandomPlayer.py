import numpy as np
from Card import Card

class RandomPlayer:

    def __init__(self, id: int, starting_fortune: float, raise_factor: float) -> None:
        self.id = id
        self.current_bet = 0
        self.actions = ["fold", "raise", "call", "check"]
        self.balance = starting_fortune
        self.raise_factor = raise_factor
        self.hand = np.zeros(104) 

    """
    Returns a str from ["fold", "raise", "call", "check"] or a subset depending on the current round, with uniform probability.
    """
    def act(self, round_number: int, table_highest_bet: float):

        previous_bet = self.current_bet


        # Preflop; valid actions are raise, call, and fold.
        if round_number == 0:

            # If another player has a higher bet, raise, call, or fold.
            if table_highest_bet > self.current_bet:
                action = np.random.choice(self.actions[:3])
                if action == "raise":
                    self.__raise(table_highest_bet)
                elif action == "call":
                    self.__call(table_highest_bet)
                elif action == "fold":
                    self.__fold()

            # Otherwise, currently the highest bet or matching, raise or fold.
            else:
                action = np.random.choice(self.actions[:2])
                if action == "raise":
                    self.__raise(table_highest_bet)
                elif action == "fold":
                    self.__fold()

            return action, previous_bet

        # Flop, turn, or river; valid actions are fold, raise, call, or check depending on the table highest bet.
        else:

            # If another player has a higher bet, raise, call, or fold.
            if table_highest_bet > self.current_bet:
                action = np.random.choice(self.actions[:3])
                if action == "raise":
                    self.__raise(table_highest_bet)
                elif action == "call":
                    self.__call(table_highest_bet)
                elif action == "fold":
                    self.__fold()

            # Otherwise, currently the highest bet or matching, raise, fold, or check.
            else:
                action = np.random.choice(self.actions[:2] + self.actions[3:])
                if action == "raise":
                    self.__raise(table_highest_bet)
                elif action == "fold":
                    self.__fold()
                elif action == "check":
                    self.__check()
                
            return action, previous_bet
        
    def bet(self, amount: float):
        self.current_bet = min(self.balance, amount)

            
    
    def __raise(self, table_highest_bet: float):
        self.current_bet = min(table_highest_bet * self.raise_factor, self.balance)

    def __call(self, table_highest_bet: float):
        self.current_bet = min(table_highest_bet, self.balance)

    def __fold(self):
        self.balance -= self.current_bet
        self.current_bet = 0

    def __check(self):
        pass


    def is_all_in(self):
        return self.current_bet == self.balance

    def get_current_bet(self):
        return self.current_bet
    
    def add_to_hand(self, card: Card):
        self.hand[card.suit * 13 + card.rank - 1] += 1