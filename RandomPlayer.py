import numpy as np
from Card import Card

class RandomPlayer:

    def __init__(self, id: int, starting_fortune: float, raise_factor: float) -> None:
        self.id = id
        self.current_bet = 0
        self.previous_bet = 0
        self.actions = ["fold", "raise", "call", "check"]   # TODO CHANGE
        self.balance = starting_fortune
        self.raise_factor = raise_factor
        self.hand = [0] * 104
        self.original_fortune = starting_fortune
        self.readable_hand = []

    """
    Returns a str from ["fold", "raise", "call", "check"] or a subset depending on the current round, with uniform probability.
    """
    def act(self, round_number: int, table_highest_bet: float, no_raises: bool):

        previous_bet = self.current_bet


        # Preflop; valid actions are raise, call, and fold.
        if round_number == 0:

            # If another player has a higher bet, raise, call, or fold.
            if table_highest_bet > self.current_bet:
                if no_raises:
                    action = np.random.choice([self.actions[0]] + [self.actions[2]], p=(0.30, 0.70))
                else:
                    action = np.random.choice(self.actions[:3], p=(0.60, 0.15, 0.25)) #.15/.3/.55
                
                if action == "raise":
                    self.__raise(table_highest_bet)
                elif action == "call":
                    self.__call(table_highest_bet)
                elif action == "fold":
                    self.__fold()

            # Otherwise, player bet is equal 
            else:
                if no_raises:
                    action = np.random.choice([self.actions[0]] + [self.actions[2]], p=(0.50, 0.50))
                else:   # only applies to big blind
                    action = np.random.choice(self.actions[:2], p=(0.1, 0.9))
                
                if action == "raise":
                    self.__raise(table_highest_bet)
                elif action == "call":
                    self.__call(table_highest_bet)
                elif action == "fold":
                    self.__fold()

            return action, previous_bet

        # Flop, turn, or river; valid actions are fold, raise, call, or check depending on the table highest bet.
        else:

            # If another player has a higher bet, raise, call, or fold.
            if table_highest_bet > self.current_bet:
                action = np.random.choice(self.actions[:3], p=(0.15, 0.3, 0.55))
                if action == "raise":
                    self.__raise(table_highest_bet)
                elif action == "call":
                    self.__call(table_highest_bet)
                elif action == "fold":
                    self.__fold()

            # Otherwise, currently the highest bet or matching, raise, fold, or check.
            else:
                action = np.random.choice(self.actions[:2] + self.actions[3:], p=(0.60, 0.15, 0.25)) # 0.15, 0.3, 0.55
                if action == "raise":
                    self.__raise(table_highest_bet)
                elif action == "fold":
                    self.__fold()
                elif action == "check":
                    self.__check()
                
            return action, previous_bet
        
    def bet(self, amount: float):
        self.current_bet = min(self.balance, amount)
        self.balance -= self.current_bet
    
    def __raise(self, table_highest_bet: float):
        self.current_bet = min(table_highest_bet * self.raise_factor, self.balance)
        self.balance -= self.current_bet

    def __call(self, table_highest_bet: float):
        self.current_bet = min(table_highest_bet, self.balance)
        self.balance -= self.current_bet

    def __fold(self):
        self.current_bet = 0

    def __check(self):
        pass


    def is_all_in(self):
        return self.balance == 0 and self.current_bet > 0

    def get_current_bet(self):
        return self.current_bet
    
    def add_to_hand(self, cards):
        for i, card in enumerate(cards):
            self.hand[card.suit * 13 + card.rank - 1 + 52 * i] += 1
        self.readable_hand.extend(cards)