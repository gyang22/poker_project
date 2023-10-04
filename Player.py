from Card import Card

"""
Class representing a player of a game, with a name and hand of cards. Cards can be hidden or shown.
"""
class Player:

    def __init__(self, name):
        self.name = name
        self.hand = []
        self.hidden = []


    """
    Prints the current cards in the player's hand, first argument determines whether all cards are shown or only non-hidden ones.
    """
    def show_hand(self, show_all=False):
        if not show_all:
            print([self.hand[i] if not self.hidden[i] else "~ Hidden ~" for i in range(len(self.hand))])
        else:
            print(self.hand)
        
    
    """
    Adds a card to the player's hand, with optional second argument to hide the card.
    """
    def add_to_hand(self, card, is_hidden=False):
        if type(card) != Card or type(is_hidden) != bool:
            raise Exception("Only Card objects can be added to the hand.")
        self.hand.append(card)
        self.hidden.append(is_hidden)


    """
    Removes and returns the specified card at the index from the player's hand, zero-indexed.
    """
    def remove_from_hand(self, index):
        self.hidden.pop(index)
        return self.hand.pop(index)


    def __str__(self):
        return f"{self.name} | {len(self.hand)} card(s)"
    

    def __repr__(self):
        return f"{self.name} | {len(self.hand)} card(s)"
    


"""
Class representing a specialized form of Player who can make bets.
"""
class BettingPlayer(Player):

    def __init__(self, name, cash):
        super().__init__(name)
        self.balance = cash
        self.current_bet = 0

    
    def bet(self, amount):
        amount = min(amount, self.balance)
        self.balance -= amount
        self.current_bet += amount
        return amount

    