from Deck import Deck
from Player import Player, BettingPlayer

class BlackjackGame:

    def __init__(self, *players):
        self.deck = Deck()
        self.dealer = Player("Dealer")
        self.players = []
        for p in players:
            if type(p) == Player:
                self.players.append(p)
            else:
                raise Exception(f"Expected arguments to be Players, instead received {type(p)}.")
    

    def display_game(self):
        print("Dealer:")
        print(self.dealer)
        self.dealer.show_hand()
        for i, p in enumerate(self.players):
            print(f"Player {i + 1}:")
            print(p)
            p.show_hand()



if __name__ == "__main__":
    l = Player("Lisa")
    b = Player("Bart")
    game = BlackjackGame(l, b)

    l.add_to_hand(game.deck.deal(), True)
    l.add_to_hand(game.deck.deal())
    l.add_to_hand(game.deck.deal(), True)
    l.add_to_hand(game.deck.deal())
    b.add_to_hand(game.deck.deal())

    game.display_game()

