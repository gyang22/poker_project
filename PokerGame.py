from Deck import Deck
from Player import Player, BettingPlayer

class PokerGame:

    FLOP_SIZE = 3
    HAND_SIZE = 5

    RANKINGS = {"Royal Flush": 10,
                "Straight Flush": 9,
                "Four of a Kind": 8,
                "Full House": 7,
                "Flush": 6,
                "Straight": 5,
                "Three of a Kind": 4,
                "Two Pair": 3,
                "Pair": 2,
                "High Card": 1}


    def __init__(self, min_bet, max_bet, *players):
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.deck = Deck()
        self.board = []
        self.players = []
        for p in players:
            self.players.append(p)
        self.pot = 0


    """
    Flops three cards onto the board, if the board already has cards on it, an exception is raised.
    """
    def flop(self):
        if len(self.board) != 0:
            raise Exception("The board is not empty, cannot flop cards.")
        for _ in range(PokerGame.FLOP_SIZE):
            self.board.append(self.deck.deal())


    """
    Turns the next card onto the board after the flop, if the board does not already have FLOP_SIZE cards an exception is raised.
    """
    def turn(self):
        if len(self.board) != PokerGame.FLOP_SIZE:
            raise Exception(f"The board has the wrong number of cards on it; expected {PokerGame.FLOP_SIZE} but there are {len(self.board)}.")
        self.board.append(self.deck.deal())


    """
    Reveals the next card onto the board after the turn, if the board does not already have FLOP_SIZE + 1 cards an exception is raised.
    """
    def river(self):
        if len(self.board) != PokerGame.FLOP_SIZE + 1:
            raise Exception(f"The board has the wrong number of cards on it; expected {PokerGame.FLOP_SIZE + 1} but there are {len(self.board)}.")
        self.board.append(self.deck.deal())


    """
    Displays the current board.
    """
    def display_board(self):
        print(self.board)


    """
    Plays one complete round of poker, each player is allowed to make moves and the winning player is awarded the amount betted.
    """
    def play(self):
        self.__pre_flop()
        for p in self.players:
            print(f"{p.name}: {p.hand}")
        self.__betting_round()
        self.flop()
        self.display_board()
        self.__betting_round()
        self.turn()
        self.display_board()
        self.__betting_round()
        self.river()
        self.display_board()
        best_hands = []
        for p in self.players:
            best_hand = self.find_hand(p.hand + self.board)
            print(f"Best hand possible for {p.name} is {best_hand[0]} with high card {best_hand[1]}.")
            best_hands.append(best_hand)
        winning_hand = max(best_hands, key = lambda x: PokerGame.RANKINGS[x[0]])
        winning_player = self.players[best_hands.index(winning_hand)]
        print(f"Winner of this round is {winning_player.name} with {winning_hand[0]}")
        self.clear_game()
        return winning_player, winning_hand
    
    
    """
    Clears the deck, board, and player hands.
    """
    def clear_game(self):
        self.deck = Deck()
        for p in self.players:
            p.hand = []
        self.board = []


    def __pre_flop(self):
        self.deck.shuffle("random")
        print("Pre-flop")
        for player in self.players:
            player.add_to_hand(self.deck.deal())
            player.add_to_hand(self.deck.deal())

    
    def __betting_round(self):
        # print("Betting Round")
        # current_max_bet = 0
        # def betting_helper(current_max):
        #     for player in self.players:
        #         desired_bet = 0
        #         while desired_bet < max(self.min_bet, current_max):
        #             desired_bet = input(f"{player.name} bet amount: ")
        #             if desired_bet < self.min_bet:
        #                 print("Too low!")
        #         actual_bet = player.bet(desired_bet)
        #         self.pot += actual_bet
        #         current_max_bet = max(current_max_bet, actual_bet)

        pass 

    """
    Compares two hands based on standard poker rules, returns positive number if hand one is larger, negative if smaller, zero if equal.
    """
    def compare_hand(self, hand1, hand2):
        return PokerGame.RANKINGS[hand1[0]] - PokerGame.RANKINGS[hand2[0]]


    """
    Finds and returns the best hand in a list of seven cards.
    """
    def find_hand(self, cards):
        greatest_card = max(cards, key = lambda card: card.rank if card.rank != 1 else 14)
        if self.__is_flush(cards) and self.__is_straight_royal(cards):
            return ("Royal Flush", greatest_card)
        elif self.__is_flush(cards) and self.__is_straight(cards):
            return ("Straight Flush", greatest_card)
        elif self.__is_n_kind(cards, 4):
            return ("Four of a Kind", greatest_card)
        elif self.__is_full_house(cards):
            return ("Full House", greatest_card)
        elif self.__is_flush(cards):
            return ("Flush", greatest_card)
        elif self.__is_straight(cards):
            return ("Straight", greatest_card)
        elif self.__is_n_kind(cards, 3):
            return ("Three of a Kind", greatest_card)
        elif self.__is_two_pair(cards):
            return ("Two Pair", greatest_card)
        elif self.__is_n_kind(cards, 2):
            return ("Pair", greatest_card)
        else:
            return ("High Card", greatest_card)


    """
    Returns true if the seven cards passed have five cards of all the same suit.
    """
    def __is_flush(self, cards):
        suits = [card.suit for card in cards]
        for i in range(Deck.NUM_SUITS):
            if suits.count(i) == 5:
                return True
        return False

    """
    Returns true if the seven cards passed in can form a full house.
    """
    def __is_full_house(self, cards):
        ranks = [card.rank for card in cards]
        counts = {}
        for c in ranks:
            counts[c] = 1 + counts.get(c, 0)
        return 2 in counts.values() and 3 in counts.values()


    """
    Returns true if the seven cards passed in can form n of a kind.
    """
    def __is_n_kind(self, cards, n):
        ranks = [card.rank for card in cards]
        counts = {}
        for c in ranks:
            counts[c] = 1 + counts.get(c, 0)
        return n in counts.values()


    """
    Checks for a two pair in the list of seven cards.
    """
    def __is_two_pair(self, cards):
        ranks = [card.rank for card in cards]
        counts = {}
        for c in ranks:
            counts[c] = 1 + counts.get(c, 0)
        return list(counts.values()).count(2) == 2
    

    """
    Returns true if the seven cards passed in can be arranged into five cards of ascending order.
    """
    def __is_straight(self, cards):
        ranks = [card.rank for card in cards]
        ranks.sort()
        for i in range(len(cards) - PokerGame.HAND_SIZE + 1):
            sublist = ranks[i:i + PokerGame.HAND_SIZE]
            if sublist[0] == 10:
                sublist.pop(0)
                sublist.append(14)
            if all(sublist[i] == sublist[i + 1] + 1 for i in range(len(sublist) - 1)):
                return True
        return False
    

    """
    Returns true if the seven cards passed in are strictly straight for a royal flush.
    """
    def __is_straight_royal(self, cards):
        ranks = [card.rank if card.rank != 1 else 14 for card in cards]
        ranks.sort()
        for i in range(len(cards) - PokerGame.HAND_SIZE + 1):
            sublist = ranks[i:i + PokerGame.HAND_SIZE]
            if sublist[0] != 10:
                continue
            if all(sublist[i] == sublist[i + 1] + 1 for i in range(len(sublist) - 1)):
                return True
        return False


if __name__ == "__main__":
    a = BettingPlayer("Abby", 100)
    b = BettingPlayer("Brian", 100)
    c = BettingPlayer("Callux", 100)
    game = PokerGame(1, 2, a, b, c)
    
    results = {}
    for i in range(650000):
        result = game.play()
        results[result[1][0]] = 1 + results.get(result[1][0], 0)
    print(results)


