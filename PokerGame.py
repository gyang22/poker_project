from Deck import Deck
from Player import Player, BettingPlayer
import numpy as np

class PokerGame:

    FLOP_SIZE = 3
    HAND_SIZE = 5
    HOLE_AND_COMMUNITY_SIZE = 7
    NUM_PLAYERS = 6

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


    def __init__(self, min_bet, max_bet, players):
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
        print("Flop")
        self.display_board()
        self.next_odds()
        self.__betting_round()

        self.turn()
        print("Turn")
        self.display_board()
        self.next_odds()
        self.__betting_round()

        self.river()
        print("River")
        self.display_board()
        
        best_hands = []
        for p in self.players:
            best_hand = self.find_hand(p.hand + self.board)
            print(f"Best hand possible for {p.name} is {best_hand[0]} with {best_hand[1]}.")
            best_hands.append(best_hand)
        winning_hand = self.best_hand(best_hands)
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
    Finds and returns the best hand in a list of seven cards.
    """
    def find_hand(self, cards):
        flush_possibility = self.find_flush(cards)
        straight_possibility = self.find_straight(cards)
        longest_sequence = self.most_of_a_kind(cards)
        two_pair_possibility = self.find_two_pair(cards)
        full_house_possibility = self.find_full_house(cards)

        if flush_possibility and self.find_straight(flush_possibility) and self.find_straight(flush_possibility)[0].rank == 1:
            return ("Royal Flush", flush_possibility)
        elif flush_possibility and self.find_straight(flush_possibility):
            return ("Straight Flush", flush_possibility)
        elif longest_sequence[1] == 4:
            return ("Four of a Kind", longest_sequence[0])
        elif full_house_possibility:
            return ("Full House", full_house_possibility)
        elif flush_possibility:
            return ("Flush", flush_possibility)
        elif straight_possibility:
            return ("Straight", straight_possibility)
        elif longest_sequence[1] == 3:
            return ("Three of a Kind", longest_sequence[0])
        elif two_pair_possibility:
            return ("Two Pair", two_pair_possibility)
        elif longest_sequence[1] == 2:
            return ("Pair", longest_sequence[0])
        else:
            cards.sort(reverse = True, key = lambda x: x.rank if x.rank != 1 else 14)
            return ("High Card", cards[:5])


    """
    Returns a list of flush cards if the seven cards passed have five cards of all the same suit, otherwise empty list. The list is 
    sorted in descending order.
    """
    def find_flush(self, cards):
        cards.sort(reverse = True, key = lambda x: x.rank if x.rank != 1 else 14)
        suits = {}
        for card in cards:
            suits[card.suit] = suits.get(card.suit, []) + [card]
        for hand in suits.values():
            if len(hand) >= 5:
                return hand[:5]
        return []
    

    """
    Returns the most possible number of matching cards of a kind, along with the respective hand.
    """
    def most_of_a_kind(self, cards):
        cards.sort(reverse = True, key = lambda x: x.rank if x.rank != 1 else 14)
        ranks = {}
        for card in cards:
            ranks[card.rank] = ranks.get(card.rank, []) + [card]
        longest = len(max(list(ranks.values()), key = len))
        most_kind = list(ranks.values())
        most_kind.sort(reverse = True, key = lambda x: x[0].rank if x[0].rank != 1 else 14)
        for l in most_kind:
            if len(l) == longest:
                most_kind = l
                break
        n_found = len(most_kind)
        burner = cards.copy()
        while len(most_kind) < 5:
            added = burner.pop()
            most_kind.append(added) if added.rank != most_kind[0].rank else most_kind
        return most_kind, n_found
    

    """
    Returns the hand if a two pair can be arranged in the list of cards, with one additional random card at the end, or empty list.
    """
    def find_two_pair(self, cards):
        cards.sort(reverse = True, key = lambda x: x.rank if x.rank != 1 else 14)
        ranks = {}
        for card in cards:
            ranks[card.rank] = ranks.get(card.rank, []) + [card]
        pairings = ranks.values()
        result = []
        if [len(l) for l in pairings].count(2) == 2:
            result = [p if len(p) == 2 else [] for p in pairings]
            result = [card for pair in result for card in pair]
            result.sort(reverse = True, key = lambda x: x.rank if x.rank != 1 else 14)
            for rank, cards in ranks.items():
                if rank != result[0].rank:
                    result.append(cards[0])
                    break
        return result


    """
    Returns the hand if the cards passed in can be arranged into five cards of ascending order, otherwise empty list.
    """
    def find_straight(self, cards):
        burner = cards.copy()
        burner.sort(reverse = True, key = lambda x: x.rank if x.rank != 1 else 14)
        if burner[0].rank == 1:
            burner.append(burner[0])
        for card in burner:
            if card.rank == 1:
                card.rank = 14
            else:
                break
        for i in range(len(burner) - PokerGame.HAND_SIZE + 1):
            sublist = burner[i:i + PokerGame.HAND_SIZE]
            if all(sublist[i].rank == sublist[i + 1].rank + 1 for i in range(len(sublist) - 1)):
                for card in sublist:
                    if card.rank == 14:
                        card.rank = 1
                    else:
                        break
                return sublist
        self.deck.toggle_value(14, 1)
        return []
    

    """
    Returns the hand if the cards passed in can be arranged into a full house, otherwise empty list.
    """
    def find_full_house(self, cards):
        ranks = {}
        for card in cards:
            ranks[card.rank] = ranks.get(card.rank, []) + [card]
        pairings = list(ranks.values())
        pairings.sort(reverse = True, key = lambda x: x[0].rank)
        result = []
        for pair in pairings:
            if len(pair) == 3:
                result.extend(pair)
                break
        for pair in pairings:
            if len(pair) == 2:
                result.extend(pair)
                break
        return result if len(result) == PokerGame.HAND_SIZE else []


    """
    Returns the best hand in a list of hands, first ranking by hand strength, then top card, and other tiebreaking rules. If it is a tie,
    the list is expanded to include both hands.
    """
    def best_hand(self, hands):
        best = hands[0]
        for hand in hands:
            if PokerGame.RANKINGS[hand[0]] == PokerGame.RANKINGS[best[0]]:
                self.deck.toggle_value(1, 14)
                for i in range(PokerGame.HAND_SIZE):
                    if hand[1][i].rank > best[1][i].rank:
                        best = hand
                        break
                    elif hand[1][i].rank < best[1][i].rank:
                        break
                self.deck.toggle_value(14, 1)
            elif PokerGame.RANKINGS[hand[0]] > PokerGame.RANKINGS[best[0]]:
                best = hand
        self.deck.toggle_value(14, 1)
        return best
    

    def next_odds(self):
        wins = [0 for i in range(len(self.players))]
        counter = 0
        for next_card in self.deck.cards:
            counter += 1
            best_hands = [self.find_hand(p.hand + self.board + [next_card]) for p in self.players]
            winning_hand = self.best_hand(best_hands)
            for i, hand in enumerate(best_hands):
                if winning_hand == hand:
                    wins[i] += 1
        for i in range(len(wins)):
            print(f"Player {i + 1} has a {wins[i] * 100 / counter}% chance of getting a winning scenario on the next card.")


if __name__ == "__main__":
    a = BettingPlayer("Abby", 100)
    b = BettingPlayer("Brian", 100)
    c = BettingPlayer("Callux", 100)
    game = PokerGame(1, 2, [a, b, c])
    
    game.play()


