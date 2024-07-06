import numpy as np
import random
from treys import Deck, Evaluator, Card


class PokerEnvironment:

    def __init__(self, num_players=6, starting_chips=1000):
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.deck = Deck()
        self.evaluator = Evaluator()
        self.small_blind = 10
        self.big_blind = 20
        self.current_round_bets = 0
        self.actions_taken = 0
        self.final_round = False
        self.reset()


    def reset(self):
        self.deck.shuffle()
        self.hands = [self.deck.draw(2) for _ in range(self.num_players)]
        self.board = []
        self.pot = 0
        self.bets = np.zeros(self.num_players)
        self.chips = np.full(self.num_players, self.starting_chips)
        self.active_players = np.ones(self.num_players, dtype=bool)
        self.current_player = 0
        self.round_ended = False
        self.actions_taken = 0
        self.final_round = False
        self.initial_chips = np.copy(self.chips) 
        self.setup_blinds()
        return self.get_state(self.current_player)
    
    def setup_blinds(self):
        small_blind_player = self.current_player
        big_blind_player = (self.current_player + 1) % self.num_players

        self.bets[small_blind_player] = min(self.small_blind, self.chips[small_blind_player])
        self.chips[small_blind_player] -= self.bets[small_blind_player]
        self.bets[big_blind_player] = min(self.big_blind, self.chips[big_blind_player])
        self.chips[big_blind_player] -= self.bets[big_blind_player]
        self.pot = self.bets[small_blind_player] + self.bets[big_blind_player]
        self.current_player = (big_blind_player + 1) % self.num_players

        self.actions_taken = 2

    def get_state(self, player_index):
        padded_board = self.board + [-1] * (5 - len(self.board))
        state = {
            'hand': self.hands[player_index],
            'board': padded_board,
            'pot': self.pot,
            'player_index': player_index,
            'bets': self.bets,
            'chips': self.chips,
            'active_players': self.active_players
        }
        return state
    
    def step(self, action, raise_amount):
        if not self.active_players[self.current_player]:
            raise Exception("not active")
        

        max_bet = np.max(self.bets)
        current_bet = self.bets[self.current_player]
        remaining_chips = self.chips[self.current_player]

        # folding
        if action == 0:
            self.active_players[self.current_player] = False
        # call/check
        elif action == 1:
            call_amount = max_bet - current_bet
            if call_amount <= remaining_chips:
                self.bets[self.current_player] += call_amount
                self.chips[self.current_player] -= call_amount
            else: # all in
                self.bets[self.current_player] += remaining_chips
                self.chips[self.current_player] = 0
        # raising
        elif action == 2 and raise_amount > 0:
            raise_amount = min(raise_amount, remaining_chips)
            total_bet = max_bet + raise_amount
            if total_bet <= remaining_chips + current_bet:
                self.bets[self.current_player] = total_bet
                self.chips[self.current_player] -= (total_bet - current_bet)
            else: # all in
                self.bets[self.current_player] += remaining_chips
                self.chips[self.current_player] = 0

        self.pot += self.bets[self.current_player] - current_bet
        self.current_player = (self.current_player + 1) % self.num_players
        self.actions_taken += 1
        #print(self.bets)
        # check if betting round is over
        active_bets = self.bets[self.active_players]
        all_acted = self.actions_taken >= np.sum(self.active_players)
        max_bet_condition = all(
            (bet == max_bet) or (bet == self.starting_chips) for bet in active_bets
        )

        if all_acted and max_bet_condition and not self.final_round and len(active_bets) > 1:
            self.actions_taken = 0
            self.round_ended = True
            if len(self.board) == 0:
                self.board.extend(self.deck.draw(3))  # flop
            elif len(self.board) == 3:
                self.board.extend(self.deck.draw(1))  # turn
            elif len(self.board) == 4:
                self.board.extend(self.deck.draw(1))  # river
                self.final_round = True  
            else:
                self.round_ended = False 

    
            max_bet_condition = False
            all_acted = False
   
        done = (self.final_round and all_acted and max_bet_condition) or np.sum(self.active_players) == 1

        if done:
            if np.sum(self.active_players) == 1:
                winner = np.argmax(self.active_players)
                rewards = np.zeros(self.num_players)
                rewards[winner] = self.pot
                rewards -= self.bets
              
            else:
                rewards = self.evaluate()
          
        else:
            rewards = np.zeros(self.num_players)
        
        return self.get_state(self.current_player), rewards, done


    def evaluate(self):
        if len(self.board) != 5:
            return np.zeros(self.num_players)

        scores = [self.evaluator.evaluate(self.board, hand) if active else float('inf') for hand, active in zip(self.hands, self.active_players)]
        winner = np.argmin(scores)
        rewards = np.zeros(self.num_players)
        if self.active_players[winner]:
            rewards[winner] = self.pot
       
        rewards -= self.bets  

        return rewards
    
    def encode_card(card):
        if card is None:
            return [-1, -1]  # use -1 to represent missing cards
        elif isinstance(card, int):
            suit = Card.get_suit_int(card)
            rank = Card.get_rank_int(card)
            return [suit, rank]
        else:
            print(card)
            raise ValueError("Unexpected card type")


    def flatten_state(state):
        hand = np.array([PokerEnvironment.encode_card(card) for card in state['hand']]).flatten()
        board = np.array([PokerEnvironment.encode_card(card) for card in state['board']]).flatten()
        pot = np.array(state['pot']).flatten()
        player_index = np.array(state['player_index']).flatten()
        bets = np.array(state['bets']).flatten()
        chips = np.array(state['chips']).flatten()
        active_players = np.array(state['active_players'], dtype=np.float32).flatten()

        return np.concatenate((hand, board, pot, player_index, bets, chips, active_players))