import numpy as np
import gymnasium as gym
from PokerGame import PokerGame
from Card import Card
import torch.nn.functional as F
import torch


class PokerGameEnvironment(gym.Env):

    NUM_PLAYERS = 6
    MIN_BET = 10
    SMALL_BLIND = 10
    BIG_BLIND = 20
    NUM_CARDS = 52

    def __init__(self, game: PokerGame, player_id: int):

        """
        0 - fold
        1 - call
        2 - raise
        3 - check
        4 - bet
        """
        self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Dict({
            "chips": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "current_bet": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "previous_bet": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "pot_size": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "other_player_bets": gym.spaces.Box(low=0, high=np.inf, shape=(self.NUM_PLAYERS - 1,), dtype=np.float32),
            "player_cards": gym.spaces.Box(low=0, high=1, shape=(self.NUM_CARDS * 2,), dtype=np.int8),
            "community_cards": gym.spaces.Box(low=0, high=1, shape=(self.NUM_CARDS * 5,), dtype=np.int8)
        })

        # Player observable state
        self.state = {
            "chips": np.array([1000], dtype=np.float32),
            "current_bet": np.array([0], dtype=np.float32),
            "previous_bet": np.array([0], dtype=np.float32),
            "pot_size": np.array([0], dtype=np.float32),
            "other_player_bets": np.zeros(shape=(self.NUM_PLAYERS - 1,), dtype=np.float32),
            "player_cards": np.zeros(shape=(self.NUM_CARDS * 2,), dtype=np.int8),
            "community_cards": np.zeros(shape=(self.NUM_CARDS * 5), dtype=np.int8)
        }

        self.player_id = player_id
        self.game = game
        self.round_num = 0
        self.game_data = []

    def flatten_state(self):
        return np.concatenate([v.flatten() for v in self.state.values()])


    def verify_action_validity(self, action):
        return True

    def raise_amount(self):
        pass

    def bet_amount(self):
        pass

    def preflop(self, button):
        no_raises = False
        reward = 0
        done = False
        raises = 0
        for current_player in range(self.NUM_PLAYERS):
            self.game.players[current_player].add_to_hand([self.game.deck.deal() for _ in range(2)])
        
        current_player = (button + 1) % self.NUM_PLAYERS

        # Small blind mandatory bet
        self.game.players[current_player].bet(self.SMALL_BLIND)
        print(f"Player {current_player} small blind {self.SMALL_BLIND}")
        self.game.pot += self.SMALL_BLIND
        current_player = self.next_player(current_player)

        # Big blind mandatory bet
        self.game.players[current_player].bet(self.BIG_BLIND)
        print(f"Player {current_player} big blind {self.BIG_BLIND}")
        self.game.pot += self.BIG_BLIND
        self.game.highest_bet = self.BIG_BLIND
        raise_player = current_player
        current_player = self.next_player(current_player)

        while current_player != raise_player:
            if current_player in self.game.folded_indices:
                current_player = self.next_player(current_player)
                continue

            player = self.game.players[current_player]
            if current_player != self.player_id:
                action, previous_bet = player.act(self.round_num, self.game.highest_bet, no_raises)
                if action == "raise":
                    self.game.highest_bet = player.get_current_bet()
                    self.game.pot += self.game.highest_bet - previous_bet
                    raise_player = current_player
                    raises += 1
                elif action == "call":
                    self.game.highest_bet = player.get_current_bet()
                    self.game.pot += self.game.highest_bet - previous_bet
                    
                elif action == "fold":
                    self.game.folded_indices.add(current_player)
                print(f"Player {current_player} " + str(action))
                
            else:       
                player.previous_bet = player.current_bet
                self.update_state()
                logits = player.select_action(self.flatten_state())
                possibilities = F.softmax(logits, dim=1)
                action = torch.argmax(possibilities.flatten()).item()

                print(f"Agent player {current_player} " + str(action))
                
                if action == 3 or action == 4:
                    reward -= player.balance
                    action = 0
                if action == 0:
                    reward -= player.current_bet
                    done = True
                elif action == 1:
                    self.game.pot += self.game.highest_bet - player.current_bet
                    player.bet(self.game.highest_bet - player.current_bet)
                elif action == 2 and not no_raises:
                    raise_factor = max(1, torch.log(possibilities.flatten())[2] + 2)
                    self.game.pot += min(self.game.highest_bet * raise_factor, player.balance) - player.current_bet
                    player.bet(min(self.game.highest_bet * raise_factor, player.balance) - player.current_bet)
                    raise_player = current_player
                    raises += 1
                else:
                    reward -= player.balance
                    action = 0
                    done = True
                
                current_data = [self.flatten_state(), action, reward, None, done]
                self.game_data.append(current_data)
                
            if self.check_win():
                print("Pot size", self.game.pot)
                winner = set(range(self.NUM_PLAYERS)).difference(self.game.folded_indices).pop()
                print(winner, "wins")
                done = True
        
            if self.check_bets() or raises >= self.NUM_PLAYERS - len(self.game.folded_indices):
                no_raises = True
            
            if done:
                break
            current_player = self.next_player(current_player)
        
        return done, reward

    def normal_round(self, button: int):
        if self.round_num == 1:
            self.game.flop() # flop the cards onto the board
        elif self.round_num == 2:
            self.game.turn()    
        else:
            self.game.river()
            
        reward = 0
        done = False
        raises = 0
        no_raises = False
        checking = True

        current_player = (button + 1) % self.NUM_PLAYERS

        raise_player = current_player

        while current_player != raise_player:
            if current_player in self.game.folded_indices:
                current_player = self.next_player(current_player)
                continue
            
            player = self.game.players[current_player]
            if current_player != self.player_id:
                action, previous_bet = player.act(self.round_num, self.game.highest_bet, no_raises)
                if action == "raise":
                    self.game.highest_bet = player.get_current_bet()
                    self.game.pot += self.game.highest_bet - previous_bet
                    raise_player = current_player
                    raises += 1
                    checking= False
                elif action == "call":
                    self.game.highest_bet = player.get_current_bet()
                    self.game.pot += self.game.highest_bet - previous_bet
                elif action == "fold":
                    self.game.folded_indices.add(current_player)
                elif action == "check" and checking:
                    continue
                else:
                    action = "call"
                    player.__call(self.game.highest_bet)
                    self.game.highest_bet = player.get_current_bet()
                    self.game.pot += self.game.highest_bet - previous_bet
                print(f"Player {current_player} " + str(action))

            else:   # current player is our player
                player.previous_bet = player.current_bet
                self.update_state()
                logits = player.select_action(self.flatten_state())
                possibilities = F.softmax(logits, dim=1)
                action = torch.argmax(possibilities.flatten()).item()

                print(f"Agent player {current_player} " + str(action))
                
                if action == 0: # fold
                    reward -= player.current_bet
                    done = True
                elif action == 1: # call
                    self.game.pot += self.game.highest_bet - player.current_bet
                    player.bet(self.game.highest_bet - player.current_bet)

                elif action == 2 and not no_raises: # raise
                    raise_factor = max(1, torch.log(possibilities.flatten())[2] + 2)
                    self.game.pot += min(self.game.highest_bet * raise_factor, player.balance) - player.current_bet
                    player.bet(min(self.game.highest_bet * raise_factor, player.balance) - player.current_bet)
                    raise_player = current_player
                    raises += 1
                elif action == 3 and checking: # check
                    pass
                else:   # checking if not allowed, raising if not allowed
                    reward -= player.balance
                    action = 0
                    done = True
                
                current_data = [self.flatten_state(), action, reward, None, done]
                self.game_data.append(current_data)

            if self.check_win():
                print("Pot size", self.game.pot)
                winner = set(range(self.NUM_PLAYERS)).difference(self.game.folded_indices).pop()
                print(winner, "wins")
                done = True
        
            if self.check_bets() or raises >= self.NUM_PLAYERS - len(self.game.folded_indices):
                no_raises = True
            
            if done:
                break
            current_player = self.next_player(current_player)

        return done, reward

    def check_win(self):
        return len(self.game.players) - len(self.game.folded_indices) == 1

    def check_bets(self):
        return all([i in self.game.folded_indices or p.is_all_in() or p.current_bet == self.game.highest_bet for i, p in enumerate(self.game.players)]) 

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)
  
    def play(self, button: int):
        cum_reward = 0
        print(f"Round number {self.round_num}")
        done, reward = self.preflop(button)
        cum_reward += reward
        for _ in range(3):
            if not done:
                self.round_num += 1
                print(f"Round number {self.round_num}")
                done, reward = self.normal_round(button)
                cum_reward += reward
                
        if not done:
            best_hands = []
            remaining_players = []
            for i, p in enumerate(self.game.players):
                if i not in self.game.folded_indices:
                    best_hand = self.game.find_hand(p.readable_hand + self.game.board)
                    print(f"Best hand possible for {i} is {best_hand[0]} with {best_hand[1]}.")
                    best_hands.append(best_hand)
                    remaining_players.append(i)
            winning_hand = self.game.best_hand(best_hands)
            winning_player = remaining_players[best_hands.index(winning_hand)]
            print(f"Winner of this round is {winning_player} with {winning_hand[0]}")
        done = True
        
        return self.game_data, reward
    
    def reset(self):
        self.state = {
            "chips": np.array([1000], dtype=np.float32),
            "current_bet": np.array([0], dtype=np.float32),
            "previous_bet": np.array([0], dtype=np.float32),
            "pot_size": np.array([0], dtype=np.float32),
            "other_player_bets": np.zeros(shape=(self.NUM_PLAYERS - 1,), dtype=np.float32),
            "player_cards": np.zeros(shape=(self.NUM_CARDS * 2,), dtype=np.int8),
            "community_cards": np.zeros(shape=(self.NUM_CARDS * 5), dtype=np.int8)
        }
        self.game.clear_game()
        self.round_num = 0
        self.game_data = []
        return self.state
    
    def array_to_cards(self, array: np.ndarray):
        hand = []
        for i in range(len(array) // 52):
            card = array[52*i:52*(i+1)]
            suit = card.to_list().index(1) // 13
            rank = card.to_list().index(1) % 13
            hand.append(Card(suit, rank))

        return hand
    
    def cards_to_array(self, cards, community=False):
        if community:
            card_array = [0] * 5 * self.NUM_CARDS
            for i, card in enumerate(cards):
                card_array[card.suit * 13 + card.rank - 1 + 52 * i] = 1
            return np.array(card_array, dtype=np.int8)
        else:
            card_array = [0] * len(cards) * self.NUM_CARDS
            for i, card in enumerate(cards):
                card_array[card.suit * 13 + card.rank - 1 + 52 * i] = 1
            return np.array(card_array, dtype=np.int8)

    def render(self, mode="human"):
        print(f"Player chips: {self.state['chips'].item()}")
        print(f"Current bet: {self.state['current_bet'].item()}")
        print(f"Pot size: {self.state['pot_size'].item()}")
        print(f"Player cards: {[card for card in self.array_to_cards(self.state['player_cards'])]}")
        print(f"Community cards: {[card for card in self.array_to_cards(self.state['community_cards'])]}")

    def close(self):
        pass
        
    def update_state(self):
        player = self.game.players[self.player_id]
        other_bets = [p.current_bet for p in self.game.players]
        other_bets.pop(self.player_id)

        state = {
            "chips": np.array([player.balance], dtype=np.float32),
            "current_bet": np.array([player.current_bet], dtype=np.float32),
            "previous_bet": np.array([player.previous_bet], dtype=np.float32),
            "pot_size": np.array([self.game.pot], dtype=np.float32),
            "other_player_bets": np.array(other_bets, dtype=np.float32),
            "player_cards": self.cards_to_array(player.readable_hand),
            "community_cards": self.cards_to_array(self.game.board, community=True)
        }

        self.state = state

    def next_player(self, current_player):
        return (current_player + 1) % self.NUM_PLAYERS
