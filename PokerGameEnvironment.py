import numpy as np
import gymnasium as gym
from PokerGame import PokerGame


class PokerGameEnvironment(gym.Env):

    NUM_PLAYERS = 6
    MIN_BET = 10
    SMALL_BLIND = 10
    BIG_BLIND = 20
    NUM_CARDS = 52

    def __init__(self):
        super.__init__()

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

    def flatten_state(self):
        return np.array([val for val in self.state.values()]).flatten()


    def verify_action_validity(self, action):
        return True

    def raise_amount(self):
        pass

    def bet_amount(self):
        pass
        
    def step(self, action):
        assert self.action_space.contains(action), f"{action} is not a valid action."
        assert self.verify_action_validity(action), f"{action} cannot be done according to the game rules."

        chips = self.state["chips"].item()
        cur_bet = self.state["current_bet"].item()
        previous_bet = cur_bet
        pot = self.state["pot_size"].item()
        other_bets = self.state["other_player_bets"]
        player_cards = self.state["player_cards"]
        community_cards = self.state["community_cards"]

        # Fold ends game and sets reward to lose current bet
        if action == 0:
            reward = -cur_bet
            done = True
        # Call gives no immediate reward and updates the player's bet to the current max bet
        elif action == 1:
            reward = 0
            done = False
            cur_bet = max(other_bets)
            pot += max(0, cur_bet - previous_bet)
            chips -= max(0, cur_bet - previous_bet)
        # Raises by the amount determined by another function and updates the state information
        elif action == 2:
            reward = 0
            done = False
            cur_bet += self.raise_amount()
            pot += max(0, cur_bet - previous_bet)
            chips -= max(0, cur_bet - previous_bet)
        # Check changes nothing about the player's own state and passes to next player
        elif action == 3:
            reward = 0
            done = False
        # Bet if there are no other bets and change state information
        elif action == 4:
            reward = 0
            done = False
            cur_bet += self.bet_amount()
            pot += max(0, cur_bet - previous_bet)
            chips -= max(0, cur_bet - previous_bet)
        

        self.state = {
            "chips": np.array([chips], dtype=np.float32),
            "current_bet": np.array([cur_bet], dtype=np.float32),
            "previous_bet": np.array([previous_bet], dtype=np.float32),
            "pot_size": np.array([pot], dtype=np.float32),
            "other_player_bets": other_bets,
            "player_cards": player_cards,
            "community_cards": community_cards
        }

        return self.state, reward, done, False, {}
    

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
        return self.state
    

    def array_to_cards(self, array):
        return 0
    
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
        

    def update_state(self, game: PokerGame, player_index: int):
        player = game.players[player_index]
        other_bets = [p.current_bet for p in game.players]
        other_bets.pop(player_index)

        state = {
            "chips": np.array([player.balance], dtype=np.float32),
            "current_bet": np.array([player.current_bet], dtype=np.float32),
            "previous_bet": np.array([player.previous_bet], dtype=np.float32),
            "pot_size": np.array([game.pot], dtype=np.float32),
            "other_player_bets": np.array(other_bets, dtype=np.float32),
            "player_cards": self.cards_to_array(player.readable_hand),
            "community_cards": self.cards_to_array(game.board, community=True)
        }

        self.state = state

        




