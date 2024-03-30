from PokerGameEnvironment import PokerGameEnvironment
import tensorflow as tf
from PokerGame import PokerGame
from RandomPlayer import RandomPlayer



random_training_data = {player_id: [] for player_id in range(6)}

def simulate_game(game: PokerGame, button_position: int):

    folded_indexes = set()

    player_outcomes = {}
    player_data = {p.id: [] for p in game.players}

    print("Preflop")
    # Preflop round; button_position + 1 makes small blind bet and button_position + 2 makes big blind bet
    active_player = (button_position + 1) % game.NUM_PLAYERS
    round_number = 0

    # Deal cards
    for p in game.players:
        p.add_to_hand(game.deck.deal())
        p.add_to_hand(game.deck.deal())
    

    # Small blind mandatory bet
    game.players[active_player].bet(game.min_bet)
    game.pot += game.min_bet
    active_player = (active_player + 1) % game.NUM_PLAYERS

    # Big blind mandatory bet
    game.players[active_player].bet(game.max_bet) 
    game.pot += game.max_bet
    active_player = (active_player + 1) % game.NUM_PLAYERS
    

    round_finished = False
    table_highest_bet = game.max_bet

    while not round_finished:
        if active_player not in folded_indexes:
            action, player_previous_bet = game.players[active_player].act(round_number=round_number, table_highest_bet=table_highest_bet)
            print(active_player, action)
            if action == "raise" or action == "call":
                table_highest_bet = game.players[active_player].get_current_bet()
                game.pot += table_highest_bet - player_previous_bet
            elif action == "fold":
                folded_indexes.add(active_player)
                player_outcomes[active_player] = -1 * player_previous_bet

        if check_win(players=game.players, excluded_players=folded_indexes):
            print("Pot size", game.pot)
            winner = post_win(excluded_players=folded_indexes)
            print(winner, "wins")
            player_outcomes[winner] = game.pot
            return player_outcomes
        
        if check_bets(players=game.players, table_highest_bet=table_highest_bet, excluded_players=folded_indexes):
            round_finished = True

        active_player = (active_player + 1) % game.NUM_PLAYERS

    print("Pot size", game.pot)
    
    print("Flop")
    # Flop round, begin again from button_position + 1
    active_player = (button_position + 1) % game.NUM_PLAYERS
    round_number = 1
    round_finished = False
    game.flop()

    turns = 0
    remaining = game.NUM_PLAYERS - len(folded_indexes)

    while not round_finished:
        if active_player not in folded_indexes:
            action, player_previous_bet = game.players[active_player].act(round_number=round_number, table_highest_bet=table_highest_bet)
            print(active_player, action)
            if action == "raise" or action == "call":
                table_highest_bet = game.players[active_player].get_current_bet()
                game.pot += table_highest_bet - player_previous_bet
            elif action == "fold":
                folded_indexes.add(active_player)
                player_outcomes[active_player] = -1 * player_previous_bet
            turns += 1
        
        if check_win(players=game.players, excluded_players=folded_indexes):
            print("Pot size", game.pot)
            winner = post_win(excluded_players=folded_indexes)
            print(winner, "wins")
            player_outcomes[winner] = game.pot
            return player_outcomes
        
        if turns >= remaining and check_bets(players=game.players, table_highest_bet=table_highest_bet, excluded_players=folded_indexes):
            round_finished = True

        active_player = (active_player + 1) % game.NUM_PLAYERS

    print("Pot size", game.pot)

    print("Turn")
    # Turn round, begin again from button_position + 1
    active_player = (button_position + 1) % game.NUM_PLAYERS
    round_number = 2
    round_finished = False
    game.turn()

    turns = 0
    remaining = game.NUM_PLAYERS - len(folded_indexes)

    while not round_finished:
        if active_player not in folded_indexes:
            action, player_previous_bet = game.players[active_player].act(round_number=round_number, table_highest_bet=table_highest_bet)
            print(active_player, action)
            if action == "raise" or action == "call":
                table_highest_bet = game.players[active_player].get_current_bet()
                game.pot += table_highest_bet - player_previous_bet
            elif action == "fold":
                folded_indexes.add(active_player)
                player_outcomes[active_player] = -1 * player_previous_bet
            turns += 1

        if check_win(players=game.players, excluded_players=folded_indexes):
            print("Pot size", game.pot)
            winner = post_win(excluded_players=folded_indexes)
            print(winner, "wins")
            player_outcomes[winner] = game.pot
            return player_outcomes
        
        if turns >= remaining and check_bets(players=game.players, table_highest_bet=table_highest_bet, excluded_players=folded_indexes):
            round_finished = True

        active_player = (active_player + 1) % game.NUM_PLAYERS

    print("Pot size", game.pot)

    print("River")
    # River round, begin again from button_position + 1
    active_player = (button_position + 1) % game.NUM_PLAYERS
    round_number = 3
    round_finished = False
    game.river()

    turns = 0
    remaining = game.NUM_PLAYERS - len(folded_indexes)

    while not round_finished:
        if active_player not in folded_indexes:
            action, player_previous_bet = game.players[active_player].act(round_number=round_number, table_highest_bet=table_highest_bet)
            print(active_player, action)
            if action == "raise" or action == "call":
                table_highest_bet = game.players[active_player].get_current_bet()
                game.pot += table_highest_bet - player_previous_bet
            elif action == "fold":
                folded_indexes.add(active_player)
                player_outcomes[active_player] = -1 * player_previous_bet
            turns += 1

        if check_win(players=game.players, excluded_players=folded_indexes):
            print("Pot size", game.pot)
            winner = post_win(excluded_players=folded_indexes)
            print(winner, "wins")
            player_outcomes[winner] = game.pot
            return player_outcomes
        
        if turns >= remaining and check_bets(players=game.players, table_highest_bet=table_highest_bet, excluded_players=folded_indexes):
            round_finished = True

        active_player = (active_player + 1) % game.NUM_PLAYERS


    return player_outcomes


def post_win(excluded_players):
    return set(range(6)).difference(excluded_players).pop()


def check_win(players, excluded_players):
    return len(players) - len(excluded_players) == 1

def check_bets(players, table_highest_bet, excluded_players):
    return all([i in excluded_players or p.is_all_in() or p.get_current_bet() == table_highest_bet for i, p in enumerate(players)])


game_iterations = 1000
player_starting_fortune = 100.0
default_raise_factor = 1.2

button = 0

for i in range(game_iterations):
    print(f"Playing game {i + 1} / {game_iterations}...")
    players = [RandomPlayer(id=int(c), starting_fortune=player_starting_fortune, raise_factor=default_raise_factor) for c in "012345"]
    game = PokerGame(min_bet=1.0, max_bet=2.0, players=players)
    game.deck.shuffle(type="random")
    button = (button + 1) % game.NUM_PLAYERS
    data = simulate_game(game, button)
    print(data)
    




"""
Generates playing data from random play, with incomplete information.

Data is of the format:
(
((player_cards), (community_cards), pot_size, (player_chips), round_num, (all_previous_actions)),
((next_player_cards), (next_community_cards), next_pot_size, (next_player_chips), next_round_num, (updated_actions)),
round_action,
round_reward,
end_of_game,
button_position
)
"""

