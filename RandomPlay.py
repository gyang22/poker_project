from PokerGameEnvironment import PokerGameEnvironment
import tensorflow as tf
from PokerGame import PokerGame
from RandomPlayer import RandomPlayer
import numpy as np



def simulate_game(game: PokerGame, button_position: int):

    folded_indexes = set()

    player_outcomes = {p.id: 0.0 for p in game.players}
    player_data = {p.id: [] for p in game.players}

    print("Preflop")
    # Preflop round; button_position + 1 makes small blind bet and button_position + 2 makes big blind bet
    active_player = (button_position + 1) % game.NUM_PLAYERS
    round_number = 0

    # Deal cards
    for p in game.players:
        p_cards = [game.deck.deal() for _ in range(2)]
        p.add_to_hand(p_cards)
    

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

    raises = 0

    while not round_finished:
        if active_player not in folded_indexes:

            player = game.players[active_player]

            # Record the player hand as well as the unknown 5 community cards
            cards_state = player.hand + [0] * 260

            # Record the balance, pot size, current bet, and other players' bets
            fortune_state = [player.balance, game.pot, player.current_bet]
            fortune_state.extend([p.current_bet if p.id not in folded_indexes else 0.0 for p in game.players])

            action, player_previous_bet = game.players[active_player].act(round_number=round_number, table_highest_bet=table_highest_bet)
            print(active_player, action)

            # Record the round number, relative position to the button, and action taken
            logistic_state = [round_number / 3.0, abs(active_player - button_position) / 5.0, action_to_int(action) / 3.0]

            # Aggregate state data
            game_state = cards_state + fortune_state + logistic_state

            player_data[active_player].append(game_state)

            if action == "raise":
                table_highest_bet = game.players[active_player].get_current_bet()
                game.pot += table_highest_bet - player_previous_bet
                raises = 0
            elif action == "call":
                table_highest_bet = game.players[active_player].get_current_bet()
                game.pot += table_highest_bet - player_previous_bet
                raises += 1
            elif action == "fold":
                folded_indexes.add(active_player)
                player_outcomes[active_player] = -1 * player_previous_bet

        if check_win(players=game.players, excluded_players=folded_indexes):
            print("Pot size", game.pot)
            winner = post_win(excluded_players=folded_indexes)
            print(winner, "wins")
            player_outcomes[winner] = game.pot
            return player_outcomes, player_data
        
        if check_bets(players=game.players, table_highest_bet=table_highest_bet, excluded_players=folded_indexes) or raises >= 6 - len(folded_indexes) - 1:
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
    raises = 0

    while not round_finished:
        if active_player not in folded_indexes:

            # Record the player hand, 3 flopped cards, and the unknown 2 community cards
            board = [0] * 260
            for i, card in enumerate(game.board):
                board[card.suit * 13 + card.rank - 1 + 52 * i] += 1
            cards_state = player.hand + board

            # Record the balance, pot size, current bet, and other players' bets
            fortune_state = [player.balance, game.pot, player.current_bet]
            fortune_state.extend([p.current_bet if p.id not in folded_indexes else 0.0 for p in game.players])

            action, player_previous_bet = game.players[active_player].act(round_number=round_number, table_highest_bet=table_highest_bet)
            print(active_player, action)

            # Record the round number, relative position to the button, and action taken
            logistic_state = [round_number / 3.0, abs(active_player - button_position) / 5.0, action_to_int(action) / 3.0]

            # Aggregate state data
            game_state = cards_state + fortune_state + logistic_state

            player_data[active_player].append(game_state)
            
            if action == "raise":
                table_highest_bet = game.players[active_player].get_current_bet()
                game.pot += table_highest_bet - player_previous_bet
                raises = 0
            elif action == "call":
                table_highest_bet = game.players[active_player].get_current_bet()
                game.pot += table_highest_bet - player_previous_bet
                raises += 1
            elif action == "fold":
                folded_indexes.add(active_player)
                player_outcomes[active_player] = -1 * player_previous_bet
            turns += 1
        
        if check_win(players=game.players, excluded_players=folded_indexes):
            print("Pot size", game.pot)
            winner = post_win(excluded_players=folded_indexes)
            print(winner, "wins")
            player_outcomes[winner] = game.pot
            return player_outcomes, player_data
        
        if turns >= remaining and check_bets(players=game.players, table_highest_bet=table_highest_bet, excluded_players=folded_indexes) or raises >= 6 - len(folded_indexes) - 1:
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
    raises = 0

    while not round_finished:
        if active_player not in folded_indexes:

            # Record the player hand, 4 known cards, and the unknown 1 community card
            board = [0] * 260
            for i, card in enumerate(game.board):
                board[card.suit * 13 + card.rank - 1 + 52 * i] += 1
            cards_state = player.hand + board

            # Record the balance, pot size, current bet, and other players' bets
            fortune_state = [player.balance, game.pot, player.current_bet]
            fortune_state.extend([p.current_bet if p.id not in folded_indexes else 0.0 for p in game.players])

            action, player_previous_bet = game.players[active_player].act(round_number=round_number, table_highest_bet=table_highest_bet)
            print(active_player, action)    

            # Record the round number, relative position to the button, and action taken
            logistic_state = [round_number / 3.0, abs(active_player - button_position) / 5.0, action_to_int(action) / 3.0]

            # Aggregate state data
            game_state = cards_state + fortune_state + logistic_state

            player_data[active_player].append(game_state)

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
            return player_outcomes, player_data
        
        if turns >= remaining and check_bets(players=game.players, table_highest_bet=table_highest_bet, excluded_players=folded_indexes) or raises >= 6 - len(folded_indexes) - 1:
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
    raises = 0

    while not round_finished:
        if active_player not in folded_indexes:

            # Record the player hand and 5 community cards
            board = [0] * 260
            for i, card in enumerate(game.board):
                board[card.suit * 13 + card.rank - 1 + 52 * i] += 1
            cards_state = player.hand + board

            # Record the balance, pot size, current bet, and other players' bets
            fortune_state = [player.balance, game.pot, player.current_bet]
            fortune_state.extend([p.current_bet if p.id not in folded_indexes else 0.0 for p in game.players])

            action, player_previous_bet = game.players[active_player].act(round_number=round_number, table_highest_bet=table_highest_bet)
            print(active_player, action)

            # Record the round number, relative position to the button, and action taken
            logistic_state = [round_number / 3.0, abs(active_player - button_position) / 5.0, action_to_int(action) / 3.0]

            # Aggregate state data
            game_state = cards_state + fortune_state + logistic_state

            player_data[active_player].append(game_state)

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
            return player_outcomes, player_data
        
        if turns >= remaining and check_bets(players=game.players, table_highest_bet=table_highest_bet, excluded_players=folded_indexes) or raises >= 6 - len(folded_indexes) - 1:
            round_finished = True

        active_player = (active_player + 1) % game.NUM_PLAYERS

    best_hands = []
    remaining_players = []
    for i, p in enumerate(game.players):
        if p.id not in folded_indexes:
            best_hand = game.find_hand(p.readable_hand + game.board)
            print(f"Best hand possible for {p.id} is {best_hand[0]} with {best_hand[1]}.")
            best_hands.append(best_hand)
            remaining_players.append(i)
    winning_hand = game.best_hand(best_hands)
    winning_player = remaining_players[best_hands.index(winning_hand)]
    print(f"Winner of this round is {winning_player} with {winning_hand[0]}")
    player_outcomes[winning_player] = game.pot
    for i in range(6):
        if i not in folded_indexes and i != winning_player:
            player_outcomes[i] = -1 * game.players[i].current_bet


    return player_outcomes, player_data

def action_to_int(action: str):
    if action == "fold":
        return 0
    elif action == "raise":
        return 1
    elif action == "call":
        return 2
    elif action == "check":
        return 3
    else:
        raise Exception("Invalid action.")


def post_win(excluded_players):
    return set(range(6)).difference(excluded_players).pop()


def check_win(players, excluded_players):
    return len(players) - len(excluded_players) == 1

def check_bets(players, table_highest_bet, excluded_players):
    return all([i in excluded_players or p.is_all_in() or p.get_current_bet() == table_highest_bet for i, p in enumerate(players)])


random_training_data = {player_id: [] for player_id in range(6)}


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
    player_outcomes, player_data = simulate_game(game, button)
    for i in range(6):
        outcome = player_outcomes[i]
        for state in player_data[i]:
            state.append(outcome)
        random_training_data[i].extend(player_data[i])



# Save to file
for i in range(6):
    file = np.array(random_training_data[i])
    np.save(f"player_{i}_state_data.npy", file)

    
