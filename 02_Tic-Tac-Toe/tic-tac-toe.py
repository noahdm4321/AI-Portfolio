import random

# Function to initialize the game
def initialize_game():
    global board, player, computer
    board = [' '] * 9
    player = 'X'
    computer = 'O'

# Function to print the board
def print_board(board):
    print('\n   A   B   C')
    print('1  ' + board[0] + ' | ' + board[1] + ' | ' + board[2] + ' ')
    print('  ―――――――――――')
    print('2  ' + board[3] + ' | ' + board[4] + ' | ' + board[5] + ' ')
    print('  ―――――――――――')
    print('3  ' + board[6] + ' | ' + board[7] + ' | ' + board[8] + ' \n')

# Function to check if the board is full
def is_board_full(board):
    return ' ' not in board

# Function to check if the current move is a winning move
def is_winner(board, player):
    winning_combinations = [
        [board[0], board[1], board[2]],
        [board[3], board[4], board[5]],
        [board[6], board[7], board[8]],
        [board[0], board[3], board[6]],
        [board[1], board[4], board[7]],
        [board[2], board[5], board[8]],
        [board[0], board[4], board[8]],
        [board[2], board[4], board[6]]
    ]
    return [player, player, player] in winning_combinations

# Function to get all possible moves on the board
def get_possible_moves(board):
    return [i for i, x in enumerate(board) if x == ' ']

# Function to make a move
def make_move(board, position, player):
    board[position] = player

# Function to undo a move
def undo_move(board, position):
    board[position] = ' '

# Function to implement the rule-based strategy
def rule_based_strategy(board, possible_moves, player, computer):
    position = None

    # Rule 1: Check if the computer has a winning move
    for move in possible_moves:
        make_move(board, move, computer)
        if is_winner(board, computer):
            position = move
            undo_move(board, move)
            break
        undo_move(board, move)

    # Rule 2: Check if the player has a winning move and block it
    if position is None:
        for move in possible_moves:
            make_move(board, move, player)
            if is_winner(board, player):
                position = move
                undo_move(board, move)
                break
            undo_move(board, move)

    # Rule 3: Check if the computer can win in two different ways
    if position is None:
        for move in possible_moves:
            make_move(board, move, computer)
            winning_moves = 0
            for next_move in possible_moves:
                make_move(board, next_move, computer)
                if is_winner(board, computer):
                    winning_moves += 1
                undo_move(board, next_move)
            if winning_moves >= 2:
                position = move
                undo_move(board, move)
                break
            undo_move(board, move)

    # Rule 4: Take the center spot if available
    if position is None and 4 in possible_moves:
        position = 4

    # Rule 5: Randomly select a move
    if position is None:
        position = random.choice(possible_moves)

    return position

# Main game loop
def play_game():
    global player_score, computer_score

    while True:
        print_board(board)

        possible_moves = get_possible_moves(board)
        while True:
            move = input('Enter your move (ex. A1): ')
            if len(move) == 2 and move[0].isalpha() and move[1].isdigit():
                column = ord(move[0].upper()) - ord('A')
                row = int(move[1]) - 1
                position = row * 3 + column
                if position in possible_moves:
                    break
            print('Invalid move. Try again.')

        make_move(board, position, player)

        if is_winner(board, player):
            print_board(board)
            print('You win!')
            player_score += 1
            break

        if is_board_full(board):
            print_board(board)
            print('It\'s a tie!')
            break

        possible_moves = get_possible_moves(board)
        # Rule-based strategy for the computer player
        position = rule_based_strategy(board, possible_moves, player, computer)
        make_move(board, position, computer)

        if is_winner(board, computer):
            print_board(board)
            print('Computer wins!')
            computer_score += 1
            break

        if is_board_full(board):
            print_board(board)
            print('It\'s a tie!')
            break

# Function to play the game
def main():
    global player_score, computer_score
    player_score = 0
    computer_score = 0
    print('Welcome to Tic Tac Toe!')

    while True:
        initialize_game()
        play_game()

        print(f"Current Score - You: {player_score} | Computer: {computer_score}\n")

        play_again = input("Do you want to play again? [yes]: ").lower()
        if play_again in ['yes', 'no', 'y', 'n', '']:
            if play_again in ['no', 'n']:
                break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()