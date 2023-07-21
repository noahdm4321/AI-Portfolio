import random

# Function to initialize the game
def initialize_game():
    # Set up the global variables for the game board, player, and computer.
    global board, player, computer
    board = [' '] * 9  # Initialize the board as an empty list of length 9, representing a 3x3 grid.
    player = 'X'  # Player is assigned 'X'.
    computer = 'O'  # Computer is assigned 'O'.

# Function to print the board
def print_board(board):
    # Display the Tic Tac Toe board along with row and column labels.
    print('\n   A   B   C')
    print('1  ' + board[0] + ' | ' + board[1] + ' | ' + board[2] + ' ')
    print('  ―――――――――――')
    print('2  ' + board[3] + ' | ' + board[4] + ' | ' + board[5] + ' ')
    print('  ―――――――――――')
    print('3  ' + board[6] + ' | ' + board[7] + ' | ' + board[8] + ' \n')

# Function to check if the board is full
def is_board_full(board):
    # Check if there are any empty spaces left in the board.
    return ' ' not in board

# Function to check if the current move is a winning move
def is_winner(board, player):
    # Define all possible winning combinations of three marks in a row, column, or diagonal.
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
    # Check if the specified player has any winning combination.
    return [player, player, player] in winning_combinations

# Function to get all possible moves on the board
def get_possible_moves(board):
    # Get a list of all available (empty) positions on the board.
    return [i for i, x in enumerate(board) if x == ' ']

# Function to make a move
def make_move(board, position, player):
    # Place the player's mark at the specified position on the board.
    board[position] = player

# Function to undo a move
def undo_move(board, position):
    # Reset the specified position on the board to an empty space.
    board[position] = ' '

# Function to implement the rule-based strategy
def rule_based_strategy(board, possible_moves, player, computer):
    position = None

    # Rule 1: Check if the computer has a winning move
    for move in possible_moves:
        # Try placing the computer's mark at each available position to see if it wins.
        make_move(board, move, computer)
        if is_winner(board, computer):
            position = move
            undo_move(board, move)
            break
        undo_move(board, move)

    # Rule 2: Check if the player has a winning move and block it
    if position is None:
        for move in possible_moves:
            # Try placing the player's mark at each available position to see if it wins.
            make_move(board, move, player)
            if is_winner(board, player):
                position = move
                undo_move(board, move)
                break
            undo_move(board, move)

    # Rule 3: Check if the computer can win in two different ways
    if position is None:
        for move in possible_moves:
            # Try placing the computer's mark at each available position to see if it creates two winning chances.
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
    # Access the global player_score and computer_score variables
    global player_score, computer_score

    while True:
        # Print the current state of the board
        print_board(board)

        # Get a list of possible moves for the current board configuration
        possible_moves = get_possible_moves(board)

        # Get and validate the human player's move until it's a valid move
        while True:
            move = input('Enter your move (ex. A1): ')
            if len(move) == 2 and move[0].isalpha() and move[1].isdigit():
                # Convert the user input into the corresponding position on the board.
                column = ord(move[0].upper()) - ord('A')
                row = int(move[1]) - 1
                position = row * 3 + column
                if position in possible_moves:
                    break
            print('Invalid move. Try again.')

        # Make the human player's move on the board
        make_move(board, position, player)

        # Check if the human player wins
        if is_winner(board, player):
            print_board(board)
            print('You win!')
            player_score += 1
            break

        # Check if the board is full, resulting in a tie
        if is_board_full(board):
            print_board(board)
            print('It\'s a tie!')
            break

        # Get a new list of possible moves for the current board configuration
        possible_moves = get_possible_moves(board)

        # Let the computer player make its move using a rule-based strategy
        position = rule_based_strategy(board, possible_moves, player, computer)
        make_move(board, position, computer)

        # Check if the computer player wins
        if is_winner(board, computer):
            print_board(board)
            print('Computer wins!')
            computer_score += 1
            break

        # Check if the board is full, resulting in a tie
        if is_board_full(board):
            print_board(board)
            print('It\'s a tie!')
            break

# Function to play the game
def main():
    # Access the global player_score and computer_score variables
    global player_score, computer_score
    player_score = 0
    computer_score = 0

    print('Welcome to Tic Tac Toe!')

    while True:
        # Initialize the game board
        initialize_game()

        # Start playing the game
        play_game()

        # Display the current scores
        print(f"Current Score - You: {player_score} | Computer: {computer_score}\n")

        # Ask if the player wants to play again
        play_again = input("Do you want to play again? [yes]: ").lower()
        if play_again in ['yes', 'no', 'y', 'n', '']:
            if play_again in ['no', 'n']:
                break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    # Start the main game loop
    main()