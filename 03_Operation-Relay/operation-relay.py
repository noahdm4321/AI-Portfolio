# Import the necessary functions from the Q-Learning model file
from q_learning import get_state, get_action

# Define the game parameters
target_number = 20


# Function to handle the human player's turn
def human_player_turn(current_number):
    print(f"\nCurrent number: {current_number}")
    valid_input = False
    while not valid_input:
        operation = input("Choose an operation (+, -, *, /): ")
        number = int(input("Choose a number (1-5 for + and -, 2-5 for * and /): "))
        if operation in ["+", "-", "*", "/"] and (
            (operation in ["+", "-"] and 1 <= number <= 5)
            or (operation in ["*", "/"] and 2 <= number <= 5)
        ):
            valid_input = True
        else:
            print("Invalid input. Please try again.")

    # Perform the chosen operation
    if operation == "+":
        next_number = current_number + number
    elif operation == "-":
        next_number = current_number - number
    elif operation == "*":
        next_number = current_number * number
    elif operation == "/":
        next_number = current_number // number

    # Print the math of the human player's move
    print(f"Your move: {current_number} {operation} {number} = {next_number}")

    return next_number


# Function to play the game against the computer
def play_game():
    prev_number = 1
    current_number = 1  # Start with 1

    while current_number != target_number:
        # Human player's turn
        current_number = human_player_turn(current_number)
        if current_number == target_number:
            print("\nCongratulations! You won the game!")
            break

        # Computer player's turn
        state = get_state(prev_number, current_number)
        operation, number = get_action(state)

        # Perform the chosen operation
        if operation == "+":
            next_number = current_number + number
        elif operation == "-":
            next_number = current_number - number
        elif operation == "*":
            next_number = current_number * number
        elif operation == "/":
            next_number = current_number // number

        # Update the state for the next turn
        prev_number = current_number
        current_number = next_number

        # Print the math of the computer player's move
        print(f"Computer's move: {prev_number} {operation} {number} = {current_number}")

        if current_number == target_number:
            print("\nThe computer won the game!")


# Play the game
play_game()
