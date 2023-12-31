# Import necessary libraries
import pickle
import random
import os


# Define Q-Learning method for AI
class QLearningModel:
    def __init__(self):
        # Define the game parameters
        self.operations = ["*", "+"]  # Available mathematical operations
        self.num_range = list(range(1, 6))  # Allowed numbers to use in operations
        self.actions = [
            (op, num) for op in self.operations for num in self.num_range
        ]  # All possible actions (operation, number) pairs
        self.q_table = {}  # Q-table to store state-action values
        self.alpha = 0.1  # Learning rate for Q-learning algorithm
        self.gamma = 0.8  # Discount factor for future rewards
        self.epsilon = 0.1  # Exploration rate for choosing random actions vs. exploiting learned values

    def get_action(self, current_number):
        # Get list of valid moves for the given current number
        valid_actions = [
            (op, num)
            for op, num in self.actions
            if self.check_valid_action(current_number, op, num)
        ]
        if current_number not in self.q_table:
            self.q_table[current_number] = {action: 0 for action in valid_actions}

        # Choose winning move if available to end the game
        winning_moves = [
            (op, num)
            for op, num in valid_actions
            if self.check_winning_move(current_number, op, num)
        ]
        if winning_moves:
            return random.choice(winning_moves)

        # Use epsilon-greedy strategy to either explore randomly or exploit learned values
        if random.uniform(0, 1) < self.epsilon:
            # Choose a random action
            return random.choice(valid_actions)
        else:
            # Choose the action with the highest Q-value for exploitation
            return max(
                self.q_table[current_number], key=self.q_table[current_number].get
            )

    def check_winning_move(self, current_number, operation, number):
        # Check if the given action leads to a winning move (reaching the target_number without exceeding it)
        if operation == "*":
            next_number = current_number * number
        elif operation == "+":
            next_number = current_number + number

        return (
            next_number == target_number
            and next_number <= target_number
            and next_number != current_number
        )

    def check_valid_action(self, current_number, operation, number):
        # Check if the given action is valid (does not exceed the target_number and does not result in the same number)
        if operation == "*":
            next_number = current_number * number
        elif operation == "+":
            next_number = current_number + number

        return next_number <= target_number and next_number != current_number

    def update_q_values(self, moves, reward):
        # Update the Q-values in the Q-table based on the game moves and the reward received
        key_numbers = list(moves.keys())
        for current_number, action in sorted(moves.items()):
            operation, number = action
            if key_numbers.index(current_number) < len(key_numbers) - 1:
                next_number = key_numbers[key_numbers.index(current_number) + 1]
                # Q-value update using Q-learning algorithm
                self.q_table[current_number][operation, number] += self.alpha * (
                    reward
                    + self.gamma * max(self.q_table[next_number].values())
                    - self.q_table[current_number][operation, number]
                )
            else:
                # If the game is won, the last state-action pair gets a higher reward (game end)
                self.q_table[current_number][operation, number] += self.alpha * (
                    reward * 2 - self.q_table[current_number][operation, number]
                )

    def q_learning(self):
        # Perform Q-learning to update the Q-table based on the game's state-action pairs
        current_number = 1  # Start with 1
        player = 1
        p1_moves = {}  # Store player 1's moves
        p2_moves = {}  # Store player 2's moves

        while current_number != target_number:
            operation, number = self.get_action(current_number)

            # Update the state based on the chosen action
            if operation == "*":
                next_number = current_number * number
            elif operation == "+":
                next_number = current_number + number

            # Add game data for the current player's move
            if player == 1:
                p1_moves[current_number] = [operation, number]
                player += 1
            else:
                p2_moves[current_number] = [operation, number]
                player -= 1

            # Update the Q-value of the previous state-action pair
            if next_number == target_number:
                if player == 2:  # Player 1 made the winning move
                    self.update_q_values(p1_moves, 1)  # Player 1 gets reward 1
                    self.update_q_values(p2_moves, 0)  # Player 2 gets no reward (lost)
                else:  # Player 2 made the winning move
                    self.update_q_values(p1_moves, 0)  # Player 1 gets no reward (lost)
                    self.update_q_values(p2_moves, 1)  # Player 2 gets reward 1
                break

            # Move to the next state
            current_number = next_number

    def train(self, num_episodes):
        # Train the Q-learning model by running Q-learning for the specified number of episodes
        for _ in range(num_episodes):
            self.q_learning()

    def save(self, filename):
        # Save the Q-table to a binary file using pickle
        with open(filename, "wb") as file:
            pickle.dump(self.q_table, file)

    def load(self, filename):
        # Load the Q-table from a binary file using pickle
        with open(filename, "rb") as file:
            self.q_table = pickle.load(file)


# Function to handle the human player's turn
def human_player_turn(current_number):
    print(f"\nCurrent number: {current_number}")
    valid_input = False
    while not valid_input:
        input_str = input("Choose an operation (+ or *) and a number (1-5): ")

        # Separate the input string into operation and number parts
        operation = input_str[0]
        number_input = input_str[1:]

        try:
            number = int(number_input)
            # Check if the chosen operation and number are valid
            if (operation == "*" and 2 <= number <= 5) or (
                operation == "+" and 1 <= number <= 5
            ):
                # Check if the result of the chosen operation exceeds the target number
                if operation == "+" and (current_number + number > target_number):
                    print(
                        f"{current_number} + {number} = {current_number + number} > {target_number}. Please try again."
                    )
                    continue
                elif operation == "*" and (current_number * number > target_number):
                    print(
                        f"{current_number} * {number} = {current_number * number} > {target_number}. Please try again."
                    )
                    continue
                else:
                    valid_input = True
            # If the operation is multiplication by 1, disallow it
            elif operation == "*" and number == 1:
                print("You cannot multiply by 1! The current number must change.")
                continue
            else:
                print("Invalid input. Please try again.")
                continue
        except ValueError:
            print("Invalid number. Please enter a valid number.")
            continue

    # Return the valid operation and number chosen by the human player
    return operation, number


# Function to play the game against the computer
def play_game():
    # Initialize game parameters and scores
    global env, player_score, computer_score
    current_number = 1  # Start with 1
    comp_moves = {}
    hum_moves = {}

    # Continue the game until the target number is reached
    while current_number != target_number:
        # Human player's turn
        operation, number = human_player_turn(current_number)

        # If the current state and action are not in the Q-Learning model, add them with an initial value of 0
        if (
            current_number not in env.q_table
            or (operation, number) not in env.q_table[current_number]
        ):
            env.q_table[current_number] = {(operation, number): 0}

        # Perform the chosen operation and update the current number
        if operation == "*":
            next_number = current_number * number
        elif operation == "+":
            next_number = current_number + number

        # Print the mathematical operation of the human player's move
        print(f"Your move: {current_number} {operation} {number} = {next_number}")
        hum_moves[current_number] = [
            operation,
            number,
        ]  # Add game data for the computer player's move

        # Check if the human player has reached the target number
        if next_number == target_number:
            print("\nCongratulations! You won the game!")
            player_score += 1
            # Update the Q-Learning model using the game data (human wins with a reward of 10)
            env.update_q_values(hum_moves, 10)
            env.update_q_values(comp_moves, 0)
            break

        # Update the state for the next turn
        current_number = next_number

        # Computer player's turn
        operation, number = env.get_action(current_number)

        # Perform the chosen operation and update the current number
        if operation == "*":
            next_number = current_number * number
        elif operation == "+":
            next_number = current_number + number

        # Print the mathematical operation of the computer player's move
        print(f"Computer's move: {current_number} {operation} {number} = {next_number}")
        comp_moves[current_number] = [
            operation,
            number,
        ]  # Add game data for the computer player's move

        # Check if the computer player has reached the target number
        if next_number == target_number:
            print("\nThe computer won the game!")
            computer_score += 1
            # Update the Q-Learning model using the game data (computer wins with a reward of 10)
            env.update_q_values(hum_moves, 0)
            env.update_q_values(comp_moves, 10)
            break

        # Update the state for the next turn
        current_number = next_number


def main():
    # Check if the user wants to load the Q-values from a file
    load = input("Do you want to load the winning strategy file? [yes]: ").lower()
    if load in ["yes", "y", ""]:
        # Load Q-values if available, or train the model
        q_values_file = os.path.join(os.path.dirname(__file__), "winning_srat.pkl")

        try:
            env.load(q_values_file)
            print("Loaded Q-values from file.")
        except FileNotFoundError:
            print("No existing Q-values found. Training new model...")
            env.train(
                num_episodes=1000000
            )  # Train model with a large number of episodes
            env.save(q_values_file)  # Save the trained Q-values to a file
            print("Q-values saved to file.")

        print("Finished! Let's play.")
    else:
        # Check if the user wants to train a new model
        choice = input("Do you want to train a new model? [yes]: ").lower()
        if choice in ["yes", "y", ""]:
            print("Training the model...")
            env.train(
                num_episodes=1000
            )  # Train model with a moderate number of episodes
            print("Finished! Let's play.")
        else:
            print("Loading game with no training! Let's play.")
    play_game()

    # After the first game, ask the user if they want to play again or quit
    while True:
        print(f"Current Score - You: {player_score} | Computer: {computer_score}")
        choice = input("Do you want to play again? [yes]: ").lower()
        if choice in ["yes", "no", "y", "n", ""]:
            if choice in ["yes", "y", ""]:
                play_game()  # Play another game
            elif load in ["yes", "y", ""]:
                env.save(q_values_file)  # Save the Q-values to the file before quitting
                break
            else:
                break
        else:
            print("\nInvalid choice. Please try again.")


# Play the game
if __name__ == "__main__":
    target_number = 30
    player_score = 0
    computer_score = 0
    env = QLearningModel()  # Initialize the Q-Learning model
    main()  # Start the main game loop
