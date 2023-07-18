import pickle
import random

# Define Q-Learning method for AI
class QLearningModel():
    def __init__(self):
        # Define the game parameters
        self.operations = ['*', '+']
        self.num_range = list(range(1, 6))
        self.actions = [(op, num) for op in self.operations for num in self.num_range]
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.6  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def check_winning_move(self, current_number, operation, number):
        if operation == '*':
            next_number = current_number * number
        elif operation == '+':
            next_number = current_number + number

        return next_number == target_number and next_number <= target_number and next_number != current_number


    def get_action(self, current_number):
        valid_actions = [(op, num) for op, num in self.actions if self.check_valid_action(current_number, op, num)]
        if random.uniform(0, 1) < self.epsilon:
            # Choose a random action
            return random.choice(valid_actions)
        else:
            # Choose the action with the highest Q-value, or a winning move if available
            if current_number not in self.q_table:
                self.q_table[current_number] = {action: 0 for action in self.actions}
    
            winning_moves = [(op, num) for op, num in valid_actions if self.check_winning_move(current_number, op, num)]
            if winning_moves:
                return random.choice(winning_moves)
        
            return max(self.q_table[current_number], key=self.q_table[current_number].get)

    def check_valid_action(self, current_number, operation, number):
        if operation == '*':
            next_number = current_number * number
        elif operation == '+':
            next_number = current_number + number

        return next_number <= target_number and next_number != current_number

    def get_reward(self, current_number):
        if current_number == target_number:
            return 1  # Reward for winning the game
        else:
            return 0  # No reward

    def q_learning(self):
        current_number = 1  # Start with 1
        while current_number != target_number:
            action = self.get_action(current_number)
            operation, number = action

            # Update the state based on the chosen action
            if operation == '*':
                next_number = current_number * number
            elif operation == '+':
                next_number = current_number + number

            # Check if the action is valid (within the game rules)
            if next_number <= target_number and next_number != current_number:
                # Update the Q-value of the previous state-action pair
                if current_number not in self.q_table:
                    self.q_table[current_number] = {action: 0 for action in self.actions}
                if next_number not in self.q_table:
                    self.q_table[next_number] = {action: 0 for action in self.actions}
                self.q_table[current_number][action] += self.alpha * (
                            self.get_reward(next_number) + self.gamma * max(self.q_table[next_number].values()) -
                            self.q_table[current_number][action])

                # Move to the next state
                current_number = next_number

    def train(self, num_episodes):
        for _ in range(num_episodes):
            self.q_learning()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.q_table = pickle.load(file)

# Function to handle the human player's turn
def human_player_turn(current_number):
    print(f"\nCurrent number: {current_number}")
    valid_input = False
    while not valid_input:
        operation = input("Choose an operation (+ or *): ")
        number_input = input("Choose a number (1-5): ")
        try:
            number = int(number_input)
            if (operation == "*" and 2 <= number <= 5) or (operation == "+" and 1 <= number <= 5):
                if operation == "+" and (current_number + number > target_number):
                    print(f"{current_number} + {number} = {current_number + number}\nResult is greater than {target_number}. Please try again.")
                    continue
                elif operation == "*" and (current_number * number > target_number):
                    print(f"{current_number} * {number} = {current_number * number}\nResult is greater than {target_number}. Please try again.")
                    continue
                else:
                    valid_input = True
            elif operation == "*" and number == 1:
                print("You cannot multiply by 1. Please try again.")
                continue
            else:
                print("Invalid input. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue


    # Perform the chosen operation
    if operation == "*":
        next_number = current_number * number
    elif operation == "+":
        next_number = current_number + number

    # Print the math of the human player's move
    print(f"Your move: {current_number} {operation} {number} = {next_number}")

    return next_number

# Function to play the game against the computer
def play_game():
    # Define the game parameters
    global env, player_score, computer_score
    current_number = 1  # Start with 1
    game_data = []

    while current_number != target_number:
        # Human player's turn
        current_number = human_player_turn(current_number)
        if current_number == target_number:
            print("\nCongratulations! You won the game!")
            player_score += 1
            break

        # Computer player's turn
        operation, number = env.get_action(current_number)

        # Perform the chosen operation
        if operation == "*":
            next_number = current_number * number
        elif operation == "+":
            next_number = current_number + number

        # Print the math of the computer player's move
        print(f"Computer's move: {current_number} {operation} {number} = {next_number}")

        # Update the state for the next turn
        current_number = next_number

        if current_number == target_number:
            print("\nThe computer won the game!")
            computer_score += 1
            game_data.append((current_number, None))  # Add game data for the computer player's last move
        else:
            game_data.append((current_number, (operation, number)))  # Add game data for the computer player's move

    # Update the Q-Learning model using the game data
    for current_number, action in game_data:
        if action:
            env.q_table[current_number][action] += env.alpha * 1000 * (env.get_reward(current_number) + max(env.q_table[current_number].values()) - env.q_table[current_number][action])


def main():
    play_game()
    while True:
        print(f"Current Score - You: {player_score} | Computer: {computer_score}")
        choice = input("Do you want to play again? [yes]: ").lower()
        if choice in ['yes', 'no', 'y', 'n', '']:
            if choice in ['yes', 'y', '']:
                play_game()
            else:
                env.save(q_values_file)
                break
        else:
            print("\nInvalid choice. Please try again.")


# Play the game
if __name__ == '__main__':
    target_number = 30
    player_score = 0
    computer_score = 0
    env = QLearningModel()

    # Load Q-values if available, or train the model
    q_values_file = '/03_Operation-Relay/q_values.pkl'
    try:
        env.load(q_values_file)
        print("Loaded Q-values from file.")
    except FileNotFoundError:
        print("No existing Q-values found. Training the model...")
        env.train(num_episodes=1000000)  # Train model
        env.save(q_values_file)
        print("Q-values saved to file.")

    print("Finished! Let's play.")
    main()