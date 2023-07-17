import os
import random

class MathEnv:
    def __init__(self):
        self.q_table = {}

    def get_valid_moves(self, current_number):
        valid_moves = []
        for operation in ['+', '-', '*', '/']:
            if operation in ['+', '-']:
                numbers = range(1, 6)
            else:
                numbers = range(2, 6)
            valid_moves.extend(
                [
                    (operation, number, current_number + number)
                    if operation == '+'
                    else (operation, number, current_number - number)
                    if operation == '-'
                    else (operation, number, current_number * number)
                    if operation == '*'
                    else (operation, number, current_number / number)
                    for number in numbers
                    if (result := current_number + number) <= 20 and result == int(result)
                ]
            )
        return valid_moves

    def update_q_table(self, state, action, next_state, reward):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.get_valid_moves(state)}
        if next_state is not None:
            next_q_values = self.q_table[next_state].values()
            self.q_table[state][action] += 0.1 * (reward + 0.9 * max(next_q_values, default=0) - self.q_table[state][action])


    def get_best_move(self, current_number):
        valid_moves = self.get_valid_moves(current_number)
        current_number_q_table = self.q_table.get(current_number, {})
        return max(valid_moves, key=lambda move: current_number_q_table.get(move, 0))

class QLearningAgent:
    def __init__(self):
        self.q_table = {}

    def choose_action(self, state):
        if state not in self.q_table:
            env = MathEnv()  # Create an instance of MathEnv
            valid_moves = env.get_valid_moves(state)  # Call get_valid_moves on the instance
            self.q_table[state] = {action: 0 for action in valid_moves}
        return max(self.q_table[state], key=self.q_table[state].get)


    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = random.randint(1, 10)

            while state != 20:
                valid_moves = env.get_valid_moves(state)
                if valid_moves:
                    action = self.choose_action(state)
                    next_state = random.choice(valid_moves)[2]
                    reward = len(valid_moves)
                    env.update_q_table(state, action, next_state, reward)
                    state = next_state
                else:
                    break


            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed.")

        print("Training completed.")

def main():
    env = MathEnv()
    agent = QLearningAgent()
    
    agent.train(env, num_episodes=10000)

    current_number = 1

    while current_number < 20:
        print("\nCurrent number:", current_number)

        # Player's turn
        valid_move = False
        while not valid_move:
            player_input = input("Enter your move (ex. +2): ").strip()
            operation = player_input[0]
            number = int(player_input[1])

            if operation in ['+', '-'] and number in range(1, 6):
                result = current_number + number if operation == '+' else current_number - number
            elif operation in ['*', '/'] and number in range(2, 6):
                result = current_number * number if operation == '*' else current_number / number
            else:
                print("Invalid move. Try again.")
                continue

            if result > 20:
                print(f"Invalid move! {current_number} {operation} {number} = {result} > 20\n")
                print("Current number:", current_number)
            elif result != int(result):
                print(f"Invalid move! {current_number} {operation} {number} = {result} (not a whole number)\n")
                print("Current number:", current_number)
            else:
                valid_move = True
                print(f"{current_number} {operation} {number} = {result}")
                current_number = int(result)

        if current_number == 20:
            print("Congratulations, you win!")
            break

        # Computer's turn
        print("\nComputer's turn...")
        operation, number, result = env.get_best_move(current_number)
        print(f"{current_number} {operation} {number} = {result}")
        current_number = result

        if current_number == 20:
            print("I win! You lose! Haha.")
            break

        # Update Q-table after each round
        valid_moves = env.get_valid_moves(current_number)
        next_states = [(current_number, op, num) for op, num, _ in valid_moves]
        reward = len(valid_moves)
        for state in next_states:
            agent.q_table[state] = {action: 0 for action in env.get_valid_moves(state)}
        agent.update_q_table((current_number, operation, number), (operation, number), next_states[0], reward)

    os.system("pause")


if __name__ == "__main__":
    main()