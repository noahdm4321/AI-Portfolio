import random

# Initialize the Q-table with zeros
q_table = {}

# Define the learning parameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate

# Define the game parameters
target_number = 20

# Define the possible operations and ranges of numbers for each operation
operations = ["+", "-", "*", "/"]
add_sub_range = list(range(1, 6))
mul_div_range = list(range(2, 6))


# Define the state, action, and reward functions
def get_state(prev_number, current_number):
    return (prev_number, current_number)


def get_action(state):
    if random.uniform(0, 1) < epsilon:
        # Choose a random action
        return random.choice(operations), random.choice(add_sub_range + mul_div_range)
    else:
        # Choose the action with the highest Q-value
        if state not in q_table:
            actions = [
                (op, num)
                for op in operations
                for num in (add_sub_range + mul_div_range)
            ]
            q_table[state] = {action: 0 for action in actions}
        return max(q_table[state], key=q_table[state].get)


def get_reward(current_number):
    if current_number == target_number:
        return 1  # Reward for winning the game
    else:
        return 0  # No reward


# Perform Q-Learning
def q_learning():
    prev_number = 1
    current_number = 2  # Start with 2
    state = get_state(prev_number, current_number)

    while current_number != target_number:
        action = get_action(state)
        operation, number = action

        # Update the state based on the chosen action
        if operation == "+":
            next_number = current_number + number
        elif operation == "-":
            next_number = current_number - number
        elif operation == "*":
            next_number = current_number * number
        elif operation == "/":
            if current_number % number == 0:
                next_number = current_number // number
            else:
                continue  # Skip this action as it would result in a non-whole number

        # Check if the action is valid (within the game rules)
        if next_number <= target_number and next_number != current_number:
            next_state = get_state(current_number, next_number)

            # Update the Q-value of the previous state-action pair
            actions = [
                (op, num)
                for op in operations
                for num in (add_sub_range + mul_div_range)
            ]
            if state not in q_table:
                q_table[state] = {action: 0 for action in actions}
            if next_state not in q_table:
                q_table[next_state] = {action: 0 for action in actions}
            q_table[state][action] += alpha * (
                get_reward(next_number)
                + gamma * max(q_table[next_state].values())
                - q_table[state][action]
            )

            # Move to the next state
            prev_number = current_number
            current_number = next_number
            state = next_state


# Train the Q-Learning model
num_episodes = 10000
for _ in range(num_episodes):
    q_learning()
