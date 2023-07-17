import random

class MathEnv:
    def __init__(self):
        self.current_number = random.randint(1, 10)
        self.result = 0
        self.num_iterations = 0

    def reset(self):
        # Reset the environment state
        self.current_number = random.randint(1, 10)
        self.result = 0
        self.num_iterations = 0
        return self.current_number

    def step(self, action):
        # Perform the chosen action and update the state, reward, and done flag
        operation = ['+', '-', '*', '/'][action]
        if operation in ['+', '-']:
            number = random.randint(1, 5)
        else:
            number = random.randint(2, 5)

        if operation == '+':
            self.result = self.current_number + number
        elif operation == '-':
            self.result = self.current_number - number
        elif operation == '*':
            self.result = self.current_number * number
        else:  # division
            self.result = self.current_number / number

        self.num_iterations += 1
        done = self.result == 20 or self.num_iterations >= 100
        reward = self.num_iterations  # The smaller the score, the better

        return self.result, reward, done, {}

    def render(self):
        # Display the current state
        print(f"Current Number: {self.current_number}, Result: {self.result}")

class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = [[0] * num_actions]

    def choose_action(self, state):
        # Select an action using epsilon-greedy policy
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        # Update the Q-table based on the Q-learning update rule
        q_value = (1 - self.learning_rate) * self.q_table[state][action] + \
                  self.learning_rate * (reward + self.discount_factor * max(self.q_table[next_state]))
        self.q_table[state][action] = q_value

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed.")

        print("Training completed.")

# Usage example
env = MathEnv()
num_states = 10  # Adjust according to the range of current_number in the environment
num_actions = 4  # '+', '-', '*', '/' operations

agent = QLearningAgent(num_actions)
agent.train(env, num_episodes=1000)

# After training, you can use the learned Q-table to play the game
state = env.reset()
done = False

while not done:
    action = np.argmax(agent.q_table[state])
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()

env.render()