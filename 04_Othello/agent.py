from copy import deepcopy
from othello import Othello
from mcts_UCB1 import MonteCarloTreeSearch
import numpy as np
from tensorflow import keras
from keras import layers
from keras.models import load_model

class OthelloAgent:
    """
    OthelloAgent class represents an intelligent agent for playing the game of Othello. It utilizes a combination of a neural network model and the Monte Carlo Tree Search (MCTS) algorithm for decision-making.

    Attributes:
    - exploration_constant (float): Controls the exploration in the Monte Carlo Tree Search.
    - num_simulations (int): Number of simulations to perform in each iteration of Monte Carlo Tree Search.
    - model (keras.Sequential): Neural network model used for predicting optimal moves based on the game state.
    - mcts (MonteCarloTreeSearch): Instance of the MonteCarloTreeSearch class for strategic decision-making.
    
    Methods:
    - create_model(): Creates the neural network model for the AI agent using Keras.
    - get_state_representation(othello_state): Converts Othello state to a format suitable for the neural network input.
    - train_agent(num_episodes=10): Trains the agent using the Monte Carlo Tree Search (MCTS).
    - determine_next_move(othello_state): Determines the next move based on the current Othello state.
    - save_model(filepath): Saves the neural network model and the Monte Carlo Tree for future use.
    - load_model(filepath): Loads a saved neural network model.
    """
    def __init__(self, exploration_constant=1.41, num_simulations=1000):
        # Initialize OthelloAgent with exploration constant and number of simulations for MCTS
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations

        # Build the neural network model and initialize Monte Carlo Tree Search
        self.model = self.create_model()
        self.mcts = MonteCarloTreeSearch(exploration_constant)

    def create_model(self):
        """Create the neural network model for the AI agent using Keras."""
        # Define a simple feedforward neural network using Keras
        model = keras.Sequential([
            layers.Flatten(input_shape=(8, 8)),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

        # Compile the model with mean squared error loss and Adam optimizer
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)
        return model

    def get_state_representation(self, othello_state):
        """Convert Othello state to a format suitable for input to the neural network"""
        return np.array(othello_state.board).reshape((8, 8, 1))

    def train_agent(self, num_episodes=1000):
        """Train the agent using the Monte Carlo Tree Search (MCTS)."""
        print(f"Training episode 1/{num_episodes}...")
        for episode in range(num_episodes):
            if episode % 100 == 0 and episode != 0:
                print(f"Training episode {episode}/{num_episodes}...")

            # Initialize a new game of Othello
            othello = Othello()
            othello.board[3][3] = 2
            othello.board[3][4] = 1
            othello.board[4][3] = 1
            othello.board[4][4] = 2
            states, actions, rewards = [], [], []

            while othello.has_legal_move() and sum(othello.num_tiles) != othello.n ** 2:
                # Use MCTS to select an action
                state_representation = self.get_state_representation(othello)
                selected_action = self.mcts.search(othello, self.num_simulations)

                # Play the selected action
                if othello.is_legal_move(selected_action):
                    othello.board[selected_action[0]][selected_action[1]] = othello.current_player + 1
                    othello.num_tiles[othello.current_player] += 1

                # Calculate reward based on winner
                if not othello.has_legal_move() or sum(othello.num_tiles) == othello.n ** 2:
                    # Reward for most tiles
                    player_tiles = sum(row.count(1) for row in othello.board)
                    opponent_tiles = sum(row.count(2) for row in othello.board)
                    # Considering adding additional rewards for corner tiles
                    reward = player_tiles - opponent_tiles
                reward=0

                # Store the current state, action, and immediate reward
                states.append(state_representation)
                actions.append(selected_action)
                rewards.append(reward)

            # Convert lists to numpy arrays
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)

            # Train the neural network using states as input, actions as targets, and rewards as rewards
            self.model.fit(states, actions, sample_weight=rewards, epochs=1, verbose=0)

    def determine_next_move(self, othello_state):
        """Determine the next best move for the agent based on model."""
        # Given the current Othello state, use the trained model to predict the best move
        legal_actions = othello_state.get_legal_moves()
        q_values = []
        for i in legal_actions:
            game_state = deepcopy(othello_state) # Create copy of game
            game_state.board[i[0]][i[1]] = 1  # Assume the current player is 1 for visualization
            state_representation = self.get_state_representation(game_state)
            q_values.append(self.model.predict(np.array([state_representation]), verbose=0)[0])
        
        # Find the index of the action with the highest Q-value
        best_action_index = np.argmax(q_values)
        # print(f'{["{:.4f}".format(float(value)) for value in q_values]} - {best_action_index}')

        # Get legal actions and select the best action
        best_action = legal_actions[best_action_index]
        # print(f'{legal_actions} - {best_action}')

        return best_action

    def save_model(self, filepath):
        """Save the model."""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load a saved model."""
        self.model = load_model(filepath)