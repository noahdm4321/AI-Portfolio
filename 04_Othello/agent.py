from othello import Othello
from mcts import MonteCarloTreeSearch
import numpy as np
import keras
from keras import layers
from keras import models

MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
             (0, -1),           (0, +1),
             (+1, -1), (+1, 0), (+1, +1)]

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
            keras.Input(shape=(8,8,3)), # Input shape: 8x8 board with 3 channels
            # Convolutional layers to extract features
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(axis=-1),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(axis=-1),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(axis=-1),
            layers.Activation('relu'),
            # Flatten the output to feed into dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(64, activation='softmax')  # Output probability field over board
        ])

        # Compile the model with mean squared error loss and SGD optimizer
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()])
        return model

    def get_state_representation(self, othello_state):
        """Convert Othello state to a format suitable for input to the neural network"""
        board = np.array(othello_state.board).reshape((8, 8))
        state_array = np.zeros((8, 8, 3), dtype=np.int8)
        state_array[board == 1, 0] = 1
        state_array[board == -1, 1] = 1
        state_array[board == 0, 2] = 1
        return state_array

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
            states = []

            while othello.has_legal_move() and sum(othello.num_tiles) != othello.n ** 2:
                # Use MCTS to select an action
                state_representation = self.get_state_representation(othello)
                selected_action = self.mcts.search(othello, self.num_simulations)

                # Play the selected action (modified functions from othello.py)
                if othello.is_legal_move(selected_action):
                    # make_move
                    othello.board[selected_action[0]][selected_action[1]] = othello.current_player + 1
                    othello.num_tiles[othello.current_player] += 1
                    # flip_tiles
                    curr_tile = othello.current_player + 1 
                    for direction in MOVE_DIRS:
                        # has_tile_to_flip
                        i = 1
                        if othello.current_player in (0, 1) and \
                        othello.is_valid_coord(selected_action[0], selected_action[1]):
                            curr_tile = othello.current_player + 1
                            while True:
                                row = selected_action[0] + direction[0] * i
                                col = selected_action[1] + direction[1] * i
                                if not othello.is_valid_coord(row, col) or \
                                    othello.board[row][col] == 0:
                                    return False
                                elif othello.board[row][col] == curr_tile:
                                    break
                                else:
                                    i += 1
                        # flip_tiles
                        if i > 1:
                            i = 1
                            while True:
                                row = selected_action[0] + direction[0] * i
                                col = selected_action[1] + direction[1] * i
                                if othello.board[row][col] == curr_tile:
                                    break
                                else:
                                    othello.board[row][col] = curr_tile
                                    othello.num_tiles[othello.current_player] += 1
                                    othello.num_tiles[(othello.current_player + 1) % 2] -= 1
                                    i += 1

                    if othello.current_player == 0:
                        othello.current_player += 1
                    else:
                        othello.current_player += -1

                # Store the current state, and immediate reward
                states.append(state_representation)

            # Determine the winner
            player_tiles = sum(row.count(2) for row in othello.board)
            opponent_tiles = sum(row.count(1) for row in othello.board)
            if player_tiles > opponent_tiles:
                reward = player_tiles
            elif player_tiles < opponent_tiles:
                reward = 0-player_tiles
            else:
                reward = 0
            num_state = len(states)
            rewards = [reward]*num_state

            # Convert lists to numpy arrays
            states = np.array(states)
            rewards = np.array(rewards)

            # Train the neural network using states as input, rewards as targets
            self.model.fit(states, rewards, batch_size=num_state, epochs=1, verbose=0)

    def determine_next_move(self, othello_state):
        """Determine the next best move for the agent based on model."""
        # Given the current Othello state, use the trained model to predict the best move
        legal_actions = othello_state.get_legal_moves()

        # Initialize the variables
        q_values = []

        # Iterate through legal_actions
        state_representation = self.get_state_representation(othello_state)
        q_value = self.model.predict(np.array([state_representation]), verbose=0)[0]
        for move in legal_actions:
            q_values.append(q_value.reshape(8, 8)[move[0]][move[1]])

        q_values = np.array(q_values)
        print(q_values)
        best_move = legal_actions[np.argmax(q_values)]
        print(best_move)

        return best_move

    def save_model(self, filepath):
        """Save the model."""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load a saved model."""
        self.model = models.load_model(filepath)