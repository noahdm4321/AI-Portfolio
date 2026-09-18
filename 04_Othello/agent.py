from copy import deepcopy
from othello import Othello
from mcts import MonteCarloTreeSearch
import numpy as np
import random
import keras
from keras import layers
from keras import models

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
     - train_agent(num_episodes=10, draw=False): Trains the agent using the Monte Carlo Tree Search (MCTS).
    - determine_next_move(othello_state): Determines the next move based on the current Othello state.
    - save_model(filepath): Saves the neural network model and the Monte Carlo Tree for future use.
    - load_model(filepath): Loads a saved neural network model.
    """
    def __init__(self, exploration_constant=1.41, num_simulations=50):
        # Initialize OthelloAgent with exploration constant and number of simulations for MCTS.
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations
        self.random_mode = False

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
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
        return model

    def get_state_representation(self, othello_state):
        """Convert Othello state to a format suitable for input to the neural network"""
        return np.array(othello_state.board).reshape((8, 8, 1))

    def train_agent(self, num_episodes=100, draw=False):
        """Train the agent using the Monte Carlo Tree Search (MCTS).

        draw (bool): If True, draw the board with turtle graphics during
                     training for debugging. Defaults to False.
        """
        if draw:
            import turtle as _turtle
            _turtle.tracer(False)
            _turtle.delay(50)

        print(f"Training episode 1/{num_episodes}...")
        for episode in range(num_episodes):
            if episode % 10 == 0 and episode != 0:
                print(f"Training episode {episode}/{num_episodes}...")

            if draw:
                import turtle as _turtle
                _turtle.Screen().clear()
                othello_draw = Othello()
                othello_draw.draw_board()
                othello_draw.initialize_board()
                _turtle.update()

            # Initialize a new game of Othello
            othello = Othello()
            othello.board[3][3] = 2
            othello.board[3][4] = 1
            othello.board[4][3] = 1
            othello.board[4][4] = 2
            states, actions, players = [], [], []

            while sum(othello.num_tiles) < othello.n ** 2:
                if not othello.has_legal_move():
                    # Current player can't move — pass to opponent.
                    # In Othello, the game continues with the opponent's turn.
                    othello.current_player = 1 - othello.current_player
                    if not othello.has_legal_move():
                        break  # Both players can't move — game over
                    continue

                # Use MCTS to select an action
                state_representation = self.get_state_representation(othello)
                selected_action = self.mcts.search(othello, self.num_simulations)

                # Play the selected action (includes tile flipping via make_move)
                if selected_action is not None and othello.is_legal_move(selected_action):
                    othello.move = selected_action
                    othello.make_move(draw=draw)

                if draw:
                    import turtle as _turtle
                    _turtle.update()

                # Store the current state, action, and the player who made it.
                # The reward is computed from each player's own perspective
                # after the episode ends (see below).
                states.append(state_representation)
                actions.append(selected_action)
                players.append(othello.current_player)

                # Switch player
                othello.current_player = 1 - othello.current_player

            # Reward for most tiles, computed from player 0's perspective.
            # (player 0 = black/1, player 1 = white/2)
            player_tiles = sum(row.count(1) for row in othello.board)
            opponent_tiles = sum(row.count(2) for row in othello.board)
            # Normalize to [-1, 1] so the sample_weight stays bounded
            base_reward = (player_tiles - opponent_tiles) / (othello.n ** 2)

            # Convert lists to numpy arrays
            states = np.array(states)
            actions = np.array(actions)

            # Build a per-sample sample_weight from each move's own perspective.
            rewards = np.array([
                base_reward if p == 0 else -base_reward
                for p in players
            ], dtype=np.float32)

            # Train the neural network using states as input, actions as targets, rewards as weight
            self.model.fit(states, actions, sample_weight=rewards, epochs=1, verbose=0)

    def determine_next_move(self, othello_state):
        """Determine the next best move for the agent based on model.
        
        Uses a single batched model.predict() call to evaluate all legal
        moves simultaneously, which is significantly faster than calling
        predict() once per move (the original implementation).
        """
        # Given the current Othello state, use the trained model to predict the best move
        legal_actions = othello_state.get_legal_moves()
        if not legal_actions:
            return None

        if self.random_mode:
            return random.choice(legal_actions)

        # Prepare all board states after applying each legal move
        state_batch = []
        for action in legal_actions:
            game_state = deepcopy(othello_state)  # Create copy of game
            # Apply the move properly (includes tile flipping), no turtle graphics
            game_state.move = action
            game_state.make_move(draw=False)
            state_representation = self.get_state_representation(game_state)
            state_batch.append(state_representation)

        # Single batched forward pass for all legal moves
        state_batch = np.array(state_batch)
        q_values = self.model.predict(state_batch, verbose=0).flatten()

        # The model is trained per-player: No sign flip is needed.
        # Select the action with the highest Q-value (from current player's perspective)
        best_action_index = int(np.argmax(q_values))

        return legal_actions[best_action_index]

    def save_model(self, filepath):
        """Save the model."""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load a saved model."""
        self.model = models.load_model(filepath)