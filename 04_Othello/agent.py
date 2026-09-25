from othello import Othello
from mcts import MonteCarloTreeSearch
import numpy as np
import random
import keras
from keras import layers
from keras import models

class OthelloAgent:
    """
    OthelloAgent class represents an intelligent agent for playing the game of Othello. It uses a neural-network value model for move selection, with Monte Carlo Tree Search (MCTS) for training and fallback decisions.

    Attributes:
    - exploration_constant (float): Controls the exploration in the Monte Carlo Tree Search.
    - num_simulations (int): Number of simulations to perform in each iteration of Monte Carlo Tree Search.
    - model (keras.Sequential): Neural network value model trained through self-play.
    - mcts (MonteCarloTreeSearch): Instance of the MonteCarloTreeSearch class for training and fallback decisions.

    Methods:
    - create_model(): Creates the neural network model for the AI agent using Keras.
    - get_state_representation(othello_state): Converts Othello state to a format suitable for the neural network input.
    - train_agent(num_episodes=10, draw=False): Trains the agent using the Monte Carlo Tree Search (MCTS).
    - determine_next_move(othello_state): Determines the next move using the value model or MCTS fallback.
    - get_move_values(othello_state): Returns the cached values used for the current decision.
    - save_model(filepath): Saves the neural network model and the Monte Carlo Tree for future use.
    - load_model(filepath): Loads a saved neural network model.
    """
    def __init__(self, exploration_constant=1.41, num_simulations=50):
        # Initialize OthelloAgent with exploration constant and number of simulations for MCTS.
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations
        self.random_mode = False
        self.value_model_ready = False

        self.last_move_values = {}
        self.last_move_source = None
        self.last_move_state = None

        self.use_mcts_fallback = False
        self.certainty_threshold = 0.001

        # Build the neural network model and initialize Monte Carlo Tree Search
        self.model = self.create_model()
        self.mcts = MonteCarloTreeSearch(exploration_constant)

    def create_model(self):
        """Create the neural network model for the AI agent using Keras."""
        # Define a simple feedforward neural network using Keras
        model = keras.Sequential([
            layers.Flatten(input_shape=(8, 8, 2)),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

        # Compile the model with mean squared error loss and Adam optimizer
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
        return model

    def get_state_representation(self, othello_state):
        """Convert Othello state to a format suitable for input to the neural network.

        Returns an (8, 8, 2) tensor:
          - Channel 0: 1 where the current player has pieces, 0 elsewhere
          - Channel 1: 1 where the opponent has pieces, 0 elsewhere
        """
        current_player = othello_state.current_player
        my_tile = current_player + 1
        opp_tile = 1 - current_player + 1
        board = np.array(othello_state.board, dtype=np.float32)
        my_channel = (board == my_tile).astype(np.float32)
        opp_channel = (board == opp_tile).astype(np.float32)
        return np.stack([my_channel, opp_channel], axis=-1)

    @staticmethod
    def _state_key(othello_state):
        return (
            tuple(tuple(row) for row in othello_state.board),
            othello_state.current_player,
        )

    def _clear_move_cache(self):
        self.last_move_values = {}
        self.last_move_source = None
        self.last_move_state = None

    def _cache_move_values(self, othello_state, values, source):
        self.last_move_state = (
            self._state_key(othello_state),
            self.random_mode,
            self.value_model_ready,
        )
        self.last_move_values = dict(values)
        self.last_move_source = source

    def _predict_move_values(self, othello_state, legal_actions):
        board_snapshot = [row[:] for row in othello_state.board]
        tiles_snapshot = othello_state.num_tiles[:]
        player_snapshot = othello_state.current_player
        move_snapshot = getattr(othello_state, "move", None)
        state_batch = []

        def restore_state():
            for row, snapshot in zip(othello_state.board, board_snapshot):
                row[:] = snapshot
            othello_state.num_tiles[:] = tiles_snapshot
            othello_state.current_player = player_snapshot
            othello_state.move = move_snapshot

        try:
            for action in legal_actions:
                restore_state()
                othello_state.move = action
                othello_state.make_move(draw=False)
                state_batch.append(self.get_state_representation(othello_state))
        finally:
            restore_state()

        predictions = self.model.predict(
            np.asarray(state_batch, dtype=np.float32), verbose=0
        )
        values = np.asarray(predictions).reshape(-1)
        return {
            action: float(value)
            for action, value in zip(legal_actions, values)
        }

    def get_move_values(self, othello_state):
        """Return the values used for the most recent decision at this state."""
        state_key = self._state_key(othello_state)
        cache_key = (state_key, self.random_mode, self.value_model_ready)
        if self.last_move_state == cache_key and self.last_move_values:
            return dict(self.last_move_values), self.last_move_source

        legal_actions = othello_state.get_legal_moves()
        if not legal_actions:
            self._clear_move_cache()
            return {}, None

        if self.value_model_ready:
            values = self._predict_move_values(othello_state, legal_actions)
            source = "random" if self.random_mode else "value_model"
        else:
            _, values = self.mcts.search_with_values(
                othello_state, self.num_simulations
            )
            source = "random" if self.random_mode else "mcts"

        self._cache_move_values(othello_state, values, source)
        return dict(values), source

    def train_agent(self, num_episodes=100, draw=False):
        """Train the agent using the Monte Carlo Tree Search (MCTS).

        draw (bool): If True, draw the board with turtle graphics during
                     training for debugging. Defaults to False.
        """
        if draw:
            import turtle as _turtle
            _turtle.tracer(False)
            _turtle.delay(50)

        self._clear_move_cache()
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
            states, players = [], []

            while sum(othello.num_tiles) < othello.n ** 2:
                if not othello.has_legal_move():
                    # Current player can't move — pass to opponent.
                    # In Othello, the game continues with the opponent's turn.
                    othello.current_player = 1 - othello.current_player
                    if not othello.has_legal_move():
                        break  # Both players can't move — game over
                    continue

                # Use MCTS to select an action
                selected_action = self.mcts.search(othello, self.num_simulations)

                # Play the selected action (includes tile flipping via make_move)
                if selected_action is not None and othello.is_legal_move(selected_action):
                    othello.move = selected_action
                    othello.make_move(draw=draw)
                    states.append(self.get_state_representation(othello))
                    players.append(othello.current_player)

                if draw:
                    import turtle as _turtle
                    _turtle.update()

                # Switch player
                othello.current_player = 1 - othello.current_player

            # Terminal value for most tiles, computed from player 0's perspective.
            # (player 0 = black/1, player 1 = white/2)
            player_tiles = sum(row.count(1) for row in othello.board)
            opponent_tiles = sum(row.count(2) for row in othello.board)
            # Normalize to [-1, 1] so training targets stay bounded
            base_value = (player_tiles - opponent_tiles) / (othello.n ** 2)

            # Convert lists to numpy arrays
            states = np.array(states, dtype=np.float32)
            values = np.array([
                base_value if p == 0 else -base_value
                for p in players
            ], dtype=np.float32).reshape(-1, 1)

            if len(states) > 0:
                # Train the network to estimate the final outcome of each resulting state.
                self.model.fit(states, values, epochs=1, verbose=0)
                self.value_model_ready = True

    def determine_next_move(self, othello_state):
        """Determine the next move using the value model, with MCTS as fallback."""
        legal_actions = othello_state.get_legal_moves()
        if not legal_actions:
            self._clear_move_cache()
            return None

        if self.random_mode:
            move = random.choice(legal_actions)
            self.get_move_values(othello_state)
            return move

        if not self.value_model_ready:
            move, values = self.mcts.search_with_values(
                othello_state, self.num_simulations
            )
            self._cache_move_values(othello_state, values, "mcts")
            return move

        values = self._predict_move_values(othello_state, legal_actions)
        ranked = sorted(values.values(), reverse=True)
        gap = (ranked[0] - ranked[1]) if len(ranked) >= 2 else 0.0

        # Defer to MCTS when certainty is below threshold.
        if self.use_mcts_fallback and gap < self.certainty_threshold:
            move, mcts_values = self.mcts.search_with_values(
                othello_state, self.num_simulations
            )
            self._cache_move_values(othello_state, mcts_values, "mcts")
            return move

        move = max(legal_actions, key=values.get)
        self._cache_move_values(othello_state, values, "value_model")
        return move

    def save_model(self, filepath):
        """Save the model."""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load a saved model."""
        self.model = models.load_model(filepath)
        self.value_model_ready = True
        self._clear_move_cache()
