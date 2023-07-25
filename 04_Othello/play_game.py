import copy
import os
import turtle
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse

# Load your Othello game implementation
from othello import Othello


class Node:
    def __init__(self, state, action=None, parent=None):
        """
        Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.

        Parameters:
            state (np.ndarray): The state of the game (board) represented as a 2D numpy array.
            action: The action taken to reach this state (i.e., the move played).
            parent: The parent node in the MCTS tree.
        """
        self.state = state
        self.action = action
        self.total_value = 0
        self.visits = 0
        self.parent = parent
        self.children = []


class MonteCarloTreeSearch:
    def __init__(self, game, agent, exploration_constant=1.41):
        """
        Implements the Monte Carlo Tree Search (MCTS) algorithm to find the best move for the AI agent.

        Parameters:
            game (Othello): The Othello game instance.
            agent (OthelloAgent): The AI agent playing the game.
            exploration_constant (float): The exploration constant used in the UCT formula for node selection.
        """
        self.game = game
        self.agent = agent
        self.exploration_constant = exploration_constant
        self.root = None

    def search(self, num_iterations):
        """
        Executes the MCTS algorithm by performing simulations and backpropagation.

        Parameters:
            num_iterations (int): The number of iterations to run the MCTS algorithm.
        """
        self.root = Node(self.game.get_state(), parent=None)
        for _ in range(num_iterations):
            winner = self._simulate()
            self._backpropagate(self.root, winner)

    def _simulate(self):
        """
        Conducts a single simulation by traversing the tree, expanding nodes, and performing a random rollout.

        Returns:
            int: The winner of the simulation: 0 for the AI agent, 1 for the user, or None for a tie.
        """
        node = self._select_node()
        game_copy = copy.deepcopy(self.game)  # Create a copy of the game
        game_copy.board = node.state.tolist()  # Set the copied state
        if game_copy.is_game_over():
            return game_copy.get_winner()
        if not node.children:
            self._expand_node(node)
        winner = self._simulate_random(node)
        return winner

    def _simulate_random(self, node):
        """
        Conducts a random rollout (simulation) from a given node until the game ends.

        Parameters:
            node (Node): The node to start the rollout from.

        Returns:
            int: The winner of the simulation: 0 for the AI agent, 1 for the user, or None for a tie.
        """
        game_copy = copy.deepcopy(self.game)
        game_copy.set_state(node.state)
        while not game_copy.is_game_over():
            moves = game_copy.get_legal_moves()
            move = np.random.choice(moves)
            game_copy.make_move(move)
        return game_copy.get_winner()

    def _select_node(self):
        """
        Selects a node for expansion based on the UCT (Upper Confidence Bound for Trees) algorithm.

        Returns:
            Node: The selected node for expansion.
        """
        node = self.root
        while node.children:
            if not all(child.visits for child in node.children):
                return self._expand_node(node)
            node = self._uct_select(node)
        return node

    def _expand_node(self, node):
        """
        Expands a node by adding its child nodes representing legal moves in the game.

        Parameters:
            node (Node): The node to expand.
        """
        legal_moves = self.game.get_legal_moves_for_state(node.state)
        for move in legal_moves:
            new_state = self.game.make_move_for_state(move, copy_state=True)
            new_node = Node(new_state, move, parent=node)
            node.children.append(new_node)

    def _backpropagate(self, node, winner):
        """
        Updates the values and visit counts in the MCTS tree nodes during backpropagation.

        Parameters:
            node (Node): The node to start the backpropagation from.
            winner: The winner of the simulation: 0 for the AI agent, 1 for the user, or None for a tie.
        """
        while node is not None:
            node.visits += 1
            if winner is not None:
                node.total_value += 1 if winner == self.agent else -1
            node = node.parent

    def _uct_select(self, node):
        """
        Selects a child node based on the UCT (Upper Confidence Bound for Trees) formula.

        Parameters:
            node (Node): The parent node to select a child from.

        Returns:
            Node: The selected child node.
        """
        log_total_visits = math.log(node.visits)
        selected_node = None
        best_score = float("-inf")
        for child in node.children:
            exploit_score = child.total_value / (child.visits + 1e-6)
            explore_score = math.sqrt(log_total_visits / (child.visits + 1e-6))
            score = exploit_score + self.exploration_constant * explore_score
            if score > best_score:
                selected_node = child
                best_score = score
        return selected_node

    def get_best_move(self):
        """
        Gets the best move for the AI agent based on the MCTS simulations.

        Returns:
            tuple: The coordinates (row, col) of the best move on the game board.
        """
        if not self.root:
            raise ValueError("MCTS has not been performed yet.")
        best_child = max(self.root.children, key=lambda node: node.visits)
        return best_child.action


class OthelloAgent:
    def __init__(self):
        self.model = None

    def create_model(self):
        """
        Creates the neural network model for the AI agent using Keras.

        The model architecture includes Convolutional Neural Network (CNN) layers
        followed by fully connected layers.

        The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss.

        The model will be stored in the self.model attribute for later use.
        """
        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(8, 8, 1))
        )
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.compile(
            optimizer=Adam(learning_rate=0.001), loss=mse, metrics=["accuracy"]
        )
        self.model = model

    def train_agent(self, game, mcts_iterations=100, epochs=10, batch_size=32):
        """
        Trains the agent using the Monte Carlo Tree Search (MCTS) for data collection and neural network training.

        Parameters:
            game (Othello): The Othello game instance used for training data collection.
            mcts_iterations (int): The number of MCTS iterations for each training sample.
            epochs (int): The number of epochs for neural network training.
            batch_size (int): The batch size for neural network training.
        """
        self.create_model()

        all_states = []
        all_moves = []
        for _ in range(mcts_iterations):
            mcts = MonteCarloTreeSearch(game, self)
            mcts.search(mcts_iterations)

            # Accumulate states and corresponding moves from the MCTS search
            states, moves = self._get_states_and_moves_from_tree(mcts.root)
            all_states.extend(states)
            all_moves.extend(moves)

        # Train the agent on the accumulated states and moves
        all_states = np.array(all_states).reshape(-1, 8, 8, 1)
        all_moves = np.array(all_moves)
        self.model.fit(all_states, all_moves, epochs=epochs, batch_size=batch_size, verbose=1)

    def _get_states_and_moves_from_tree(self, node):
        """
        Collects states and corresponding moves from the tree for training the agent.

        Parameters:
            node (Node): The root node of the Monte Carlo Tree.

        Returns:
            list: List of states (2D numpy arrays) from the tree.
            list: List of corresponding moves (actions) from the tree.
        """
        states = []
        moves = []
        for child in node.children:
            if child.visits > 0:
                states.append(child.state)
                moves.append(child.action)
        return states, moves

    def save_file(self, file_name):
        self.model.save(file_name)

    def load_file(self, file_name):
        self.model = load_model(file_name)

    def predict_move(self, game):
        state = np.array(game.board).reshape(1, 8, 8, 1)
        prediction = self.model.predict(state, verbose=0)[0]
        return np.argmax(prediction[0]), np.argmax(prediction[1])


class OthelloGame:
    def __init__(self):
        self.game = Othello()
        self.agent = OthelloAgent()

    def run(self, agent_file):
        """
        Runs the Othello game with an AI agent.

        Parameters:
            agent_file (str): The file path to save or load the trained AI agent.

        If the agent_file exists, the AI agent will be loaded from the file.
        Otherwise, a new AI agent will be trained and saved to the specified file.
        """
        if os.path.exists(agent_file):
            self.agent.load_file(agent_file)
            print("Loaded model from file.")
        else:
            print("No existing model found. Training new model...")
            self.agent.create_model()
            # Train the agent using the game data
            self.agent.train_agent(self.game)
            # Save the trained model
            self.agent.save_file(agent_file)
            print("Model saved to file.")

        print("Finished! Let's play.")
        # Create Othello board with turtle
        self.game.draw_board()
        self.game.initialize_board()

        # Run modified game file with ai agent
        if self.game.current_player not in (0, 1):
            print("Error: unknown player. Quit...")
            return

        self.game.current_player = 0
        turtle.onscreenclick(self.play)
        turtle.mainloop()

    def play(self, x, y):
        """
        Plays a turn in the Othello game.

        Parameters:
            x (float): The x-coordinate of the user's click on the game board.
            y (float): The y-coordinate of the user's click on the game board.
        """
        # Play the user's turn
        if self.game.has_legal_move():
            self.game.get_coord(x, y)
            if self.game.is_legal_move(self.game.move):
                turtle.onscreenclick(None)
                self.game.make_move()
            else:
                return

        # Play the computer's turn
        while True:
            self.game.current_player = 1
            if self.game.has_legal_move():
                mcts = MonteCarloTreeSearch(self.game, self.game.current_player)
                mcts.search(num_iterations=100)  # Perform MCTS search
                self.game.move = mcts.get_best_move()
                self.game.make_move()
                self.game.current_player = 0
                if not self.game.has_legal_move() or sum(self.game.num_tiles) == self.game.n**2:
                    break
            else:
                break

        # Switch back to the user's turn
        self.game.current_player = 0

        # Check whether the game is over
        if (
            not self.game.has_legal_move()
            or sum(self.game.num_tiles) == self.game.n**2
        ):
            turtle.onscreenclick(None)
            print("-----------")
            self.game.report_result()
            print("Thanks for playing Othello!")
            os.system("pause")
            turtle.bye()
            return self.game.get_winner()
        else:
            turtle.onscreenclick(self.play)


if __name__ == "__main__":
    choice = input("Do you want to load the AI? [yes]: ")
    if choice in ["yes", "y", ""]:
        # Create the neural network model
        agent_file = os.path.join(os.path.dirname(__file__), "winning_strat.h5")
        OthelloGame().run(agent_file)
    else:
        print("OK! Let's play.")
        # Create Othello board with turtle
        game = Othello()
        game.draw_board()
        game.initialize_board()
        # Run modified game file without ai agent
        game.run()
