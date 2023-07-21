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
    def __init__(self, state, action=None):
        self.state = state
        self.action = action
        self.total_value = 0
        self.visits = 0
        self.children = []


class MonteCarloTreeSearch:
    def __init__(self, game, agent, exploration_constant=1.41):
        self.game = game
        self.agent = agent
        self.exploration_constant = exploration_constant
        self.root = None

    def search(self, num_iterations):
        state = np.array(self.game.board).reshape(1, 8, 8, 1)
        self.root = Node(state)
        for _ in range(num_iterations):
            self._simulate()

    def _simulate(self):
        node = self._select_node()
        if node.state.is_game_over():
            return self._backpropagate(node, node.state.get_winner())

        if not node.children:
            self._expand_node(node)
            winner = self._simulate_random(node)
        else:
            winner = self._simulate()

        return self._backpropagate(node, winner)

    def _select_node(self):
        node = self.root
        while node.children:
            if not all(child.visits for child in node.children):
                return self._expand_node(node)
            node = self._uct_select(node)
        return node

    def _expand_node(self, node):
        legal_moves = node.state.get_legal_moves()
        for move in legal_moves:
            new_state = node.state.make_move(move)
            new_node = Node(new_state, move)
            node.children.append(new_node)
        return np.random.choice(node.children)

    def _simulate_random(self, node):
        game_copy = self.game.copy()
        game_copy.set_state(node.state)
        while not game_copy.is_game_over():
            moves = game_copy.get_legal_moves()
            move = np.random.choice(moves)
            game_copy.make_move(move)
        return game_copy.get_winner()

    def _backpropagate(self, node, winner):
        while node is not None:
            node.visits += 1
            if winner is not None:
                node.total_value += 1 if winner == self.agent else -1
            node = node.parent

    def _uct_select(self, node):
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
        if not self.root:
            raise ValueError("MCTS has not been performed yet.")
        best_child = max(self.root.children, key=lambda node: node.visits)
        return best_child.action


class OthelloAgent:
    def __init__(self):
        self.model = None

    def create_model(self):
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
        self.create_model()
        self.model.summary()

        for _ in range(mcts_iterations):
            mcts = MonteCarloTreeSearch(game, self)
            mcts.search(mcts_iterations)

            # Get the best move from the MCTS search
            best_move = mcts.get_best_move()

            # Perform a game simulation using the selected move
            game_copy = game.copy()
            game_copy.make_move(best_move)

            # Train the agent with the new game state
            self.model.fit(
                np.array(game_copy.board).reshape(1, 8, 8, 1),
                np.array([best_move]),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )

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
                self.game.move = self.agent.predict_move(self.game)
                moves = self.game.get_legal_moves()
                print(self.game.move)
                print(moves)
                self.game.make_move()
                self.game.current_player = 0
                if self.game.has_legal_move():
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
        else:
            turtle.onscreenclick(self.play)


if __name__ == "__main__":
    choice = input("Do you want to load the AI? [yes]: ")
    if choice in ["yes", "y", ""]:
        # Create the neural network model
        agent_file = os.path.join(os.path.dirname(__file__), "othello_agent.h5")
        OthelloGame().run(agent_file)
    else:
        print("OK! Let's play.")
        # Create Othello board with turtle
        game = Othello()
        game.draw_board()
        game.initialize_board()
        # Run modified game file without ai agent
        game.run()
