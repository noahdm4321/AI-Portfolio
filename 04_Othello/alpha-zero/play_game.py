import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from othello import Othello
import turtle
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D
from keras.optimizers import Adam
from keras.losses import mse, categorical_crossentropy

class OthelloGame:
    """
    OthelloGame class manages the execution and user interface of the Othello game. It uses the Othello class for game logic and the OthelloAgent class for AI decision-making.

    Attributes:
    - game (Othello): Instance of the Othello class for game logic.
    - networks (OthelloNetworks): Instance of the OthelloNetworks class for AI decision-making.

    Methods:
    - run(agent_file): Runs the Othello game with optional AI opponent.
    - play(x, y): Handles player moves and manages the game flow.
    """
    def __init__(self):
        # Initialize OthelloGame with instances of Othello and OthelloAgent.
        self.game = Othello()
        networks = OthelloNetworks()
        self.policy_network = networks.policy_network
        self.value_network = networks.value_network
        self.user_score = 0
        self.computer_score = 0

    def run(self):
        ''' Method: run
            Parameters: self
            Returns: nothing
            Does: If agent model is saved, load it. Otherwise, train new model and save it. Then draws the board and start the game, sets the user to be the first player, and then alternate back and forth between the user and the computer until the game is over.
        '''
        # Check if a saved model exists
        policy = 'policy_network_weights.keras'
        value = 'value_network_weights.keras'
        try:
            networks = OthelloNetworks()
            networks.load_weights(policy, value)
            self.policy_network = networks.policy_network
            self.value_network = networks.value_network
            print("Loaded existing model.")
        except (OSError, IOError):
            print("No existing model found. Training a new model...")
            # Train the agent
            self_play = SelfPlay(Othello(), self.policy_network, mcts_simulations=10)
            print("Self-play game completed. Data collected.")
            training_loop = TrainingLoop(self.policy_network, self.value_network, optimizer=Adam(), discount_factor=0.95)

            # Self-Play and Training Loop
            for _ in range(10): 
                states, policy_targets, value_targets = self_play.play_game()
                training_loop.train(states, policy_targets, value_targets)
                training_loop.update_networks()

            print("New model trained and saved.")
        print("Finished! Let's play.")

        # Create Othello board with turtle
        print("Draw Board")
        self.game.draw_board()
        self.game.initialize_board()

        if self.game.current_player not in (0, 1):
            print("Error: unknown player. Quit...")
            return

        self.game.current_player = 0
        print('Your turn.')
        turtle.onscreenclick(self.play)
        turtle.mainloop()

    def play(self, x, y):
        ''' Method: play
            Parameters: self, x (float), y (float)
            Returns: nothing
            Does: Plays alternately between the user's turn and the computer's turn. The user plays the first turn. For the user's turn, gets the user's move by their click on the screen, and makes the move if it is legal; otherwise, waits indefinitely for a legal move to make. For the computer's turn, get the computer's move by querying agent. If one of the two players (user/computer) does not have a legal move, switches to another player's turn. When both of them have no more legal moves or the board is full, reports the result, saves the user's score and ends the game.
        '''
        # Play the user's turn
        if self.game.has_legal_move():
            self.game.get_coord(x, y)
            if self.game.is_legal_move(self.game.move):
                turtle.onscreenclick(None)
                self.game.make_move()
            else:
                return

        # Play the Computer's turn
        self.game.current_player = 1
        while self.game.has_legal_move():
            print('Computer\'s turn.')

            # Use agent to determine best move
            computer_move = self.get_computer_move()

            # Make the move on the board
            self.game.move = computer_move
            self.game.make_move()
            self.game.current_player = 0
            if self.game.has_legal_move():
                break
            self.game.current_player = 1

        # Switch back to the user's turn
        self.game.current_player = 0

        # Check whether the game is over
        if not self.game.has_legal_move() or sum(self.game.num_tiles) == self.game.n**2:
            turtle.onscreenclick(None)
            print("-----------")
            self.game.report_result()
            
            # Update and display scores
            if self.game.num_tiles[0] > self.game.num_tiles[1]:
                self.user_score += 1
            elif self.game.num_tiles[0] < self.game.num_tiles[1]:
                self.computer_score += 1
            print()
            print(f"User Score: {self.user_score} | Computer Score: {self.computer_score}")
            
            # Offer the player a chance to replay
            self.replay()
        else:
            print('Your turn.')
            turtle.onscreenclick(self.play)

    def replay(self):
        """Offers the player a chance to replay the game."""
        choice = input("Do you want to replay the game? [yes]: ").lower()
        if choice in ["yes", "y", ""]:
            turtle.clearscreen()
            self.game = Othello()

            print("Draw New Board")
            self.game.draw_board()
            self.game.initialize_board()

            self.game.current_player = 0
            print('Your turn.')
            turtle.onscreenclick(self.play)
            turtle.mainloop()
        else:
            print("Thanks for playing Othello!")
            turtle.bye()

    def get_computer_move(self):
        # Use the policy network to determine the computer's move
        state_input = np.array(self.game.board).reshape(1, 8, 8, 1)
        action_probs = self.policy_network.predict(state_input)[0]

        # Get legal moves
        legal_moves = self.game.get_legal_moves()

        # Choose a move based on the action probabilities
        chosen_move_index = np.random.choice(len(action_probs), p=action_probs)
        computer_move = legal_moves[chosen_move_index]

        return computer_move

# Define Policy Network
class OthelloNetworks:
    """
    OthelloNetworks class defines the neural networks used for AI decision-making in the Othello game.

    Attributes:
    - input_shape: The shape of the input state for the neural networks.
    - policy_network: Neural network for predicting move probabilities.
    - value_network: Neural network for predicting the value of a given game state.

    Methods:
    - build_policy_network(num_actions): Builds the policy network architecture.
    - build_value_network(): Builds the value network architecture.
    - load_weights(policy_weights_path, value_weights_path): Loads pre-trained weights for the networks.
    """
    def __init__(self, num_actions=64):
        self.input_shape = (8, 8, 1)
        self.policy_network = self.build_policy_network(num_actions)
        self.value_network = self.build_value_network()

    def build_policy_network(self, num_actions=64):
        """
        Builds the policy network architecture.

        Parameters:
        - num_actions: Number of possible actions in the Othello game.

        Returns:
        - model: Compiled policy network model.
        """
        input_state = Input(shape=self.input_shape, name='input_state')
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_state)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output_probs = Dense(num_actions, activation='softmax', name='output_probs')(x)

        model = Model(inputs=input_state, outputs=output_probs)
        model.compile(optimizer=Adam(), loss=categorical_crossentropy)

        return model

    def build_value_network(self):
        """
        Builds the value network architecture.

        Returns:
        - model: Compiled value network model.
        """
        input_state = Input(shape=self.input_shape, name='input_state')
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_state)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output_value = Dense(1, activation='tanh', name='output_value')(x)

        model = Model(inputs=input_state, outputs=output_value)
        model.compile(optimizer=Adam(), loss=mse)

        return model

    def load_weights(self, policy_weights_path, value_weights_path):
        """
        Loads pre-trained weights for the policy and value networks.

        Parameters:
        - policy_weights_path: Path to pre-trained weights for the policy network.
        - value_weights_path: Path to pre-trained weights for the value network.
        """
        # Load pre-trained weights for the policy network
        self.policy_network.load_weights(policy_weights_path)

        # Load pre-trained weights for the value network
        self.value_network.load_weights(value_weights_path)

# Define Monte Carlo Tree Node
class MCTSNode:
    """
    MCTSNode class represents a node in the Monte Carlo Tree Search (MCTS) tree.

    Attributes:
    - state: The game state associated with the node.
    - parent: The parent node in the MCTS tree. None if the node is the root.
    - children: List of child nodes.
    - visits: Number of times the node has been visited during the simulation.
    - value: Accumulated value of the node's state during simulations.

    Parameters:
    - state: The game state associated with the node.
    - parent: The parent node in the MCTS tree. Default is None for the root node.
    """
    def __init__(self, state, parent = None, move=None):
        self.state = state  # Othello object for node state
        self.state = parent # Predicessor node
        self.move = move    # Move made to get to current node
        self.children = []  # List of nodes accessible from current node
        self.visits = 0     # Number of children in node
        self.value = 0      # Winning games - lossing games

# Define Monte Carlo Tree Search
class MCTS:
    """
    MCTS class implements the Monte Carlo Tree Search algorithm.

    Attributes:
    - root_node: The root node of the MCTS tree.
    - policy_network: The neural network used for action selection.

    Parameters:
    - root_node: The initial node representing the current game state.
    - policy_network: The neural network for guiding exploration and exploitation.
    """
    def __init__(self, root_node, policy_network):
        self.root_node = root_node
        self.policy_network = policy_network

    def select(self, node):
        """
        Select a child node using the Upper Confidence Bound 1 (UCB1) formula.

        Parameters:
        - node: The current node in the tree.

        Returns:
        - selected_node: The selected child node based on UCB1 values.
        """
        exploration_weight = 1.41
        legal_moves = node.state.get_legal_moves()
        
        if legal_moves:
            # Use the policy network to get action probabilities
            state_input = (8, 8, 1)
            action_probs = self.policy_network.predict(state_input)[0]

            # Calculate UCB1 values for each legal move
            ucb_values = []
            for move in legal_moves:
                move_index = move[0] * node.state.n + move[1]
                ucb_value = (node.children[move_index].value / node.children[move_index].visits + exploration_weight * action_probs[move_index] * np.sqrt(np.log(node.visits) / node.children[move_index].visits))
                ucb_values.append(ucb_value)

            # Select the move with the highest UCB1 value
            selected_move_index = legal_moves[np.argmax(ucb_values)]
            return node.children[selected_move_index]

        return None

    def expand(self, node):
        """
        Expand the current node by adding a child based on a legal move.

        Parameters:
        - node: The current node in the tree.

        Returns:
        - new_node: The newly created child node.
        """
        legal_moves = node.state.get_legal_moves()
        for move in legal_moves:
            new_state = node.state.make_move(move)
            new_node = MCTSNode(new_state, node, move)
            node.children.append(new_node)
        return node

    def accumulate_value(self, node):
        """
        Calculate the cumulative value based on endstates reachable from the given node.
        """
        # Check if the node represents an endstate
        if not node.state.has_legal_move():
            return node.state.get_winner()

        # If not an endstate, accumulate values from children
        accumulated_value = 0
        for child in node.children:
            accumulated_value += child.value

        return accumulated_value

    def backpropagate(self, node, value):
        """
        Backpropagate the outcome value up the tree.

        Parameters:
        - node: The node to start the backpropagation from.
        - value: The outcome value to propagate.
        """
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
        return node

    def run(self, iterations=1):
        """
        Run the Monte Carlo Tree Search for a specified number of iterations.

        Parameters:
        - iterations: The number of iterations to run the search.

        Returns:
        - best_move: The best move found based on visit counts.
        """
        for _ in range(iterations):
            print(self.root_node)
            print(self.root_node.state)
            node = self.root_node
            first_iteration = True

            # Selection and expansion
            while node != self.root_node or first_iteration:
                first_iteration = False

                if node.state.has_legal_move() and node.children == []:
                    # Expand node children
                    node = self.expand(node)
                    node = self.select(node)
                elif any(child.visits == 0 for child in node.children):
                    # Explore all unvisited children
                    node = [child for child in node.children if child.visits == 0][0]
                else:
                    # Backpropagate when all children have been visited at least once
                    cumulative_value = self.accumulate_value(node)
                    node = self.backpropagate(node, cumulative_value)

        # Select the best move based on visit counts
        try:
            best_child = max(node.children, key=lambda child: child.value)
            return best_child.move
        except ValueError:
            # Handle the case when there are no children
            return None

# Self-Play Function
class SelfPlay:
    """
    SelfPlay class manages the self-play process for generating training data for the Othello AI.

    Attributes:
    - game_class: The class representing the Othello game.
    - policy_network: The neural network used for move selection during self-play.
    - mcts_simulations: The number of Monte Carlo Tree Search (MCTS) simulations to perform.

    Parameters:
    - game_class: The class representing the Othello game.
    - policy_network: The neural network used for move selection during self-play.
    - mcts_simulations: The number of MCTS simulations to perform during move selection.
    """
    def __init__(self, game_class, policy_network, mcts_simulations=1):
        self.game_class = game_class
        self.policy_network = policy_network
        self.mcts_simulations = mcts_simulations

    def play_game(self):
        """
        Play a self-play game of Othello and generate training data.

        Returns:
        - states: List of game states during the self-play.
        - policy_targets: List of policy targets for each game state.
        - value_targets: List of value targets indicating the outcome of the game.
        """
        game = self.game_class
        game.board[3][3] = 2
        game.board[3][4] = 1
        game.board[4][3] = 1
        game.board[4][4] = 2
        states = []
        policy_targets = []
        value_targets = []

        while not game.is_game_over():
            # Use MCTS to select a move
            root_node = MCTSNode(game)
            mcts = MCTS(root_node, self.policy_network)
            selected_move = mcts.run(self.mcts_simulations)

            # Store the current state
            states.append(game)

            # Get the policy targets from the MCTS probabilities
            state_input = np.array(game.board).reshape(1, 8, 8, 1)
            mcts_probs = self.policy_network.predict(state_input, verbose=0)[0]
            policy_targets.append(np.copy(mcts_probs))

            # Make the selected move and continue
            game.make_move(selected_move)

        # Update the value targets based on the winner of the game
        winner = game.get_winner()
        if winner is not None:
            value_targets = [1.0 if i == winner else -1.0 for i in range(game.num_players)]
        else:
            value_targets = [0.0, 0.0]

        print("Self-play game completed.")
        return states, policy_targets, value_targets

# Training Loop
class TrainingLoop:
    """
    TrainingLoop class manages the training loop for updating the policy and value networks.

    Attributes:
    - policy_network: The neural network for move selection during training.
    - value_network: The neural network for predicting the value of game states during training.
    - optimizer: The optimizer used for updating the network weights.
    - discount_factor: The discount factor for calculating discounted rewards in the value network.

    Parameters:
    - policy_network: The neural network for move selection during training.
    - value_network: The neural network for predicting the value of game states during training.
    - optimizer: The optimizer used for updating the network weights.
    - discount_factor: The discount factor for calculating discounted rewards in the value network.
    """
    def __init__(self, policy_network, value_network, optimizer, discount_factor=0.95):
        self.policy_network = policy_network
        self.value_network = value_network
        self.optimizer = optimizer
        self.discount_factor = discount_factor

    def train(self, states, policy_targets, value_targets, epochs=1, batch_size=32):
        """
        Train the policy and value networks using the provided training data.

        Parameters:
        - states: List of game states during training.
        - policy_targets: List of policy targets for each game state.
        - value_targets: List of value targets indicating the outcome of each game state.
        - epochs: Number of training epochs.
        - batch_size: Batch size for training.
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]

                # Prepare batch data
                batch_states = [states[i] for i in batch_indices]
                batch_policy_targets = [policy_targets[i] for i in batch_indices]
                batch_value_targets = [value_targets[i] for i in batch_indices]

                print(f"Training Batch {start // batch_size + 1}/{len(states) // batch_size}")
                print("Batch Shapes:", np.array(batch_states).shape, np.array(batch_policy_targets).shape, np.array(batch_value_targets).shape)

                # Train policy network
                self.policy_network.train_on_batch(np.array(batch_states), np.array(batch_policy_targets))

                # Train value network
                self.value_network.train_on_batch(np.array(batch_states), np.array(batch_value_targets))

    def update_networks(self):
        """
        Update the target networks with the current weights.

        Saves the weights of the policy and value networks to files.
        """
        # Update target networks with current weights
        self.policy_network.save_weights('policy_network_weights.keras')
        self.value_network.save_weights('value_network_weights.keras')

if __name__ == "__main__":
    choice = input("Do you want to load the AI? [yes]: ")
    if choice in ["yes", "y", ""]:
        game = OthelloGame()
        game.run()
    else:
        print("OK! Let's play.")
        # Create Othello board with turtle
        game = Othello()
        game.draw_board()
        game.initialize_board()
        # Run game file without ai agent
        game.run()
