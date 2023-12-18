import random
from othello import Othello

class MonteCarloTreeSearch:
    """
    MonteCarloTreeSearch class represents the core of a Monte Carlo Tree Search algorithm for decision-making in the context of the game of Othello. It explores possible actions in a game tree, evaluates their effectiveness through simulations, and selects the best action based on a balance between exploitation and exploration.

    Attributes:
    - exploration_constant (float): Controls the balance between exploration and exploitation in the tree.
    - root_node (Node): Represents the root of the Monte Carlo tree, starting from the current game state.

    Methods:
    - search(root_state, num_simulations): Performs Monte Carlo Tree Search to find the best action.
    - selection(node): Selects the best child node until a terminal or unexpanded node is reached.
    - expansion(node): Expands the tree by adding a child node for an untried action.
    - simulation(node): Simulates a game from the given node until a terminal state is reached.
    - backpropagation(node, result): Backpropagates the result of a simulation up the tree.
    - best_child(node): Selects the best child based on the cumulative reward strategy.
    """
    def __init__(self, exploration_constant=None):
        # No need for an exploration constant in the cumulative reward strategy
        pass

    def search(self, root_state: Othello, num_simulations):
        """Perform Monte Carlo Tree Search to find the best action."""
        root_node = Node(root_state)
        for _ in range(num_simulations):
            selected_node = self.selection(root_node)
            expanded_node = self.expansion(selected_node)
            simulation_result = self.simulation(expanded_node)
            self.backpropagation(expanded_node, simulation_result)

        # Return the action with the highest average value
        best_child = self.best_child(root_node)
        return best_child.action

    def selection(self, node):
        """Select the best child node until a terminal or unexpanded node is reached."""
        while not node.is_terminal() and node.is_fully_expanded():
            node = self.best_child(node)
        return node

    def expansion(self, node):
        """Expand the tree by adding a child node for an untried action."""
        legal_actions = node.get_untried_actions()
        if legal_actions:
            action = random.choice(legal_actions)
            new_state = node.get_state()

            # Play the selected action
            if new_state.is_legal_move(action):
                new_state.board[action[0]][action[1]] = new_state.current_player + 1
                new_state.num_tiles[new_state.current_player] += 1

            # Change player
            new_state.current_player = 1 if node.state.current_player == 0 else 0
            new_node = Node(new_state, parent=node, action=action)
            node.add_child(new_node)
            return new_node
        else:
            return node

    def simulation(self, node):
        """Simulate a game from the given node until a terminal state is reached."""
        state = node.get_state()
        while not state.has_legal_move() or sum(state.num_tiles) == state.n ** 2:
            legal_actions = state.get_legal_moves()
            if legal_actions:
                action = random.choice(legal_actions)

                # Play the selected action
                if state.is_legal_move(action):
                    state.board[action[0]][action[1]] = state.current_player + 1
                    state.num_tiles[state.current_player] += 1

                # Change player
                state.current_player = 1 if state.current_player == 0 else 0
            else:
                break

        # Calculate reward based on winner
        if not state.has_legal_move() or sum(state.num_tiles) == state.n ** 2:
            player_tiles = sum(row.count(1) for row in state.board)
            opponent_tiles = sum(row.count(2) for row in state.board)
            # Possibly want to add rewards for getting corner and edge tiles
            reward = player_tiles - opponent_tiles
        reward=0

        return reward

    def backpropagation(self, node, result):
        """Backpropagate the result of a simulation up the tree."""
        while node is not None:
            node.update(result)
            node = node.parent

    def best_child(self, node):
        """Select the best child based on UCB1 formula."""
        children = node.children
        if not children:
            return None
        return max(children, key=lambda child: child.get_value())


class Node:
    """
    The Node class represents a node in the Monte Carlo Tree used in the context of the Othello game. Each node encapsulates a state of the game, along with information about its parent, the action that led to its creation, and its children in the tree. The class provides methods for checking terminal and fully expanded states, retrieving untried and legal actions, accessing the current state, adding child nodes, updating node information during backpropagation, and calculating the node's value based on average reward and visit count.

    Attributes:
    - state (Othello): The Othello game state represented by this node.
    - parent (Node): The parent node of this node in the tree. Defaults to None for the root node.
    - action (tuple): The action that led to this node from its parent.
    - children (list): List of child nodes.
    - visits (int): Number of times this node has been visited.
    - value (float): Accumulated value associated with this node.

    Methods:
    - is_terminal(): Checks if the node represents a terminal state in the Othello game.
    - is_fully_expanded(): Checks if all possible actions from this node have been tried.
    - get_untried_actions(): Returns a list of actions that haven't been tried from this node.
    - get_legal_actions(): Returns a list of legal actions from this node.
    - get_state(): Returns the state represented by this node.
    - add_child(child): Adds a child node to this node.
    - update(result): Updates the visit count and value of this node during backpropagation.
    - get_value(): Calculates the value of this node based on the average reward and visit count, avoiding division by zero.
    """
    def __init__(self, state: Othello, parent=None, action=None):
        # Initialize a node in the Monte Carlo Tree.
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_terminal(self):
        """Check if the node represents a terminal state."""
        return not self.state.get_legal_moves() or sum(self.state.num_tiles) == self.state.n ** 2

    def is_fully_expanded(self):
        """Check if all possible actions from this node have been tried."""
        return self.get_untried_actions() == []

    def get_untried_actions(self):
        """Get a list of actions that haven't been tried from this node."""
        legal_actions = self.state.get_legal_moves()
        tried_actions = [child.action for child in self.children]
        return list(set(legal_actions) - set(tried_actions))

    def get_legal_actions(self):
        """Get a list of legal actions from this node."""
        return self.state.get_legal_moves()

    def get_state(self):
        """Get the state represented by this node."""
        return self.state

    def add_child(self, child):
        """Add a child node to this node."""
        self.children.append(child)

    def update(self, result):
        """Update the visit count and value of this node during backpropagation."""
        self.visits += 1
        self.value += result

    def get_value(self):
        """Calculate the value of this node based on the average reward and visit count."""
        return self.value / (self.visits + 1e-6)  # Avoid division by zero
