import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mcts_UCB1 import MonteCarloTreeSearch, Node
from othello import Othello

class Visualizer:
    def __init__(self, root_state):
        # Create an instance of the Othello game
        self.root_node = Node(root_state)
        self.fig, self.ax = plt.subplots()
        self.node_positions = {}
        self.node_sizes = {}
        self.draw_tree(self.root_node)

    def draw_tree(self, node, x=0, y=0, level=1):
        if node is not None:
            self.node_positions[node] = (x, y)
            self.node_sizes[node] = 1 / (level * 2)

            # Draw the node
            self.ax.add_patch(Rectangle((x, y), self.node_sizes[node], self.node_sizes[node], fill=True, color='lightblue'))
            self.ax.text(x + self.node_sizes[node] / 2, y + self.node_sizes[node] / 2, f'{node.get_value():.2f}', ha='center', va='center', fontsize=8)

            # Draw edges and recurse for children
            for child in node.children:
                child_x = x + self.node_sizes[node]
                child_y = y + self.node_sizes[node] / 2
                self.ax.plot([x + self.node_sizes[node], child_x], [y + self.node_sizes[node] / 2, child_y], color='black')
                self.draw_tree(child, child_x, child_y, level + 1)

    def show(self):
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    # Create an instance of the Othello game
    othello_game = Othello()
    othello_game.board[3][3] = 2
    othello_game.board[3][4] = 1
    othello_game.board[4][3] = 1
    othello_game.board[4][4] = 2

    # Create an instance of the Monte Carlo Tree Search
    mcts = MonteCarloTreeSearch()

    # Execute MCTS
    mcts.search(othello_game, num_simulations=1000)

    # Create a Visualizer instance and show the visualization
    visualizer = Visualizer(othello_game)
    visualizer.show()