from copy import deepcopy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import matplotlib.pyplot as plt
from othello import Othello
from agent import OthelloAgent

def visualize_agent(agent, game_state, action=None, state_size=8):
    # Create an empty grid for the heatmap
    q_values_grid = np.zeros((state_size, state_size))
    legal_actions = game_state.get_legal_moves()

    # Iterate through all possible positions on the board
    for i in range(state_size):
        for j in range(state_size):
            othello_state = deepcopy(game_state) # Create copy of game
            othello_state.board[i][j] = 1  # Assume the current player is 1 for visualization
            state_representation = agent.get_state_representation(othello_state)

            # Predict Q-values using the agent's model
            q_values = agent.model.predict(np.array([state_representation]), verbose=0)[0]

            # Store the Q-value for the selected action (assuming there's only one action)
            q_values_grid[i][j] = q_values[0]
            if (i,j) in legal_actions:
                if (i,j) == action:
                    plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='red', fill=False))
                else:
                    plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black', fill=False))
            elif game_state.board[i][j] > 0:
                plt.gca().add_patch(plt.Circle((j, i), 0.4, color='grey', fill=True))

    # Plot the heatmap
    plt.imshow(q_values_grid, cmap='viridis', interpolation='nearest')
    plt.title('Q-Values Heatmap')
    plt.colorbar(label='Q-Values')
    plt.show()


if __name__ == "__main__":
    # Instantiate your OthelloAgent
    agent = OthelloAgent()

    # Create a copy of the current board state
    othello_state = Othello()
    othello_state.board[3][3] = 2
    othello_state.board[3][4] = 1
    othello_state.board[4][3] = 1
    othello_state.board[4][4] = 2

    # Load a trained model
    agent.load_model('agent_model.keras')

    # Visualize Q-values
    visualize_agent(agent, othello_state)