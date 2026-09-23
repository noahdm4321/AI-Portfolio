import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from othello import Othello
from agent import OthelloAgent

def visualize_agent(agent, game_state, action=None, state_size=8):
    state_size = getattr(game_state, "n", state_size)
    legal_actions = game_state.get_legal_moves()
    move_values, value_source = agent.get_move_values(game_state)
    values_grid = np.full((state_size, state_size), np.nan, dtype=np.float32)

    for move, value in move_values.items():
        if move in legal_actions and np.isfinite(value):
            values_grid[move[0], move[1]] = value

    if action is None and value_source != "random" and move_values:
        action = max(move_values, key=move_values.get)

    plt.figure()
    image = plt.imshow(
        values_grid, cmap="viridis", interpolation="nearest"
    )
    finite_values = values_grid[np.isfinite(values_grid)]
    if finite_values.size:
        image.set_clim(finite_values.min(), finite_values.max())

    axis = plt.gca()
    axis.set_xlim(-0.5, state_size - 0.5)
    axis.set_ylim(state_size - 0.5, -0.5)
    axis.set_xticks(np.arange(-0.5, state_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, state_size, 1), minor=True)
    axis.grid(
        which="minor", color="black", linestyle="-", linewidth=0.8, zorder=0.5
    )
    axis.tick_params(which="minor", bottom=False, left=False)

    for row in range(state_size):
        for col in range(state_size):
            cell = game_state.board[row][col]
            if cell != 0:
                axis.add_patch(
                    plt.Circle(
                        (col, row), 0.4, color="gray", fill=True
                    )
                )

            if (row, col) == action:
                axis.add_patch(
                    plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        color="red",
                        fill=False,
                        linewidth=3,
                    )
                )

    if value_source == "value_model":
        title = "Model Values"
    elif value_source == "mcts":
        title = "MCTS Values"
    elif value_source == "random":
        value_label = (
            "Value Model" if getattr(agent, "value_model_ready", False)
            else "MCTS"
        )
        title = f"Random Selection ({value_label} Values)"
    else:
        title = "Move Values"

    plt.title(title)
    colorbar = plt.colorbar(image)
    colorbar.set_label("Value Scale")
    plt.xticks(range(state_size))
    plt.yticks(range(state_size))
    axis.set_aspect("equal")
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

    # Load the current medium-difficulty model.
    agent.load_model('agent_model_1000.keras')

    # Visualize values for legal moves.
    visualize_agent(agent, othello_state)
