# 04_Othello Game Documentation

## Overview

This is an Othello (Reversi) implementation featuring:
- **Visual gameplay** using Python's `turtle` graphics
- **AI opponent** using Monte Carlo Tree Search (MCTS), with neural-network self-play training
- **Training capability** to improve the AI through self-play

---

## Architecture

### Core Modules

| File | Purpose |
|------|---------|
| `board.py` | Base `Board` class: board representation, drawing, coordinate conversion |
| `othello.py` | `Othello` class (inherits `Board`): game logic, move validation, tile flipping |
| `mcts.py` | Monte Carlo Tree Search implementation with `Node` class |
| `agent.py` | `OthelloAgent`: neural network + MCTS for AI decision-making |
| `play_game.py` | `OthelloGame`: main entry point, orchestrates human vs AI gameplay |
| `agent_visualizer.py` | Debug tool to visualize Q-values as heatmap |

---

## Game Logic (`othello.py` + `board.py`)

### Board Representation
- **8x8 grid** (configurable via `n` parameter)
- `board[row][col]` values:
  - `0` = empty
  - `1` = black tile (player 0 / human)
  - `2` = white tile (player 1 / computer)
- `num_tiles = [black_count, white_count]`
- `current_player` = `0` (human) or `1` (computer)

### Initial Setup
```
Initial 4 tiles (center):
  (3,3)=white  (3,4)=black
  (4,3)=black  (4,4)=white
```

### Move Directions (8-way)
```python
MOVE_DIRS = [(-1,-1), (-1,0), (-1,+1),
             (0,-1),           (0,+1),
             (+1,-1), (+1,0),  (+1,+1)]
```

### Key Methods

#### `is_legal_move(move)`
Returns `True` if:
1. Move is within bounds and on empty square
2. At least one direction has opponent tiles to flip (`has_tile_to_flip`)

#### `has_tile_to_flip(move, direction)`
Scans in a direction until:
- Hits board edge or empty square → `False`
- Hits own tile → `True` (if at least one opponent tile was passed)
- Hits opponent tile → continue scanning

#### `make_move(draw=True)`
1. Places current player's tile at `self.move`
2. Increments `num_tiles[current_player]`
3. Calls `flip_tiles(draw)`
4. **Does NOT switch players** (caller responsibility)

#### `flip_tiles(draw=True)`
For each direction with tiles to flip:
- Walks outward from move position
- Flips opponent tiles to current player's color
- Updates `num_tiles` counts
- Optionally draws via turtle (`draw=False` for headless MCTS)

#### `has_legal_move()` / `get_legal_moves()`
Scans entire board for legal moves

#### `play(x, y)` (visual mode)
Human clicks → `get_coord` converts to (row,col) → validates → makes move → computer's turn (random move) → repeat

---

## Monte Carlo Tree Search (`mcts.py`)

### `MonteCarloTreeSearch` Class
- `exploration_constant`: UCB1 exploration weight (default ~1.41)
- `agent`: Optional `OthelloAgent` for NN-guided rollouts

### `search(root_state, num_simulations)`
1. Create root `Node` from `root_state`
2. Loop `num_simulations` times:
   - **Selection**: Traverse tree via UCB1 until terminal or unexpanded node
   - **Expansion**: Pick random untried action, create child node
   - **Simulation**: Play out game to terminal (random or NN-guided)
   - **Backpropagation**: Update visit counts and values up the tree
3. Return best action from root (highest average value)

### `Node` Class
Represents a game state in the search tree:
- `state`: `Othello` instance (deep-copied)
- `parent`, `action`, `children`
- `visits`, `value` (accumulated reward)
- Methods: `is_terminal()`, `is_fully_expanded()`, `get_untried_actions()`, `update()`, `get_value()`

### Terminal State Detection
Game is over when:
- Board is full (`sum(num_tiles) == n²`)
- **Both** players have no legal moves (pass-pass)

### Reward Calculation
```
reward = (root_player_tiles - opponent_tiles) / n²  # from the search root player's perspective
```

Used by MCTS rollouts so selection remains correct when either player is at the root. Training rewards are separately normalized to `[-1, 1]` and weighted from each recorded player's perspective.
---

## Neural Network Agent (`agent.py`)

### `OthelloAgent` Class
- `exploration_constant`, `num_simulations` for MCTS
- `random_mode`: When `True`, `determine_next_move()` returns random legal actions (set for easy difficulty)
- `model`: Keras Sequential neural network
- `mcts`: `MonteCarloTreeSearch` instance (random rollouts for training/fallback; optional agent-guided rollouts via `agent=self`)

### Network Architecture
```
Input: (8, 8, 1)  # board state
Flatten → Dense(512, relu) → Dense(1024, relu) → Dense(1, linear)
Output: Single scalar value estimate for a resulting board state
Loss: MSE, Optimizer: Adam
```

### `get_state_representation(othello_state)`
Converts `board` (8x8) to (8, 8, 1) numpy array for model input

### `random_mode`
- When `True`, `determine_next_move()` selects uniformly random legal actions instead of using the neural network.
- Set automatically for **easy** difficulty; manually settable for testing.

### `train_agent(num_episodes=100, draw=False)`
Self-play training loop:
1. Initialize fresh `Othello` game
2. If `draw=True`: set up turtle board with `tracer(False)` for fast rendering
3. While game not over:
   - If no legal moves: pass turn; if both players pass, game ends
   - Use MCTS to select action (`mcts.search(othello, num_simulations)`)
   - Apply move with `draw=draw`
   - Store the **resulting board state** and the player who made the move
   - Switch player
4. After the episode ends, compute a normalized terminal value:
   - `base_value = (player_tiles - opponent_tiles) / n²` (tile differential for player 0, in `[-1, 1]`)
5. Build one scalar target per recorded move:
   - `target = +base_value` for player 0's resulting states
   - `target = -base_value` for player 1's resulting states
   - Each target is the final outcome from the mover's perspective
6. Train model: `model.fit(states, targets, epochs=1)`
   - The network learns a bounded value estimate for resulting board states
   - When `draw=True`, the board is redrawn between episodes for debugging.

### `determine_next_move(othello_state)`
1. Get all legal actions.
2. If `random_mode` is True, return a random legal action (used for easy difficulty).
3. If a value model has been trained in this session, snapshot the mutable game state, apply each legal move without drawing, and batch-predict the resulting board values.
4. Restore the original state and return the legal action with the highest predicted value.
5. If no value model is ready, use MCTS from the current player's perspective as a fallback.

---

## Game Flow (`play_game.py`)

### `OthelloGame.run(train=False, draw=False, difficulty="easy")`
1. **Easy**: set `agent.random_mode = True`, skip model loading, use random selection
2. **Medium**: load `agent_model_1000.keras`, optionally train and save
3. **Hard**: load `agent_model_10000.keras`, optionally train and save
4. If `train=True` and no model exists: train fresh model, save to difficulty's file
5. Draw board via turtle, initialize 4 center tiles
6. Set `current_player = 0` (human), bind `play()` to click events
7. Enter `turtle.mainloop()`

### `OthelloGame.play(x, y)` — Event Handler
**Human turn:**
- Convert click to (row,col) via `get_coord`
- If legal: `make_move()`, disable click handler
- Else: return (wait for valid click)

**Computer turn:**
- Set `current_player = 1`
- While computer has legal moves:
  - Get move from `agent.determine_next_move(game)`
  - Apply move via `make_move()`
  - Switch to human, check if human has moves → break
  - Else switch back to computer (continue loop)
- Switch back to human

**Game over check:**
- No legal moves for human OR board full → report result, update scores, offer replay

---

## Headless Mode for MCTS

Critical optimization: both `make_move(draw=False)` and `flip_tiles(draw=False)` skip all turtle graphics. This allows MCTS to run thousands of simulations rapidly without GUI overhead.

---

## Key Design Patterns

1. **Inheritance**: `Othello` extends `Board` for graphics + game logic separation
2. **State copying**: `deepcopy` used extensively in MCTS/agent to avoid mutation bugs
3. **Player switching**: Explicit `current_player = 1 - current_player` (not automatic in `make_move`)
4. **Pass handling**: When player has no moves, switch to opponent; game ends only when both pass
5. **Value-model inference**: Agent evaluates legal resulting states in one batch; MCTS is used for training and fallback decisions

---

## Files to Modify for Experiments

| Goal | Files to Change |
|------|-----------------|
| Board size | `Othello(n=...)` in `othello.py`, `play_game.py`, `agent.py` |
| MCTS simulations | `num_simulations` in `OthelloAgent.__init__` |
| Exploration constant | `exploration_constant` in `OthelloAgent.__init__` |
| Network architecture | `create_model()` in `agent.py` |
| Training episodes | `train_agent(num_episodes=...)` call in `play_game.py` |
| Draw board during training | `draw=True` in `train_agent()` via `run()` |
| AI difficulty (model file) | `difficulty` parameter in `run()`: `easy` (random), `medium` (`agent_model_1000.keras`), `hard` (`agent_model_10000.keras`) |
| Random mode for easy | `agent.random_mode = True` in `run()` |
| Reward function | `simulation()` in `mcts.py`, reward calc in `train_agent()` |
| Visualization | `agent_visualizer.py` |

---

## Running the Game

```bash
cd 04_Othello
# Play with existing model (using virtual environment)
/mnt/Data/noahm/Documents/VS Code/AI-Portfolio/.venv/bin/python play_game.py
# First select difficulty (easy/medium/hard), then optionally train a new AI

# Train new model at a specific difficulty
/mnt/Data/noahm/Documents/VS Code/AI-Portfolio/.venv/bin/python play_game.py
# Select difficulty, answer "yes" to training prompt
# Optionally answer "yes" to draw board during training for debugging
```

**Startup flow:**
1. Select difficulty (easy / medium / hard)
2. For medium/hard: choose whether to train a new AI
3. If training: agent loads existing model, trains, saves, then starts game
4. If not training: agent loads any existing model for compatibility and uses MCTS until a value model is trained
5. **Easy mode**: no model loaded; AI uses random move selection

**Note**: Random-rollout MCTS is used for training and fallback decisions; keep `num_simulations` modest (10-50) to control runtime.