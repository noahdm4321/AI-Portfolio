import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from othello import Othello
from agent import OthelloAgent
import turtle
from agent_visualizer import visualize_agent as visualize

class OthelloGame:
    """
    OthelloGame class manages the execution and user interface of the Othello game. It uses the Othello class for game logic and the OthelloAgent class for AI decision-making.

    Attributes:
    - game (Othello): Instance of the Othello class for game logic.
    - agent (OthelloAgent): Instance of the OthelloAgent class for AI decision-making.

    Methods:
    - run(agent_file): Runs the Othello game with optional AI opponent.
    - play(x, y): Handles player moves and manages the game flow.
    """
    def __init__(self):
        # Initialize OthelloGame with instances of Othello and OthelloAgent.
        self.game = Othello()
        self.agent = OthelloAgent()
        self.user_score = 0
        self.computer_score = 0

    def run(self):
        ''' Method: run
            Parameters: self
            Returns: nothing
            Does: If agent model is saved, load it. Otherwise, train new model and save it. Then draws the board and start the game, sets the user to be the first player, and then alternate back and forth between the user and the computer until the game is over.
        '''
        # Check if a saved model exists
        filepath = os.path.dirname(os.path.realpath(__file__)) + '\\agent_model.keras'
        try:
            self.agent.load_model(filepath)
            print("Loaded existing model.")
        except (ValueError):
            print("No existing model found. Training a new model...")
            # Train the agent
            self.agent.train_agent()
            # Save the trained model
            self.agent.save_model(filepath)
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
            computer_move = self.agent.determine_next_move(self.game)
            visualize(self.agent, self.game, computer_move)  # Uncomment to visuallize q-values for each decision.

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
