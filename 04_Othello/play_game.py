import os
import turtle
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.losses import mse
# Load your Othello game implementation
from othello import Othello

# Define the neural network model for the Othello agent
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(8,8,1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='relu'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=mse, metrics=['accuracy'])
    return model

# Train the Othello agent
def train_agent(model, game_data, epochs=10):
    board = np.array(game_data.board).reshape(1,8,8,1)
    moves = game_data.get_legal_moves()
    try:
        move = np.random.choice(moves)
    except ValueError:
        move = (0,0)
    model.fit(board, np.array([move]), epochs=epochs, batch_size=1, verbose=0)

# Save the trained model
def save(model, file_name):
    model.save(file_name)

# Load the trained model
def load(file_name):
    global model
    model = load_model(file_name)

def run():
    ''' Method: run
        Parameters: game_data
        Returns: nothing
        Does: Starts the game, sets the user to be the first player,
                and then alternate back and forth between the user and 
                the computer until the game is over.
    '''
    global game_data
    if game_data.current_player not in (0,1):
        print('Error: unknown player. Quit...')
        return
        
    game_data.current_player = 0
    turtle.onscreenclick(play)
    turtle.mainloop()

def play(x, y):
    ''' Method: play
        Parameters: self, x (float), y (float)
        Returns: nothing
        Does: Plays alternately between the user's turn and the computer's
                turn. The user plays the first turn. For the user's turn, 
                gets the user's move by their click on the screen, and makes 
                the move if it is legal; otherwise, waits indefinitely for a 
                legal move to make. For the computer's turn, just makes a 
                random legal move. If one of the two players (user/computer)
                does not have a legal move, switches to another player's 
                turn. When both of them have no more legal moves or the 
                board is full, reports the result, and ends the game.

                About the input: (x, y) are the coordinates of where 
                the user clicks.
    '''
    global game_data, model
    # Play the user's turn
    if game_data.has_legal_move():
        game_data.get_coord(x, y)
        if game_data.is_legal_move(game_data.move):
            turtle.onscreenclick(None)
            game_data.make_move()
        else:
            return

    # Play the computer's turn
    while True:
        game_data.current_player = 1
        if game_data.has_legal_move():
            state = np.array(game_data.board).reshape(1,8,8,1)
            prediction = model.predict(state, verbose=0)[0]
            game_data.move = (np.argmax(prediction[0]),np.argmax(prediction[1]))
            moves = game_data.get_legal_moves()
            print(game_data.move)
            print(moves)
            game_data.make_move()
            game_data.current_player = 0
            if game_data.has_legal_move():  
                break
        else:
            break
        
    # Switch back to the user's turn
    game_data.current_player = 0

    # Check whether the game is over
    if not game_data.has_legal_move() or sum(game_data.num_tiles) == game_data.n ** 2:
        turtle.onscreenclick(None)
        print('-----------')
        game_data.report_result()
        print('Thanks for playing Othello!')
        os.system("pause")
        turtle.bye()
    else:
        turtle.onscreenclick(play)    
    
if __name__ == '__main__':
    # Load your Othello game data (states and actions)
    game_data = Othello()

    choice = input("Do you want to load the AI? [yes]: ")
    if choice in ['yes', 'y', '']:
        # Create the neural network model
        agent_file = os.path.join(os.path.dirname(__file__), "othello_agent.h5")
        try:
            load(agent_file)
            print("Loaded model from file.")
        except FileNotFoundError:
            print("No existing model found. Training new model...")
            model = create_model()
            # Train the agent using the game data
            train_agent(model, game_data)
            # Save the trained model
            save(model, agent_file)
            print("Model saved to file.")

        print("Finished! Let's play.")
        # Create Othello board with turtle
        game_data.draw_board()
        game_data.initialize_board()
        # Run modified game file with ai agent
        run()
    else:
        # Create Othello board with turtle
        game_data.draw_board()
        game_data.initialize_board()
        # Run modified game file without ai agent
        game_data.run()
