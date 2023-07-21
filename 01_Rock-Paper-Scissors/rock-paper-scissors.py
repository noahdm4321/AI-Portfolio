# Import necessary libraries
import os
import random

# Function to get the user's choice for Rock, Paper, or Scissors
def get_user_choice():
    while True:
        choice = input("Enter your choice (rock, paper, or scissors): ").lower()
        if choice in ['rock', 'r', 'paper', 'p', 'scissors', 's']:
            if choice == 'r':  # Convert the single letter choice to the full word
                return 'rock'
            elif choice == 'p':
                return 'paper'
            elif choice == 's':
                return 'scissors'
            else:
                return choice
        else:
            print("Invalid choice. Please try again.")

# Function to get the computer's choice randomly from 'rock', 'paper', or 'scissors'
def get_computer_choice():
    choices = ['rock', 'paper', 'scissors']
    return random.choice(choices)

# Function to determine the winner of the game
def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return 'tie'
    elif (
        (user_choice == 'rock' and computer_choice == 'scissors') or
        (user_choice == 'paper' and computer_choice == 'rock') or
        (user_choice == 'scissors' and computer_choice == 'paper')
    ):
        return 'user'  # User wins if their choice beats the computer's choice
    else:
        return 'computer'  # Computer wins in all other cases

# Function to ask the user if they want to play again
def play_again():
    while True:
        choice = input("Do you want to play again? [yes]: ").lower()
        if choice in ['yes', 'no', 'y', 'n', '']:
            if choice in ['yes', 'y', '']:
                return True  # Return True if the user wants to play again
            else:
                return False  # Return False if the user does not want to play again
        else:
            print("Invalid choice. Please try again.")

# Main function to execute the game
def main():
    print("Let's play Rock Paper Scissors!")
    user_score = 0
    computer_score = 0

    while True:
        print(f"\nCurrent Score - You: {user_score} | Computer: {computer_score}")

        user_choice = get_user_choice()  # Get the user's choice
        computer_choice = get_computer_choice()  # Get the computer's choice

        print(f"\nYou chose: {user_choice}")
        print(f"The computer chose: {computer_choice}\n")

        winner = determine_winner(user_choice, computer_choice)  # Determine the winner
        if winner == 'tie':
            print("It's a tie!")  # Display if it's a tie
        elif winner == 'user':
            print("You win!")  # Display if the user wins
            user_score += 1  # Increase the user's score by 1
        else:
            print("Computer wins!")  # Display if the computer wins
            computer_score += 1  # Increase the computer's score by 1

        if not play_again():  # Ask if the user wants to play again and break the loop if not
            break

    print(f"\nFinal Score - You: {user_score} | Computer: {computer_score}")
    os.system("pause")  # Pause the script so the user can see the final score before closing

if __name__ == '__main__':
    main()  # Start the game by calling the main function if the script is executed directly