import os
import random

def get_user_choice():
    while True:
        choice = input("Enter your choice (rock, paper, or scissors): ").lower()
        if choice in ['rock', 'r', 'paper', 'p', 'scissors', 's']:
            if choice == 'r':
                return 'rock'
            elif choice == 'p':
                return 'paper'
            elif choice == 's':
                return 'scissors'
            else:
                return choice
        else:
            print("Invalid choice. Please try again.")

def get_computer_choice():
    choices = ['rock', 'paper', 'scissors']
    return random.choice(choices)

def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return 'tie'
    elif (
        (user_choice == 'rock' and computer_choice == 'scissors') or
        (user_choice == 'paper' and computer_choice == 'rock') or
        (user_choice == 'scissors' and computer_choice == 'paper')
    ):
        return 'user'
    else:
        return 'computer'

def play_again():
    while True:
        choice = input("Do you want to play again? [yes]: ").lower()
        if choice in ['yes', 'no', 'y', 'n', '']:
            if choice in ['yes', 'y', '']:
                return True
            else:
                return False
        else:
            print("Invalid choice. Please try again.")

def main():
    print("Let's play Rock Paper Scissors!")
    user_score = 0
    computer_score = 0

    while True:
        print(f"\nCurrent Score - You: {user_score} | Computer: {computer_score}")

        user_choice = get_user_choice()
        computer_choice = get_computer_choice()

        print(f"\nYou chose: {user_choice}")
        print(f"The computer chose: {computer_choice}\n")

        winner = determine_winner(user_choice, computer_choice)
        if winner == 'tie':
            print("It's a tie!")
        elif winner == 'user':
            print("You win!")
            user_score += 1
        else:
            print("Computer wins!")
            computer_score += 1

        if not play_again():
            break

    print(f"\nFinal Score - You: {user_score} | Computer: {computer_score}")
    os.system("pause")

if __name__ == '__main__':
    main()