# Mapping

## Goal:
##### The goal of the game is to trap the ai program by creating a maze that it cannot solve. The maze must have a solution though. 

## Rules:
1. Start with the number 20x20 gid. Found in the `maze.txt` file. 
2. Fill in spaces on the grid that the ai cannot move to (the walls of the maze) with hash symbols (`#`). There must be at least one path from the start (`s`) to the finish (`f`). 
3. Save the `maze.txt` file. 
4. Run maze.py
3. The ai will then attempt to solve the maze without exploring more than half of the maze.
4. If they ai cannot find a solution to the maze without looking at more than 200 squares, you win.
5. If the ai can find the solution, the ai wins. 

## AI Showcase:
##### This program involves transition models and breadth first search algorithms. 