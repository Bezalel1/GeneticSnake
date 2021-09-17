# GeneticSnake

### solving the snake game with genetic algorithm

## Problem Definition

Train an agent to play the **Snake Game** using **Genetic Algorithm**.

### Genetic Algorithm

Genetic Algorithm (GA) is a search-based optimization technique based on the principles of Genetics and Natural
Selection. It is frequently used to find optimal or near-optimal solutions to difficult problems which otherwise would
take a lifetime to solve. It is frequently used to solve optimization problems, in research, and machine learning.

The genetic algorithm's progress:

1. Create first generation, then:
2. Run all players.
3. Evaluate for each player is score(=evaluation).
4. Select the best players(=selection).
5. Create a new generation from the best players(=crossover).
6. Create a mutations(=mutation).
7. Repeat steps 2-6.

## Analysis and Design of Problem

### Genes

The chromosomes represented by a neural network for each player, for evaluate the best direction of motion after each
step, the snake sends its current state to a feedforward neural network, and the neural network returns the best
direction according to the weighs.

### Fitness

Evaluate the score of each player:

* A step towards the apple: score +1
* A step that moves away from apple: score -1.5
* Eating the apple: score +10

### Selection

Select the best players randomly, for each player the higher the player's score, the more likely he is to be selected.

### Crossover

Generate a new generation: 1/3 from the parents generation and 2/3 new generation with mixed chromosomes from the
selected parents.

### Mutation

Mutation 1/2 of the new generation.

### Performance

![plot](saved%20data/fig.png)

* Right:
    - blue - parents&mutation mean.
    - black - parent mean.
    - red - child mead.
    - green - child&mutation mean.
    - yellow - total mean.
* Left: total maximum.

The max Score reached around 65%-75% of the game table.

## Code Structure

play.py - to start training the snake game using genetic algorithm.

genetic.py - the genetic algorithm and the players.

neural_network.py - simple version of neural network which represent the gen of each snake player.

helper.py - Data and State classes that helping represent the data of the snake for feeding to the neural network.

snake.py - contains the logic for creating snake game using pygame.

setting - main setting for the project.