import numpy as np
from typing import NamedTuple
import random
import pickle
# internal imports
import snake
from setting import setting
from neural_network import NN
from snake import Snake, Point


class State(NamedTuple):
    snake_body: list[Point]
    apple_location: snake.Point
    head_direction: int
    length: int


class Data:

    def __init__(self, mode, w, h) -> None:
        super().__init__()
        self.w, self.h = w, h
        self.wall_distance_idx = 0  # 0-3, r,l,u,d
        self.body_distance_idx = mode  # 4-7, r,l,u,d
        self.head_direction_idx = mode * 2  # 8-11
        self.tail_direction_idx = self.head_direction_idx + 4  # 12-15
        self.apple_distance_x = self.tail_direction_idx + 4  # 16
        self.apple_distance_y = self.apple_distance_x + 1  # 17
        self.length_idx = self.apple_distance_y + 1  # 18
        self.X = np.zeros((self.length_idx + 1,))

    def __getitem__(self, key):
        return self.X[key]

    def __setitem__(self, key, value):
        self.X[key] = value

    def __len__(self):
        return len(self.X)

    def setX(self, state: State) -> None:
        """
        prepare vector X with the current parameters [wall,body,apple,length,...]
            for the neural network prediction
        """
        x, y = int(state.snake_body[0].x), int(state.snake_body[0].y)

        # distance to wall
        self.X[self.wall_distance_idx:self.wall_distance_idx + 4] = self.w - x, x, y, self.h - y

        # distance to body
        self.X[self.body_distance_idx:self.body_distance_idx + 4] = 0  # np.inf
        for x_ in range(x + snake.block_size, self.w, snake.block_size):  # RIGHT
            if Point(x_, y) in state.snake_body:
                self.X[self.body_distance_idx + snake.RIGHT] = x_ - x
        for x_ in range(0, x, snake.block_size):  # LEFT
            if Point(x_, y) in state.snake_body:
                self.X[self.body_distance_idx + snake.LEFT] = x - x_
        for y_ in range(0, y, snake.block_size):  # UP
            if Point(x, y_) in state.snake_body:
                self.X[self.body_distance_idx + snake.UP] = y_ - y
        for y_ in range(y + snake.block_size, self.h, snake.block_size):  # DOWN
            if Point(x, y_) in state.snake_body:
                self.X[self.body_distance_idx + snake.DOWN] = y_ - y

        # head direction
        self.X[self.head_direction_idx + state.head_direction] = 1

        # tail direction
        tail_direction = None
        if state.snake_body[-1].x < state.snake_body[-2].x:
            tail_direction = snake.RIGHT
        elif state.snake_body[-1].x > state.snake_body[-2].x:
            tail_direction = snake.LEFT
        elif state.snake_body[-1].y > state.snake_body[-2].y:
            tail_direction = snake.UP
        elif state.snake_body[-1].y < state.snake_body[-2].y:
            tail_direction = snake.DOWN
        self.X[self.tail_direction_idx + tail_direction] = 1

        # distance to apple
        self.X[self.apple_distance_x] = state.apple_location.x - state.snake_body[0].x
        self.X[self.apple_distance_y] = state.apple_location.y - state.snake_body[0].y

        # length
        self.X[self.length_idx] = state.length

    def __str__(self) -> str:
        """
        print details of X for debugging
        :return: X in string representation
        """
        data = '--- data ---\n'
        data += f'distance to wall: right={self[snake.RIGHT]}, '
        data += f'left={self[snake.LEFT]}, up={self[snake.UP]}, down={self[snake.DOWN]}\n'
        i = self.body_distance_idx
        data += f'distance to body: right={self[i + snake.RIGHT]},'
        data += f' left={self[i + snake.LEFT]}, up={self[i + snake.UP]}, down={self[i + snake.DOWN]}\n'
        direction = {0: 'right', 1: 'left', 2: 'up', 3: 'down'}
        data += f'direction={direction[int(np.argmax(self.X[self.head_direction_idx:self.head_direction_idx + 4]))]}, '
        data += f'tail direction=' \
                f'{direction[int(np.argmax(self.X[self.tail_direction_idx:self.tail_direction_idx + 4]))]}\n '
        data += f'distance to apple x={self.X[self.apple_distance_x]}, '
        data += f'distance to apple y={self.X[self.apple_distance_y]}\n'
        data += f'length={self.X[self.length_idx]}\n'
        return data


class Player(NN):

    def __init__(self, W: np.ndarray, snake_: Snake, data: Data) -> None:
        super().__init__(W)
        self.steps = 0
        self.snake: Snake = snake_
        self.data = data

    def play(self) -> bool:
        game_over, direction = False, random.randint(0, 4)  # snake.RIGHT
        self.snake.restart(direction)
        score_, steps = 0, 0

        while not game_over:
            game_over, score = self.snake.play_step(direction)
            self.data.setX(State(self.snake.snake, self.snake.food, self.snake.direction, self.snake.score + 3))
            direction = self.predict(self.data.X.reshape((1, -1)))
            self.steps += 1

            # if snake stack in endless loop
            steps += 1
            if score != score_:
                score_, steps = score, 0
            elif steps >= score ** 2 + 10000:
                print('steps large=', steps)
                return True

        return False


class Agent:

    def __init__(self, gen_sizes: list, gui: bool = True) -> None:
        super().__init__()
        # init sizes
        self.layers = setting['layers']
        self.mode = setting['vision mode']
        self.gen_sizes = gen_sizes
        # init objects
        self.snake = Snake(gui=gui)
        self.data = Data(self.mode, self.snake.w, self.snake.h)
        # init generation
        self.generation: np.ndarray = np.array(
            [Player(NN.init_W(self.layers), self.snake, self.data) for _ in range(gen_sizes[0])])
        self.fitness: np.ndarray = np.empty((gen_sizes[0],), dtype=np.float64)

        self.history = []

    def train(self, load=False, save=False, file_name='generation.npy'):
        if load:
            self.load(file_name)
            try:
                with open('history', 'rb') as f:
                    self.history = pickle.load(f)
            except Exception as e:
                print(e)

        for curr_gen_idx, size in enumerate(self.gen_sizes[:-1]):
            print(f'---------------  generation {curr_gen_idx}  -------------------')
            for j, player in enumerate(self.generation):
                stack_in_loop = player.play()
                self.fitness[j] = self.evaluation(player, stack_in_loop)
            self.crossover(curr_gen_idx)
            self.mutation(curr_gen_idx)

        if save:
            self.save(file_name)
            try:
                with open('history', 'wb') as f:
                    pickle.dump(self.history, f)

            except Exception as e:
                print(e)

    def evaluation(self, player: Player, stack_in_loop: bool) -> float:
        steps, apple = player.steps, self.snake.score
        # fitness = steps + 2 ** apple + 500 * apple ** 2.1 - 0.25 * steps ** 1.3 * apple ** 1.2
        # fitness = -steps + 2 ** apple + apple ** 2 * 500 + steps * apple * 100
        # if stack_in_loop:
        #     fitness = 0
        # print(steps, fitness, stack_in_loop)
        # fitness = apple / steps
        # fitness = steps
        # if not stack_in_loop:
        #     fitness += apple ** 4 * steps
        return apple
        # return - player.steps / max(player.snake.score ** 3, 1)
        # return player.steps  # + player.snake.score ** 3

    def selection(self, next_gen_idx: int) -> np.ndarray:
        fitness = self.fitness.copy() + np.abs(np.min(self.fitness))
        sum_fitness = float(np.sum(fitness))
        probability = fitness / sum_fitness if sum_fitness != 0 else 1
        idx = np.arange(self.gen_sizes[next_gen_idx - 1])
        return np.random.choice(idx, self.gen_sizes[next_gen_idx], p=probability)

    def crossover(self, curr_gen_idx):
        # -------
        print(f'mean={np.mean(self.fitness)}, max={np.max(self.fitness)}')
        x = self.gen_sizes[curr_gen_idx] // 3
        print(f'{np.mean(self.fitness[:x])}, {np.mean(self.fitness[x:2 * x])}, {np.mean(self.fitness[2 * x:])}')
        self.history.append([np.mean(self.fitness), np.max(self.fitness)])
        # -------

        x_len = self.gen_sizes[curr_gen_idx + 1] // 3
        chosen_dna = self.generation[self.selection(curr_gen_idx + 1)]
        next_generation = [Player(player.W, self.snake, self.data) for player in chosen_dna[:x_len]]
        self.fitness = np.empty((self.gen_sizes[curr_gen_idx + 1]), dtype=np.float64)

        for _ in range(x_len, self.gen_sizes[curr_gen_idx + 1]):
            parent1_idx, parent2_idx = random.randint(0, x_len - 1), random.randint(0, x_len - 1)
            parent1, parent2 = chosen_dna[parent1_idx], chosen_dna[parent2_idx]
            W = []
            for w1, w2 in zip(parent1.W, parent2.W):
                # generate indexes
                id_ = np.array(np.meshgrid(np.arange(w1.shape[0]), np.arange(w1.shape[1]))).T.reshape((-1, 2))
                np.random.shuffle(id_)
                idx, idy, x = id_[:, 0], id_[:, 1], id_.shape[0] // 2
                # copy to new DNA
                w_ = np.empty(w1.shape, dtype=np.float64)
                w_[idx[:x], idy[:x]] = w1[idx[:x], idy[:x]]
                w_[idx[x:], idy[x:]] = w2[idx[x:], idy[x:]]
                W.append(w_)
            next_generation.append(Player(np.array(W, dtype=np.object), self.snake, self.data))
        self.generation = np.array(next_generation)

    def mutation(self, i):
        x = self.gen_sizes[i + 1] // 3
        shapes = np.array([np.array(w.shape) for w in self.generation[0].W])

        for i in range(2 * x, 3 * x):
            k = random.randint(0, len(shapes) - 1)
            size = (shapes[k][0] * shapes[k][1]) // 2
            x = np.random.randint(0, shapes[k][0], size=size)
            y = np.random.randint(0, shapes[k][1], size=size)
            self.generation[i].W[k][x, y] += random.random()

    def save(self, file_name):
        """save the weights of the last generation to file"""
        save_object = np.array([player.W for player in self.generation], dtype=np.object)
        np.save(file_name, save_object, allow_pickle=True)

    def load(self, file_name):
        """load the last generation from a file"""
        load_object = np.load(file_name, allow_pickle=True)[:self.gen_sizes[0]]
        self.generation = np.array([Player(W, self.snake, self.data) for W in load_object])
