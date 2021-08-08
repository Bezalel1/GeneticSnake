import numpy as np
# from abc import ABC, abstractmethod
# from threading import Thread, Lock
from typing import NamedTuple
# import time
import random
# internal imports
import snake_
from setting import setting
from neural_network import NN
from snake_ import Snake, Point


class State(NamedTuple):
    snake_body: list[Point]
    apple_location: snake_.Point
    head_direction: int
    length: int
    game_over: bool


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
        """
        x, y = int(state.snake_body[0].x), int(state.snake_body[0].y)

        # distance to wall
        self.X[self.wall_distance_idx:self.wall_distance_idx + 4] = self.w - x, x, y, self.h - y

        # distance to body
        self.X[self.body_distance_idx:self.body_distance_idx + 4] = 0  # np.inf
        for x_ in range(x + snake_.block_size, self.w, snake_.block_size):  # RIGHT
            if Point(x_, y) in state.snake_body:
                self.X[self.body_distance_idx + snake_.RIGHT] = x_ - x
        for x_ in range(0, x, snake_.block_size):  # LEFT
            if Point(x_, y) in state.snake_body:
                self.X[self.body_distance_idx + snake_.LEFT] = x - x_
        for y_ in range(0, y, snake_.block_size):  # UP
            if Point(x, y_) in state.snake_body:
                self.X[self.body_distance_idx + snake_.UP] = y_ - y
        for y_ in range(y + snake_.block_size, self.h, snake_.block_size):  # DOWN
            if Point(x, y_) in state.snake_body:
                self.X[self.body_distance_idx + snake_.DOWN] = y_ - y

        # head direction
        self.X[self.head_direction_idx + state.head_direction] = 1

        # tail direction
        tail_direction = None
        if state.snake_body[-1].x < state.snake_body[-2].x:
            tail_direction = snake_.RIGHT
        elif state.snake_body[-1].x > state.snake_body[-2].x:
            tail_direction = snake_.LEFT
        elif state.snake_body[-1].y > state.snake_body[-2].y:
            tail_direction = snake_.UP
        elif state.snake_body[-1].y < state.snake_body[-2].y:
            tail_direction = snake_.DOWN
        self.X[self.tail_direction_idx + tail_direction] = 1

        # distance to apple
        self.X[self.apple_distance_x] = state.apple_location.x - state.snake_body[0].x
        self.X[self.apple_distance_y] = state.apple_location.y - state.snake_body[0].y

        # length
        self.X[self.length_idx] = state.length

    def __str__(self) -> str:
        data = '--- data ---\n'
        data += f'distance to wall: right={self[snake_.RIGHT]}, '
        data += f'left={self[snake_.LEFT]}, up={self[snake_.UP]}, down={self[snake_.DOWN]}\n'
        i = self.body_distance_idx
        data += f'distance to body: right={self[i + snake_.RIGHT]},'
        data += f' left={self[i + snake_.LEFT]}, up={self[i + snake_.UP]}, down={self[i + snake_.DOWN]}\n'
        direction = {0: 'right', 1: 'left', 2: 'up', 3: 'down'}
        data += f'direction={direction[int(np.argmax(self.X[self.head_direction_idx:self.head_direction_idx + 4]))]}, '
        data += f'tail direction={direction[int(np.argmax(self.X[self.tail_direction_idx:self.tail_direction_idx + 4]))]}\n '
        data += f'distance to apple x={self.X[self.apple_distance_x]}, '
        data += f'distance to apple y={self.X[self.apple_distance_y]}\n'
        data += f'length={self.X[self.length_idx]}\n'
        return data


class Player(NN):

    def __init__(self, W: np.ndarray, snake: Snake, data: Data, mode: int) -> None:
        super().__init__(W)
        self.mode = mode
        self.score, self.steps = 0, 0
        self.snake: Snake = snake
        self.data = data

    def play(self):
        game_over, direction = False, random.randint(0, 4)  # snake_.RIGHT
        self.snake.restart(direction)

        while not game_over:
            game_over, _ = self.snake.play_step(direction)
            self.data.setX(State(self.snake.snake, self.snake.food, self.snake.direction, self.snake.score + 3, False))
            direction = self.predict(self.data.X.reshape((1, -1)))
            self.steps += 1


class Agent:

    def __init__(self, gen_sizes: list, gui=True) -> None:
        super().__init__()
        # init sizes
        self.layers = setting['layers']
        self.mode = setting['vision mode']
        self.gen_sizes = gen_sizes
        # init objects
        self.snake = Snake(gui=gui)
        self.data = Data(self.mode, self.snake.w, self.snake.h)
        # init generation
        self.generation = [Player(NN.init_W(self.layers), self.snake, self.data, self.mode) for _ in
                           range(gen_sizes[0])]
        self.fitness = {}

    def train(self):
        for i, size in enumerate(self.gen_sizes[:-1]):
            # print data
            print(f'---------------  generation {i}  -------------------')
            for player in self.generation:
                player.play()
                self.fitness[player] = self.evaluation(player)
            self.crossover(i)
            self.mutation(i)

        self.save('640 x 480')

    def evaluation(self, player: Player):
        steps, apple = player.steps, self.snake.score
        return steps + 2 ** apple + apple ** 2.1 * 500 - apple ** 2.1 * (0.25 * steps) ** 1.3
        # return - player.steps / max(player.snake.score ** 3, 1)
        # return player.steps  # + player.snake.score ** 3

    def crossover(self, i):
        x_len = min(self.gen_sizes[i] // 3, self.gen_sizes[i + 1] // 3)
        prev_generation = sorted(self.generation, key=lambda player: self.fitness[player], reverse=True)[:x_len]
        self.generation = [Player(player.W, self.snake, self.data, self.mode) for player in prev_generation]
        # -------
        tmp = []
        # tmp1 = []
        for p in prev_generation:
            tmp.append(self.fitness[p])
        print(f'mean={np.mean(tmp)}, max={tmp[0]}')
        # -------
        self.fitness = {}

        for _ in range(x_len, self.gen_sizes[i + 1]):
            parent1_idx, parent2_idx = random.randint(0, x_len - 1), random.randint(0, x_len - 1)
            parent1, parent2 = self.generation[parent1_idx], self.generation[parent2_idx]
            W = []
            for w1, w2 in zip(parent1.W, parent2.W):
                # generate indexes
                id = np.array(np.meshgrid(np.arange(w1.shape[0]), np.arange(w1.shape[1]))).T.reshape((-1, 2))
                # np.random.shuffle(id)
                idx, idy, x = id[:, 0], id[:, 1], id.shape[0] // 2
                # copy to new DNA
                w_ = np.empty(w1.shape)
                w_[idx[:x], idy[:x]] = w1[idx[:x], idy[:x]]
                w_[idx[x:], idy[x:]] = w2[idx[x:], idy[x:]]
                W.append(w_)
            self.generation.append(Player(np.array(W, dtype=np.object), self.snake, self.data, self.mode))

    def mutation(self, i):
        x = self.gen_sizes[i + 1] // 3
        shapes = [w.shape for w in self.generation[0].W]
        for i in range(2 * x, 3 * x):
            k = random.randint(0, len(shapes) - 1)
            x = random.randint(0, shapes[k][0] - 1)
            y = random.randint(0, shapes[k][1] - 1)
            self.generation[i].W[k][x, y] += random.random()

    def save(self, file_name):
        np.save(file_name, np.array([player.W for player in self.generation], dtype=np.object), allow_pickle=True)

    def load(self, file_name):
        weights = np.load(file_name, allow_pickle=True)
        self.generation = [Player(W, self.snake, self.data, self.mode) for W in weights]


if __name__ == '__main__':
    pass
    # layers = setting['layers']
    # agent = Agent([20, 20, 20], layers)
    # agent.train()

    # p = Player(NN.init_W(setting['layers']), 4)
    # print(p.data[2])

    # d = Data(4)
    # d[0] = 3
    # print(d[0])
    # for x in d:
    #     print(x)
