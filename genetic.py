import numpy as np
import random
import pickle
import time
# internal imports
import neural_network
import snake
from setting import setting
from neural_network import NN
from snake import Snake, Point
from helper import State, Data


class Player(NN):

    def __init__(self, W: np.ndarray, snake_: Snake, data: Data) -> None:
        super().__init__(W, activation=neural_network.linear)
        self.steps = 0
        self.snake: Snake = snake_
        self.data = data

    def play(self):
        game_over, direction = False, random.randint(0, 4)  # random.randint(0, 4)  # snake.RIGHT
        self.snake.restart(direction)
        score, steps, steps_all = 0, 0, 0

        # model = tf.keras.models.Sequential()
        # model.add(Dense(10, input_shape=(19,), activation='relu'))
        # model.add(Dense(8, activation='relu'))
        # model.add(Dense(4, activation='softmax'))
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # print(self.data.X.shape)
        # print('l', ())

        while not game_over:
            x0, y0 = self.snake.head.x, self.snake.head.y
            x, y = self.snake.food.x, self.snake.food.y
            game_over, _ = self.snake.play_step(direction, simple_mode=False)
            x1, y1 = self.snake.head.x, self.snake.head.y

            if abs(x1 - x) < abs(x0 - x) or abs(y1 - y) < abs(y0 - y):
                self.steps += 1
                # print('not eat:', [abs(x1 - x), abs(x0 - x)], [abs(y1 - y), abs(y0 - y)])
                # print('not eat:', [(x0, y0), (x1, y1), (x, y)])
            # elif score_ < self.snake.score:
            #     self.steps += 11
            #     score_ = self.snake.score
            # print('eat')
            else:
                # print('=not eat:', [abs(x1 - x), abs(x0 - x)], [abs(y1 - y), abs(y0 - y)])
                self.steps -= 1.5

            self.data.setX2(State(self.snake.snake, self.snake.food, self.snake.direction, self.snake.score + 3))
            # direction = int(np.argmax(model.predict(self.data.X.reshape((1, -1)))))
            direction = int(self.predict(self.data.X.reshape((1, -1))))

            # if snake stack in endless loop
            steps += 1
            steps_all += 1
            if score != self.snake.score:
                score, steps = self.snake.score, 0
            elif steps >= score ** 2 + 200:

                # print(
                #     f'steps={steps} apple={score} fitness={score * 10 + self.steps} self.steps={self.steps} steps_all='
                #     f'{steps_all} mean steps={steps_all / score if score != 0 else 1}')
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
        # self.generation = None
        self.generation: np.ndarray = np.array(
            [Player(NN.init_W(self.layers), self.snake, self.data) for _ in range(gen_sizes[0])])
        self.fitness: np.ndarray = np.empty((gen_sizes[0],), dtype=np.float64)

        self.history = []
        self.mean = 0
        self.best_player, self.best_fitness = self.generation[0], 0

    def train(self, load=False, save=False):
        if load:
            self.load()

        for curr_gen_idx, size in enumerate(self.gen_sizes[:-1]):
            print(f'---------------  generation {curr_gen_idx}  -------------------')
            for j, player in enumerate(self.generation):
                stack_in_loop = player.play()
                # print(type(self.evaluation(player,stack_in_loop)))
                self.fitness[j] = self.evaluation(player, stack_in_loop)
            self.crossover(curr_gen_idx)
            self.mutation(curr_gen_idx)
            # time.sleep(3)

        if save:
            self.save()

    def evaluation(self, player: Player, stack_in_loop: bool) -> float:
        steps, apple = player.steps, self.snake.score
        # fitness = steps + (2 ** apple) + 500 * (apple ** 2.1) - 0.25 * (steps ** 1.3) * (apple ** 1.2)
        # fitness = steps * apple + (2 ** apple) + 2000 * ((apple + 1) ** 4) + (apple + 1) ** 4 / steps
        # fitness = steps + (2 ** apple) + 500 * (apple ** 2.1) - 0.25 * (steps ** 1.3) * (apple ** 1.2)
        # fitness = apple * steps
        # if stack_in_loop:
        #     fitness = 0
        # print(f'1000=>score={apple}, steps={steps}, fitness={fitness}')
        fitness = steps + apple * 10

        if apple >= 43:
            print(f'apple=>score={apple}, steps={steps}, fitness={fitness}')

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_player = player

        # if steps >= 700:
        #     print(f'1000=>score={apple}, steps={steps}, fitness={fitness}')
        return fitness

    def selection(self, next_gen_idx: int) -> np.ndarray:
        # best_dna_idx = np.arange(self.fitness.shape[0])

        # best_dna_idx = np.argwhere(self.fitness >= 0).reshape((-1,))
        best_dna_idx = np.argwhere(self.fitness >= 80).reshape((-1,))
        # print(self.fitness[best_dna_idx])
        print(f'best len={best_dna_idx.shape}')
        # best_dna_idx = np.argsort(self.fitness)[-self.fitness.shape[0]:]
        fitness = self.fitness[best_dna_idx]
        # fitness += np.abs(np.min(fitness))
        probability = fitness / float(np.sum(fitness))
        if np.inf in probability:
            print(probability)
            print(fitness)
        return np.random.choice(best_dna_idx, self.gen_sizes[next_gen_idx], p=probability)

    def crossover(self, curr_gen_idx):
        chosen_dna = self.selection(curr_gen_idx + 1)

        # -------
        print(f'mean={np.mean(self.fitness)}, max={np.max(self.fitness)}')
        x = self.gen_sizes[curr_gen_idx] // 3
        print('chosen_dna mean:', np.mean(self.fitness[chosen_dna]))
        print(f'mean: {np.mean(self.fitness[:x // 2])}, {np.mean(self.fitness[x // 2:x])}, '
              f'{np.mean(self.fitness[x:2 * x])}, {np.mean(self.fitness[2 * x:])}')
        print(f'max: {np.max(self.fitness[:x // 2])}, {np.max(self.fitness[x // 2:x])},'
              f' {np.max(self.fitness[x:2 * x])}, {np.max(self.fitness[2 * x:])}')
        print(f'best: {np.argwhere((chosen_dna < x // 2)).shape[0]}, '
              f'{np.argwhere((chosen_dna >= x // 2) & (chosen_dna < x)).shape[0]}, '
              f'{np.argwhere((chosen_dna >= x) & (chosen_dna <= 2 * x)).shape[0]}, '
              f'{np.argwhere(chosen_dna > 2 * x).shape[0]}')
        self.history.append([np.mean(self.fitness), np.max(self.fitness), np.mean(self.fitness[:x // 2]),
                             np.mean(self.fitness[x // 2:x]), np.mean(self.fitness[x:2 * x]),
                             np.mean(self.fitness[2 * x:])])
        # -------

        x_len = self.gen_sizes[curr_gen_idx + 1] // 3
        # chosen_dna = self.selection(curr_gen_idx + 1)
        chosen_dna = self.generation[self.selection(curr_gen_idx + 1)]
        next_generation = [Player(player.W, self.snake, self.data) for player in chosen_dna[:x_len]]
        self.fitness = np.empty((self.gen_sizes[curr_gen_idx + 1]), dtype=np.float64)

        for i in range(x_len, self.gen_sizes[curr_gen_idx + 1]):
            parent1_idx, parent2_idx = random.randint(0, x_len - 1), random.randint(0, x_len - 1)
            parent1, parent2 = chosen_dna[parent1_idx], chosen_dna[parent2_idx]
            # size = self.gen_sizes[curr_gen_idx + 1]
            # parent1, parent2 = chosen_dna[i % size], chosen_dna[i % (size + 1)]
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

        def _mutation(start, end, eps=1.):
            shapes = np.array([np.array(w.shape) for w in self.generation[0].W])

            for j in range(start, end):
                k = random.randint(0, len(shapes) - 1)
                size = (shapes[k][0] * shapes[k][1]) // 5
                x = np.random.randint(0, shapes[k][0], size=size)
                y = np.random.randint(0, shapes[k][1], size=size)
                self.generation[j].W[k][x, y] += np.random.uniform(low=-1, high=1, size=(size,)) * eps

        x_idx = self.gen_sizes[i + 1] // 3

        _mutation(x_idx // 2, x_idx, 0.01)
        _mutation(2 * x_idx, 3 * x_idx, 0.01)

    def save(self):
        """save the weights of the last generation to file"""
        save_object = np.array([player.W for player in self.generation], dtype=np.object)
        file_name = '|'.join(str(x) for x in setting['layers'])
        np.save(file_name + '.npy', save_object, allow_pickle=True)

        try:
            with open('history', 'wb') as f:
                pickle.dump(self.history, f)
        except Exception as e:
            print(e)
        try:
            with open('best_snake', 'wb') as f:
                pickle.dump((self.best_player.W, self.best_fitness), f)
        except Exception as e:
            print('save:', e)

    def load(self):
        """load the last generation from a file"""
        file_name = '|'.join(str(x) for x in setting['layers'])
        load_object = np.load(file_name + '.npy', allow_pickle=True)[:self.gen_sizes[0]]
        self.generation = np.array([Player(W, self.snake, self.data) for W in load_object])

        try:
            with open('history', 'rb') as f:
                self.history = pickle.load(f)
        except Exception as e:
            print(e)

        try:
            with open('best_snake', 'rb') as f:
                W, self.best_fitness = pickle.load(f)
                self.best_player = Player(W, self.snake, self.data)
        except Exception as e:
            print('load:', e)

    def demonstration(self):
        try:
            with open('best_snake', 'rb') as f:
                self.best_player = Player(pickle.load(f)[0], self.snake, self.data)
                self.best_player.play()
        except Exception as e:
            print(e)
