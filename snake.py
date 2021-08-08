from abc import ABC
from tkinter import Tk, Label, Canvas
import random
from threading import Thread
# internal modules
import numpy as np
import neural_network
from setting import setting

# constant vars
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
# USER_MODE, AUTO_MODE = 0, 1
# global vars
width, height, part_size = setting['width'], setting['height'], setting['part_size']


class Gui(Tk):
    def __init__(self) -> None:
        super().__init__()

        # window setting
        self.title("Snake")
        self.resizable(False, False)

        # colors
        self.snake_color = '#36FF26'
        self.food_color = '#B31E1A'
        self.background_color = '#272314'

        # graphic elements
        self.label = Label(self, text=f'Score: {0}', font=('ariel', 40))
        self.label.pack()
        self.canvas = Canvas(self, bg=self.background_color, height=height, width=width)
        self.canvas.pack()
        self.update()
        self.snake_body = []
        self.food_body = None

        # sizes
        self.window_width = self.winfo_width()
        self.window_height = self.winfo_height()
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        x = int(self.screen_width / 2 - self.window_width / 2)
        y = int(self.screen_height / 2 - self.window_height / 2)
        self.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

    def refresh(self):
        self.label.config(text=f'Score: {0}', font=('ariel', 40))
        for x in self.snake_body:
            self.canvas.delete(x)
        self.canvas.delete(self.food_body)
        self.snake_body = []
        self.food_body = None

    def next(self, x: int, y: int, del_tail=False) -> None:
        if del_tail:
            self.canvas.delete(self.snake_body[-1])
            del self.snake_body[-1]

        self.snake_body.insert(0, self.canvas.create_rectangle(x, y, x + part_size, y + part_size,
                                                               fill=self.snake_color, tag='snake'))

    def eat(self, x: int, y: int, score: int) -> None:
        self.canvas.delete(self.food_body)
        self.food_body = self.canvas.create_oval(x, y, x + part_size, y + part_size, fill=self.food_color, tag='food')
        self.label.config(text=f'Score: {score}', font=('ariel', 40))


from genetic import Player


class Snake:

    def __init__(self, genetic_sizes: list) -> None:
        super().__init__()
        self.score, self.length, self.direction = 0, 3, RIGHT
        self.speed = setting['speed']
        self.coordinates, self.food_loc = [((i + 5) * part_size, 5 * part_size) for i in
                                           range(self.length - 1, -1, -1)], ()
        self.gui = Gui()
        self.layers = np.array(setting['layers'])
        self.mode = setting['vision mode']
        self.X = np.zeros((1, self.mode + 9))
        self.genetic_sizes, self.i, self.j = genetic_sizes, 0, 0
        self.eat()
        for (x, y) in self.coordinates[::-1]:
            self.gui.next(x, y, False)

        # user mode
        # self.lock = True
        # self.gui.bind('<Up>', lambda event: self.turn(UP))
        # self.gui.bind('<Down>', lambda event: self.turn(DOWN))
        # self.gui.bind('<Right>', lambda event: self.turn(RIGHT))
        # self.gui.bind('<Left>', lambda event: self.turn(LEFT))

        # self.current = Individual(Individual.init_W(setting['layers']))

        # genetic data
        # X = [wall[up,down,right,left], body[up,down,right,left], apple[up,down,right,left], head, tail, len],
        ## X = [wall[left,strait,right], body[left,strait,right], apple[left,strait,right], head, tail, len],
        x, y = self.coordinates[0]
        self.tail_direction = RIGHT
        self.X[0, 0:4] = y, height - y, x, width - x
        self.X[0, self.mode + self.direction] = self.X[0, self.mode + 4 + self.tail_direction] = 1
        self.X[0, -1] = self.length

        # statistic
        self.max_fit = 0
        self.steps = 0
        self.W_fit_history = []
        self.gen = [Player(Player.init_W(setting['layers'])) for i in range(self.genetic_sizes[self.j])]
        self.prev_gen = None
        self.W_shapes = np.array([w.shape for w in self.gen[0].W])
        self.tmp_ = 0

    def next(self) -> None:
        x, y = self.coordinates[0]
        self.turn(int(self.gen[self.i].predict(self.X)))
        if self.direction == LEFT:
            print('!left!')
        # print(self.direction)

        # change head in location
        if self.direction == UP:
            y -= part_size
            self.X[0, UP] -= part_size
            self.X[0, DOWN] += part_size
        elif self.direction == DOWN:
            y += part_size
            self.X[0, UP] += part_size
            self.X[0, DOWN] -= part_size
        elif self.direction == RIGHT:
            x += part_size
            self.X[0, RIGHT] -= part_size
            self.X[0, DOWN] += part_size
        elif self.direction == LEFT:
            x -= part_size
            self.X[0, RIGHT] += part_size
            self.X[0, DOWN] -= part_size

        # change tail direction
        self.X[0, self.mode + 4 + self.tail_direction] = 0
        if self.coordinates[-1][1] > self.coordinates[-2][1]:
            self.tail_direction = UP
        elif self.coordinates[-1][1] < self.coordinates[-2][1]:
            self.tail_direction = DOWN
        elif self.coordinates[-1][0] < self.coordinates[-2][0]:
            self.tail_direction = RIGHT
        elif self.coordinates[-1][0] > self.coordinates[-2][0]:
            self.tail_direction = LEFT
        self.X[0, self.mode + 4 + self.tail_direction] = 1

        self.steps += 1
        if self.collision(x, y):
            self.return_y(True)
            self.W_fit_history.append((self.evaluation(), self.gen[self.i].W))
            self.i += 1
            # print('----------------game over------------------')
            if self.i == self.genetic_sizes[self.j]:
                print(f'gen {self.j}')
                self.j += 1
                self.i = 0
                # self.restart()
                if self.j == len(self.genetic_sizes):
                    self.gui.destroy()
                    return
                else:
                    self.crossover()
            self.restart()
        else:
            self.return_y(False)
            self.coordinates.insert(0, (x, y))

            if x == self.food_loc[0] and y == self.food_loc[1]:
                self.score += 1
                self.length += 1
                self.eat()
                self.gui.next(x, y, del_tail=False)
            else:
                del self.coordinates[-1]
                self.gui.next(x, y, del_tail=True)

            # self.lock = True

        self.gui.after(self.speed, self.next)

    def return_y(self, collision: bool):
        # self.X.append([self.X, ~collision])
        r, y = random.randint(0, 1), np.zeros((1,)).reshape((-1,))
        if collision:
            if self.direction == UP or self.direction == DOWN:
                y[0] = r + 2
            elif self.direction == RIGHT or self.direction == LEFT:
                y[0] = r
        else:
            y[0] = self.direction
        self.gen[self.i].fit(self.X, y, max_iter=1000)
        # self.gen[self.i].backpropagation(self.gen[self.i].feedforward(self.X), y)

    def turn(self, new_direction: int) -> None:
        """change the direction"""
        # user mode
        # if not self.lock:
        #     return
        # print(f'currenr={self.i},direction={self.direction}, new_direction={new_direction}')
        self.X[0, self.mode + self.direction] = 0
        if new_direction == UP:
            if self.direction != DOWN:
                self.tmp_ += self.direction != new_direction
                self.direction = new_direction
                # self.lock = False
        elif new_direction == DOWN:
            if self.direction != UP:
                self.tmp_ += self.direction != new_direction
                self.direction = new_direction
                # self.lock = False
        elif new_direction == RIGHT:
            if self.direction != LEFT:
                self.tmp_ += self.direction != new_direction
                self.direction = new_direction
                # self.lock = False
        elif new_direction == LEFT:
            if self.direction != RIGHT:
                self.tmp_ += self.direction != new_direction
                self.direction = new_direction
                # self.lock = False
        self.X[0, self.mode + self.direction] = 1

    def eat(self) -> None:
        """
        find new location for the food
        """
        while True:
            x = random.randint(0, width / part_size - 1) * part_size
            y = random.randint(0, height / part_size - 1) * part_size
            if (x, y) not in self.coordinates:
                break

        self.food_loc = (x, y)
        self.gui.eat(x, y, self.score)
        self.X[0, -1] += 1

    def start(self) -> None:
        self.next()
        # self.gui.mainloop()

    def restart(self) -> None:
        # print(f'tmp={self.tmp_}')
        self.tmp_ = 0
        self.gui.refresh()
        self.direction, self.score, self.length, self.lock = RIGHT, 0, 3, True
        self.coordinates, self.food_loc = [((i) * part_size, 5 * part_size) for i in
                                           range(self.length - 1, -1, -1)], ()
        self.eat()
        for (x, y) in self.coordinates[::-1]:
            self.gui.next(x, y, False)
        if self.j == 0:
            self.current = Player(Player.init_W(setting['layers']))
        # else:
        #     self.current = Individual()
        # X = [wall[up,down,right,left], body[up,down,right,left], apple[up,down,right,left], head, tail, len],
        ## X = [wall[left,strait,right], body[left,strait,right], apple[left,strait,right], head, tail, len],
        self.X = np.zeros((1, self.mode + 9))
        x, y = self.coordinates[0]
        self.tail_direction = RIGHT
        self.X[0, 0:4] = y, height - y, x, width - x
        self.X[0, self.mode + self.direction] = self.X[0, self.mode + 4 + self.tail_direction] = 1
        self.X[0, -1] = self.length
        # print(self.coordinates)
        # print(self.X)
        # statistic
        self.steps = 0

    def collision(self, x: int, y: int) -> bool:
        return x < 0 or x > width or y < 0 or y > height or (x, y) in self.coordinates

    def evaluation(self):
        return self.steps

    def crossover(self):
        self.W_fit_history = sorted(self.W_fit_history, key=lambda x: x[0], reverse=True)
        tmp = []
        for i in range(self.genetic_sizes[self.j - 1]):
            tmp.append(self.W_fit_history[i][0])
        print(f'mean={np.mean(tmp)}, max={tmp[0]}')
        self.prev_gen = self.gen[:10]
        self.gen = []

        for i in range(10):
            self.gen.append(self.prev_gen[i])

        for i in range(10, self.genetic_sizes[self.j]):  # self.j
            parent1, parent2 = np.random.randint(10, size=2)
            parent1, parent2 = self.prev_gen[parent1], self.prev_gen[parent2]
            W = []
            for shape_, w1, w2 in zip(self.W_shapes, parent1.W, parent2.W):
                w_ = np.empty(shape_)
                w_[:shape_[0] // 2, :shape_[1] // 2] = w1[:shape_[0] // 2, :shape_[1] // 2]
                w_[shape_[0] // 2:, shape_[1] // 2:] = w2[shape_[0] // 2:, shape_[1] // 2:]
                W.append(w_)
            self.gen.append(Player(np.array(W, dtype=np.object)))
        self.mutation()

    def mutation(self):
        x = self.genetic_sizes[self.j] // 3
        for i in range(2 * x, 3 * x):
            k = random.randint(0, len(self.W_shapes) - 1)
            x = random.randint(0, self.W_shapes[k][0] - 1)
            y = random.randint(0, self.W_shapes[k][1] - 1)
            self.gen[i].W[k][x, y] += random.random()


if __name__ == '__main__':
    snake = Snake([50, 50, 50, 50, 50, 50, 50, 50])
    snake.start()
    snake.gui.mainloop()
    # np.save('last_gen', np.array(snake.gen,dtype=np.object),allow_pickle=True)
