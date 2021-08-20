import numpy as np
from typing import NamedTuple
import snake
from snake import Point


class State(NamedTuple):
    """
    represent the snake is state
    """
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
        # self.apple_loc_idx = self.length_idx + 1  # 02-6238238
        self.X = np.zeros((self.length_idx + 1,))

    def __getitem__(self, key):
        return self.X[key]

    def __setitem__(self, key, value):
        self.X[key] = value

    def __len__(self):
        return len(self.X)

    def setX(self, state: State):
        self.X = np.zeros((6,))
        # self.X[-1] = state.length / ((self.h * self.w) / snake.block_size ** 2)
        x, y, block = state.snake_body[0].x, state.snake_body[0].y, snake.block_size
        x_apple, y_apple = state.apple_location.x, state.apple_location.y
        if state.head_direction == snake.RIGHT:
            # 0.
            if x + block == self.w or Point(x + block, y) in state.snake_body:
                self.X[0] = 1
            # 1.
            if y == 0 or Point(x, y - block) in state.snake_body:
                self.X[1] = 1
            # 2.
            if y + block == self.h or Point(x, y + block) in state.snake_body:
                self.X[2] = 1
            # 3.
            self.X[3] = x < x_apple
            # 4.
            self.X[4] = y > y_apple
            # 5.
            self.X[5] = y < y_apple
        elif state.head_direction == snake.LEFT:
            # 0.
            if x == 0 or Point(x - block, y) in state.snake_body:
                self.X[0] = 1
            # 1.
            if y + block == self.h or Point(x, y + block) in state.snake_body:
                self.X[1] = 1
            # 2.
            if y == 0 or Point(x, y - block) in state.snake_body:
                self.X[2] = 1
            # 3.
            self.X[3] = x > x_apple
            # 4.
            self.X[4] = y < y_apple
            # 5.
            self.X[5] = y > y_apple
        elif state.head_direction == snake.UP:
            # 0.
            if y == 0 or Point(x, y - block) in state.snake_body:
                self.X[0] = 1
            # 1.
            if x == 0 or Point(x - block, y) in state.snake_body:
                self.X[1] = 1
            # 2.
            if x + block == self.w or Point(x + block, y) in state.snake_body:
                self.X[2] = 1
            # 3.
            self.X[3] = y > y_apple
            # 4.
            self.X[4] = x > x_apple
            # 5.
            self.X[5] = x < x_apple
        elif state.head_direction == snake.DOWN:
            # 0.
            if y + block == self.h or Point(x, y + block) in state.snake_body:
                self.X[0] = 1
            # 1.
            if x + block == self.h or Point(x + block, y) in state.snake_body:
                self.X[1] = 1
            # 2.
            if x == 0 or Point(x - block, y) in state.snake_body:
                self.X[2] = 1
            # 3.
            self.X[3] = y < y_apple
            # 4.
            self.X[4] = x < x_apple
            # 5.
            self.X[5] = x > x_apple

    def setX2(self, state: State):
        self.X = np.zeros((6,))
        self.X[-1] = state.length / ((self.h * self.w) / snake.block_size ** 2)
        x, y, block = state.snake_body[0].x, state.snake_body[0].y, snake.block_size
        x_apple, y_apple = state.apple_location.x, state.apple_location.y

        if state.head_direction == snake.RIGHT:
            # 0.
            self.X[0] = self.w - x
            for p in state.snake_body:
                if y == p.y and x < p.x:
                    self.X[0] = min(self.X[0], p.x - x)
            self.X[0] /= self.w
            # 1. u
            self.X[1] = y + snake.block_size
            for p in state.snake_body:
                if x == p.x and y > p.y:
                    self.X[1] = min(self.X[1], y - p.y)
            self.X[1] /= self.h
            # 2. d
            self.X[2] = self.h - y
            for p in state.snake_body:
                if x == p.x and y < p.y:
                    self.X[2] = min(self.X[2], p.y - y)
            self.X[2] /= self.h
            # 3.
            self.X[3] = x_apple - x
            self.X[3] /= self.w
            # 4.
            self.X[4] = y - y_apple
            self.X[4] /= self.h
        elif state.head_direction == snake.LEFT:
            # 0.
            self.X[0] = x + snake.block_size
            for p in state.snake_body:
                if y == p.y and x > p.x:
                    self.X[0] = min(self.X[0], x - p.x)
            self.X[0] /= self.w
            # 1. l->down
            self.X[1] = self.h - y
            for p in state.snake_body:
                if x == p.x and y < p.y:
                    self.X[1] = min(self.X[1], p.y - y)
            self.X[1] /= self.h
            # 2. r->up
            self.X[2] = y + snake.block_size
            for p in state.snake_body:
                if x == p.x and y > p.y:
                    self.X[2] = min(self.X[2], y - p.y)
            self.X[2] /= self.h
            # 3.
            self.X[3] = x - x_apple
            self.X[3] /= self.w
            # 4.
            self.X[4] = y_apple - y
            self.X[4] /= self.h
        elif state.head_direction == snake.UP:
            # 0.
            self.X[0] = y + snake.block_size
            for p in state.snake_body:
                if x == p.x and y > p.y:
                    self.X[0] = min(self.X[0], y - p.y)
            self.X[0] /= self.h
            # 1. l->left
            self.X[1] = x + snake.block_size
            for p in state.snake_body:
                if y == p.y and x > p.x:
                    self.X[1] = min(self.X[1], x - p.x)
            self.X[1] /= self.w
            # 2. r->right
            self.X[2] = self.w - x
            for p in state.snake_body:
                if y == p.y and x < p.x:
                    self.X[2] = min(self.X[2], p.x - x)
            self.X[2] /= self.w
            # 3.
            self.X[3] = y - y_apple
            self.X[3] /= self.h
            # 4.
            self.X[4] = x - x_apple
            self.X[4] /= self.w
        elif state.head_direction == snake.DOWN:
            # 0.
            self.X[0] = self.h - y
            for p in state.snake_body:
                if x == p.x and y < p.y:
                    self.X[0] = min(self.X[0], p.y - y)
            self.X[0] /= self.h
            # 1. l->right
            self.X[1] = self.w - x
            for p in state.snake_body:
                if y == p.y and x < p.x:
                    self.X[1] = min(self.X[1], p.x - x)
            self.X[1] /= self.w
            # 2. r->left
            self.X[2] = x + snake.block_size
            for p in state.snake_body:
                if y == p.y and x > p.x:
                    self.X[2] = min(self.X[2], x - p.x)
            self.X[2] /= self.w
            # 3.
            self.X[3] = y_apple - y
            self.X[3] /= self.h
            # 4.
            self.X[4] = x_apple - x
            self.X[4] /= self.w

    def setX1(self, state: State) -> None:
        """
        prepare vector X with the current parameters [wall,body,apple,length,...]
            for the neural network prediction
        """
        x, y = int(state.snake_body[0].x), int(state.snake_body[0].y)
        w, h = self.w, self.h

        # distance to wall
        self.X[self.wall_distance_idx:self.wall_distance_idx + 4] = (w - x) / w, x / w, y / h, (h - y) / h

        # distance to body
        self.X[self.body_distance_idx:self.body_distance_idx + 4] = 1  # np.inf
        if state.head_direction != snake.LEFT:
            for x_ in range(x + snake.block_size, self.w, snake.block_size):  # RIGHT
                if Point(x_, y) in state.snake_body:
                    self.X[self.body_distance_idx + snake.RIGHT] = (x_ - x) / w
                    break
        if state.head_direction != snake.RIGHT:
            for x_ in range(0, x, snake.block_size):  # LEFT
                if Point(x_, y) in state.snake_body:
                    self.X[self.body_distance_idx + snake.LEFT] = (x - x_) / w
        if state.head_direction != snake.DOWN:
            for y_ in range(0, y, snake.block_size):  # UP
                if Point(x, y_) in state.snake_body:
                    self.X[self.body_distance_idx + snake.UP] = (y_ - y) / h
        if state.head_direction != snake.UP:
            for y_ in range(y + snake.block_size, self.h, snake.block_size):  # DOWN
                if Point(x, y_) in state.snake_body:
                    self.X[self.body_distance_idx + snake.DOWN] = (y_ - y) / h
                    break

        # head direction
        self.X[self.head_direction_idx:self.tail_direction_idx] = 0
        self.X[self.head_direction_idx + state.head_direction] = 1

        # tail direction
        self.X[self.tail_direction_idx:self.apple_distance_x] = 0
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
        self.X[self.apple_distance_x] = (state.apple_location.x - state.snake_body[0].x) / w
        self.X[self.apple_distance_y] = (state.apple_location.y - state.snake_body[0].y) / h

        # length
        self.X[self.length_idx] = state.length / (w * h)

        # apple location
        # self.X[self.apple_loc_idx] = state.apple_location.x / w
        # self.X[self.apple_loc_idx + 1] = state.apple_location.y / h

    def setX3(self, state: State):
        self.X = np.zeros((10,))
        self.X[-1] = state.length / ((self.h * self.w) / snake.block_size ** 2)

        x, y, block = state.snake_body[0].x, state.snake_body[0].y, snake.block_size
        w, h = self.w, self.h
        x_apple, y_apple = state.apple_location.x, state.apple_location.y

        if state.head_direction == snake.RIGHT:
            # 0,1=>Straight
            self.X[0] = x + block == w or Point(x + block, y) in state.snake_body
            self.X[1] = x + block * 2 >= w or Point(x + block * 2, y) in state.snake_body
            # 2,3=>Left (up)
            self.X[2] = y == 0 or Point(x, y - block) in state.snake_body
            self.X[3] = y - block <= 0 or Point(x, y - block * 2) in state.snake_body
            # 4,5=>Right (down)
            self.X[4] = y + block == h or Point(x, y + block) in state.snake_body
            self.X[5] = y + block * 2 >= h or Point(x, y + block * 2) in state.snake_body
            # 6,7,8=>Apple
            self.X[6], self.X[7], self.X[8] = x < x_apple, y > y_apple, y < y_apple
            # 10,11,12,13=>Diag
            # self.X[9] = Point(x + block, y - block) in state.snake_body or x + block == w or y - block < 0
            # self.X[10] = Point(x + block * 2, y - block * 2) in state.snake_body \
            #              or x + block * 2 >= w or y - block * 2 < 0
            # self.X[11] = Point(x + block, y + block) in state.snake_body or x + block == w or y + block == h
            # self.X[12] = Point(x + block * 2, y + block * 2) in state.snake_body \
            #              or x + block * 2 >= w or y + block * 2 >= h
        elif state.head_direction == snake.LEFT:
            # 0,1=>Straight
            self.X[0] = x == 0 or Point(x - block, y) in state.snake_body
            self.X[1] = x - block <= 0 or Point(x - block * 2, y) in state.snake_body
            # 2,3=>Left (down)
            self.X[2] = y + block == h or Point(x, y + block) in state.snake_body
            self.X[3] = y + block * 2 >= h or Point(x, y + block * 2) in state.snake_body
            # 4,5=> Right (up)
            self.X[4] = y == 0 or Point(x, y - block) in state.snake_body
            self.X[5] = y - block <= 0 or Point(x, y - block * 2) in state.snake_body
            # 6,7,8=>Apple
            self.X[6], self.X[7], self.X[8] = x > x_apple, y < y_apple, y > y_apple
            # 10,11,12,13=>Diag
            # self.X[9] = Point(x - block, y + block) in state.snake_body or x == 0 or y + block == h
            # self.X[10] = Point(x - block * 2, y + block * 2) in state.snake_body or x - block <= 0 or y + 2 * block >= h
            # self.X[11] = Point(x - block, y - block) in state.snake_body or x == 0 or y == 0
            # self.X[12] = Point(x - block * 2, y - block * 2) in state.snake_body or x - block <= 0 or y - block <= 0
        elif state.head_direction == snake.UP:
            # 0,1=>Straight
            self.X[0] = y == 0 or Point(x, y - block) in state.snake_body
            self.X[1] = y - block <= 0 or Point(x, y - block * 2) in state.snake_body
            # 2,3=>Left (left)
            self.X[2] = x == 0 or Point(x - block, y) in state.snake_body
            self.X[3] = x - block <= 0 or Point(x - block * 2, y) in state.snake_body
            # 4,5=> Right (right)
            self.X[4] = x + block == w or Point(x + block, y) in state.snake_body
            self.X[5] = x + block * 2 >= w or Point(x + block * 2, y) in state.snake_body
            # 6,7,8=>Apple
            self.X[6], self.X[7], self.X[8] = y > y_apple, x > x_apple, x < x_apple
            # 10,11,12,13=>Diag
            # self.X[9] = Point(x - block, y - block) in state.snake_body or x == 0 or y == 0
            # self.X[10] = Point(x - block * 2, y - block * 2) in state.snake_body or x - block <= 0 or y - block <= 0
            # self.X[11] = Point(x + block, y - block) in state.snake_body or x + block == w or y == 0
            # self.X[12] = Point(x + block * 2, y - block * 2) in state.snake_body or x + 2 * block >= w or y - block <= 0
        elif state.head_direction == snake.DOWN:
            # 0,1=>Straight
            self.X[0] = y + block == h or Point(x, y + block) in state.snake_body
            self.X[1] = y + block * 2 >= h or Point(x, y + block * 2) in state.snake_body
            # 2,3=>Left (right)
            self.X[2] = x + block == h or Point(x + block, y) in state.snake_body
            self.X[3] = x + block * 2 >= h or Point(x + block * 2, y) in state.snake_body
            # 4,5=> Right (left)
            self.X[4] = x == 0 or Point(x - block, y) in state.snake_body
            self.X[5] = x - block <= 0 or Point(x - block * 2, y) in state.snake_body
            # 6,7,8=>Apple
            self.X[6], self.X[7], self.X[8] = y < y_apple, x < x_apple, x > x_apple
            # 10,11,12,13=>Diag
            # self.X[9] = Point(x + block, y + block) in state.snake_body or x + block == w or y + block == h
            # self.X[10] = Point(x + block * 2,
            #                    y + block * 2) in state.snake_body or x + 2 * block >= w or y + 2 * block >= h
            # self.X[11] = Point(x - block, y + block) in state.snake_body or x == 0 or y + block == h
            # self.X[12] = Point(x - block * 2, y + block * 2) in state.snake_body or x - block <= 0 or y + 2 * block >= h

    def setX4(self, state):
        x, y, block = state.snake_body[0].x, state.snake_body[0].y, snake.block_size
        w, h = self.w, self.h
        x_apple, y_apple = state.apple_location.x, state.apple_location.y

        X = np.zeros((w // block, h // block))
        for p in state.snake_body:
            X[int(p.x // w), int(p.y // h)] = 1
        X = X.reshape((-1,))
        self.X = np.hstack((X, [x_apple / w, y_apple / h, state.length / ((h * w) / block ** 2)]))

    def setX5(self, state: State):
        self.X = np.zeros((12,))
        self.X[-1] = state.length / ((self.h * self.w) / snake.block_size ** 2)

        x, y, block = state.snake_body[0].x, state.snake_body[0].y, snake.block_size
        w, h = self.w, self.h
        x_apple, y_apple = state.apple_location.x, state.apple_location.y

        if state.head_direction == snake.RIGHT:
            # 0,1=>Straight
            self.X[0] = x + block == w or Point(x + block, y) in state.snake_body
            self.X[1] = x + block * 2 >= w or Point(x + block * 2, y) in state.snake_body
            self.X[2] = x + block * 3 >= w or Point(x + block * 3, y) in state.snake_body
            # 2,3=>Left (up)
            self.X[3] = y == 0 or Point(x, y - block) in state.snake_body
            self.X[4] = y - block <= 0 or Point(x, y - block * 2) in state.snake_body
            self.X[5] = y - block * 2 <= 0 or Point(x, y - block * 3) in state.snake_body
            # 4,5=>Right (down)
            self.X[6] = y + block == h or Point(x, y + block) in state.snake_body
            self.X[7] = y + block * 2 >= h or Point(x, y + block * 2) in state.snake_body
            self.X[8] = y + block * 3 >= h or Point(x, y + block * 3) in state.snake_body
            # 6,7,8=>Apple
            self.X[9], self.X[10], self.X[11] = x < x_apple, y > y_apple, y < y_apple
            # 10,11,12,13=>Diag
            # self.X[9] = Point(x + block, y - block) in state.snake_body or x + block == w or y - block < 0
            # self.X[10] = Point(x + block * 2, y - block * 2) in state.snake_body \
            #              or x + block * 2 >= w or y - block * 2 < 0
            # self.X[11] = Point(x + block, y + block) in state.snake_body or x + block == w or y + block == h
            # self.X[12] = Point(x + block * 2, y + block * 2) in state.snake_body \
            #              or x + block * 2 >= w or y + block * 2 >= h
        elif state.head_direction == snake.LEFT:
            # 0,1=>Straight
            self.X[0] = x == 0 or Point(x - block, y) in state.snake_body
            self.X[1] = x - block <= 0 or Point(x - block * 2, y) in state.snake_body
            self.X[2] = x - block * 2 <= 0 or Point(x - block * 3, y) in state.snake_body
            # 2,3=>Left (down)
            self.X[3] = y + block == h or Point(x, y + block) in state.snake_body
            self.X[4] = y + block * 2 >= h or Point(x, y + block * 2) in state.snake_body
            self.X[5] = y + block * 3 >= h or Point(x, y + block * 3) in state.snake_body
            # 4,5=> Right (up)
            self.X[6] = y == 0 or Point(x, y - block) in state.snake_body
            self.X[7] = y - block <= 0 or Point(x, y - block * 2) in state.snake_body
            self.X[8] = y - block * 2 <= 0 or Point(x, y - block * 3) in state.snake_body
            # 6,7,8=>Apple
            self.X[9], self.X[10], self.X[11] = x > x_apple, y < y_apple, y > y_apple
            # 10,11,12,13=>Diag
            # self.X[9] = Point(x - block, y + block) in state.snake_body or x == 0 or y + block == h
            # self.X[10] = Point(x - block * 2, y + block * 2) in state.snake_body or x - block <= 0 or y + 2 * block >= h
            # self.X[11] = Point(x - block, y - block) in state.snake_body or x == 0 or y == 0
            # self.X[12] = Point(x - block * 2, y - block * 2) in state.snake_body or x - block <= 0 or y - block <= 0
        elif state.head_direction == snake.UP:
            # 0,1=>Straight
            self.X[0] = y == 0 or Point(x, y - block) in state.snake_body
            self.X[1] = y - block <= 0 or Point(x, y - block * 2) in state.snake_body
            self.X[2] = y - block + 2 <= 0 or Point(x, y - block * 3) in state.snake_body
            # 2,3=>Left (left)
            self.X[3] = x == 0 or Point(x - block, y) in state.snake_body
            self.X[4] = x - block <= 0 or Point(x - block * 2, y) in state.snake_body
            self.X[5] = x - block * 2 <= 0 or Point(x - block * 3, y) in state.snake_body
            # 4,5=> Right (right)
            self.X[6] = x + block == w or Point(x + block, y) in state.snake_body
            self.X[7] = x + block * 2 >= w or Point(x + block * 2, y) in state.snake_body
            self.X[8] = x + block * 3 >= w or Point(x + block * 3, y) in state.snake_body
            # 6,7,8=>Apple
            self.X[9], self.X[10], self.X[11] = y > y_apple, x > x_apple, x < x_apple
            # 10,11,12,13=>Diag
            # self.X[9] = Point(x - block, y - block) in state.snake_body or x == 0 or y == 0
            # self.X[10] = Point(x - block * 2, y - block * 2) in state.snake_body or x - block <= 0 or y - block <= 0
            # self.X[11] = Point(x + block, y - block) in state.snake_body or x + block == w or y == 0
            # self.X[12] = Point(x + block * 2, y - block * 2) in state.snake_body or x + 2 * block >= w or y - block <= 0
        elif state.head_direction == snake.DOWN:
            # 0,1=>Straight
            self.X[0] = y + block == h or Point(x, y + block) in state.snake_body
            self.X[1] = y + block * 2 >= h or Point(x, y + block * 2) in state.snake_body
            self.X[2] = y + block * 3 >= h or Point(x, y + block * 3) in state.snake_body
            # 2,3=>Left (right)
            self.X[3] = x + block == h or Point(x + block, y) in state.snake_body
            self.X[4] = x + block * 2 >= h or Point(x + block * 2, y) in state.snake_body
            self.X[5] = x + block * 3 >= h or Point(x + block * 3, y) in state.snake_body
            # 4,5=> Right (left)
            self.X[6] = x == 0 or Point(x - block, y) in state.snake_body
            self.X[7] = x - block <= 0 or Point(x - block * 2, y) in state.snake_body
            self.X[8] = x - block * 3 <= 0 or Point(x - block * 3, y) in state.snake_body
            # 6,7,8=>Apple
            self.X[9], self.X[10], self.X[11] = y < y_apple, x < x_apple, x > x_apple
            # 10,11,12,13=>Diag
            # self.X[9] = Point(x + block, y + block) in state.snake_body or x + block == w or y + block == h
            # self.X[10] = Point(x + block * 2,
            #                    y + block * 2) in state.snake_body or x + 2 * block >= w or y + 2 * block >= h
            # self.X[11] = Point(x - block, y + block) in state.snake_body or x == 0 or y + block == h
            # self.X[12] = Point(x - block * 2, y + block * 2) in state.snake_body or x - block <= 0 or y + 2 * block >= h

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
                f'{direction[int(np.argmax(self.X[self.tail_direction_idx:self.tail_direction_idx + 4]))]}\n'
        data += f'distance to apple x={self.X[self.apple_distance_x]}, '
        data += f'distance to apple y={self.X[self.apple_distance_y]}\n'
        data += f'length={self.X[self.length_idx]}\n'
        return data
