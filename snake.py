import pygame
import random
from collections import namedtuple
# internal imports
from setting import setting

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

RIGHT, LEFT, UP, DOWN = 0, 1, 2, 3

Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

block_size = setting['block_size']
speed = setting['speed']


class Snake:

    def __init__(self, direction=RIGHT, gui=True):
        self.w = setting['width']
        self.h = setting['height']
        # init display
        self.gui = gui
        if gui:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.direction, self.head, self.snake, self.score, self.food = None, None, None, None, None
        self.restart(direction)

    def restart(self, direction=RIGHT):
        # init game state
        self.direction = direction
        # self.head = Point(self.w / 2, self.h / 2)
        # self.snake = [self.head,
        #               Point(self.head.x - BLOCK_SIZE, self.head.y),
        #               Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        if direction == RIGHT or direction == UP or direction == DOWN:
            self.head = Point(self.w / 2, self.h / 2)
            self.snake = [self.head,
                          Point(self.head.x - block_size, self.head.y),
                          Point(self.head.x - 2 * block_size, self.head.y)]
        elif direction == LEFT:
            self.head = Point(self.w / 2 - 2 * block_size, self.h / 2)
            self.snake = [self.head,
                          Point(self.head.x + block_size, self.head.y),
                          Point(self.head.x + 2 * block_size, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - block_size) // block_size) * block_size
        y = random.randint(0, (self.h - block_size) // block_size) * block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, direction):
        # 1. set direction
        if direction == RIGHT:
            if self.direction != LEFT:
                self.direction = direction
        elif direction == LEFT:
            if self.direction != RIGHT:
                self.direction = direction
        elif direction == UP:
            if self.direction != DOWN:
                self.direction = direction
        elif direction == DOWN:
            if self.direction != UP:
                self.direction = direction

        # 2. move
        self._move()  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        if self.gui:
            self._update_ui()
        self.clock.tick(speed)
        # 6. return game over and score
        return game_over, self.score

    def _is_collision(self):
        return self.head.x > self.w - block_size or self.head.x < 0 or self.head.y > self.h - block_size or \
               self.head.y < 0 or self.head in self.snake[1:]

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, block_size, block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, block_size, block_size))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self):
        x = self.head.x
        y = self.head.y
        if self.direction == RIGHT:
            x += block_size
        elif self.direction == LEFT:
            x -= block_size
        elif self.direction == DOWN:
            y += block_size
        elif self.direction == UP:
            y -= block_size

        self.head = Point(x, y)
