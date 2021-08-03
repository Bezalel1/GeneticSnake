from tkinter import Tk, Label, Canvas
import random
# internal modules
from setting import setting

# constant vars
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
# global vars
width, height, part_size = setting['width'], setting['height'], setting['part_size']
score, length = 0, 3
direction = RIGHT


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
        self.label = Label(self, text=f'Score: {score}', font=('ariel', 40))
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

    def next(self, x: int, y: int, del_tail=False) -> None:
        if del_tail:
            self.canvas.delete(self.snake_body[-1])
            del self.snake_body[-1]

        self.snake_body.insert(0, self.canvas.create_rectangle(x, y, x + part_size, y + part_size,
                                                               fill=self.snake_color, tag='snake'))

    def eat(self, x: int, y: int) -> None:
        self.canvas.delete(self.food_body)
        self.food_body = self.canvas.create_oval(x, y, x + part_size, y + part_size, fill=self.food_color, tag='food')
        self.label.config(text=f'Score: {score}', font=('ariel', 40))


class Game:
    def __init__(self) -> None:
        super().__init__()

        self.speed = setting['speed']
        self.coordinates, self.food_loc = [(i * part_size, 0) for i in range(length - 1, -1, -1)], ()
        self.gui = Gui()
        self.eat()
        for (x, y) in self.coordinates[::-1]:
            self.gui.next(x, y, False)

        self.lock = True
        self.gui.bind('<Up>', lambda event: self.turn(UP))
        self.gui.bind('<Down>', lambda event: self.turn(DOWN))
        self.gui.bind('<Right>', lambda event: self.turn(RIGHT))
        self.gui.bind('<Left>', lambda event: self.turn(LEFT))

    def next(self) -> None:
        x, y = self.coordinates[0]
        if direction == UP:
            y -= part_size
        elif direction == DOWN:
            y += part_size
        elif direction == RIGHT:
            x += part_size
        elif direction == LEFT:
            x -= part_size
        if self.collision(x, y):
            print('----------------game over------------------')
            self.gui.destroy()
            return

        self.coordinates.insert(0, (x, y))

        if x == self.food_loc[0] and y == self.food_loc[1]:
            global score, length
            score += 1
            length += 1
            self.eat()
            self.gui.next(x, y, del_tail=False)
        else:
            del self.coordinates[-1]
            self.gui.next(x, y, del_tail=True)

        self.lock = True

        self.gui.after(self.speed, self.next)

    def turn(self, new_direction: int) -> None:
        """change the direction"""
        if not self.lock:
            return
        global direction
        if new_direction == UP:
            if direction != DOWN:
                direction = new_direction
                self.lock = False
        if new_direction == DOWN:
            if direction != UP:
                direction = new_direction
                self.lock = False
        if new_direction == RIGHT:
            if direction != LEFT:
                direction = new_direction
                self.lock = False
        if new_direction == LEFT:
            if direction != RIGHT:
                direction = new_direction
                self.lock = False

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
        self.gui.eat(x, y)

    def start(self) -> None:
        self.next()
        self.gui.mainloop()

    def collision(self, x: int, y: int) -> bool:
        return x < 0 or x > width or y < 0 or y > height or (x, y) in self.coordinates


Game().start()
