from genetic import Agent
from snake import Snake

# train without GUI
agent = Agent([20 for _ in range(2)], gui=False)  # 1000:20
agent.train(load=True, save=False)

# demonstration with a GUI
while True:
    agent.snake = Snake(gui=True)
    agent.demonstration()

