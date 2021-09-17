from genetic import Agent

# train without GUI
agent = Agent([2000 for _ in range(50)], gui=False)  # 1000:20
agent.train(load=False, save=True)

# demonstration with a GUI
agent = Agent([20 for _ in range(3)], gui=True)
agent.train(load=True, save=False)
