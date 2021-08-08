from genetic import Agent

for i in range(1):
    print(f'************************  iteration {i}  ********************')
    agent = Agent([1000 for _ in range(3)], gui=False)
    # agent.load('generation.npy')
    agent.train()
    agent.save('generation.npy')
