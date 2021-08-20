from genetic import Agent
import matplotlib.pyplot as plt
import numpy as np

# agent = Agent([20 for _ in range(3)], gui=True)
# agent.demonstration()
# agent = Agent([500 for _ in range(3)], gui=True)  # 1000:20
# agent.train(load=True, save=False)

agent = Agent([2000 for _ in range(5)], gui=False)  # 1000:20
agent.train(load=True, save=True)

data = np.array(agent.history)
plt.style.use('seaborn')
fig = plt.figure(figsize=(16, 12))
fig_0, fig_1 = fig.add_subplot(121), fig.add_subplot(122)
fig_0.plot(range(data.shape[0]), data[:, 2], c='b', label='mean 1')  # mean 1
fig_0.plot(range(data.shape[0]), data[:, 3], c='k', label='mean 2')  # mean 2
fig_0.plot(range(data.shape[0]), data[:, 4], c='r', label='mean 3')  # mean 3
fig_0.plot(range(data.shape[0]), data[:, 5], c='g', label='mean 4')  # mean 3
fig_0.plot(range(data.shape[0]), data[:, 0], c='y', alpha=0.5, linewidth=8, label='mean')  # mean
fig_0.legend()
fig_0.set_title('Mean')
plt.xlabel(xlabel='Generation')
plt.ylabel(ylabel='Fitness')
fig_1.plot(range(data.shape[0]), data[:, 1], c='r', label='max')  # max
fig_1.legend()
fig_1.set_title('Max')
plt.xlabel(xlabel='Generation')
plt.ylabel(ylabel='Fitness')
plt.savefig('fig')
plt.show()
