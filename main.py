from genetic import Agent
from setting import setting
import matplotlib.pyplot as plt
import numpy as np

# setting['speed'] = 35
i = 100
j = 5
k = 1000
# agent = Agent([k for _ in range(1)], gui=False)
# agent.train(load=False, save=True)
print(f'************************   iteration {i}   ************************')
agent = Agent([k for _ in range(50)], gui=False)
agent.train(load=True, save=True)

plt.style.use('seaborn')
data = np.array(agent.history)
plt.style.use('seaborn')
fig = plt.figure(figsize=(16, 12))
fig_0, fig_1 = fig.add_subplot(121), fig.add_subplot(122)
fig_0.plot(range(data.shape[0]), data[:, 0], c='b', label='mean')  # mean
fig_0.legend()
fig_0.set_title('Mean')
fig_1.plot(range(data.shape[0]), data[:, 1], c='r', label='max')  # max
fig_1.legend()
fig_1.set_title('Max')
# fig_1.xlabel(xlabel='Generation')
# fig_1.ylabel(ylabel='Fitness')
plt.savefig('fig', dpi=fig.dpi)
plt.show()
