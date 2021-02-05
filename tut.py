from matplotlib import pyplot as plt
import numpy as np
import torch

# means = np.array([3, 5, 1])
means = torch.tensor([[3], [5], [1]])

stds = np.array([1.3, 2.6, 0.78])
stds = torch.tensor([[1.3], [2.6], [0.78]])
# if stds.ndim != 1:
#     print(stds)
# stds = stds.numpy()
# torch.reshape(means-stds, (1,-1)).numpy()
print(means)
print(stds)
print(torch.reshape(means-stds, (1,-1)))
print(torch.reshape(means-stds, (1,-1)).numpy()[0])

l = torch.reshape(means-stds, (1,-1)).numpy()[0]
u = torch.reshape(means+stds, (1,-1)).numpy()[0]

fig = plt.figure(figsize=(12, 9))
plt.plot(means)
plt.fill_between(range(3),l, u, alpha=.1)

plt.show()