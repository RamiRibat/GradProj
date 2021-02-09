import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# seed = 100
# torch.manual_seed(seed)
# np.random.seed(seed)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MLPModel(nn.Module):  # Rami (Done)

    def __init__(self, ip_dim, op_dim, hidden_sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()

        layers = []
        layers.append(nn.Linear(ip_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_sizes)):
            if i < len(hidden_sizes):
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                layers.append(nn.ReLU())
            # else: # if i == len(hidden_sizes)-1
            #     layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))

        self.net = nn.Sequential(*layers)
        print(self.net)
        self.mu_layer = nn.Linear(hidden_sizes[-1], op_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], op_dim)

    def forward(self, ip, deterministic=False, with_logprob=True):
        # st+1 ~ f(st+1|st,at;omega) = N(mu,std|st,at;omega)
        net_out = self.net(ip)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        inv_std = torch.exp(-log_std)

        # Transition distribution
        delta = Normal(mu, std).rsample()

        return delta, mu, log_std, std, inv_std


class MLPModelEnsemble(nn.Module):
    def __init__(self, ip_dim, op_dim, hidden_sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.delta1 = MLPModel(ip_dim, op_dim, hidden_sizes, activation, output_activation)
        self.delta2 = MLPModel(ip_dim, op_dim, hidden_sizes, activation, output_activation)
        self.delta3 = MLPModel(ip_dim, op_dim, hidden_sizes, activation, output_activation)

    def forward(self, ip, deterministic=False, with_logprob=True):
        delta1 = self.delta1(ip)
        delta2 = self.delta2(ip)
        delta3 = self.delta3(ip)
        return delta1, delta2, delta3




def compute_loss_model(train_x, train_y): # Rami (Done)

    x, y = train_x, train_y
    x = x.reshape(len(train_x),1)
    y = y.reshape(len(train_y),1)

    delta1, mu1, log_std1, std1, inv_std1 = model(x)[0]
    delta2, mu2, log_std2, std2, inv_std2 = model(x)[1]
    delta3, mu3, log_std3, std3, inv_std3 = model(x)[2]

    loss_func = torch.nn.MSELoss()
    loss_delta1 = (loss_func(y, mu1)*inv_std1 + abs(log_std1)).mean()
    loss_delta2 = (loss_func(y, mu2)*inv_std2 + abs(log_std2)).mean()
    loss_delta3 = (loss_func(y, mu3)*inv_std3 + abs(log_std3)).mean()

    loss = (loss_delta1 + loss_delta2 + loss_delta3)/3

    return loss_delta1, loss_delta2, loss_delta3, loss



def updateModel(train_x, train_y): # Rami (Done)

    # print("Model updating..")
    # Run one gradient descent step for model
    clip = 1
    model_optimizer.zero_grad()
    loss_delta1, loss_delta2, loss_delta3, loss = compute_loss_model(train_x, train_y)
    # loss_delta1.backward()
    for loss in [loss_delta1, loss_delta2, loss_delta3]:
        loss.backward() # Descent
        torch.nn.utils.clip_grad_norm_(model.delta1.parameters(), clip)
    Adam(model.delta1.parameters(), lr=0.001).step()
    Adam(model.delta2.parameters(), lr=0.001).step()
    Adam(model.delta3.parameters(), lr=0.001).step()
    return loss_delta1, loss_delta2, loss_delta3, loss


# Plotting
def plotter(fig, ax, i, loss):
    with torch.no_grad():
        # Initialize plot
        # f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # plt.cla()

        # Get upper and lower confidence bounds
        # lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), '.', label='Train Data', color = '#808080')
        ax.plot(orig_x.numpy(), orig_y.numpy(), label= 'True Model', color = 'k', linewidth=3)

        # Plot predictive means as blue line
        # mean1 = observed_pred[0][1]
        # std1 = observed_pred[0][2]
        l1 = torch.reshape(mu1-std1, (1,-1)).numpy()[0]
        u1 = torch.reshape(mu1+std1, (1,-1)).numpy()[0]
        ax.plot(test_x.numpy(), mu1, label = 'Model Blue', color = '#0000FF', linewidth=2) # Blue
        ax.fill_between(test_x.numpy(), l1, u1, alpha=0.25, color = '#0000FF')

        # mean2 = observed_pred[1][1]
        # std2 = observed_pred[1][2]
        # l2 = torch.reshape(mu2-std2, (1,-1)).numpy()[0]
        # u2 = torch.reshape(mu2+std2, (1,-1)).numpy()[0]
        # ax.plot(test_x.numpy(), mu2, label = 'Model Green', color = '#006633', linewidth=2) # green
        # ax.fill_between(test_x.numpy(), l2, u2, alpha=0.25, color = '#006633')
        # #
        # # mean3 = observed_pred[2][1]
        # # std3= observed_pred[2][2]
        # l3 = torch.reshape(mu3-std3, (1,-1)).numpy()[0]
        # u3 = torch.reshape(mu3+std3, (1,-1)).numpy()[0]
        # ax.plot(test_x.numpy(), mu3, label = 'Model Red', color = '#FF0000', linewidth=2) # Red
        # ax.fill_between(test_x.numpy(), l3, u3, alpha=0.25, color = '#FF0000')

        ax.set_title(f'Epoch {i}, Loss {loss}', fontsize=25)
        ax.set_ylim([-2, 2])
        ax.set_xlim([-3, 3])

        for label in ax.xaxis.get_ticklabels():
            # label is a Text instance
            label.set_fontsize(15)

        for label in ax.yaxis.get_ticklabels():
            # line is a Line2D instance
            label.set_fontsize(15)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)


        # ax.legend(['Observed Data', 'Ground Truth', 'Model Blue', 'Model Green', 'Model Red'], fontsize=15)
        ax.legend(loc='lower left', fontsize=15)

        # plt.draw()
        # plt.pause(0.01)
        # # fig.clear()
        # plt.close()












model = MLPModelEnsemble(1, 1, [100, 100])
model_optimizer = Adam(model.parameters(), lr=0.001) # Rami



# Training data is 100 points in [0,1] inclusive regularly spaced
orig_x = torch.tensor(torch.linspace(-3.14, 3.14, 1000))
orig_y = torch.sin(orig_x * (2 * math.pi))

train_x = torch.cat(
                    [
                        torch.tensor(torch.linspace(-1, -0.5, 500)),
                        torch.tensor(torch.linspace(0.5, 1, 1000)),
                        torch.tensor(torch.linspace(2.15, 2.75, 1000)),
                    ],
                    axis=0
                    )
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)


BATCH_SIZE = 64
EPOCHS = 1000


for i in range(EPOCHS):
    indexes = random.sample(range(0, 2500), BATCH_SIZE)
    inx = torch.tensor(indexes)

    loss_delta1, loss_delta2, loss_delta3, loss = updateModel(train_x[inx], train_y[inx])

    with torch.no_grad():
        test_x = torch.linspace(-3, 3, 600)
        delta1, delta2, delta3 = model(test_x.reshape(len(test_x),1))
        del1, mu1, log_std1, std1, inv_std1 = delta1
        del2, mu2, log_std2, std2, inv_std2 = delta2
        del3, mu3, log_std3, std3, inv_std3 = delta3

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    anime = FuncAnimation(fig, plotter(fig, ax, i, round(loss.item(), 4)))
    plt.tight_layout()
    plt.draw()
    plt.pause(5/EPOCHS)
    fig.clear()
    if i < EPOCHS-1:
        plt.close()
    else:
        plt.show()
