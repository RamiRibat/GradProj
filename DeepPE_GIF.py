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
        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
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
    # loss_delta1 = (loss_func(y, mu1)*inv_std1 + torch.abs(log_std1)).mean()
    loss_delta1 = (torch.square(y - mu1)*inv_std1).mean() + (torch.abs(log_std1)).mean()
    loss_delta2 = (loss_func(y, mu2)*inv_std2 + torch.abs(log_std2)).mean()
    loss_delta3 = (loss_func(y, mu3)*inv_std3 + torch.abs(log_std3)).mean()

    loss = (loss_delta1 + loss_delta2 + loss_delta3)/3

    return loss_delta1, loss_delta2, loss_delta3, loss



def updateModel(train_x, train_y): # Rami (Done)

    # print("Model updating..")
    # Run one gradient descent step for model
    clip = 1
    # model_optimizer.zero_grad()
    model_optimizer1.zero_grad()
    model_optimizer2.zero_grad()
    model_optimizer3.zero_grad()
    loss_delta1, loss_delta2, loss_delta3, loss = compute_loss_model(train_x, train_y)
    # loss_delta1.backward()
    for loss in [loss_delta1, loss_delta2, loss_delta3]:
        loss.backward() # Descent
        # nn.utils.clip_grad_norm_(model.parameters(), clip)
        nn.utils.clip_grad_norm_(model.delta1.parameters(), clip)
        nn.utils.clip_grad_norm_(model.delta2.parameters(), clip)
        nn.utils.clip_grad_norm_(model.delta3.parameters(), clip)
    # model_optimizer.step()
    model_optimizer1.step()
    model_optimizer2.step()
    model_optimizer3.step()
    # Adam(model.delta1.parameters(), lr=0.001).step()
    # Adam(model.delta2.parameters(), lr=0.001).step()
    # Adam(model.delta3.parameters(), lr=0.001).step()
    return loss_delta1, loss_delta2, loss_delta3, loss


# Plotting
def plotter(fig, i, loss):
    with torch.no_grad():
        # Plot training data as black stars
        axs[0].plot(train_x.numpy(), train_y.numpy(), marker='.', mec='#606060', label='Train Data', mfc = '#808080')
        axs[0].plot(orig_x.numpy(), orig_y.numpy(), label= 'True Model', color = 'k', linewidth=3)

        # Plot predictive means as blue line
        l1 = torch.reshape(mu1-std1, (1,-1)).numpy()[0]
        u1 = torch.reshape(mu1+std1, (1,-1)).numpy()[0]
        axs[0].plot(test_x.numpy(), mu1, label = 'Model Blue', color = '#0000FF', linewidth=2) # Blue
        axs[0].fill_between(test_x.numpy(), l1, u1, alpha=0.25, color = '#0000FF')
        axs[0].grid(True)
        axs[1].plot(test_x.numpy(), std1, label = 'Model Blue STD', color = '#0000FF', linewidth=2) # Blue
        axs[1].grid(True)
        axs[2].plot(model1_losses, label = 'Model Blue Error', color = '#0000FF', linewidth=2) # Blue
        axs[2].grid(True)

        l2 = torch.reshape(mu2-std2, (1,-1)).numpy()[0]
        u2 = torch.reshape(mu2+std2, (1,-1)).numpy()[0]
        axs[0].plot(test_x.numpy(), mu2, label = 'Model Green', color = '#006633', linewidth=2) # green
        axs[0].fill_between(test_x.numpy(), l2, u2, alpha=0.25, color = '#006633')
        axs[0].grid(True)
        axs[1].plot(test_x.numpy(), std2, label = 'Model Green STD', color = '#006633', linewidth=2) # Blue
        axs[1].grid(True)
        axs[2].plot(model2_losses, label = 'Model Green Error', color = '#006633', linewidth=2) # Blue
        axs[2].grid(True)

        l3 = torch.reshape(mu3-std3, (1,-1)).numpy()[0]
        u3 = torch.reshape(mu3+std3, (1,-1)).numpy()[0]
        axs[0].plot(test_x.numpy(), mu3, label = 'Model Red', color = '#FF0000', linewidth=2) # Red
        axs[0].fill_between(test_x.numpy(), l3, u3, alpha=0.25, color = '#FF0000')
        axs[0].grid(True)
        axs[1].plot(test_x.numpy(), std3, label = 'Model Red STD', color = '#FF0000', linewidth=2) # Blue
        axs[1].grid(True)
        axs[2].plot(model3_losses, label = 'Model Red Error', color = '#FF0000', linewidth=2) # Blue
        axs[2].grid(True)


        axs[0].set_title(f'Epoch {i}', fontsize=15)
        axs[0].set_ylim([-3, 3])
        axs[0].set_xlim([-3, 3])
        axs[1].set_title(f'STD', fontsize=15)
        axs[1].set_ylim([0.75, 1.25])
        axs[1].set_xlim([-3, 3])
        axs[2].set_title(f'Loss {round(loss_delta1.item(), 4)}', fontsize=15)
        # axs[2].set_ylim([-2, 2])
        # axs[2].set_xlim([-3, 3])

        # for label in axs[0].xaxis.get_ticklabels():
        #     # label is a Text instance
        #     label.set_fontsize(15)

        # for label in axs[0].yaxis.get_ticklabels():
        #     # line is a Line2D instance
        #     label.set_fontsize(15)

        for axis in ['top','bottom','left','right']:
            axs[0].spines[axis].set_linewidth(1)


        # ax.legend(['Observed Data', 'Ground Truth', 'Model Blue', 'Model Green', 'Model Red'], fontsize=15)
        # axs[0].legend(loc='lower left', fontsize=5)













model = MLPModelEnsemble(1, 1, [256, 256])
model_optimizer = Adam(model.parameters(), lr=0.001)
model_optimizer1 = Adam(model.delta1.parameters(), lr=0.001)
model_optimizer2 = Adam(model.delta2.parameters(), lr=0.001)
model_optimizer3 = Adam(model.delta3.parameters(), lr=0.001)



# Training data is 100 points in [0,1] inclusive regularly spaced
orig_x = torch.tensor(torch.linspace(-3.14, 3.14, 1000))
orig_y = torch.sin(orig_x * (2 * math.pi))

train_x = torch.cat(
                    [
                        torch.tensor(torch.linspace(-1, -0.5, 1000)),
                        torch.tensor(torch.linspace(0.5, 1, 1000)),
                        # torch.tensor(torch.linspace(2.15, 2.75, 1000)),
                    ],
                    axis=0
                    )
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)


BATCH_SIZE = 64
EPOCHS = 500

model1_losses = []
model2_losses = []
model3_losses = []
for i in range(1, EPOCHS+1):
    indexes = random.sample(range(0, 2000), BATCH_SIZE)
    inx = torch.tensor(indexes)

    loss_delta1, loss_delta2, loss_delta3, loss = updateModel(train_x[inx], train_y[inx])
    model1_losses.append(loss_delta1)
    model2_losses.append(loss_delta2)
    model3_losses.append(loss_delta3)

    with torch.no_grad():
        test_x = torch.linspace(-3, 3, 600)
        delta1, delta2, delta3 = model(test_x.reshape(len(test_x),1))
        del1, mu1, log_std1, std1, inv_std1 = delta1
        del2, mu2, log_std2, std2, inv_std2 = delta2
        del3, mu3, log_std3, std3, inv_std3 = delta3

    # fig = plt.figure(figsize=(8, 10))
    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 2]}, figsize=(8, 10))
    anime = FuncAnimation(fig, plotter(fig, i, round(loss.item(), 4)))
    plt.tight_layout()
    plt.grid(True)
    plt.draw()
    plt.pause(5/EPOCHS)
    if i < EPOCHS:
        fig.clear()
        plt.close()
    else:
        plt.savefig('DeepPEs.png')
        plt.show()
