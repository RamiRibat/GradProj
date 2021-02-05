import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam

from matplotlib import pyplot as plt

seed = 111
torch.manual_seed(seed)
np.random.seed(seed)

LOG_STD_MAX = 0.1
LOG_STD_MIN = -20


class MLPModel(nn.Module): # Rami (Done)

    def __init__(self, ip_dim, op_dim, hidden_sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()

        # layers = []
        # layers.append(torch.nn.Linear(ip_dim, hidden_sizes[0]))
        # layers.append(torch.nn.ReLU())
        # for i in range(0, len(hidden_sizes)-1):
        #     layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        #     layers.append(torch.nn.ReLU())

        # self.net = nn.Sequential(*layers, nn.Linear(hidden_sizes[-1], 1))
        # print('self.net: ', self.net)
        # self.mu_layer = nn.Linear(1, op_dim)
        # self.log_std_layer = nn.Linear(1, op_dim)

        layers = [torch.nn.Linear(1, 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, 200),
                torch.nn.ReLU(),
                torch.nn.Linear(200, 100)]

        self.net = nn.Sequential(*layers)
        print('self.net: ', self.net)
        self.mu_layer = nn.Linear(100, op_dim)
        self.log_std_layer = nn.Linear(100, op_dim)

    def forward(self, ip, deterministic=False, with_logprob=True):
        # st+1 ~ f(st+1|st,at;omega) = N(mu,std|st,at;omega)
        net_out = self.net(ip)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) 
        # Transition distribution 
        delta = Normal(mu, std)
        return delta.rsample(), mu, std


class MLPModelEnsemble(nn.Module):
    def __init__(self, ip_dim, op_dim, hidden_sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.delta1 = MLPModel(ip_dim, op_dim, hidden_sizes, activation, output_activation)
        self.delta2 = MLPModel(ip_dim, op_dim, hidden_sizes, activation, output_activation)
        self.delta3 = MLPModel(ip_dim, op_dim, hidden_sizes, activation, output_activation)
        # self.delta4 = MLPModel(ip_dim, hidden_sizes, activation, output_activation)
        # self.delta5 = MLPModel(ip_dim, hidden_sizes, activation, output_activation)

    def forward(self, ip, deterministic=False, with_logprob=True):
        delta1 = self.delta1(ip)
        delta2 = self.delta2(ip)
        delta3 = self.delta3(ip)
        # delta4 = self.delta4(ip)
        # delta5 = self.delta5(ip)
        # delta = (delta1 + delta2 + delta3)/3
        return delta1, delta2, delta3


def compute_loss_model(train_x, train_y): # Rami (Done)

    x, y = train_x, train_y
    x = x.reshape(len(train_x),1)
    y = y.reshape(len(train_y),1)

    loss_func = torch.nn.MSELoss()
    loss_delta1 = loss_func(y, dyn_models(x)[0][0]) # delta.rsample()
    loss_delta2 = loss_func(y, dyn_models(x)[1][0])
    loss_delta3 = loss_func(y, dyn_models(x)[2][0])
    
    # loss_delta1 = ((y - model(x))[0]**2).mean()
    # loss_delta2 = ((y - model.delta2(x))**2).mean()
    # loss_delta3 = ((y - model.delta3(x))**2).mean()
    loss_model = (loss_delta1 + loss_delta2 + loss_delta3)/3

    return loss_delta1, loss_delta2, loss_delta3


def updateModel(train_x, train_y): # Rami (Done)

    # print("Model updating..")
    # Run one gradient descent step for model
    model_optimizer.zero_grad()
    loss_delta1, loss_delta2, loss_delta3 = compute_loss_model(train_x, train_y)
    for loss in [loss_delta1, loss_delta2, loss_delta3]:
        loss.backward() # Descent
    Adam(dyn_models.delta1.parameters(), lr=0.001).step()
    Adam(dyn_models.delta2.parameters(), lr=0.001).step()
    Adam(dyn_models.delta3.parameters(), lr=0.001).step()
    # model_optimizer.step()

# print('Plotting')
def plotModel():
    with torch.no_grad():
        observed_pred = dyn_models(test_x.reshape(len(test_x),1))
        # Initialize plot
        # f, ax = plt.subplots(1, 1, figsize=(4, 3))
        # fig = plt.figure(figsize=(8, 6))

        # ax = fig.add_subplot(1, 1, 1)

        # Get upper and lower confidence bounds
        # lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        # ax.plot(train_x.numpy(), train_y.numpy(), '.', color = '#00CC66')
        # ax.plot(orig_x.numpy(), orig_y.numpy(), color = '#006633', linewidth=2.5)

        # Plot predictive means as blue line
        mean1 = observed_pred[0][1]
        std1 = observed_pred[0][2]
        l1 = torch.reshape(mean1-std1, (1,-1)).numpy()[0]
        u1 = torch.reshape(mean1+std1, (1,-1)).numpy()[0]
        ax.plot(test_x.numpy(), mean1, color = '#FF0000', linewidth=4)
        ax.fill_between(test_x.numpy(), l1, u1, alpha=0.5, color = '#FF0000')

        # mean2 = observed_pred[1][1]
        # std2 = observed_pred[1][2]
        # l2 = torch.reshape(mean2-std2, (1,-1)).numpy()[0]
        # u2 = torch.reshape(mean2+std2, (1,-1)).numpy()[0]
        # ax.plot(test_x.numpy(), mean2, color = '#009900', linewidth=4)
        # ax.fill_between(test_x.numpy(), l2, u2, alpha=0.5, color = '#009900')

        mean3 = observed_pred[2][1]
        std3= observed_pred[2][2]
        l3 = torch.reshape(mean3-std3, (1,-1)).numpy()[0]
        u3 = torch.reshape(mean3+std3, (1,-1)).numpy()[0]
        ax.plot(test_x.numpy(), mean3, color = '#0000FF', linewidth=4)
        ax.fill_between(test_x.numpy(), l3, u3, alpha=0.5, color = '#0000FF')


        # Shade between the lower and upper confidence bounds
        # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        # ax.set_ylabel('$\mathregular{s_{t+1}}$', fontsize=20)
        # ax.set_xlabel('$\mathregular{s_t}$, $\mathregular{a_t}$', fontsize=20)
        # # ax.set_title('Gaussian Process', fontsize=25)
        # ax.set_ylim([-2, 2])
        # ax.set_xlim([-2, 2])

        # for label in ax.xaxis.get_ticklabels():
        #     # label is a Text instance
        #     label.set_fontsize(15)

        # for label in ax.yaxis.get_ticklabels():
        #     # line is a Line2D instance
        #     label.set_fontsize(15)

        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_linewidth(1)


        # ax.legend(['Observed Data', 'Ground Truth', 'Model 1', 'Model 2'], fontsize=15)
        # ax.legend(['Observed Data', 'Function', 'Mean'], fontsize=15)

    # plt.pause(1e-17)
    # time.sleep(10)



# Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.Tensor(torch.linspace(0, 1, 100)
orig_x = torch.Tensor(torch.linspace(-3.14, 3.14, 1000))
orig_y = torch.sin(orig_x * (2 * math.pi))

train_x = torch.cat(
                    [
                        torch.Tensor(torch.linspace(-1.5, -0.5, 2000)),
                        torch.Tensor(torch.linspace(0.5, 1.5, 2000)),
                        # torch.Tensor(torch.linspace(1, 1.5, 4000)),
                    ],
                    axis=0
                    )
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)


# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad():
    test_x = torch.linspace(-3, 3, 600)



dyn_models = MLPModelEnsemble(1, 1, [200, 100])
model_optimizer = Adam(dyn_models.parameters(), lr=0.001) # Rami


BATCH_SIZE = 64
EPOCH = 100


# plt.show()
# fig = plt.figure(figsize=(8, 6))
# ax0 = fig.add_subplot(1, 1, 1)
# ax = fig.add_subplot(1, 1, 1)

# ax.set_ylabel('$\mathregular{s_{t+1}}$', fontsize=20)
# ax.set_xlabel('$\mathregular{s_t}$, $\mathregular{a_t}$', fontsize=20)
# ax.set_title('Gaussian Process', fontsize=25)
# ax.set_ylim([-2, 2])
# ax.set_xlim([-2, 2])
# ax.plot(train_x.numpy(), train_y.numpy(), '.', color = '#00CC66')
# ax.plot(orig_x.numpy(), orig_y.numpy(), color = '#006633', linewidth=2.5)

for i in range(EPOCH):
    print('Epoch: ', i)
    indexes = random.sample(range(0, 4000), BATCH_SIZE)
    updateModel(train_x[indexes], train_y[indexes])

#     ax.set_title(f'Epoch {i}', fontsize=20)
#     ax.set_ylim([-2, 2])
#     ax.set_xlim([-2, 2])
#     ax.plot(train_x.numpy(), train_y.numpy(), '.', color = '#00CC66')
#     ax.plot(orig_x.numpy(), orig_y.numpy(), color = '#006633', linewidth=2.5)
# plotModel()
    # plt.pause(0.001)
    # time.sleep(0.1)
    # ax.clear()

# plt.show()



# print('Plotting')
def plotModel():
with torch.no_grad():
    observed_pred = dyn_models(test_x.reshape(len(test_x),1))
    # Initialize plot
    # f, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(1, 1, 1)

    # Get upper and lower confidence bounds
    # lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), '.', color = '#00CC66')
    ax.plot(orig_x.numpy(), orig_y.numpy(), color = '#006633', linewidth=2.5)

    # Plot predictive means as blue line
    mean1 = observed_pred[0][1]
    std1 = observed_pred[0][2]
    l1 = torch.reshape(mean1-std1, (1,-1)).numpy()[0]
    u1 = torch.reshape(mean1+std1, (1,-1)).numpy()[0]
    ax.plot(test_x.numpy(), mean1, color = '#FF0000', linewidth=4)
    ax.fill_between(test_x.numpy(), l1, u1, alpha=0.5, color = '#FF0000')

    # mean2 = observed_pred[1][1]
    # std2 = observed_pred[1][2]
    # l2 = torch.reshape(mean2-std2, (1,-1)).numpy()[0]
    # u2 = torch.reshape(mean2+std2, (1,-1)).numpy()[0]
    # ax.plot(test_x.numpy(), mean2, color = '#009900', linewidth=4)
    # ax.fill_between(test_x.numpy(), l2, u2, alpha=0.5, color = '#009900')

    mean3 = observed_pred[2][1]
    std3= observed_pred[2][2]
    l3 = torch.reshape(mean3-std3, (1,-1)).numpy()[0]
    u3 = torch.reshape(mean3+std3, (1,-1)).numpy()[0]
    ax.plot(test_x.numpy(), mean3, color = '#0000FF', linewidth=4)
    ax.fill_between(test_x.numpy(), l3, u3, alpha=0.5, color = '#0000FF')


    # Shade between the lower and upper confidence bounds
    # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

    ax.set_ylabel('$\mathregular{s_{t+1}}$', fontsize=20)
    ax.set_xlabel('$\mathregular{s_t}$, $\mathregular{a_t}$', fontsize=20)
    # # ax.set_title('Gaussian Process', fontsize=25)
    ax.set_ylim([-2, 2])
    ax.set_xlim([-2, 2])

    for label in ax.xaxis.get_ticklabels():
        # label is a Text instance
        label.set_fontsize(15)

    for label in ax.yaxis.get_ticklabels():
        # line is a Line2D instance
        label.set_fontsize(15)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)


    ax.legend(['Observed Data', 'Ground Truth', 'Model 1', 'Model 2'], fontsize=15)
    # ax.legend(['Observed Data', 'Function', 'Mean'], fontsize=15)

plt.show()
