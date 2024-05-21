# Burgers equation.
# 09 / 14 / 2021
# Edgar A. M. O.

import torch
import torch.nn as nn
import numpy as np
from random import uniform


class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f):
        # x & t from boundary conditions:
        self.x_u = torch.tensor(X_u[:, 0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True)
        self.t_u = torch.tensor(X_u[:, 1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True)

        # x & t from collocation points:
        self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True)
        self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True)

        # boundary solution:
        self.u = torch.tensor(u, dtype=torch.float32)

        # null vector to test against f:
        self.null = torch.zeros((self.x_f.shape[0], 1))

        # initialize net:
        self.create_net()

        # this optimizer updates the weights and biases of the net:
        self.optimizer = torch.optim.LBFGS(self.net.parameters(),
                                           lr=1,
                                           max_iter=50000,
                                           max_eval=50000,
                                           history_size=50,
                                           tolerance_grad=1e-05,
                                           tolerance_change=0.5 * np.finfo(float).eps,
                                           line_search_fn="strong_wolfe")

        # typical MSE loss (this is a function):
        self.loss = nn.MSELoss()

        # loss :
        self.ls = 0

        # iteration number:
        self.iter = 0

    def create_net(self):
        """ net takes a batch of two inputs: (n, 2) --> (n, 1) """
        self.net = nn.Sequential(
            nn.Linear(2, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 1)
        )

    def net_u(self, x, t):
        u = self.net(torch.hstack((x, t)))
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        
        u_tt = torch.autograd.grad(
            u_t, t,
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True)[0]

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True)[0]

        f = u_xx + u_tt + 2 * torch.pi **2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * t)

        return f

    def plot(self):
        """ plot the solution on new data """

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        x = torch.linspace(-1, 1, 200)
        t = torch.linspace(-1, 1, 200)

        # x & t grids:
        X, T = torch.meshgrid(x, t, indexing=None)

        # x & t columns:
        xcol = X.reshape(-1, 1)
        tcol = T.reshape(-1, 1)

        # one large column:
        usol = self.net_u(xcol, tcol)

        # reshape solution:
        U = usol.reshape(x.numel(), t.numel())

        # transform to numpy:
        xnp = x.numpy()
        tnp = t.numpy()
        Unp = U.detach().numpy()

        # plot:
        fig = plt.figure(figsize=(9, 4.5))
        ax = fig.add_subplot(111)

        h = ax.imshow(Unp,
                      interpolation='nearest',
                      cmap='rainbow',
                      extent=[tnp.min(), tnp.max(), xnp.min(), xnp.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        plt.savefig('./2D_poisson_pinn.png')
        plt.show()

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # u & f predictions:
        u_prediction = self.net_u(self.x_u, self.t_u)
        f_prediction = self.net_f(self.x_f, self.t_f)

        # losses:
        u_loss = self.loss(u_prediction, self.u)
        f_loss = self.loss(f_prediction, self.null)
        self.ls = u_loss + f_loss

        # derivative with respect to net's weights:
        self.ls.backward()

        # increase iteration count:
        self.iter += 1

        # print report:
        if not self.iter % 100:
            print('Epoch: {0:}, Loss: {1:6.3f}'.format(self.iter, self.ls))

        return self.ls

    def train(self):
        """ training loop """
        self.net.train()
        self.optimizer.step(self.closure)


if __name__ == '__main__':

    N_u = 100  # number of data points in the boundaries
    N_f = 100  # number of collocation points

    # X_u_train: a set of pairs (x, t) located at:
    # x = 1, t = [-1, 1]
    # x = -1, t = [-1, 1]
    # t = -1, x = [-1, 1]
    # t = 1, x = [-1, 1]
    x_right = np.ones((N_u // 4, 1), dtype=float)
    x_left = np.ones((N_u // 4, 1), dtype=float) * (-1)
    t_up = np.ones((N_u // 4, 1), dtype=float)
    t_down = np.ones((N_u // 4, 1), dtype=float) * (-1)

    t_right = 2 * np.random.rand(N_u // 4, 1) - 1
    t_left = 2 * np.random.rand(N_u // 4, 1) - 1
    x_up = 2 * np.random.rand(N_u // 4, 1) - 1
    x_down = 2 * np.random.rand(N_u // 4, 1) - 1

    # stack uppers, lowers and zeros:
    X_right = np.hstack((x_right, t_right))  # 右边界的[x,t].T
    X_left = np.hstack((x_left, t_left))  # 左边界的[x,t].T
    X_up = np.hstack((x_up, t_up))  # 上边界的[x,t].T
    X_down = np.hstack((x_down, t_down)) # 下边界的[x,t].T

    # each one of these three arrays has 2 columns, 
    # now we stack them vertically, the resulting array will also have 2 
    # columns and 100 rows:
    X_u_train = np.vstack((X_right, X_left, X_up, X_down))  # 将边界值拼成(100,2)，作为边界采样点的输入值

    # shuffle X_u_train:
    index = np.arange(0, N_u)
    np.random.shuffle(index)
    X_u_train = X_u_train[index, :]

    # make X_f_train:
    X_f_train = np.zeros((N_f, 2), dtype=float)  # (10000,2)
    for row in range(N_f):
        x = uniform(-1, 1)  # x range
        t = uniform(0, 1)  # t range

        X_f_train[row, 0] = x
        X_f_train[row, 1] = t  # (10000,2) 在定义域内随机采样x,t值，作为内部采样点的输入值

    # add the boundary points to the collocation points:
    X_f_train = np.vstack((X_f_train, X_u_train))  # (10100,2) 内部+边界采样点 作为输入值

    # make u_train
    u_right = np.zeros((N_u // 4, 1), dtype=float)
    u_left = np.zeros((N_u // 4, 1), dtype=float)
    u_up = -np.sin(np.pi * x_up)  # 随机取值作为边界上的真实值，不是真正burgers方程的值，所以无法进行预测
    u_down = -np.sin(np.pi * x_down)

    # stack them in the same order as X_u_train was stacked:
    u_train = np.vstack((u_right, u_left, u_up, u_down))

    # match indices with X_u_train
    u_train = u_train[index, :]  # (100,1) 边界采样点 作为标签

    # pass data sets to the PINN:
    pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train)

    pinn.train()

    pinn.plot()
