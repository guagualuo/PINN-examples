import torch
import torch.nn as nn
from torch import autograd
import numpy as np

import sys
sys.path.append('../')
from utils import TimeDependentFFN


class GalerkinFFN(TimeDependentFFN):
    def __init__(self, activation, n_hidden, n_nodes, inputs):
        super(GalerkinFFN, self).__init__(activation, n_hidden, n_nodes, inputs)
    
    def forward(self, x, t):
        xt = super(GalerkinFFN, self).forward(x, t)
        return (x**2 -1.0) * xt


class Burger1DNN:
    def __init__(self, N_init, N_bc, N_pde, net, device):
        self.net = net
        self.iteration = 0
        self.N_init, self.N_bc, self.N_pde = N_init, N_bc, N_pde
        self.device = device
    
    def pde(self, x, t):
        u = self.net(x, t)
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        return u_t + u*u_x - (0.01/np.pi)*u_xx
    
    def initial_condtion(self, x, t):
        return self.net(x, t) + torch.sin(np.pi*x)
    
    def boundary_condition(self, x, t):
        return self.net(x, t)
    
    def generate_samples(self):
        device = self.device
        self.x_init = (2.0*torch.rand((self.N_init, 1), requires_grad=True) - 1.0).to(device)
        self.t_init = (torch.zeros((self.N_init, 1), requires_grad=True)).to(device)

        self.t_bc = (torch.rand((self.N_bc, 1), requires_grad=True)).to(device)
        self.x_bc = (torch.ones((self.N_bc, 1))).to(device)
        self.x_bc[:int(self.N_bc/2)] = -1
        self.x_bc.requires_grad = True

        self.x_pde = (2.0*torch.rand((self.N_pde, 1), requires_grad=True) - 1.0).to(device)
        self.t_pde = (torch.rand((self.N_pde, 1), requires_grad=True)).to(device)
    
    def train(self, optimizer, metric, max_epochs, callbacks=[]):
        while self.iteration < max_epochs:
            for callback in callbacks: callback.on_batch_begin()
            optimizer.zero_grad()

            # loss from boundary and initial condition
            mse_u = metric(self.initial_condtion(self.x_init, self.t_init), torch.zeros_like(self.t_init)) + \
            metric(self.boundary_condition(self.x_bc, self.t_bc), torch.zeros_like(self.t_bc))

            # loss from PDE
            mse_f = metric(self.pde(self.x_pde, self.t_pde), torch.zeros_like(self.t_pde))

            loss = mse_u + mse_f
            
            with autograd.no_grad():
                for callback in callbacks: callback.on_loss_end(**{'training loss': loss.item()})
            # BP
            loss.backward(retain_graph=True)
            optimizer.step()
            with autograd.no_grad():
                for callback in callbacks: callback.on_step_end()
            self.iteration += 1

