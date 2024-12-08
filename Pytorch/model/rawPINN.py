"""
@author: WenXuan Yuan
Email: wenxuan.yuan@qq.com
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm


class Sin(torch.nn.Module):  # sin激活函数
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.sin()
        return x


def save_checkpoint(model, optimizer, scheduler, save_dir):
    """save model and optimizer"""

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_dir)


def load_checkpoint(model, optimizer, scheduler, model_dir):
    """load model and optimizer"""

    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('model in "{}" loaded!'.format(model_dir))

    return model, optimizer, scheduler


class rawPINN(nn.Module):
    def __init__(self, MLP_layer_structure):

        super(rawPINN, self).__init__()

        # model_input channels of layer includes input_channels and hidden_channels of cells
        self.MLP_layer_structure = MLP_layer_structure
        self.MLP = nn.Sequential()

        # defining MLP structure based on MLP_layer_structure
        for layer_index in range(len(self.MLP_layer_structure)):
            if layer_index == len(self.MLP_layer_structure) - 1:
                continue
            elif layer_index == len(self.MLP_layer_structure) - 2:
                self.MLP.add_module('Linear_layer{}'.format(layer_index),
                                    nn.Linear(self.MLP_layer_structure[layer_index],
                                              self.MLP_layer_structure[layer_index + 1],
                                              bias=True))
            else:
                self.MLP.add_module('Linear_layer{}'.format(layer_index),
                                    nn.Linear(self.MLP_layer_structure[layer_index],
                                              self.MLP_layer_structure[layer_index + 1]))
                self.MLP.add_module('Activation_Tanh{}'.format(layer_index), Sin())

    def forward(self, x, t, F=False):
        if F == True:
            mlp_input = torch.cat((x, t), dim=2)
        else:
            mlp_input = torch.cat((x, t), dim=1)  # [t_num * x_num, 2]
        mlp_output = self.MLP(mlp_input)  # [t_num * x_num, 1]
        return mlp_output

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_normal_(module.weight.train_data, mode='fan_out')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in self.LSTM.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=1)


def Gen_Derivatives(u, x, t):
    """Generator of Derivatives"""
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), retain_graph=True)[0]
    return u, u_t, u_x, u_xxx


class loss_fn(torch.nn.Module):
    def __init__(self, parameters):
        super(loss_fn, self).__init__()
        self.k_1 = int(parameters['k_1'])
        self.k_2 = int(parameters['k_2'])
        self.d = int(parameters['d'])
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def initial_loss(self, pred_initial_boundary_Y, initial_boundary_Y):
        init_loss = self.mae_loss(pred_initial_boundary_Y.cuda(), initial_boundary_Y.cuda())
        return init_loss

    def b_loss(self, pred_end_Y, end_Y):
        end_loss = self.mae_loss(pred_end_Y.cuda(), end_Y.cuda())
        return end_loss

    def phy_loss(self, output, x, t):
        y, y_t, y_x, y_xxx = Gen_Derivatives(output, x, t)
        f = y_t + self.d * y_x + self.k_1 * y * y_x + self.k_2 * y * y * y_x + y_xxx
        phy_loss = self.mae_loss(f.cuda(), torch.zeros_like(f).cuda())
        return phy_loss


if __name__ == '__main__':
    print('Debugging rawPINN network')

    domain_data = scio.loadmat('../data/domain_data/domain_7.mat')

    MLP_layer_structure = [2, 40, 30, 50, 40, 1]
    parameters = domain_data['D_parameter']

    # load train data
    t_sample_num = int(domain_data['train_num']['t_train_num'])
    x_sample_num = int(domain_data['train_num']['x_train_num'])

    train_data = domain_data['train_data'].astype(float)
    train_data = train_data.reshape((t_sample_num * x_sample_num, 2))  # [t_num, x_num, 2] -> [t_num * x_num, 2]
    Y = domain_data['train_Y0'].astype(float)
    Y_ = domain_data['train_Yn'].astype(float)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    Y0 = torch.tensor(Y, dtype=torch.float32)
    Yn = torch.tensor(Y_, dtype=torch.float32)
    print(train_data.shape)

    train_x = train_data[:, 0].reshape((t_sample_num * x_sample_num, 1))
    train_t = train_data[:, 1].reshape((t_sample_num * x_sample_num, 1))

    # Defining Model
    model = rawPINN(MLP_layer_structure=MLP_layer_structure)
    model.initialize()
    print(model)

    output = model.forward(train_x, train_t)
    print(output.shape)

    a, b = model.Time_loss(train_x, train_t)
    print(a.item())
    print(b.item())
