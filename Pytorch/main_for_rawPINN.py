import numpy as np
import torch
import torch.nn as nn
from model.rawPINN import rawPINN, loss_fn
import scipy.io as scio
from data.Gen_data import D1, D2, D3, D4, D5, D6, D7, D8
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import time
from pyDOE import *


def Gen_Derivatives(u, x, t):
    """Generator of Derivatives"""
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
    return u, u_t, u_x, u_xxx


def start_training(model, x_data_1, t_data_1, x_data_2, t_data_2, train_exact, parameters, epochs, learning_rate, save_dir, domain_order, Midway=1000):
    """Trianing Network"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.97)
    loss_list = []
    loss_function = loss_fn(parameters)

    minloss = 1.0

    mae_loss = nn.L1Loss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        t1 = time.time()
        output_1 = model(x_data_1, t_data_1, True)
        output_1 = output_1.reshape((1000, 1000))

        f_output = model(x_data_2, t_data_2)
        phyloss = loss_function.phy_loss(f_output, x_data_2, t_data_2)

        data_loss = mae_loss(output_1[:900, :], train_exact)

        all_loss = data_loss + phyloss

        all_loss.backward(retain_graph=True)

        optimizer.step()
        scheduler.step()

        loss_list.append([all_loss.item(), data_loss.item(), phyloss.item()])
        t2 = time.time()

        print('[%d/%d %d%%] loss: %.10f, data_loss: %.11f, phy_loss: %.11f, cost_time: %.3f s' % (
            (epoch + 1), epochs, ((epoch + 1) / epochs * 100.0), all_loss, data_loss, phyloss, t2 - t1))

        if all_loss.item() < minloss:
            minloss = all_loss.item()
            save_dir_ = save_dir + 'trained_model/rawPINN/rawPINN.pth'.format(domain_order, epoch)
            print('saving model')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir_)

    return loss_list


if __name__ == '__main__':
    torch.manual_seed(123)

    Domain_list = [D1, D2, D3, D4, D5, D6, D7, D8]
    domain_order = 5

    # Model Initialization parameters
    MLP_layer_structure = [2, 50, 50, 50, 50, 50, 50, 50, 1]

    domain_data = scio.loadmat('data/domain_data/domain_{}.mat'.format(str(domain_order)))
    exact = scio.loadmat('data/domain_data/domain_{}_exact.mat'.format(str(domain_order)))
    parameters = domain_data['D_parameter']

    # load train data
    train_data = domain_data['train_data'].astype(float)
    exact_data = exact['exact'].astype(float)
    Y = exact_data[0, :]
    Y_ = exact_data[-1, :]

    lb = np.array([-11.5, 0])
    ub = np.array([0.5, 0.5])
    train_phy_X = lb + (ub - lb) * lhs(2, 10000)

    train_data_1 = torch.tensor(train_data, dtype=torch.float32, requires_grad=True)
    train_data_2 = torch.tensor(train_phy_X, dtype=torch.float32, requires_grad=True)
    train_exact = torch.tensor(exact_data[:900, :], dtype=torch.float32).cuda()

    train_x_1 = train_data_1[:, :, 0].reshape(1000, 1000, 1).cuda()
    train_t_1 = train_data_1[:, :, 1].reshape(1000, 1000, 1).cuda()

    train_x_2 = train_data_2[:, 0].reshape(-1, 1).cuda()
    train_t_2 = train_data_2[:, 1].reshape(-1, 1).cuda()

    # Model Initialization
    model = rawPINN(MLP_layer_structure=MLP_layer_structure).cuda()
    model.initialize()

    # Training hyperparameters
    epochs = 1500
    learning_rate = 0.001
    save_path = 'results/'

    # Starting Training
    train_epochs_loss = start_training(
        model=model,
        x_data_1=train_x_1,
        t_data_1=train_t_1,
        x_data_2=train_x_2,
        t_data_2=train_t_2,
        train_exact=train_exact,
        parameters=parameters,
        epochs=epochs + 1,
        learning_rate=learning_rate,
        save_dir=save_path,
        domain_order=domain_order,
        Midway=0
    )

    checkpoint = torch.load('results/trained_model/rawPINN/rawPINN.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Plot Figure
    Figure_save_path = 'results/figures/rawPINN/'

    # Train loss
    plt.figure(num=1, figsize=(15, 10), dpi=80)
    plt.plot(range(len(train_epochs_loss)), train_epochs_loss)
    plt.xlabel("Epoch number")
    plt.ylabel("MSE Loss Value")
    plt.title("rawPINN Train Loss Curve")
    plt.savefig(Figure_save_path + "rawPINN_Train_Loss_Curve.png")
    plt.show()

    # Prediction Heat Figure
    td = torch.tensor(train_data, dtype=torch.float32, requires_grad=True)
    test_x = td[:, :, 0].reshape(-1, 1).cuda()
    test_t = td[:, :, 1].reshape(-1, 1).cuda()
    pred_u = model(test_x, test_t)
    u = pred_u.cpu().detach().numpy()
    np.save('results/npyfile/rawPINN/rawPINN_D{}.npy'.format(domain_order), u)
    u_new = u.reshape((1000, 1000, 1))
    plt.figure(num=2, figsize=(10, 10), dpi=80)
    plt.imshow(u_new, interpolation='nearest', cmap='YlGnBu', origin='lower', aspect='auto')
    plt.colorbar(shrink=0.70)

    plt.xticks(())
    plt.yticks(())
    plt.xlabel("x Axis")
    plt.ylabel("t Axis")
    plt.title("rawPINN Prediction")
    plt.savefig(Figure_save_path + "rawPINN_Prediction_heatmap.png")
    plt.show()

    # Last time step figure
    D = Domain_list[domain_order - 1]()
    plot_x = np.linspace(D.x_boundary[0], D.x_boundary[1], 1000)

    last_t = u_new[-1, :]
    plt.figure(num=3, figsize=(10, 10), dpi=80)
    plt.plot(plot_x, last_t, 'r--', alpha=1, linewidth=2, label='Prediction')
    plt.plot(plot_x, Y_, 'b-',  alpha=0.5, linewidth=2, label='Exact')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("train_Y0")
    plt.title("rawPINN Last time step")
    plt.savefig(Figure_save_path + "rawPINN_Last_time_step.png")
    plt.show()
