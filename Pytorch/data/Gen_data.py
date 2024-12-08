"""
@author: WenXuan Yuan
Email: wenxuan.yuan@qq.com
"""

import numpy as np
import random
import scipy.io
from matplotlib import pyplot as plt


# domain parameters:
class D1:
    """Domain 1 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 2
        self.k_2 = -1
        self.d = 2

        self.u_positive = -2
        self.u_negative = -1

        self.t = 30

        self.x_boundary = [-380, 20]


class D2:
    """Domain 2 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 6
        self.k_2 = -3
        self.d = 15

        self.u_positive = -1
        self.u_star = 3
        self.u_negative = 2

        self.t = 10

        self.x_boundary = [-70, 180]


class D3:
    """Domain 3 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 6
        self.k_2 = -1
        self.d = 10

        self.u_positive = 3
        self.u_negative = 5

        self.t = 90

        self.x_boundary = [1260, 1800]


class D4:
    """Domain 4 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 6
        self.k_2 = -1
        self.d = 10

        self.u_positive = 4
        self.u_negative = 5

        self.t = 90

        self.x_boundary = [1260, 1720]


class D5:
    """Domain 5 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 12
        self.k_2 = -1
        self.d = -30

        self.u_positive = 11
        self.u_negative = 9

        self.t = 0.3

        self.x_boundary = [-11, 0.5]


class k2gt0_D5:
    """Domain 5 parameters"""

    def __init__(self):
        self.k2gt0 = True
        self.k_1 = 16
        self.k_2 = 2
        self.d = -6

        self.u_positive = 2
        self.u_negative = -3

        self.t = 0.5

        self.x_boundary = [-18.5, 17.5]


class D6:
    """Domain 6 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 2
        self.k_2 = -1 / 2
        self.d = -3 / 2

        self.u_positive = 4
        self.u_negative = 1

        self.t = 100

        self.x_boundary = [-340, 40]


class k2gt0_D6:
    """Domain 6 parameters"""

    def __init__(self):
        self.k2gt0 = True
        self.k_1 = 2
        self.k_2 = 1
        self.d = 3

        self.u_positive = 1
        self.u_negative = -2

        self.t = 5.0

        self.x_boundary = [-1.0, 31.0]


class D7:
    """Domain 7 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 3
        self.k_2 = -1
        self.d = 0

        self.u_positive = 3
        self.u_negative = -1

        self.t = 1.0

        self.x_boundary = [-5.0, 11.0]


class D8:
    """Domain 8 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 3
        self.k_2 = -1
        self.d = 0

        self.u_positive = 0
        self.u_negative = -1

        self.t = 1.0

        self.x_boundary = [-5.0, 1.0]


def step_function(x, u_positive, u_negative):
    """Step function"""
    if x >= 0:
        return u_positive
    elif x < 0:
        return u_negative


if __name__ == '__main__':

    Domain_list = [D1, D2, D3, D4, D5, D6, D7, D8]

    D_order = 1
    D = Domain_list[D_order - 1]()
    # D_order = 5
    # D = k2gt0_D5()
    print('Start Generating Initial Data for {}'.format('Domain ' + str(D_order)))

    # Number of samples for training
    t_train_num = 1000
    x_train_num = 1000

    # Initial all_train_x based on given boundary
    x = list(np.linspace(D.x_boundary[0], D.x_boundary[1], x_train_num))
    t = list(np.linspace(0, D.t, t_train_num))

    # use x_train_num to generate train_data from all_train_x
    print('Start Generating train Data')
    train_data = np.zeros((t_train_num, x_train_num, 2))

    # Generate grid data
    for i in range(t_train_num):
        for j in range(x_train_num):
            train_data[i, j, 0] = x[j]
            train_data[i, j, 1] = t[i]

    # for i in range(t_train_num):
    #     sample_x = random.sample(x, x_train_num)
    #     train_data[i, :, 0] = sample_x
    #     train_data[i, :, 1] = t[i]
    # print(train_data)
    print(train_data.shape)

    # use step_function to initial train_Y0 at t=0
    train_X0 = train_data[0, :, 0]
    train_Y0 = np.zeros((len(train_X0), 1))
    for i in range(len(train_X0)):
        train_Y0[i][0] = step_function(train_X0[i], D.u_positive, D.u_negative)
    # print(train_Y0)
    print(train_Y0.shape)

    # # use exact_function to initial train_Yn at t=0
    # train_Xn = train_data[-1, :, 0]
    # train_Yn = np.zeros((len(train_Xn), 1))
    # for i in range(len(train_Xn)):
    #     train_Yn[i][0] = End_f(train_Xn[i])
    # # print(train_Yn)
    # print(train_Yn.shape)
    #
    # # Number of samples for testing
    # t_test_num = 100
    # x_test_num = 1001
    #
    # # Initial all_test_x based on given boundary
    # test_t = list(np.linspace(0, D.t, t_test_num))
    # test_x = list(np.linspace(D.x_boundary[0], D.x_boundary[1], x_test_num))
    #
    # # use x_test_num to generate test_data from all_test_x
    # print('Start Generating test Data')
    # test_data = np.zeros((t_test_num, x_test_num, 2))
    # for i in range(t_test_num):
    #     for j in range(x_test_num):
    #         test_data[i, j, 0] = test_x[j]
    #         test_data[i, j, 1] = test_t[i]
    # # print(test_data)
    # print(test_data.shape)
    #
    # # use step_function to initial test_Y0 at t=0
    # test_X0 = test_data[0, :, 0]
    # test_Y0 = np.zeros((len(test_X0), 1))
    # for i in range(len(test_X0)):
    #     test_Y0[i][0] = step_function(test_X0[i], D.u_positive, D.u_negative)
    # # print(test_Y0)
    # print(test_Y0.shape)
    #
    # # use exact_function to initial train_Yn at t=0
    # test_Xn = test_data[-1, :, 0]
    # test_Yn = np.zeros((len(test_Xn), 1))
    # for i in range(len(test_Xn)):
    #     test_Yn[i][0] = End_f(test_Xn[i])
    # # print(test_Yn)
    # print(test_Yn.shape)

    D_parameter = {'k_1': D.k_1, 'k_2': D.k_2, 'd': D.d}

    # saving data to .mat file
    # scipy.io.savemat('domain_data/domain_{}.mat'.format(str(D_order)),
    #                  {
    #                      'D_parameter': D_parameter,
    #                      'train_data': train_data,
    #                      'train_Y0': train_Y0,
    #                      'train_Yn': train_Yn,
    #                      'train_num': {'t_train_num': t_train_num, 'x_train_num': x_train_num},
    #                      'test_data': test_data,
    #                      'test_Y0': test_Y0,
    #                      'test_Yn': test_Yn,
    #                      'test_num': {'t_test_num': t_test_num, 'x_test_num': x_test_num},
    #                  })

    if D.k2gt0:
        print('k2gt0')
        scipy.io.savemat('domain_data/k2gt0_domain_{}.mat'.format(str(D_order)),
                         {
                             'D_parameter': D_parameter,
                             'train_data': train_data,
                             'train_Y0': train_Y0
                         })
    else:
        scipy.io.savemat('domain_data/domain_{}.mat'.format(str(D_order)),
                     {
                         'D_parameter': D_parameter,
                         'train_data': train_data,
                         'train_Y0': train_Y0
                     })

    print('Finished Generating Initial Data')
