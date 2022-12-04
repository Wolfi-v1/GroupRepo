'''
Authors:
    Name: David Schweighofer Mtr.Nr.: 11771131
    Name: Wolfgang Höller Mtr.Nr.: 11771132
    Name: Bernhard Grill Mtr.Nr.:01608707
'''

import numpy as np
import matplotlib.pyplot as plt

def approx_c_ls(AA_, y_obs_):
    # pre compute expression [A.T A]^-1 A.T
    temp_1 = np.matmul(np.linalg.inv(np.matmul(AA_.T, AA_)), AA_.T)
    # c_ls = [A.T A]^-1 A.T y_obs.T
    return np.matmul(temp_1, y_obs_.T)

def calc_ls_error(AA_, y_obs_):
    # pre compute expression A [A.T A]^-1 A.T
    temp_1 = np.matmul(AA_, np.matmul(np.linalg.inv(np.matmul(AA_.T, AA_)), AA_.T))
    # e_ls = [I - A [A.T A]^-1 A.T] y_obs
    # where B = [I - A [A.T A]^-1 A.T]
    return np.matmul((np.eye(np.shape(temp_1)[0]) - temp_1),y_obs_.T)

def get_weighting_matirx(WW_, w_):
    WW_[0][0] = w_
    WW_[WW_.shape[0]-1][WW_.shape[1]-1] = w_
    return WW_

def approx_weighted_c_ls(AA_, y_obs_, WW_):
    # c_w_LS = [A.T W A]^-1 A.T W y
    return np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(AA_.T, WW_), AA_)), AA_.T), WW_), y_obs_.T)

def calc_MSE(N_, WW_, y_obs_, c_w_ls_, AA_):
    numerator = 0
    denominator = 0
    y_estimated = np.matmul(AA_, c_w_ls_)
    for i in range(0, N_-1):
        numerator = numerator + (WW_[i][i] * (y_obs_[i] - y_estimated[i][0])**2)
        denominator = denominator + WW_[i][i]
    return numerator / (denominator * N_)


if __name__ == '__main__':
    print("Authors:\n"
          "\tName: David Schweighofer \tMtr.Nr.: 11771131\n"
          "\tName: Wolfgang Höller \tMtr.Nr.: 11771132\n"
          "\tName: Bernhard Grill \tMtr.Nr.:01608707")
    '''Input parameters'''
    x_obs = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    y_obs = np.array([[45, 37, 28, 14, 13, 3, 1, 4, 2, 9]])
    N = len(x_obs[0])
    '''5.4.1'''
    # quadratic approximation y = a x^2 + b x + c -> AA = 10 x 3
    AA = np.ones((N, 3), dtype=float)
    for i in range(N):
        AA[i][0] = x_obs[0][i]**2
        AA[i][1] = x_obs[0][i]

    c_LS = approx_c_ls(AA_=AA, y_obs_=y_obs)

    '''5.4.2'''
    e_ls = calc_ls_error(AA_=AA, y_obs_=y_obs)
    print("Check if e_ls is orthogonal to our estimation (A.c_ls). \n"
          "Following condition must be fulfilled  \n\t<e_ls, (A.c_ls)> == 0\n"
          "Computing result: \n\t<e_ls, (A.c_ls)> = {:.5e}\n"
          "This error is only a numerical calculation artifact \n"
          "\t-> e_ls is orthogonal to our estimation (A.c_ls)".format(np.inner(e_ls.T,np.matmul(AA,c_LS).T)[0][0]))

    '''5.4.3'''
    plt.scatter(x_obs, y_obs, label="Measured data", marker='x')
    plt.scatter(x_obs, np.matmul(AA,c_LS), label="LS", marker='x')
    plt.grid(axis='x', color='0.8')
    plt.grid(axis='y', color='0.8')
    plt.xticks(np.arange(0, N, 1))
    plt.yticks(np.arange(0, 50, 5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least square approximation')
    plt.legend()
    plt.show()
    plt.close()

    '''5.4.3 & 5.4.4'''
    w_range = [0.1, 1, 10, 100]
    figure, axis = plt.subplots(int(len(w_range)/2),int(len(w_range)/2))
    x = 0
    for i in range(0,len(w_range)):

        WW = get_weighting_matirx(np.eye(10), w_range[i])
        c_w_LS = approx_weighted_c_ls(AA_=AA, y_obs_=y_obs, WW_=WW)
        mse_err = calc_MSE(10, WW, y_obs[0], c_w_LS, AA)

        # Plot section
        ax = axis[i%2][x]
        ax.scatter(x_obs, y_obs, label="Measured data", marker='x')
        ax.scatter(x_obs, np.matmul(AA, c_w_LS), label="Estimation", marker='x')
        ax.set_title("Weighted LS \n(w={} mse_err = {:.4f})".format(w_range[i], mse_err), fontsize=11)
        ax.grid(axis='x', color='0.8')
        ax.grid(axis='y', color='0.8')
        ax.xaxis.set_ticks(np.arange(0, N, 1))
        ax.yaxis.set_ticks(np.arange(0, 50, 5))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        x = x + i % 2
        # Plot section
    plt.subplots_adjust(hspace=.6)
    plt.show()
    print("Conclusion from the plot:\n"
          "\tw = 100 has the lowest MSE error.\n")
