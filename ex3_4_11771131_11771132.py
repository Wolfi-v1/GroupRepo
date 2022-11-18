'''
Authors:
    Name: David Schweighofer Mtr.Nr.: 11771131
    Name: Wolfgang Höller Mtr.Nr.: 11771132
'''

import numpy as np
import matplotlib.pyplot as plt


def h_d(n, omega1, omega2):
    '''
    :param n: n-th index of h_d
    :param omega1: limit frequency 1
    :param omega2: limit frequency 2
    :return: h_d at index n
    '''
    '''h_d calculated by back transformation of ideal filter characteristics '''#
    if n == 0:
        return 0.2 + 0.8
    else:
        return 0.8*(np.sinc(n*omega2)-np.sinc(n*omega1))

def optimizer(N, omega1, omega2):
    '''
    Find the optimized coefficients for the filter
    a0, a1, ..., an
    :param N: Filter order
    :param omega1: limit frequency 1
    :param omega2: limit frequency 2
    :return: np.array of filter coefficients
    '''

    a = np.array(np.zeros(N+1))
    '''a0 = h_d(0) ->  used for a1, a2, .. an '''
    a[0] = h_d(0, omega1, omega2)
    for n in range(1, N):
        '''an = h_d(n)'''
        a[n] = h_d(n, omega1, omega2)
    return a

def H(N, omega1, omega2, steps):
    '''
    Calculate plot data the the filter behaviour H(e^jw)
    :param N: Filter order
    :param omega1: limit frequency 1
    :param omega2: limit frequency 2
    :param steps: fractal of PI steps to be plotted
    :return: x-axis and y-axis data for the filter behaviour H(e^jw)
    '''
    '''Get optimized coefficients'''
    a = optimizer(N, omega1, omega2)
    '''Generate an x-axis for plotting from -PI to PI'''
    xaxis = np.arange(-np.pi, np.pi, np.pi / steps)
    H = np.array(np.zeros(len(xaxis)))
    '''Calculate H(e^jw) according to problem filter desciption'''
    for w in range(len(xaxis)):
        '''Sum over k to N'''
        H[w] = a[0]
        for k in range(1, N+1):
            H[w] = H[w] + 2 * a[k] * np.cos(k * xaxis[w])
    return xaxis, H

if __name__ == '__main__':
    print("Authors:\n"
          "\tName: David Schweighofer \tMtr.Nr.: 11771131\n"
          "\tName: Wolfgang Höller \tMtr.Nr.: 11771132")
    '''Input parameters'''
    omega1 = np.pi/4
    omega2 = 3*np.pi/4
    filter_order = [4, 10, 100]
    '''Generate all plots for the given filter orders'''
    for N in filter_order:
        x_axis, y_axis = H(N, omega1, omega2, 1000)
        plt.plot(x_axis, y_axis, label=("N={}".format(N)))
    plt.legend()
    xtick = np.arange(-np.pi, np.pi, np.pi / 4)
    plt.xticks([-np.pi,-np.pi*3/4, -np.pi*2/4, -np.pi*1/4,0, np.pi*1/4, np.pi*2/4, np.pi*3/4, np.pi],
               [r'$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$',  r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
               )
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    plt.xlabel('\u03A9')
    plt.ylabel('$H(exp^j\u03A9)')
    plt.title('$H(exp^j\u03A9) filters')
    plt.show()
    print("Conclusion:\n"
          "\t-> higher filter orders show better cutoff performance for needed frequency band\n"
          "Findings:\n"
          "\t-> Scaling is not correct\n"
          "\t-> Frequencies are not exactly in the set boundaries\n"
          )