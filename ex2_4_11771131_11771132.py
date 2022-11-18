'''
Authors:
    Name: David Schweighofer Mtr.Nr.: 11771131
    Name: Wolfgang Höller Mtr.Nr.: 11771132
'''

import scipy.signal as sc
import matplotlib.pyplot as plt
import numpy as np

def my_deci(A, d):
    '''Problem 2.4.1'''
    # n_max = A(row)/d
    # m_max = A(col)/d
    rows , cols = np.shape(A)
    n = int((rows / d))
    m = int((cols / d))
    B = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            B[i][j] = A[i * d][j * d]
    return B


def my_gauss2D(p, sigma):
    '''Problem 2.4.2'''
    # constant pre factor according to (2.4.2)
    a = 1 / (2 * np.pi * sigma ** 2)
    G = np.zeros(shape=(p + 1, p + 1))
    for n in range(p + 1):
        for m in range(p + 1):
            # formula for the element g_n,m according to (2.4.2)
            b = ((n - p / 2) ** 2 + (m - p / 2) ** 2) / (2 * sigma ** 2)
            G[n][m] = a * np.exp(-b)
    return G

def my_nni(A,d):
    '''Problem 2.4.4'''
    A_rows, A_cols = np.shape(A)
    B_rows = int(A_rows * d)
    B_cols = int(A_cols * d)
    temp = np.zeros(shape=(A_rows, B_cols))
    B = np.zeros(shape=(B_rows, B_cols))
    # stretch the image in x direction by copying each A_col d-1 times
    for i in range(B_rows):
        for j in range(A_rows):
            temp[j][i] = A[j][int(np.floor(i/d))]
    # stretch the image in y direction by copying each A_row d-1 times
    for i in range(B_cols):
        for j in range(B_rows):
            B[i][j] = temp[int(np.floor(i/d))][j]
    return B

def my_gauss2DUp(A,d,p,sigma):
    '''Problem 2.4.5'''
    A_rows, A_cols = np.shape(A)
    n = int(A_rows * d)
    m = int(A_cols * d)
    B = np.zeros(shape=(n, m))
    # Zero padding: insert d-1 zeros between each A_row and A_col
    for i in range(int(np.floor(n/d))):
        for j in range(int(np.floor(m/d))):
            B[j*d][i*d] = A[j][i]
    # Return the convolution of matrix A with the gaussian matrix G
    return sc.convolve2d(A, my_gauss2D(p, sigma), mode='same')

if __name__ == '__main__':
    print("Authors:\n"
          "\tName: David Schweighofer \tMtr.Nr.: 11771131\n"
          "\tName: Wolfgang Höller \tMtr.Nr.: 11771132")
    A = plt.imread('ex2_4_moire_pattern.png')
    d = 2
    p = 6
    sigma = 1
    B1 = my_deci(A, d)
    B2 = my_deci(sc.convolve2d(A, my_gauss2D(p, sigma), mode='same'), d)
    print("Problem 2.4.3: Downsampling \n\tgiven that: d = {}, p = {}, sigma = {}".format(d,p,sigma))
    plt.imshow(A, cmap='gray', interpolation='lanczos')
    plt.title("2.4.3 Original image A")
    plt.show()
    plt.imshow(B1, cmap='gray', interpolation='lanczos')
    plt.title("2.4.3 a) B1 as decimation of A ")
    plt.show()
    plt.imshow(B2, cmap='gray', interpolation='lanczos')
    plt.title("2.4.3 b) B2 as decimation of the convolution of A with G")
    plt.show()
    print("Problem 2.4.3 Conclusion:\n \tThe image B1 suffers from the Aliasing effect")

    A = plt.imread('ex2_4_cameraman.png')
    d = 3
    p = 20
    sigma = 1.6
    B1 = my_nni(A,d)
    B2 = my_gauss2DUp(A,d,p,sigma)
    print("Problem 2.4.6 Upsampling: \n\tgiven that: d = {}, p = {}, sigma = {}".format(d, p, sigma))
    plt.imshow(A, cmap='gray', interpolation='lanczos')
    plt.title("2.4.6 Original image A")
    plt.show()
    plt.imshow(B1, cmap='gray', interpolation='lanczos')
    plt.title("2.4.6 a) B1 by nearest neighbour interpolation of A")
    plt.show()
    plt.imshow(B2, cmap='gray', interpolation='lanczos')
    plt.title("2.4.6 b) B2 by Gaussian interpolation Upsampling")
    plt.show()
    print("Problem 2.4.6 Conclusion: \n \tB1 -> sharper image but looks more pixilated\n"
          "\tB2 -> smoother image look therefore less details visible")