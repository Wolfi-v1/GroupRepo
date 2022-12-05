'''
Authors:
    Name: David Schweighofer Mtr.Nr.: 11771131
    Name: Wolfgang Höller Mtr.Nr.: 11771132
    Name: Bernhard Grill Mtr.Nr.:01608707
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def get_s(U_):
    s_ = np.zeros(U_, dtype=complex)
    # Define 4-QAM symbol library
    QAM_biblio = [1 + 1j,1 - 1j,-1 -1j,-1+1j]
    for i in range(U_):
        # Select a random symbol out of the library an normalize symbol with sqrt(2)
        s_[i]=QAM_biblio[np.random.randint(0,3)]/np.sqrt(2)
    return s_

def get_H(U_,N_):
    HH_ = np.zeros((U_,N_), dtype=complex)
    for u in range (U_):
        for n in range(N_):
            # Create a random complex channel an normalize symbol with sqrt(2)
            HH_[u][n] = (np.random.randn() + 1j * np.random.randn())/np.sqrt(2)
    return HH_

def get_noise(U_, sigma2):
    n_ = np.zeros(U_, dtype=complex)
    for i in range(U_):
        n_[i] = (np.random.randn() + 1j * np.random.randn())*np.sqrt(sigma2)/np.sqrt(2)
    return n_

def get_W(N_, U_, HH_):
    WW_ = np.zeros((N_, U_), dtype=complex)
    for u in range(U_):
        # HH shape: 3x4 -> H_u shape: 2x4 gives VV shape: 4x4 -> 2D Nullspace
        UU_, SS_, VVh_ = np.linalg.svd(np.delete(HH_, u, 0))
        # choose the last col vec of the nullspace as it corresponds to a zero singular value
        WW_[:, u] = VVh_.conj().T[:, -1]
    return WW_

def get_g(U_, HH_, WW_):
    g_ = np.zeros(U_, dtype=complex)
    for u in range(U_):
        g_[u] = 1/np.matmul(HH_[u,:], WW_[:,u])
    return g_

def min_dist_quantizer(s_hat):
    # return real and imaginary part according to the sign of the respective part
    # -> minimum distace quantizer for a 4-QAM symbol
    return np.sign(np.real(s_hat))+1j*np.sign(np.imag(s_hat))

if __name__ == '__main__':
    print("Authors:\n"
          "\tName: David Schweighofer \tMtr.Nr.: 11771131\n"
          "\tName: Wolfgang Höller \tMtr.Nr.: 11771132\n"
          "\tName: Bernhard Grill \tMtr.Nr.:01608707")
    '''Input parameters'''
    N = 4
    U = 3
    x_snr_range_dec = np.logspace(-1, 2,num=10)
    '''6.4.5 Function testing'''
    # Get random symbols from 4-QAM biblio
    s = get_s(U)
    # Generate random channel matrix
    HH = get_H(U, N)
    # Create random noise
    n = get_noise(U, sigma2=0.1)
    # Compute precoder Matrix
    WW = get_W(N, U, HH)
    # Compute equalizers
    g = get_g(U, HH, WW)
    # Simulate transmition
    y = np.matmul(HH, np.matmul(WW,s))+n
    # Generate equalized estimation
    s_hat = np.multiply(y,g)
    # Feed minimum distance quantizer with s_hat to get most probable original symbol
    s_hat_decided = min_dist_quantizer(s_hat)
    '''6.4.6'''
    # generate a dummy vector for SER values
    SER_vec = np.zeros(len(x_snr_range_dec), dtype=float)
    # random symbols to be transmitted
    # Symbols are constant over the simulation run
    s = get_s(U)
    for x in range(len(x_snr_range_dec)):
        # Create two counters for transmitted symbols and erroneous symbols
        symbol_errors = 0
        symbols_transmitted = 0
        for run in range(2500):
            # Get decimal sigma^2 from SNR
            sigma2_dec = 1 / x_snr_range_dec[x]
            # Generate noise vector
            n_run = get_noise(U, sigma2_dec)
            # Generate channel matrix
            HH_run = get_H(U, N)
            # Calculate precoder matrix for channel HH_run
            WW_run = get_W(N, U, HH_run)
            # Get equalizer values
            g_run = get_g(U, HH_run, WW_run)
            # calculate y for this run
            y_run = np.matmul(HH_run, np.matmul(WW_run, s)) + n_run
            # multiply y with the equalizers
            s_hat_run = np.multiply(y_run, g_run)
            # Let the min. dist. quantizer pick the estimated solution
            s_hat_run_decided = min_dist_quantizer(s_hat_run)
            # Compare all received symbols with the expected symbols
            for i in range(len(s_hat_run)):
                symbols_transmitted = symbols_transmitted + 1
                # Check if the decided symbol is equal to the original symbol
                imag_nok = np.sign(np.imag(s_hat_run_decided[i])) != np.sign(np.imag(s[i]))
                real_nok = np.sign(np.real(s_hat_run_decided[i])) != np.sign(np.real(s[i]))
                if real_nok or imag_nok:
                    symbol_errors = symbol_errors + 1
        SER_vec[x] = symbol_errors/symbols_transmitted
        print("Symbols transmitted: {} , Erroneous symbols: {}, "
              "SER: {:.2f}".format(symbols_transmitted, symbol_errors,  SER_vec[x]))
    plt.plot(x_snr_range_dec, SER_vec, label="MU-MIMO", marker='x')
    '''6.4.7'''
    # generate a dummy vector for SER_ps values
    SER_Ps_vec = np.zeros(len(x_snr_range_dec), dtype=float)
    for x  in range(len(x_snr_range_dec)):
        snr_dec = x_snr_range_dec[x]
        SER_Ps_vec[x] = scipy.special.erfc(np.sqrt(snr_dec))*(1-(1/4 *scipy.special.erfc(np.sqrt(snr_dec))))
    plt.plot(x_snr_range_dec, SER_Ps_vec, label="4-QAM - AWGN", marker='x')
    # Plot facelift
    plt.ylim(0.0008,1)
    plt.xlim(0.1, 100)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which="both", ls="-")
    plt.xlabel('SNR')
    plt.ylabel('SER')
    plt.legend()
    plt.show()