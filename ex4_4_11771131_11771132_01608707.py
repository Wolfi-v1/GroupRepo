'''
Authors:
    Name: David Schweighofer Mtr.Nr.: 11771131
    Name: Wolfgang Höller Mtr.Nr.: 11771132
    Name: Bernhard Grill Mtr.Nr.:01608707
'''

import numpy as np
import matplotlib.pyplot as plt

def generate_random_v_k(snr):
    '''
    :param snr: Signal to noise ration used for weighting
    :return: random normalized complex noise weighted with SNR
    '''
    snr_dec = 10**(snr/10)
    sigma = np.sqrt(1/snr_dec)
    vk = np.random.randn() + 1j * np.random.randn()
    # Normalization and weighting with SNR
    return vk*sigma/np.absolute(vk)

def generate_random_h():
    '''
    :return: random normalized complex channel
    '''
    h = np.random.randn() + 1j * np.random.randn()
    # Normalization
    return h / np.absolute(h)

def calc_y_k(k, h, eps , snr):
    '''
    :param k: symbol index
    :param h: complex values channel
    :param eps: frequency offset in rad
    :param snr: signal to noise ration in dB
    :return: y_k
    - v_k is random for symbol index k & k+1
    - h is constant for all simulations of one channel
    '''
    # Generate 2 random noise samples with given SNR
    v_k = generate_random_v_k(snr)
    v_k_1 = generate_random_v_k(snr)
    # Calculate r_k & r_k+1
    r_k = h*np.exp(1j * eps * k)+v_k
    r_k_1 = h*np.exp(1j * eps * (k+1))+v_k_1
    return r_k * np.conj(r_k_1)

def calc_g_m(h_set, snr ):
    gm_set = np.empty(len(h_set), dtype=complex)
    snr_dec = 10 ** (snr / 10)
    sigma2 = 1 / snr_dec
    for i in range(len(h_set)):
        gm_set[i] = np.absolute(h_set[i]) **2/(2 * np.absolute(h_set[i]) **2 + sigma2)
    return gm_set


def prob_4_4_1(eps, snr):
    '''
    :param eps: frequency offset in rad
    :param snr: signal to noise ration in dB
    :return: -arg(y_k) computed for problem 4_4_1
    '''
    yk = calc_y_k(k=1, h=generate_random_h(), eps=eps, snr=snr)
    return -np.angle(yk)

def prob_4_4_2(eps, snr):
    '''
    :param eps: frequency offset in rad
    :param snr: signal to noise ration in dB
    :return: -arg(y_k) computed for problem 4_4_2
    '''
    K = 5
    h_run = generate_random_h()
    yk = np.empty(K - 1, dtype=complex)
    for k in range(1, K):
        '''run from k = 1,..,4 -> 
        r_k([1,5]) will be used. 
        y_k([1,4]) will be generated'''
        yk[k - 1] = calc_y_k(k=k, h=h_run, eps=eps, snr=snr)
    return -np.angle(np.sum(yk))

def prob_4_4_3(eps, snr):
    '''
    :param eps: frequency offset in rad
    :param snr: signal to noise ration in dB
    :return: -arg(y_k) computed for problem 4_4_3
    '''
    # number of channels
    M = 4
    h_set = np.empty(M, dtype=complex)
    y_k = np.empty(M, dtype=complex)
    # generate M random channels
    for ch in range(M):
        h_set[ch] = generate_random_h()
    # For one symbol (k=1) generate yk outputs for all channels
    for i in range(M):
        y_k[i] = calc_y_k(k=1, h=h_set[i], eps=eps, snr=snr)
    return -np.angle(np.sum(y_k))

def prob_4_4_4(eps, snr):
    '''
    :param eps: frequency offset in rad
    :param snr: signal to noise ration in dB
    :return: -arg(y_k) computed for problem 4_4_4
    '''
    # number of channels
    M = 4
    h_set = np.empty(M, dtype=complex)
    y_k = np.empty(M, dtype=complex)
    # generate M random channels
    for ch in range(M):
        h_set[ch] = generate_random_h()
    # Calculate g_m out of the channel characteristics
    g_m = calc_g_m(h_set=h_set,snr=snr)
    # For one symbol (k=1) generate yk outputs for all channels
    for i in range(M):
        y_k[i] = g_m[i]*calc_y_k(k=1, h=h_set[i], eps=eps, snr=snr)
    return -np.angle(np.sum(y_k))

if __name__ == '__main__':
    print("Authors:\n"
          "\tName: David Schweighofer \tMtr.Nr.: 11771131\n"
          "\tName: Wolfgang Höller \tMtr.Nr.: 11771132\n"
          "\tName: Bernhard Grill \tMtr.Nr.:01608707")
    '''Input parameters'''
    set_eps = np.pi/4
    sim_runs = 2000
    # Define SNR for initial tests
    set_snr = 0
    '''4.4.1'''
    avrg_eps = 0
    for run in range(sim_runs):
        avrg_eps = avrg_eps + prob_4_4_1(eps=set_eps, snr=set_snr)
    avrg_eps = avrg_eps/sim_runs
    print('Problem 4.4.1: \n'
          '\teps_estimate_avrg = {:.5f} \u03C0 \n'
          '\teps_expected = {:.2f} \u03C0 \n'
          '\tSNR = {} dB'.format(avrg_eps/np.pi, set_eps/np.pi, set_snr))

    '''4.4.2'''
    avrg_eps = 0
    for run in range(sim_runs):
        avrg_eps = avrg_eps + prob_4_4_2(eps=set_eps, snr=set_snr)
    avrg_eps = avrg_eps / sim_runs
    print('Problem 4.4.2:\n'
          '\teps_estimate_avrg = {:.5f} \u03C0 \n'
          '\teps_expected = {:.2f} \u03C0 \n'
          '\tSNR = {} dB'.format(avrg_eps/np.pi, set_eps/np.pi, set_snr))

    '''4.4.3'''
    avrg_eps = 0
    for run in range(sim_runs):
        avrg_eps = avrg_eps + prob_4_4_3(eps=set_eps, snr=set_snr)
    avrg_eps = avrg_eps / sim_runs
    print('Problem 4.4.3:\n'
          '\teps_estimate_avrg = {:.5f} \u03C0 \n'
          '\teps_expected = {:.2f} \u03C0 \n'
          '\tSNR = {} dB'.format(avrg_eps/np.pi, set_eps/np.pi, set_snr))

    '''4.4.4'''
    avrg_eps = 0
    for run in range(sim_runs):
        avrg_eps = avrg_eps + prob_4_4_4(eps=set_eps, snr=set_snr)
    avrg_eps = avrg_eps / sim_runs
    print('Problem 4.4.4: \n'
          '\teps_estimate_avrg = {:.5f} \u03C0 \n'
          '\teps_expected = {:.2f} \u03C0 \n'
          '\tSNR = {} dB'.format(avrg_eps/np.pi, set_eps/np.pi, set_snr))

    '''4.4.5'''
    x_snr = np.arange(-20, 20, 1)
    # Define 2-D arrays for easier iteration over different functions
    avrg_err = np.zeros(shape=(4, len(x_snr)))
    y_sims = np.zeros(shape=(4, sim_runs))
    # Define function array on which the simulation runs are executed
    functions = [prob_4_4_1, prob_4_4_2, prob_4_4_3, prob_4_4_4]
    # iterate over x-axis (SNR)
    for x in range(len(x_snr)):
        # for each SNR value calculate "sim_runs" errors and average them after the simulations
        for sim in range(sim_runs):
            # f_index is used to select the functions from the set
            for f_index in range(4):
                est_eps = functions[f_index](eps=set_eps, snr=x_snr[x])
                y_sims[f_index][sim] = np.absolute(est_eps - set_eps) ** 2
        # Averaging done for one simulation run per SNR point
        for y in range(len(y_sims)):
            avrg_err[y][x] = np.sum(y_sims[y]) / len(y_sims[y])
    # add plots to the plotter with the according label of the function for the label
    for y in range(4):
        plt.scatter(x_snr, avrg_err[y], label=functions[y].__name__, marker='x')
    # Plot facelift
    plt.yscale("log")
    plt.grid(axis='x', color='0.8')
    plt.grid(axis='y', color='0.8')
    plt.xlabel('SNR in dB')
    plt.ylabel('$| \u03B5_{est} - \u03B5|^2$')
    plt.title('Frequency offset problem')
    plt.legend()
    plt.show()
    print("Conclusion from the plot:\n"
          "\t4.4.1 -> performs worse than 4.4.2 & 4.4.3 & 4.4.4 because it uses only 1 channel and 1 symbol\n"
          "\t4.4.2 -> uses 5 symbols  and 1 channel to improve the error for lower SNRs\n"
          "\t4.4.3 -> uses 4 channels and 1 symbol to improve the error for lower SNRs\n"
          "\t\t4.4.2 and 4.4.3 are similar in performance as\n"
          "\t\t\t one time more channels (receivers)\n"
          "\t\t\t one time more symbols are used\n"
          "\t\t therefore it shows that the SNR can be improved in equal ways by either using more channels or more symbols\n"
          "\t4.4.4 -> does not show an improvement to 4.4.2 and 4.4.3\n"
          "\t\tReason: g_m is a real valued coefficient -> no impact on phase -> no impact on the phase error performace\n"
          "\t\t \tg_m can be helpfull if different channel attenuations are present ant therefore weighten the impact of different channels"
          )



