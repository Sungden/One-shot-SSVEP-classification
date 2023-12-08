import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt

# def ck_signal_nh(f, fs, phase, tlen, num_of_harmonics):
#     p = tlen
#     TP = np.arange(0, p / fs, 1 / fs)
#     ref_signal = np.zeros((p,2*num_of_harmonics))

#     for h in range(num_of_harmonics):
#         Sinh1 = np.sin(2 * np.pi * h * f * TP + h * phase)
#         ref_signal[:,2*h]=Sinh1
#         Cosh1 = np.cos(2 * np.pi * h * f * TP + h * phase)
#         ref_signal[:,2*h+1]=Cosh1
#     return ref_signal.T


def ck_signal_nh(f, fs, phase, tlen, num_of_harmonics):
    p = tlen
    TP = np.arange(0, p/fs, 1/fs)
    ref_signal = np.zeros((2 * num_of_harmonics, len(TP)))

    for h in range(1, num_of_harmonics + 1):
        Sinh1 = np.sin(2 * np.pi * h * f * TP + h * phase)
        Cosh1 = np.cos(2 * np.pi * h * f * TP + h * phase)
        ref_signal[2 * h - 2, :] = Sinh1
        ref_signal[2 * h - 1, :] = Cosh1
    return ref_signal

def calculate_ssvep_template(dataset_no):
    if dataset_no == 1:
        str_dir = '/data/Bench/'
        ch_used =[47, 53, 54, 55, 56, 57, 60, 61,62] # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
        sti_f = loadmat("/data/Bench/Freq_Phase.mat")['freqs'][0]
        n_sti = len(sti_f)  # number of stimulus frequencies
        target_order = range(n_sti)
        sti_f = sti_f[target_order]
        num_of_subj = 35
        latencyDelay = 0.14  # latency
        Fs = 250
        signalLength=5
    elif dataset_no == 2:
        str_dir = '/data/BETA/'
        ch_used = [47, 53, 54, 55, 56, 57, 60, 61,62] # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
        sti_f = loadmat("/data/BETA/Freqs_Beta.mat")['freqs'][0]
        n_sti = len(sti_f)  # number of stimulus frequencies
        target_order = range(n_sti)
        sti_f = sti_f[target_order]
        num_of_subj = 70
        latencyDelay = 0.13  # latency
        Fs = 250
        signalLength=3
    else:
        str_dir ='/data/12JFPM/'
        ch_used = [0,1, 2, 3, 4, 5, 6, 7]  # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
        sti_f = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
        n_sti = len(sti_f)  # number of stimulus frequencies
        target_order = range(n_sti)
        sti_f = sti_f[target_order]
        num_of_subj = 10
        latencyDelay = 0.35  # 0.15;
        Fs = 256
        signalLength=3

    bandpass = [7, 100]
    # Design the Butterworth bandpass filter
    order = 4
    low_cutoff = bandpass[0] / (Fs / 2)
    high_cutoff = bandpass[1] / (Fs / 2)
    b1, a1 = butter(order, [low_cutoff, high_cutoff], btype='band')

    subj = [{'ssvep_template': np.zeros((len(ch_used), (signalLength-1) * Fs, n_sti))} for _ in range(num_of_subj)]
    # for item in subj:
    #     print(item,'777777777777')
    for sn in range(num_of_subj):
        if dataset_no == 1:
            data = loadmat(str_dir + 'S' + str(sn+1) + '.mat')
            eeg = data['data'][ch_used,
                  int(0.5 * Fs + latencyDelay * Fs):int(0.5 * Fs + latencyDelay * Fs) + 4 * Fs, :]
        elif dataset_no == 2:
            data = loadmat(str_dir + 'S' + str(sn+1) + '.mat')
            eegdata = data['data'][0][0][0]
            data = np.transpose(eegdata, (0, 1, 3, 2))
            eeg = data[ch_used,
                  int(0.5 * Fs + latencyDelay * Fs):int(0.5 * Fs + latencyDelay * Fs) + 2 * Fs, :]
        else:
            data = loadmat(str_dir + 's' + str(sn+1) + '.mat')
            eeg = np.double(np.transpose(data['eeg'], (1, 2, 0, 3)))
            eeg = eeg[ch_used,
                  int(latencyDelay * Fs) :int(latencyDelay * Fs) + 2 * Fs, :]

        d1_, d2_, d3_, d4_ = eeg.shape #(9, 1000, 40, 6)
        d1, d2, d3, d4 = d3_, d4_, d1_, d2_

        SSVEPdata = np.zeros((d3, d4, d2))#(9, 1000, 6)
        for i in range(d1):
            for j in range(d2):
                y0 = eeg[:, :, i, j].reshape((d3, d4))
                y = np.zeros_like(y0)
                for ch_no in range(1,d3+1):
                    # CAR
                    new_array = np.delete(range(len(ch_used)), ch_no-1)
                    y0[ch_no-1, :] = y0[ch_no-1, :] - np.mean(y0[new_array.tolist()], axis=0)
                    y[ch_no-1, :] = filtfilt(b1, a1, y0[ch_no-1, :])
                
                SSVEPdata[:, :, j] = y.reshape((d3, d4))
            mu_ssvep = np.mean(SSVEPdata, axis=2)
            subj[sn]['ssvep_template'][:, :, i] = mu_ssvep

        subj[sn]['ssvep_template'] = subj[sn]['ssvep_template'][:, :, target_order]

    filename = __file__

    if dataset_no == 1:
        savemat('th_ssvep_template_for_stcca.mat', {'subj': subj, 'filename': filename})
    elif dataset_no == 2:
        savemat('beta_ssvep_template_for_stcca.mat', {'subj': subj, 'filename': filename})
    else:
        savemat('ucsd_ssvep_template_for_stcca.mat', {'subj': subj, 'filename': filename})


