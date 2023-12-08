import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import butter, cheby1, filtfilt
from calculate_template import ck_signal_nh
from sklearn.cross_decomposition import CCA
from scipy.linalg import qr, svd, inv,solve,pinv
import logging,os
from typing import Union, Optional, Tuple
from numpy import ndarray
import scipy.linalg as slin
from sklearn.utils import check_X_y
from scipy.io import loadmat


def canoncorr_2(X:np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    
    returns: A,B,r,U,V 
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X,0)
    Y = Y - np.mean(Y,0)

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1,T11,perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0,0])))).eps*max([n,p1]))

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:,:rankX]
        T11 = T11[rankX,:rankX]

    Q2,T22,perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0,0])))).eps*max([n,p2]))

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:,:rankY]
        T22 = T22[:rankY,:rankY]

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX,rankY)
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:,:d] * np.sqrt(n-1)
    B = inv(T22) @ M[:,:d] * np.sqrt(n-1)
    r = D[:d]
    # remove roundoff errs
    r[r>=1] = 1
    r[r<=0] = 0

    if not fullReturn:
        return r
    # Put coefficients back to their full size and their correct order
    A[perm1,:] = np.vstack((A, np.zeros((p1-rankX,d))))
    B[perm2,:] = np.vstack((B, np.zeros((p2-rankY,d))))

    return A, B, r





# below is also ok
def qr_remove_mean(X: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Remove column mean and QR decomposition 

    Parameters
    ----------
    X : ndarray
        (M * N)

    Returns
    -------
    Q : ndarray
        (M * K)
    R : ndarray
        (K * N)
    P : ndarray
        (N,)
    """
    
    X_remove_mean = X - np.mean(X,0)
    
    Q, R, P = slin.qr(X_remove_mean, mode = 'economic', pivoting = True)
    
    return Q, R, P

def mldivide(A: ndarray,
             B: ndarray) -> ndarray:
    """
    A\B, Solve Ax = B

    Parameters
    ----------
    A : ndarray
    B : ndarray

    Returns
    -------
    x: ndarray
    """
    
    return slin.pinv(A) @ B

def canoncorr(X: ndarray, 
              Y: ndarray,
              force_output_UV: Optional[bool] = False) -> Union[Tuple[ndarray, ndarray, ndarray], ndarray]:
    """
    Canonical correlation analysis following matlab

    Parameters
    ----------
    X : ndarray
    Y : ndarray
    force_output_UV : Optional[bool]
        whether calculate and output A and B
    
    Returns
    -------
    A : ndarray
        if force_output_UV, return A
    B : ndarray
        if force_output_UV, return B
    r : ndarray
    """
    n, p1 = X.shape
    _, p2 = Y.shape
    
    Q1, T11, perm1 = qr_remove_mean(X)
    Q2, T22, perm2 = qr_remove_mean(Y)
    svd_X = Q1.T @ Q2
    if svd_X.shape[0]>svd_X.shape[1]:
        full_matrices=False
    else:
        full_matrices=True
        
    L, D, M = slin.svd(svd_X,
                     full_matrices=full_matrices,
                     check_finite=False,
                     lapack_driver='gesvd')
    M = M.T
    
    r = D
    
    if force_output_UV:
        A = mldivide(T11, L) * np.sqrt(n - 1)
        B = mldivide(T22, M) * np.sqrt(n - 1)
        A_r = np.zeros(A.shape)
        for i in range(A.shape[0]):
            A_r[perm1[i],:] = A[i,:]
        B_r = np.zeros(B.shape)
        for i in range(B.shape[0]):
            B_r[perm2[i],:] = B[i,:]

        return A_r, B_r, r
    else:
        return r


def fun_stcca(f_idx, num_of_trials, TW, dataset_no):
    if dataset_no == 1:
        ch_used =  [47, 53, 54, 55, 56, 57, 60, 61,62]  # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
        pha_val =loadmat("/data/Bench/Freq_Phase.mat")['phases'][0]
        sti_f = loadmat("/data/Bench/Freq_Phase.mat")['freqs'][0]
        n_sti = len(sti_f)  # number of stimulus frequencies
        target_order = range(n_sti)
        sti_f = sti_f[target_order]
        num_of_subj = 35
        latencyDelay = 0.14
        num_class = 40
        ch = 9
        Fs = 250
        n_trial = 6
        signalLength=5
        str_dir = '/data/Bench/'
        name='bench_feature_'
    elif dataset_no == 2:
        ch_used =  [47, 53, 54, 55, 56, 57, 60, 61,62] # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
        pha_val = np.array([1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
         1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1]) * np.pi
        sti_f = loadmat("/data/BETA/Freqs_Beta.mat")['freqs'][0]
        n_sti = len(sti_f)  # number of stimulus frequencies
        target_order = range(n_sti)#np.argsort(sti_f)
        sti_f = sti_f[target_order]
        num_of_subj = 70
        latencyDelay = 0.13
        num_class = 40
        ch = 9
        Fs = 250
        n_trial = 4
        signalLength=3
        str_dir = '/data/BETA/'
        name='beta_feature_'
    else:
        ch_used = [0,1, 2, 3, 4, 5, 6, 7] # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
        pha_val = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5]) * np.pi
        sti_f = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
        n_sti = len(sti_f)  # number of stimulus frequencies
        #target_order = np.argsort(sti_f)
        target_order=range(n_sti)
        sti_f = sti_f[target_order]
        num_of_subj = 10
        latencyDelay = 0.35  # or 0.35?
        num_class = 12
        ch = 8
        Fs = 256
        n_trial = 15
        signalLength=3
        str_dir ='/data/12JFPM/'
        name='ucsd_feature_'
      

    temp_len = int(1 * Fs)
    num_of_harmonics = 5  # for all cca
    num_of_subbands = 5  # for filter bank analysis

    # butterworth filter
    bandpass = [7, 70]
    Wn = [bandpass[0] / (Fs / 2), bandpass[1] / (Fs / 2)]
    b1, a1 = butter(4, Wn, btype='band')

    b2 = np.zeros((num_of_subbands, 9))
    a2 = np.zeros((num_of_subbands, 9))

    # Chebyshev filter
    for k in range(1, num_of_subbands + 1):
        bandpass1 = [8 * k, 90]
        Wn_cheby = [bandpass1[0] / (Fs / 2), bandpass1[1] / (Fs / 2)]
        b2[k-1, :], a2[k-1, :] = cheby1(4, 1, Wn_cheby, 'bandpass')
    
    # Calculate FB_coef0
    FB_coef0 = np.power(np.arange(1, num_of_subbands + 1), -1.25) + 0.25
    if dataset_no == 1:
        ssvep_template_for_stcca = loadmat('th_ssvep_template_for_stcca.mat')
    elif dataset_no == 2:
        ssvep_template_for_stcca = loadmat('beta_ssvep_template_for_stcca.mat')
    else:
        ssvep_template_for_stcca = loadmat('ucsd_ssvep_template_for_stcca.mat')


    sig_len = int((signalLength-1)*Fs)
    subj = [{'subband': [{'ssvep_template': np.zeros((len(ch_used), (signalLength-1) * Fs, n_sti))} for _ in range(num_of_subbands)]} for _ in range(num_of_subj)]
    subj_1 = [{'subband': [{'sf': np.zeros((len(ch_used)))} for _ in range(num_of_subbands)]} for _ in range(num_of_subj)]
    subj_new = [{'subband': [{'filtered_ssvep_template': np.zeros((num_class, sig_len))} for _ in range(num_of_subbands)]} for _ in range(num_of_subj)]
    for k in range(num_of_subbands):
        for sn in range(num_of_subj):
            temp = None
            ref = None
            for m in range(num_class):
                tmp = ssvep_template_for_stcca['subj'][0][sn]['ssvep_template'][0][0][:, :, m]
                tmp_sb = np.zeros_like(tmp)
                for ch_no in range(ch ):
                    tmp_sb[ch_no , :] = filtfilt(b2[k, :], a2[k , :], tmp[ch_no, :])           
                subj[sn]['subband'][k]['ssvep_template'][:, :, m] = tmp_sb

                temp = tmp_sb if temp is None else np.append(temp, tmp_sb, axis=1)
                ref0 = ck_signal_nh(sti_f[m], Fs, pha_val[m], sig_len, num_of_harmonics)
                ref = ref0 if ref is None else np.append(ref, ref0, axis=1)

            # print(temp.shape,ref.shape)
            W_x, W_y, _= canoncorr_2(temp.T, ref.T,True)
            subj_1[sn]['subband'][k]['sf'] = W_x[:, 0]
            
            for m in range(num_class):
                ssvep_temp = subj[sn]['subband'][k]['ssvep_template'][:, :, m ]
                subj_new[sn]['subband'][k ]['filtered_ssvep_template'][m , :] = np.dot(W_x[:, 0].T, ssvep_temp)

    sub_idx = np.arange(num_of_subj)
    features_stCCA = np.zeros((num_of_subj, n_trial, n_trial - 1, num_class, num_class))
    sub_acc=[]
    for sn in range(num_of_subj):
        if dataset_no == 1:
            data = loadmat(str_dir + 'S' + str(sn+1) + '.mat')
            eeg = data['data'][ch_used,int(0.5 * Fs + latencyDelay * Fs):int(0.5 * Fs + latencyDelay * Fs) + 4 * Fs, :]
        elif dataset_no == 2:
            data = loadmat(str_dir + 'S' + str(sn+1) + '.mat')
            eegdata = data['data'][0][0][0]
            data = np.transpose(eegdata, (0, 1, 3, 2))
            eeg = data[ch_used,int(0.5 * Fs + latencyDelay * Fs):int(0.5 * Fs + latencyDelay * Fs) + 2 * Fs, :]
        else:
            data = loadmat(str_dir + 's' + str(sn+1) + '.mat')
            eeg = np.double(np.transpose(data['eeg'], (1, 2, 0, 3)))
            eeg = eeg[ch_used,int(latencyDelay * Fs) :int(latencyDelay * Fs) + 2 * Fs, :]

        d1_, d2_, d3_, d4_ = eeg.shape
        d1, d2, d3, d4 = d3_, d4_, d1_, d2_
        no_of_class = d1

        subband_signal = [{'SSVEPdata': np.zeros((len(ch_used), (signalLength-1) * Fs,d2,d1))}  for _ in range(num_of_subbands)]
        for i in range(d1):
            for j in range(d2):
                y0 = eeg[:, :, i, j].reshape((d3, d4))
                y = np.zeros_like(y0)
                for ch_no in range(d3):
                    # CAR
                    y0[ch_no, :] = y0[ch_no, :] - np.mean(y0[np.concatenate([np.arange(0, ch_no), np.arange(ch_no+1, d3)])], axis=0)
                    y[ch_no, :] = filtfilt(b1, a1, y0[ch_no, :])

                for sub_band in range(num_of_subbands):
                    y_sb = np.zeros(y.shape)
                    for ch_no in range(d3 ):
                        y_sb[ch_no, :] = filtfilt(b2[sub_band, :], a2[sub_band, :], y[ch_no, :])

                    subband_signal[sub_band]['SSVEPdata'][:, :, j, i] = np.reshape(y_sb, (d3, d4))

        # Initialization
        TW_p = np.round(TW * Fs)
        n_run = d2
        for sub_band in range( num_of_subbands):
            subband_signal[sub_band]['SSVEPdata'] = subband_signal[sub_band]['SSVEPdata'][:, :, :, target_order]

        FB_coef = np.dot(FB_coef0.reshape((-1, 1)), np.ones((1, n_sti)))
        n_correct =0

        # Classify
        seq_0 = np.zeros((d2, num_of_trials))
        features_subject = np.zeros((d2, n_trial - 1, no_of_class, no_of_class))

        for run in range( d2 ):
            if num_of_trials == 1:
                seq1 = run
            elif num_of_trials == d2 :
                seq1 = np.arange(n_run)
                seq1 = seq1[seq1 != run]
            else:
                isOK = 0
                while isOK == 0:
                    seq = np.random.permutation(np.arange(d2))
                    seq1 = seq[:num_of_trials]
                    seq1 = np.sort(seq1)
                    if len(np.where(np.sum((np.outer(seq1, np.ones(d2)) - seq_0) ** 2, axis=0) == 0)[0]) == 0:
                        isOK = 1

            idx_traindata = [seq1]
            idx_testdata = np.arange(n_run)
            idx_testdata = np.delete(idx_testdata, seq1)
            subband_signal_new = [{'signal_template': np.zeros((no_of_class,len(ch_used), (signalLength-1) * Fs))}  for _ in range(num_of_subbands)]
            for ii in range(no_of_class):
                for kk in range(num_of_subbands):
                    if len(idx_traindata)> 1:
                        subband_signal_new[kk]['signal_template'][ii, :, :] = np.mean(subband_signal[kk ]['SSVEPdata'][:, :, idx_traindata , ii ], axis=2)
                    else:
                        subband_signal_new[kk ]['signal_template'][ii , :, :] = np.squeeze(subband_signal[kk ]['SSVEPdata'][:, :, idx_traindata , ii ])

            # Training stage:
            # Find the intra-subject spatial filter
            for kkk in range(num_of_subbands ):
                target_ssvep = np.zeros((d3, int(len(f_idx)*Fs)))
                target_ref = np.zeros((num_of_harmonics*2, int(len(f_idx)*Fs)))
                for fn in range(len(f_idx)):
                    tmp1 = np.reshape(subband_signal_new[kkk]['signal_template'][int(f_idx[fn]) , :, 0:temp_len], (d3, temp_len))
                    ref1 = ck_signal_nh(sti_f[int(f_idx[fn]) ], Fs, pha_val[int(f_idx[fn])], temp_len,num_of_harmonics)
                    target_ssvep[:, Fs*fn:Fs*fn+Fs] = tmp1
                    target_ref[:, Fs*fn:Fs*fn+Fs] = ref1

                W_x, W_y, _= canoncorr_2(target_ssvep.T, target_ref.T,True)
                subband_signal_new[kkk ]['Wx'] = W_x[:, 0]
                subband_signal_new[kkk]['Wy'] = W_y[:, 0]
                tar_subj_sf = W_x[:, 0]

                # Find the weights for constructing the inter-subject SSVEP template
                source_idx = sub_idx
                source_idx = source_idx[source_idx != sn]
                source_ssvep_temp0 = np.zeros((len(source_idx), temp_len * len(f_idx)))
                source_ssvep_temp = np.zeros((1, d4))

                for ssn in range(len(source_idx) ):
                    stmp = np.zeros((1, Fs * len(f_idx)))
                    for fn in range(len(f_idx) ):
                        tmp2 = subj_new[source_idx[ssn ]]['subband'][kkk ]['filtered_ssvep_template'][int(f_idx[fn ]) , 0:temp_len]
                        stmp[:, fn* temp_len:fn * temp_len+Fs] = tmp2

                    source_ssvep_temp0[ssn, :] = stmp
                
                X = source_ssvep_temp0.T
                Y = (np.expand_dims(tar_subj_sf,axis=1).T @ target_ssvep).T#(3072, 1) 
                W0 = slin.pinv(X.T @ X) @ X.T @ Y
                W_template1 = W0[:, 0]
                if np.sum(np.abs(W_template1)) == 0:
                    W_template1 = np.ones((1, 34))

                for ssn in range(len(source_idx)):
                    source_ssvep_temp = source_ssvep_temp + W_template1[ssn] * subj_new[source_idx[ssn]]['subband'][kkk ]['filtered_ssvep_template']

                source_ssvep_temp = source_ssvep_temp / np.sum(np.abs(W_template1))
                subband_signal[kkk ]['source_subject_filtered_template'] = source_ssvep_temp

            # Testing stage:
            feature_class = np.zeros((no_of_class, no_of_class))
            feature = np.zeros((len(idx_testdata), no_of_class, no_of_class))
            for run_test in range(len(idx_testdata) ):

                sig_len = int(TW_p)
                print('stCCA Processing TW %fs, No. calibration %d, No.crossvalidation %d' % (TW, len(f_idx) * len(idx_traindata), idx_testdata[run_test ]))

                for iii in range(no_of_class ):
                    itR=np.zeros((num_of_subbands,no_of_class))
                    for sub_band in range(num_of_subbands):
                        test_signal = subband_signal[sub_band ]['SSVEPdata'][:, 0:sig_len,idx_testdata[run_test] , iii ]
                        for jj in range(no_of_class):
                            template = subband_signal[sub_band]['source_subject_filtered_template'][jj ,0:sig_len]
                            ref = ck_signal_nh(sti_f[jj ], Fs, pha_val[jj ], sig_len, num_of_harmonics)

                            # print(subband_signal_new[sub_band ]['Wx'].T.shape, test_signal.shape,subband_signal_new[sub_band ]['Wy'].T.shape,ref.shape,'3333333333333')#(8,) (8, 128) (10,) (10, 128)
                            r1 = np.corrcoef(subband_signal_new[sub_band ]['Wx'].T @ test_signal,subband_signal_new[sub_band ]['Wy'].T @ ref)
                            r2 = np.corrcoef(subband_signal_new[sub_band]['Wx'].T @ test_signal, template)
                            itR[sub_band , jj ] = np.sign(r1[0, 1]) * r1[0, 1] ** 2 + np.sign(r2[0, 1]) * r2[0, 1] ** 2

                    itR1 = np.sum((itR * FB_coef), axis=0)
                    feature_class[iii, :] = itR1
                    idx = np.argmax(itR1)
                    if idx == iii:
                        n_correct = n_correct + 1
                feature[run_test , :, :] = feature_class

            seq_0[run , :] = seq1
            features_subject[run , :, :, :] = feature
        
        features_stCCA[sn,:,:,:,:]=features_subject
        accuracy = 100 * n_correct / (n_sti * n_run * len(idx_testdata))
        print(accuracy)
        sub_acc.append(accuracy)
        print(sn+1)

    print(sub_acc)
    print(np.mean(sub_acc))
    # save_path= 'feature'
    # if not os.path.exists(save_path):
    #   os.makedirs(save_path)
    # # Save results
    # save_name=name+str(TW)+'s'
    # np.save(os.path.join(save_path,save_name), features_stCCA)
    return features_stCCA,sub_acc
