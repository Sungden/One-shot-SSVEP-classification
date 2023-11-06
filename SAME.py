import numpy as np
from numpy import ndarray
from scipy.linalg import eigh, pinv, qr
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from typing import Optional, List, Tuple, Union
from scipy.signal import sosfiltfilt, cheby1, cheb1ord
from joblib import Parallel, delayed

def generate_filterbank(
    passbands: List[Tuple[float, float]],
    stopbands: List[Tuple[float, float]],
    srate: int,
    order: Optional[int] = None,
    rp: float = 0.5,
):
    filterbank = []
    for wp, ws in zip(passbands, stopbands):
        if order is None:
            N, wn = cheb1ord(wp, ws, 3, 40, fs=srate)
            sos = cheby1(N, rp, wn, btype="bandpass", output="sos", fs=srate)
        else:
            sos = cheby1(order, rp, wp, btype="bandpass", output="sos", fs=srate)

        filterbank.append(sos)
    return filterbank



def lst_kernel(S: ndarray, T: ndarray):
    P = T @ S.T @ pinv(S @ S.T)
    return P


class LST(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray):
        X = X.reshape((-1, *X.shape[-2:]))  # n_trials, n_channels, n_samples
        self.classes_ = np.unique(y)
        self.T_ = [np.mean(X[y == label], axis=0) for label in self.classes_]
        return self

    def transform(self, X: ndarray, y: ndarray):
        X = np.copy(X)
        X = X.reshape((-1, *X.shape[-2:]))  # n_trials, n_channels, n_samples
        Ts = np.zeros_like(X)
        for i, label in enumerate(self.classes_):
            Ts[y == label] = self.T_[i]
        P = np.stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(lst_kernel)(S, T) for S, T in zip(X, Ts)
            )
        )
        X = P @ X
        return X

def TRCs_estimation(data, mean_target):
    """source signal estimation using LST [1]_

    author: Ruixin Luo <ruixin_luo@tju.edu.cn>

    Created on: 2023-01-09

    update log:
        2023-09-06 by Ruixin Luo <ruixin_luo@tju.edu.cn>

    Parameters
    ----------
    data:ndarray
        Reference signal, shape(n_channel_1, n_times).
    mean_target:ndarray
        Average template, shape(n_channel_2, n_times).

    Returns
    -------
    data_after:ndarray
        Source signal, shape(n_channel_2, n_times).

    References
    ----------
    .. [1] Chiang, K. J., Wei, C. S., Nakanishi, M., & Jung, T. P. (2021, Feb 11).
           Boosting  template-based ssvep decoding by cross-domain transfer learning. J Neural Eng, 18(1), 016002.

    """

    X_a = data
    X = mean_target

    # Using the least squares method to solve aliasing matrix
    PT = lst_kernel(S=X_a, T=X)
    # source signal estimation
    data_after = PT @ X_a

    return data_after

def get_augment_noiseAfter(fs, f, Nh, n_Aug, mean_temp):
    """Artificially generated signals by SAME

    Parameters
    ----------
    fs : int
        Sampling rate.
    f : float
        Frequency of signal.
    Nh: int
        The number of harmonics.
    n_Aug: int
        The number of generated signals
    mean_temp: ndarray-like (n_channel, n_times)
        Average template.

    Returns
    -------
    data_aug : ndarray-like (n_channel, n_times, n_Aug)
        Artificially generated signals.
    """

    nChannel, nTime = mean_temp.shape
    #  Generate reference signal Yf
    Ts = 1 / fs
    n = np.arange(nTime) * Ts
    Yf = np.zeros((nTime, 2 * Nh))
    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n)
        Yf[:, iNh * 2] = y_sin
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n)
        Yf[:, iNh * 2 + 1] = y_cos

    Z = TRCs_estimation(Yf.T, mean_temp)
    # get vars of Z
    vars = np.zeros((Z.shape[0], Z.shape[0]))
    for i_c in range(nChannel):
        vars[i_c, i_c] = np.var(Z[i_c, :])

    # add noise
    data_aug = np.zeros((nChannel, nTime, n_Aug))
    for i_aug in range(n_Aug):
        # Randomly generated noise
        Datanoise = np.random.multivariate_normal(mean=np.zeros((nChannel)), cov=vars, size=nTime)
        data_aug[:, :, i_aug] = Z + 0.05 * Datanoise.T

    return data_aug


class SAME(BaseEstimator, TransformerMixin):
    """
    source aliasing matrix estimation (SAME).
    we apply SAME before filter bank analysis in the version, to be compatible with MetaBCI.
    -author: Ruixin luo
    -Created on: 2023-01-09
    -update log:
        None

    Parameters
    ----------
    fs : int
        Sampling rate.
    flist : list
        Frequency of all class.
    Nh: int
        The number of harmonics.
    n_Aug: int
        The number of generated signals
    """

    def __init__(self, n_jobs=None, fs=250, flist=None, Nh=5, n_Aug=5):
        self.n_jobs = n_jobs
        self.fs = fs
        self.Nh = Nh
        self.n_Aug = n_Aug
        self.flist = flist

    def fit(self, X: ndarray, y: ndarray):
        X = X.reshape((-1, *X.shape[-2:]))  # n_trials, n_channels, n_samples
        self.classes_ = np.unique(y)
        self.T_ = [np.mean(X[y == label], axis=0) for label in self.classes_]
        return self

    def augment(self):
        X_aug = []
        y_aug = []
        for i, label in enumerate(self.classes_):
            temp = self.T_[i]
            f = self.flist[i]
            data_aug = get_augment_noiseAfter(fs=self.fs, f=f, Nh=self.Nh, n_Aug=self.n_Aug, mean_temp=temp)
            data_aug = np.transpose(data_aug, [2, 0, 1])  # n_aug, n_channel, n_times
            X_aug.append(data_aug)
            y_aug.append(np.ones(self.n_Aug, dtype=np.int32) * label)

        X_aug = np.concatenate(X_aug, axis=0)
        y_aug = np.concatenate(y_aug, axis=0)
        return X_aug, y_aug