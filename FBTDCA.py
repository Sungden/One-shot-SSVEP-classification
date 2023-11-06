import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from functools import partial
from typing import Optional, List, Tuple, Union, cast
from scipy.linalg import solve
from scipy.signal import sosfiltfilt, cheby1, cheb1ord
from scipy.linalg import eigh, pinv, qr,eig
from scipy.stats import pearsonr
from scipy.sparse import block_diag, identity, vstack, spmatrix
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

def _ged_wong(
    Z: ndarray,
    D: Optional[ndarray] = None,
    P: Optional[ndarray] = None,
    n_components=1,
    method="type1",
):
    if method != "type1" and method != "type2":
        raise ValueError("not supported method type")

    A = Z
    if D is not None:
        A = D.T @ A
    if P is not None:
        A = P.T @ A
    A = A.T @ A
    if method == "type1":
        B = Z
        if D is not None:
            B = D.T @ Z
        B = B.T @ B
        if isinstance(A, spmatrix) or isinstance(B, spmatrix):
            D, W = eigsh(A, k=n_components, M=B)
        else:
            D, W = eigh(A, B)
    elif method == "type2":
        if isinstance(A, spmatrix):
            D, W = eigsh(A, k=n_components)
        else:
            D, W = eigh(A)

    D_exist = cast(ndarray, D)
    ind = np.argsort(D_exist)[::-1]
    D_exist, W = D_exist[ind], W[:, ind]
    return D_exist[:n_components], W[:, :n_components]

def generate_cca_references(
    freqs: Union[ndarray, int, float],
    srate,
    T,
    phases: Optional[Union[ndarray, int, float]] = None,
    n_harmonics: int = 1,
):
    if isinstance(freqs, int) or isinstance(freqs, float):
        freqs = np.array([freqs])
    freqs = np.array(freqs)[:, np.newaxis]
    if phases is None:
        phases = 0
    if isinstance(phases, int) or isinstance(phases, float):
        phases = np.array([phases])
    phases = np.array(phases)[:, np.newaxis]
    t = np.linspace(0, T, int(T * srate))

    Yf = []
    
    for i in range(n_harmonics):
        Yf.append(
            np.stack(
                [
                    np.sin(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                    np.cos(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                ],
                axis=1,
            )
        )
    Yf = np.concatenate(Yf, axis=1)
    return Yf


def robust_pattern(W: ndarray, Cx: ndarray, Cs: ndarray) -> ndarray:
    """Transform spatial filters to spatial patterns based on paper [1]_.

    Parameters
    ----------
    W : ndarray
        Spatial filters, shape (n_channels, n_filters).
    Cx : ndarray
        Covariance matrix of eeg data, shape (n_channels, n_channels).
    Cs : ndarray
        Covariance matrix of source data, shape (n_channels, n_channels).

    Returns
    -------
    A : ndarray
        Spatial patterns, shape (n_channels, n_patterns), each column is a spatial pattern.

    References
    ----------
    .. [1] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging."
           Neuroimage 87 (2014): 96-110.
    """
    # use linalg.solve instead of inv, makes it more stable
    # see https://github.com/robintibor/fbcsp/blob/master/fbcsp/signalproc.py
    # and https://ww2.mathworks.cn/help/matlab/ref/mldivide.html
    A = solve(Cs.T, np.dot(Cx, W).T).T
    return A

 
    
def xiang_dsp_kernel(
    X: ndarray, y: ndarray
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    DSP: Discriminal Spatial Patterns, only for two classes[1]
    -Author: Swolf <swolfforever@gmail.com>
    -Created on: 2021-1-07
    -Update log:

    Parameters
    ----------
    X : ndarray
        EEG data assuming removing mean, shape (n_trials, n_channels, n_samples)
    y : ndarray
        labels of EEG data, shape (n_trials, )

    Returns
    -------
    W: ndarray
        spatial filters, shape (n_channels, n_filters)
    D: ndarray
        eigenvalues in descending order
    M: ndarray
        template for all classes, shape (n_channel, n_samples)
    A: ndarray
        spatial patterns, shape (n_channels, n_filters)

    Notes
    -----
    the implementation removes regularization on within-class scatter matrix Sw.

    References
    ----------
    [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in
        a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    X, y = np.copy(X), np.copy(y)
    labels = np.unique(y)
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    # the number of each label
    n_labels = np.array([np.sum(y == label) for label in labels])
    # average template of all trials
    M = np.mean(X, axis=0)
    # class conditional template
    Ms, Ss = zip(
        *[
            (
                np.mean(X[y == label], axis=0),
                np.sum(
                    np.matmul(X[y == label], np.swapaxes(X[y == label], -1, -2)), axis=0
                ),
            )
            for label in labels
        ]
    )
    Ms, Ss = np.stack(Ms), np.stack(Ss)
    # within-class scatter matrix
    Sw = np.sum(
        Ss
        - n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
        axis=0,
    )
    Ms = Ms - M
    # between-class scatter matrix
    Sb = np.sum(
        n_labels[:, np.newaxis, np.newaxis] * np.matmul(Ms, np.swapaxes(Ms, -1, -2)),
        axis=0,
    )

    D, W = eig(Sb, Sw) #use eig if eigh may lead to numpy.linalg.LinAlgError
    ix = np.argsort(D)[::-1]  # in descending order
    D, W = D[ix], W[:, ix]
    A = robust_pattern(W, Sb, W.T @ Sb @ W)

    return W, D, M, A


def xiang_dsp_feature(
    W: ndarray, M: ndarray, X: ndarray, n_components: int = 1
) -> ndarray:
    """
    Return DSP features in paper [1]
    -Author: Swolf <swolfforever@gmail.com>
    -Created on: 2021-1-07
    -Update log:

    Parameters
    ----------
    W : ndarray
        spatial filters from csp_kernel, shape (n_channels, n_filters)
    M: ndarray
        common template for all classes, shape (n_channel, n_samples)
    X : ndarray
        eeg test data, shape (n_trials, n_channels, n_samples)
    n_components : int, optional
        length of the spatial filters, first k components to use, by default 1

    Returns
    -------
    features: ndarray
        features, shape (n_trials, n_components, n_samples)

    Raises
    ------
    ValueError
        n_components should less than half of the number of channels

    Notes
    -----
    1. instead of meaning of filtered signals in paper [1]_., we directly return filtered signals.

    References
    ----------
    [1] Liao, Xiang, et al. "Combining spatial filters for the classification of single-trial EEG in
        a finger movement task." IEEE Transactions on Biomedical Engineering 54.5 (2007): 821-831.
    """
    W, M, X = np.copy(W), np.copy(M), np.copy(X)
    max_components = W.shape[1]
    if n_components > max_components:
        raise ValueError("n_components should less than the number of channels")
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    features = np.matmul(W[:, :n_components].T, X - M)
    return features

class FilterBank(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        base_estimator: BaseEstimator,
        filterbank: List[ndarray],
        n_jobs: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.filterbank = filterbank
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: Optional[ndarray] = None, **kwargs):
        self.estimators_ = [
            clone(self.base_estimator) for _ in range(len(self.filterbank))
        ]
        X = self.transform_filterbank(X)
        for i, est in enumerate(self.estimators_):
            est.fit(X[i], y, **kwargs)
        # def wrapper(est, X, y, kwargs):
        #     est.fit(X, y, **kwargs)
        #     return est
        # self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        #     delayed(wrapper)(est, X[i], y, kwargs) for i, est in enumerate(self.estimators_))
        return self

    def transform(self, X: ndarray, **kwargs):
        X = self.transform_filterbank(X)
        feat = [est.transform(X[i], **kwargs) for i, est in enumerate(self.estimators_)]
        # def wrapper(est, X, kwargs):
        #     retval = est.transform(X, **kwargs)
        #     return retval
        # feat = Parallel(n_jobs=self.n_jobs)(
        #     delayed(wrapper)(est, X[i], kwargs) for i, est in enumerate(self.estimators_))
        feat = np.concatenate(feat, axis=-1)
        return feat

    def transform_filterbank(self, X: ndarray):
        Xs = np.stack([sosfiltfilt(sos, X, axis=-1) for sos in self.filterbank])
        return Xs


class FilterBankSSVEP(FilterBank):
    """Filter bank analysis for SSVEP."""

    def __init__(
        self,
        filterbank: List[ndarray],
        base_estimator: BaseEstimator,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.filterweights = filterweights
        super().__init__(base_estimator, filterbank, n_jobs=n_jobs)

    def transform(self, X: ndarray):  # type: ignore[override]
        features = super().transform(X)
        if self.filterweights is None:
            return features
        else:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            return np.sum(
                features * self.filterweights[np.newaxis, :, np.newaxis], axis=1
            )

def proj_ref(Yf: ndarray):
    Q, R = qr(Yf.T, mode="economic")
    P = Q @ Q.T
    return P


def aug_2(X: ndarray, n_samples: int, padding_len: int, P: ndarray, training: bool):
    X = X.reshape((-1, *X.shape[-2:]))
    n_trials, n_channels, n_points = X.shape

    # if n_points < padding_len + n_samples:
    #     raise ValueError("the length of X should be larger than l+n_samples.")
    aug_X = np.zeros((n_trials, (padding_len + 1) * n_channels, n_samples))
    if training:
        for i in range(padding_len + 1):
            aug_X[:, i * n_channels : (i + 1) * n_channels, :] = X[
                ..., i : i + n_samples
            ]
    else:
        for i in range(padding_len + 1):
            aug_X[:, i * n_channels : (i + 1) * n_channels, : n_samples - i] = X[
                ..., i:n_samples
            ]
    aug_Xp = aug_X @ P
    aug_X = np.concatenate([aug_X, aug_Xp], axis=-1)
    return aug_X


def tdca_feature(
    X: ndarray,
    templates: ndarray,
    W: ndarray,
    M: ndarray,
    Ps: List[ndarray],
    padding_len: int,
    n_components: int = 1,
    training=False,
):
    rhos = []
    for Xk, P in zip(templates, Ps):
        a = xiang_dsp_feature(
            W,
            M,
            aug_2(X, P.shape[0], padding_len, P, training=training),
            n_components=n_components,
        )
        b = Xk[:n_components, :]
        a = np.reshape(a, (-1))
        b = np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return rhos


class TDCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, padding_len: int, n_components: int = 1):
        self.padding_len = padding_len
        self.n_components = n_components

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        X -= np.mean(X, axis=-1, keepdims=True)
        self.classes_ = np.unique(y)
        self.Ps_ = [proj_ref(Yf[i]) for i in range(len(self.classes_))]

        aug_X_list, aug_Y_list = [], []
        for i, label in enumerate(self.classes_):
            aug_X_list.append(
                aug_2(
                    X[y == label],
                    self.Ps_[i].shape[0],
                    self.padding_len,
                    self.Ps_[i],
                    training=True,
                )
            )
            aug_Y_list.append(y[y == label])

        aug_X = np.concatenate(aug_X_list, axis=0)
        aug_Y = np.concatenate(aug_Y_list, axis=0)
        self.W_, _, self.M_, _ = xiang_dsp_kernel(aug_X, aug_Y)

        self.templates_ = np.stack(
            [
                np.mean(
                    xiang_dsp_feature(
                        self.W_,
                        self.M_,
                        aug_X[aug_Y == label],
                        n_components=self.W_.shape[1],
                    ),
                    axis=0,
                )
                for label in self.classes_
            ]
        )
        return self

    def transform(self, X: ndarray):
        n_components = self.n_components
        X -= np.mean(X, axis=-1, keepdims=True)
        X = X.reshape((-1, *X.shape[-2:]))
        rhos = [
            tdca_feature(
                tmp,
                self.templates_,
                self.W_,
                self.M_,
                self.Ps_,
                self.padding_len,
                n_components=n_components,
            )
            for tmp in X
        ]
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels


class FBTDCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        padding_len: int,
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.padding_len = padding_len
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TDCA(padding_len, n_components=n_components),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels,features
        
       
       
       
def _trca_kernel(X: ndarray):
    """TRCA.
    X: (n_trials, n_channels, n_samples)
    """
    X = np.reshape(X, (-1, *X.shape[-2:]))
    M, C, N = X.shape
    n_components = C
    P = vstack([identity(N) for _ in range(M)])
    P = P @ P.T
    Z = np.hstack(X).T  # type: ignore
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    return U
    
def _trca_feature(
    X: ndarray,
    templates: ndarray,
    Us: ndarray,
    n_components: int = 1,
    ensemble: bool = True,
):
    rhos = []
    if not ensemble:
        for Xk, U in zip(templates, Us):
            a = U[:, :n_components].T @ X
            b = U[:, :n_components].T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    else:
        U = Us[:, :, :n_components]
        U = np.concatenate(U, axis=-1)
        for Xk in templates:
            a = U.T @ X
            b = U.T @ Xk
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            rhos.append(pearsonr(a, b)[0])
    return rhos

            
class TRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, 
            n_components: int = 1, 
            ensemble: bool = True,
            n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.ensemble = ensemble
        self.n_jobs = n_jobs
    
    def fit(self, 
            X: ndarray, 
            y: ndarray,
            Yf: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack([np.mean(X[y==label], axis=0) for label in self.classes_])

        self.Us_ = np.stack([_trca_kernel(X[y==label]) for label in self.classes_])
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        n_components = self.n_components
        templates = self.templates_
        Us = self.Us_
        ensemble = self.ensemble
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_trca_feature, Us=Us, n_components=n_components, ensemble=ensemble))(a, templates) for a in X)
        rhos = np.stack(rhos)
        return rhos
    
    def predict(self, X: ndarray):
        feat = self.transform(X)
        labels = self.classes_[np.argmax(feat, axis=-1)]
        return labels

class FBTRCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(self, 
        filterbank: List[ndarray],
        n_components: int = 1,
        ensemble: bool = True,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.ensemble = ensemble
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            TRCA(n_components=n_components, ensemble=ensemble, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(features, (features.shape[0], len(self.filterbank), -1))
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels,features


def _scca_kernel(X: ndarray, Yf: ndarray):
    """Standard CCA (sCCA).

    This is an time-consuming implementation due to GED.

    X: (n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    n_components = min(X.shape[0], Yf.shape[0])
    Q, R = qr(Yf.T, mode="economic")
    P = Q @ Q.T
    Z = X.T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    V = pinv(R) @ Q.T @ X.T @ U  # V for Yf
    return U, V


def _mscca_feature(X: ndarray, templates: ndarray, U: ndarray, n_components: int = 1):
    rhos = []
    for Xk in zip(templates):
        a = U[:, :n_components].T @ X
        b = U[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rhos.append(pearsonr(a, b)[0])
    return np.array(rhos)


class MsCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Note: MsCCA heavily depends on Yf, thus the phase information should be included when designs Yf.

    """

    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):
        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf

        self.U_, self.V_ = _scca_kernel(
            np.concatenate(self.templates_, axis=-1), np.concatenate(self.Yf_, axis=-1)
        )
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        n_components = self.n_components
        U = self.U_
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_mscca_feature, U=U, n_components=n_components))(
                a, templates
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels


class FBMsCCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
            self,
            filterbank: List[ndarray],
            n_components: int = 1,
            filterweights: Optional[ndarray] = None,
            n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            MsCCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        # print(labels.shape,features.shape,'aaaaaaaaa')#(168,) (168, 12) aaaaaaa
        return labels, features
