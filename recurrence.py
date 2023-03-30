"""
Module for recurrence-based nonlinear time series analysis
==========================================================

"""
# adapted from Bedartha Goswami <goswami@pik-potsdam.de>

import sys
import numpy as np

from scipy.spatial.distance import pdist, squareform

# disable dive by zero warnings
np.seterr(divide="ignore")


def mi(x, maxlag, binrule="fd", pbar_on=True):
    """
    Returns the self mutual information of a time series up to max. lag.
    """
    # initialize variables
    n = len(x)
    lags = np.arange(0, maxlag, dtype="int")
    mi = np.zeros(len(lags))
    # loop over lags and get MI
    for i, lag in enumerate(lags):
        # extract lagged data
        y1 = x[:n - lag].copy()
        y2 = x[lag:].copy()
        # use np.histogram to get individual entropies
        H1, be1 = entropy1d(y1, binrule)
        H2, be2 = entropy1d(y2, binrule)
        H12, _, _ = entropy2d(y1, y2, [be1, be2])
        # use the entropies to estimate MI
        mi[i] = H1 + H2 - H12

    return mi, lags


def entropy1d(x, binrule):
    """
    Returns the Shannon entropy according to the bin rule specified.
    """
    p, be = np.histogram(x, bins=binrule, density=True)
    r = be[1:] - be[:-1]
    P = p * r
    L = np.log2(P)
    i = ~ np.isinf(L)
    H = -(P[i] * L[i]).sum()

    return H, be


def entropy2d(x, y, bin_edges):
    """
    Returns the Shannon entropy according to the bin rule specified.
    """
    p, bex, bey = np.histogram2d(x, y, bins=bin_edges, normed=True)
    r = np.outer(bex[1:] - bex[:-1], bey[1:] - bey[:-1])
    P = p * r
    H = np.zeros(P.shape)
    i = ~np.isinf(np.log2(P))
    H[i] = -(P[i] * np.log2(P[i]))
    H = H.sum()

    return H, bex, bey


def first_minimum(y):
    """
    Returns the first minimum of given data series y.
    """
    try:
        fmin = np.where(np.diff(np.sign(np.diff(y))) == 2.)[0][0] + 2
    except IndexError:
        fmin = np.nan
    return fmin



def fnn(x, tau, maxdim, r=0.10, pbar_on=True):
    """
    Returns the number of false nearest neighbours up to max dimension.
    """
    # initialize params
    sd = x.std()
    r = r * (x.max() - x.min())
    e = sd / r
    fnn = np.zeros(maxdim)
    dims = np.arange(1, maxdim + 1, dtype="int")

    # ensure that (m-1) tau is not greater than N = length(x)
    N = len(x)
    K = (maxdim + 1 - 1) * tau
    if K >= N:
        m_c = N / tau
        i = np.where(dims >= m_c)
        fnn[i] = np.nan
        j = np.where(dims < m_c)
        dims = dims[j]

    # get first values of distances for m = 1
    d_m, k_m = mindist(x, 1, tau)

    # loop over dimensions and get FNN values
    for m in dims:
        # get minimum distances for one dimension higher
        d_m1, k_m1 = mindist(x, m + 1, tau)
        # remove those indices in the m-dimensional calculations which cannot
        # occur in the m+1-dimensional arrays as the m+1-dimensional arrays are
        # smaller
        cond1 = k_m[1] > k_m1[0][-1]
        cond2 = k_m[0] > k_m1[0][-1]
        j = np.where(~(cond1 + cond2))[0]
        k_m_ = (k_m[0][j], k_m[1][j])
        d_k_m, d_k_m1 = d_m[k_m_], d_m1[k_m_]
        n_m1 = d_k_m.shape[0]
        # calculate quantities in Eq. 3.8 of Kantz, Schreiber (2004) 2nd Ed.
        j = d_k_m > 0.
        y = np.zeros(n_m1, dtype="float")
        y[j] = (d_k_m1[j] / d_k_m[j] > r)
        w = (e > d_k_m)
        num = float((y * w).sum())
        den = float(w.sum())
        # assign FNN value depending on whether denominator is zero
        if den != 0.:
            fnn[m - 1] = num / den
        else:
            #fnn[m - 1] = np.nan
            fnn[m - 1] = 0
        # assign higher dimensional values to current one before next iteration
        d_m, k_m = d_m1, k_m1
    fnn[0]=1
    return fnn, dims


def mindist(x, m, tau):
    """
    Returns the minimum distances for each point in given embedding.
    """
    z = embed(x, m, tau)
    # d = squareform(pdist(z))
    n = len(z)
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = np.max(np.abs(z[i] - z), axis=1)

    np.fill_diagonal(d, 99999999.)
    k = (np.arange(len(d)), np.argmin(d, axis=1))

    return d, k


def embed(x, m, tau):
    """
    Embeds a scalar time series in m dimensions with time delay tau.
    """
    n = len(x)
    k = n - (m - 1) * tau
    z = np.zeros((k, m), dtype="float")
    for i in range(k):
        z[i] = [x[i + j * tau] for j in range(m)]

    return z


def first_zero(y):
    """
    Returns the index of the first value which is zero in y.
    """
    try:
        fzero = np.where(y == 0.)[0][0]
    except IndexError:
        fzero = 0
    return fzero


def rp(x, m, tau, e, norm="euclidean", threshold_by="distance", normed=True):
    """Returns the recurrence plot of given time series."""
    if normed:
        x = normalize(x)
    z = embed(x, m, tau)
    D = squareform(pdist(z, metric=norm))
    R = np.zeros(D.shape, dtype="int")
    if threshold_by == "distance":
        i = np.where(D <= e)
        R[i] = 1
    elif threshold_by == "fan":
        nk = np.ceil(e * R.shape[0]).astype("int")
        i = (np.arange(R.shape[0]), np.argsort(D, axis=0)[:nk])
        R[i] = 1
    elif threshold_by == "frr":
        e = np.percentile(D, e * 100.)
        i = np.where(D <= e)
        R[i] = 1

    return R


''' simplified code for recurrence plot without embedding '''

def rp_no_emb(x, e, norm="euclidean"):
    """Returns the recurrence plot of given time series."""
    d = pdist(np.array(x)[:, None], metric=norm)
    D = squareform(d)
    R = np.zeros(D.shape, dtype="int")
    ''' distance measured by frr metric'''
    e = np.percentile(D, e * 100.)
    i = np.where(D <= e)
    R[i] = 1

    return R


def rn(x, m, tau, e, norm="euclidean", threshold_by="distance", normed=True):
    """Returns the recurrence network adjacency matrix of given time series."""
    z = embed(x, m, tau)
    D = squareform(pdist(z, metric=norm))
    np.fill_diagonal(D, np.inf)
    A = np.zeros(D.shape, dtype="int")
    if threshold_by == "distance":
        i = np.where(D <= e)
        A[i] = 1
    elif threshold_by == "fan":
        nk = np.ceil(e * A.shape[0]).astype("int")
        i = (np.arange(A.shape[0]), np.argsort(D, axis=0)[:nk])
        A[i] = 1
    elif threshold_by == "frr":
        e = np.percentile(D, e * 100.)
        i = np.where(D <= e)
        A[i] = 1

    return A


def normalize(x):
    """
    Returns the Z-score series for x.
    """
    return (x - x.mean()) / x.std()
