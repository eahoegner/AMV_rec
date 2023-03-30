"""
Import this script as a module 
"""

import numpy as np 
from itertools import chain
from scipy.spatial.distance import pdist, squareform


def det(R, lmin=None, hist=None, verb=True):
    """returns DETERMINISM for given recurrence matrix R."""
    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = diagonal_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating DET...")
    Pl = nlines.astype('float')
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx = l >= lmin
    num = l[idx] * Pl[idx]
    den = l * Pl
    DET = num.sum() / den.sum()
    return DET


def lam(R, lmin=None, hist=None, verb=True):
    """returns LAMINARITY for given recurrence matrix R."""
    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = vertical_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating LAM...")
    Pl = nlines.astype('float')
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx = l >= lmin
    num = l[idx] * Pl[idx]
    den = l * Pl
    LAM = num.sum() / den.sum()
    return LAM


def diagonal_lines_hist(R, verb=True):
    """returns the histogram P(l) of diagonal lines of length l."""
    if verb:
        print("diagonal lines histogram...")
    line_lengths = []
    for i in range(1, len(R)):
        d = np.diag(R, k=i)
        ll = _count_num_lines(d)
        line_lengths.append(ll)
    line_lengths = np.array(list(chain.from_iterable(line_lengths)))
    bins = np.arange(0.5, line_lengths.max() + 0.1, 1.)
    num_lines, _ = np.histogram(line_lengths, bins=bins)
    return num_lines, bins, line_lengths


def vertical_lines_hist(R, verb=True):
    """returns the histogram P(l) of diagonal lines of length l."""
    if verb:
        print("vertical lines histogram...")
    line_lengths = []
    for i in range(1, len(R)):
        v = R[:,i]
        ll = _count_num_lines(v)
        line_lengths.append(ll)
    line_lengths = np.array(list(chain.from_iterable(line_lengths)))
    bins = np.arange(0.5, line_lengths.max() + 0.1, 1.)
    num_lines, _ = np.histogram(line_lengths, bins=bins)
    return num_lines, bins, line_lengths


def _count_num_lines(arr):
    """returns a list of line lengths contained in given array."""
    line_lens = []
    counting = False
    l = 0
    for i in range(len(arr)):
        if counting:
            if arr[i] == 0:
                l += 1
                line_lens.append(l)
                l = 0
                counting = False
            elif arr[i] == 1:
                l += 1
                if i == len(arr) - 1:
                    l += 1
                    line_lens.append(l)
        elif not counting:
            if arr[i] == 1:
                counting = True
    return line_lens


# recurrence plot with no embedding

def rp_no_emb(x, e, norm="euclidean"):
    """Returns the recurrence plot of given time series."""
    ''' no normalization '''
    d = pdist(np.array(x)[:, None], metric=norm)
    D = squareform(d)
    R = np.zeros(D.shape, dtype="int")
    ''' distance measured by frr metric'''
    e = np.percentile(D, e * 100.)
    i = np.where(D <= e)
    R[i] = 1

    return R


# function to compute the running determinism for window w

def rundet(w, x, lm, RR):
    
    n = x.shape[0]
    bounds = w // 2
    xs = np.zeros_like(x)
    
    for i in range(bounds):
        xw = x[: i + bounds + 1]
        xs[i] = det(rp_no_emb(xw, RR), lmin=lm, verb=False) 
    
    for i in range(n - bounds, n):
        xw = x[i - bounds + 1:]
        xs[i] = det(rp_no_emb(xw, RR), lmin=lm, verb=False) 
        
    for i in range(bounds, n - bounds):
        xw = x[i - bounds : i + bounds + 1]
        xs[i] = det(rp_no_emb(xw, RR), lmin=lm, verb=False) 
        
    return xs




# function to compute the running laminarity for window w

def runlam(w, x, lm, RR):
    
    n = x.shape[0]
    bounds = w // 2
    xs = np.zeros_like(x)
    
    for i in range(bounds):
        xw = x[: i + bounds + 1]
        xs[i] = lam(rp_no_emb(xw, RR), lmin=lm, verb=False) 
    
    for i in range(n - bounds, n):
        xw = x[i - bounds + 1:]
        xs[i] = lam(rp_no_emb(xw, RR), lmin=lm, verb=False) 
        
    for i in range(bounds, n - bounds):
        xw = x[i - bounds : i + bounds + 1]
        xs[i] = lam(rp_no_emb(xw, RR), lmin=lm, verb=False) 
        
    return xs



# define function for average diagonal line length

def dll(R, lmin=None, hist=None, verb=True):
    """returns DETERMINISM for given recurrence matrix R."""
    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = diagonal_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating DET...")
    Pl = nlines.astype('float')
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx = l >= lmin
    num = l[idx] * Pl[idx]
    den = l * Pl
    DET = den.sum() / Pl.sum()
    return DET