# adapted from code provided by Vanessa Skiba

import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.optimize import curve_fit


def funcfit2(x, a, b, c):
    """ 
    model function for curve fit
    """
    if b <= 0 or c <= 0:
        return 1e8
    else:
        return a + np.power(-b * (x - c), 1 / 2)

def funcfit3(x, a, b, c):
    """ 
    model function for curve fit
    """
    if b <= 0 or c <= 0:
        return 1e8
    else:
        return a + np.power(-b * (x - c), 1 / 3)

def funcfit3_jac(x, a, b, c):
    """
    jacobian of model function
    """
    return np.array([np.ones(x.shape[0]), -(x-c) * np.power(-b * (x - c), -2/3) / 3, b * np.power(-b * (x - c), -2/3) / 3]).T


def standardise(x, time):
    """
    input: dataframe of time series
    removes trend obtained via non-linear least squares fit
    standardises to mean 0 and std 1
    """
    values = []
    for column in x.columns:
        data = x[column]
        popt31, cov = curve_fit(funcfit3, time[time > time[0]], data[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
        values.append(data - funcfit3(time, *popt31))
    detrended = pd.DataFrame(values).transpose()
    standardised = (detrended-detrended.mean())/detrended.std()
    return standardised


def standardise2(x, time):
    """
    input: time series
    removes trend obtained via non-linear least squares fit
    standardises to mean 0 and std 1
    """
    popt31, cov = curve_fit(funcfit3, time[time > time[0]], x[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
    detrended = x - funcfit3(time, *popt31)
    standardised = (detrended-detrended.mean())/detrended.std()
    return standardised


def fourier_surrogates(ts, ns):
    """
    computes Fourier surrogates of a time series
    by phase randomisation in frequency domain
    """
    # Fourier transform
    ts_fourier  = np.fft.rfft(ts)
    # phase randomisation
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    # inverse Fourier transform
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts


def kendall_tau_test(ts, ns, w, tau, EWS, bound, mode1 = 'fourier', mode2 = 'linear'):
    """
    significance test of EWS against Fourier surrogates
    
    Parameters
        ----------
        ts: detrended time series         
        ns: number of Fourier surrogates
        w: window size
        tau: slope of the linear fit
        EWS: EWS used for analysis
             var: variance
             ar: lag-1 autocorrelation
             lam: restoring rate (lambda)
        bound: cutoff at the ends of the time series, f.ex. w/2 to ensure constant window size
            
    Returns
    -------
    p-value
    """
    tlen = ts.shape[0]
    global surrogate
    if mode1 == 'fourier':
        tsf = ts - ts.mean()
        nts = fourier_surrogates(tsf, ns)
    elif mode1 == 'shuffle':
        nts = shuffle_surrogates(ts, ns)
    stat = np.zeros(ns)
    tlen = nts.shape[1]        
    if mode2 == 'linear':
        for i in range(ns):
            if EWS == "var":
                surrogate = runstd(nts[i], w)[bound : -bound]
               # print(EWS, i)
            if EWS == "ar":
                surrogate = runac(nts[i], w)[bound : -bound]
              #  print(EWS, i)
            if EWS == "lam":
                surrogate = run_fit_a_ar1(nts[i], w)[bound : -bound]   
              #  print(EWS, i)
        tlen_bounds = ts[bound : -bound].shape[0]
        stat[i] = st.linregress(np.arange(tlen_bounds), surrogate)[0]
    elif mode2 == 'kt':
        for i in range(ns):
            stat[i] = st.kendalltau(np.arange(tlen), surrogate)[0]
            
    # distinguish positive and negative trends
    if tau > 0:
        p = 1 - st.percentileofscore(stat, tau) / 100.
    else:
        p = st.percentileofscore(stat, tau) / 100.
    
    return p



def runstd(x, w):
    """
    computes running standard deviation of time series
    w: window size
    """
   n = x.shape[0]
   bounds = w // 2
   xs = np.zeros_like(x)
   for i in range(bounds):
      xw = x[: i + bounds + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan
   for i in range(n - bounds, n):
      xw = x[i - bounds + 1:]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan

   for i in range(bounds, n - bounds):
      xw = x[i - bounds : i + bounds + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.std(xw)
      else:
          xs[i] = np.nan

   return xs


def runac(x, w):
    """
    computes running lag-1 autocorrelation of time series
    w: window size
    """
   n = x.shape[0]
   bounds = w // 2
   xs = np.zeros_like(x)
   for i in range(bounds):
      xw = x[: i + bounds + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]
          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   for i in range(n - bounds, n):
      xw = x[i - bounds + 1:]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1

          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   for i in range(bounds, n - bounds):
      xw = x[i - bounds : i + bounds + 1]
      xw = xw - xw.mean()
      if np.std(xw) > 0:
          lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
          p0 = lg[0]
          p1 = lg[1]

          xw = xw - p0 * np.arange(xw.shape[0]) - p1
          xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
      else:
          xs[i] = np.nan

   return xs
