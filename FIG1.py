''' PATH '''
## Specify path for where you have located this directory:
path = '/some/path/to/these/files'

''' MODULES '''
# standard library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.optimize import curve_fit
import scipy.stats as st

# local application imports
from EWS_functions_AMV import *


''' DATA '''
# import
proxies = pd.read_csv('all_proxies3.csv')
AMV = np.loadtxt("Micheletal2022_AMVreconstruction.txt", comments='#',encoding='utf-8', dtype=float)
surrogate_PLN = pd.read_csv('SurrogateReconstructions_PowerLawNoise_RandomForest_850_1987_nasstF.csv')
surrogate_PLN = surrogate_PLN.drop(surrogate_PLN.columns[0], axis=1)
surrogate_WN = pd.read_csv('SurrogateReconstructions_WhiteNoise_RandomForest_850_1987_nasstF.csv')
surrogate_WN = surrogate_WN.drop(surrogate_WN.columns[0], axis=1)



################################################################
#---------FIG. 1 - AMV reconstruction + statistical EWS--------#
################################################################


''' DATA PREP '''
data = AMV[:,1]
time = np.arange(np.min(AMV[:,0]), np.min(AMV[:,0]) + len(AMV[:,0]))
time2 = np.arange(time[0], time[-1])

# linear fit for unprocessed data
popt31, cov = curve_fit(funcfit3, time[time > time[0]], data[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
fit = funcfit3(time2, *popt31)

# standardise data
AMV_df = pd.DataFrame(AMV)
standardised_AMV = standardise(AMV_df[AMV_df.columns[1:]], time)
detrended = standardised_AMV[1]

''' PARAMETERS '''
# window size
ws = 50
bound = ws // 2
# number of fourier surrogates for significance testing of the trend
# 1000 surrogates were used to produce the figures for the manuscript
tt_samples = 1000


''' PLOT '''
fig = plt.figure(figsize = (6,8))

ax = fig.add_subplot(211)
ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.set_xlim((time[0], time[-1]))
ax.plot(time, data, 'k-', label = r"AMV reconstruction")
ax.plot(time2,  fit, 'r-', label = 'Non-linear least squares fit')
plt.legend(loc = 1)
ax.set_ylabel('AMV reconstruction')

ax1 = fig.add_subplot(212)
ax1.text(-.15, 1, s = 'b', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax1.plot(time[bound : -bound], runstd(detrended, ws)[bound : -bound]**2, color = 'r', label = 'Variance')
p0, p1 = np.polyfit(time[bound : -bound], runstd(detrended, ws)[bound : -bound]**2, 1)
pv = kendall_tau_test(detrended, tt_samples, ws, p0, "var", bound)
if pv >= .001:
    ax1.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax1.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')

ax1.set_xlabel('Time [yr CE]')
ax1.set_ylabel('Variance (ws = %d yr)'%ws, color = 'r')
plt.legend(loc = 3)
ax1.set_xlim((time[0], time[-1]))
ax1.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
ax1.axvspan(time[:-bound][-1], time[-1], facecolor = 'none', edgecolor = 'k', hatch = '/')

ax3 = ax1.twinx()
ax3.plot(time[bound : -bound], runac(detrended, ws)[bound : -bound], color = 'b', label = 'AC1')
p0, p1 = np.polyfit(time[bound : -bound], runac(detrended, ws)[bound : -bound], 1)
pv = kendall_tau_test(detrended, tt_samples, ws, p0, "ar", bound)
if pv >= .001:
    ax3.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax3.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p < 10^{-3}$')
plt.legend(loc = 1)
ax3.set_ylabel('AC1 (ws = %d yr)'%ws, color = 'b')


fig.savefig('FIG1.pdf', bbox_inches = 'tight')



################################################################
#---------FIG. A1 - sample surrogates + statistical EWS--------#
################################################################


''' DATA PREP '''
data = surrogate_PLN["V5"]
min = np.min(np.array(surrogate_PLN["V5"]))
time = np.arange(min, min + len(surrogate_PLN))
time2 = np.arange(time[0], time[-1])
detrended = standardise2(data, time)

# parameters (ws and tt_samples) are the same as in Fig. 1

''' PANEL a '''

fig = plt.figure(figsize = (13,8))
fig.subplots_adjust(wspace=0.25)

ax = fig.add_subplot(221)
ax.plot(time, data, 'k-', label = r"Power Law Noise surrogate")
popt31, cov = curve_fit(funcfit3, time[time > time[0]], data[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
ax.plot(time2,  funcfit3(time2, *popt31), 'r-', label = 'Non-linear least squares fit')
plt.legend(loc = 1)
ax.set_xlim((time[0], time[-1]))
ax.set_ylabel('Power Law Noise surrogate')

ax1 = fig.add_subplot(223)
ax1.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax1.plot(time[bound : -bound], runstd(detrended, ws)[bound : -bound]**2, color = 'r', label = 'Variance')
p0, p1 = np.polyfit(time[bound : -bound], runstd(detrended, ws)[bound : -bound]**2, 1)
pv = kendall_tau_test(detrended, tt_samples, ws, p0, "var", bound)
if pv >= .001:
    ax1.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax1.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')

ax1.set_xlabel('Time [yr CE]')
ax1.set_ylabel('Variance (ws = %d yr)'%ws, color = 'r')
plt.legend(loc = 3)
ax1.set_xlim((time[0], time[-1]))
ax1.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
ax1.axvspan(time[:-bound][-1], time[-1], facecolor = 'none', edgecolor = 'k', hatch = '/')

ax3 = ax1.twinx()
ax3.plot(time[bound : -bound], runac(detrended, ws)[bound : -bound], color = 'b', label = 'AC1')
p0, p1 = np.polyfit(time[bound : -bound], runac(detrended, ws)[bound : -bound], 1)
pv = kendall_tau_test(detrended, tt_samples, ws, p0, "ar", bound)
if pv >= .001:
    ax3.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax3.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p < 10^{-3}$')
plt.legend(loc = 1)


''' DATA PREP '''

data = surrogate_WN["V16"]
min = np.min(np.array(surrogate_WN["V16"]))
time = np.arange(min, min + len(surrogate_WN))
time2 = np.arange(time[0], time[-1])
detrended = standardise2(data, time)

''' PANEL b '''

ax = fig.add_subplot(222)
ax.plot(time, data, 'k-', label = r"White Noise surrogate")
popt31, cov = curve_fit(funcfit3, time[time > time[0]], data[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
ax.plot(time2,  funcfit3(time2, *popt31), 'r-', label = 'Non-linear least squares fit')
plt.legend(loc = 1)
ax.set_xlim((time[0], time[-1]))
ax.set_ylabel('White Noise surrogate')

ax1 = fig.add_subplot(224)
ax1.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax1.plot(time[bound : -bound], runstd(detrended, ws)[bound : -bound]**2, color = 'r', label = 'Variance')
p0, p1 = np.polyfit(time[bound : -bound], runstd(detrended, ws)[bound : -bound]**2, 1)
pv = kendall_tau_test(detrended, tt_samples, ws, p0, "var", bound)
if pv >= .001:
    ax1.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax1.plot(time2, p0 * time2 + p1, color = 'r', ls = '--', label = '$p < 10^{-3}$')

ax1.set_xlabel('Time [yr CE]')
#ax1.set_ylabel('Variance (ws = %d yr)'%ws, color = 'r')
plt.legend(loc = 3)
ax1.set_xlim((time[0], time[-1]))
ax1.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
ax1.axvspan(time[:-bound][-1], time[-1], facecolor = 'none', edgecolor = 'k', hatch = '/')

ax3 = ax1.twinx()
ax3.plot(time[bound : -bound], runac(detrended, ws)[bound : -bound], color = 'b', label = 'AC1')
p0, p1 = np.polyfit(time[bound : -bound], runac(detrended, ws)[bound : -bound], 1)
pv = kendall_tau_test(detrended, tt_samples, ws, p0, "ar", bound)
if pv >= .001:
    ax3.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p = %.3f$'%pv)
else:
    ax3.plot(time2, p0 * time2 + p1, color = 'b', ls = '--', label = '$p < 10^{-3}$')
plt.legend(loc = 1)
ax3.set_ylabel('AC1 (ws = %d yr)'%ws, color = 'b')


fig.savefig('FigA1.pdf', bbox_inches = 'tight')



################################################################
#-----------FIG. A2 - EWS sensitivity to window size-----------#
################################################################

''' COMPUTATIONS '''

var = []
ac = []
data = standardised_AMV[1]

# loop through different window sizes

for i in np.arange(10,410,10):
    ws = i # window size
    bound = ws // 2
    time = np.arange(np.min(data), np.min(data) + len(data))
    time2 = np.arange(time[0], time[-1])
    
    popt31, cov = curve_fit(funcfit3, time[time > time[0]], data[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
    
    # store variance fit
    p0, p1 = np.polyfit(time[bound : -bound], runstd(data - funcfit3(time, *popt31), ws)[bound : -bound]**2, 1)
    var.append(p0)
    
    # store autocorrelation fit
    p0, p1 = np.polyfit(time[bound : -bound], runac(data - funcfit3(time, *popt31), ws)[bound : -bound], 1)
    ac.append(p0)
    

''' PLOT '''

fig = plt.figure(figsize=(14,4))
fig.subplots_adjust(wspace=0.25)

ax = fig.add_subplot(121)
ax.plot(np.arange(10,410,10), ac, 'ob')
ax.set_xlabel("Window size")
ax.set_ylabel("Slope of fit")
ax.set_title("AMV reconstruction lag-1 autocorrelation")

ax = fig.add_subplot(122)
ax.plot(np.arange(10,410,10), var, 'or')
ax.set_xlabel("Window size")
ax.set_ylabel("Slope of fit")
ax.set_title("AMV reconstruction variance")

plt.savefig('FigA2.pdf', bbox_inches = 'tight')
