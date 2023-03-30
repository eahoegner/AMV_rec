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
from recurrence import *
from recurrence_quantification import *


''' DATA '''
# import
AMV = np.loadtxt("Micheletal2022_AMVreconstruction.txt", comments='#',encoding='utf-8', dtype=float)

surrogate_PLN = pd.read_csv('SurrogateReconstructions_PowerLawNoise_RandomForest_850_1987_nasstF.csv')
surrogate_PLN = surrogate_PLN.drop(surrogate_PLN.columns[0], axis=1)
surrogate_WN = pd.read_csv('SurrogateReconstructions_WhiteNoise_RandomForest_850_1987_nasstF.csv')
surrogate_WN = surrogate_WN.drop(surrogate_WN.columns[0], axis=1);

# standardise
time = np.arange(np.min(AMV[:,0]), np.min(AMV[:,0]) + len(AMV[:,0]))
time2 = np.arange(time[0], time[-1])
standardised_PLN = standardise(surrogate_PLN[surrogate_PLN.columns[1:]].dropna(axis=1), time)
standardised_WN = standardise(surrogate_WN[surrogate_WN.columns[1:]].dropna(axis=1), time)
AMV_df = pd.DataFrame(AMV)
standardised_AMV = standardise(AMV_df[AMV_df.columns[1:]], time)
data_AMV = standardised_AMV[1]


''' PARAMETERS '''
ws = 250
lmin = 5
RR = 0.3
bound = ws // 2
time = np.arange(np.min(AMV[:,0]), np.min(AMV[:,0]) + len(AMV[:,0]))
time2 = np.arange(time[0], time[-1])
data_AMV = standardised_AMV[1]


''' COMPUTE RQA AMV '''
RD = rundet(ws, data_AMV, 5, 0.3)
RL = runlam(ws, data_AMV, 5, 0.3)

# extract slope
p0, p1 = np.polyfit(time[bound : -bound], RD[bound : -bound], 1)
LSL = p0
p0, p1 = np.polyfit(time[bound : -bound], RL[bound : -bound], 1)
LSD = p0

''' COMPUTE RQA SURROGATES '''

PLND = []
PLNL = []
WND = []
WNL = []

for column in standardised_PLN.columns:
    data = surrogate_PLN[column]
    runL = runlam(ws, data, lmin, RR)
    runD = rundet(ws, data, lmin, RR)
    p0, p1 = np.polyfit(time[bound : -bound], runD[bound : -bound], 1)
    PLND.append(p0)
    p0, p1 = np.polyfit(time[bound : -bound], runL[bound : -bound], 1)
    PLNL.append(p0)
    
for column in standardised_WN.columns:
    data = surrogate_WN[column]
    runL = runlam(ws, data, lmin, RR)
    runD = rundet(ws, data, lmin, RR)
    p0, p1 = np.polyfit(time[bound : -bound], runD[bound : -bound], 1)
    WND.append(p0)
    p0, p1 = np.polyfit(time[bound : -bound], runL[bound : -bound], 1)
    WNL.append(p0)


    
################################################################
#--------FIG. 5 - RQA AMV + surrogates running DET + LAM-------#
################################################################


''' PLOT '''
fig = plt.figure(figsize = (13,6))
fig.subplots_adjust(hspace=0.5)
subfigs = fig.subfigures(1,2)

''' PANEL a '''
ax = subfigs[0].add_subplot()
ax.plot(time[bound : -bound], RD[bound : -bound], color='mediumorchid')
ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
p0, p1 = np.polyfit(time[bound : -bound], RD[bound : -bound], 1)
ax.plot(time2, p0 * time2 + p1, color = 'mediumorchid', ls = '--')
ax.set_xlim((time[0], time[-1]))
ax.set_title("RQA of AMV reconstruction, 250 year running window")
ax.set_xlabel("Time [yr CE]")
ax.set_ylabel("DET", color='mediumorchid')
ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
ax.axvspan(time[:-bound][-1], time[-1], facecolor = 'none', edgecolor = 'k', hatch = '/')

ax2 = ax.twinx()
ax2.plot(time[bound : -bound], RL[bound : -bound], color='aqua')
p0, p1 = np.polyfit(time[bound : -bound], RL[bound : -bound], 1)
ax2.plot(time2, p0 * time2 + p1, color = 'aqua', ls = '--')
ax2.set_xlabel("Time [yr CE]")
ax2.set_ylabel("LAM", color='aqua')

''' PANEL b '''
ax = subfigs[1].add_subplot(211)
ax.hist(PLND, color='m', bins=100, alpha=.8)
ax.text(-.1, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax2 = ax.twiny()
ax2 = ax.twinx()
ax2.hist(PLNL, color='c', bins=100, alpha=.5)
ax2.locator_params(axis='x', nbins=6)
ax2.locator_params(axis='y', nbins=6)
plt.axvline(x = 0, color='lightgrey', linestyle='dotted')
plt.axvline(x = LSD, color='m', linestyle='dashed', label="DET AMV")
plt.axvline(x = LSL, color='c', linestyle='dashed', label="LAM AMV")
plt.title("Slope of linear fit in moving DET/LAM for PLN surrogates")
plt.legend(loc=2)

''' PANEL c '''
ax = subfigs[1].add_subplot(212)
ax.hist(WND, color='m', bins=100, alpha=.8)
ax.text(-.1, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax2 = ax.twiny()
ax2 = ax.twinx()
ax2.hist(WNL, color='c', bins=100, alpha=.5)
ax2.locator_params(axis='x', nbins=6)
ax2.locator_params(axis='y', nbins=6)
plt.axvline(x = 0, color='lightgrey', linestyle='dotted')
plt.axvline(x = LSD, color='m', linestyle='dashed', label="DET AMV")
plt.axvline(x = LSL, color='c', linestyle='dashed', label="LAM AMV")
plt.title("Slope of linear fit in moving DET/LAM for WN surrogates")
plt.legend(loc=2)

fig.savefig('FIG5.pdf', bbox_inches = 'tight')