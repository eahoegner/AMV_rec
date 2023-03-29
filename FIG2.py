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
#-------FIG. 2 - EWS in surrogates, significance testing-------#
################################################################


#--------------------------------------------------------#
#-------------------- COMPUTATIONS ----------------------#
#--------------------------------------------------------#

# calculate standardised AMV reconstruction's EWS slopes and p values

''' DATA PREP '''

time = np.arange(np.min(AMV[:,0]), np.min(AMV[:,0]) + len(AMV[:,0]))
time2 = np.arange(time[0], time[-1])
AMV_df = pd.DataFrame(AMV)
data = standardise(AMV_df[AMV_df.columns[1:]], time)[1]
standardised_PLN = standardise(surrogate_PLN[surrogate_PLN.columns[1:]].dropna(axis=1), time)
standardised_WN = standardise(surrogate_WN[surrogate_WN.columns[1:]].dropna(axis=1), time)

''' PARAMETERS '''

ws = 50 # window size
bound = ws // 2
tt_samples = 100 # number of fourier surrogates for significance testing of the trend
AMV_p = []
AMV_pv = []

p0, p1 = np.polyfit(time[bound : -bound], runstd(data, ws)[bound : -bound]**2, 1)
pv = kendall_tau_test(data, tt_samples, ws, p0, "var", bound) #correct significance testing as in Ben-Yami et al 
AMV_p.append(p0)
AMV_pv.append(pv)

p0, p1 = np.polyfit(time[bound : -bound], runac(data, ws)[bound : -bound], 1)
pv = kendall_tau_test(data, tt_samples, ws, p0, "ar", bound) #correct significance testing as in Ben-Yami et al 
AMV_p.append(p0)
AMV_pv.append(pv)



### calculate EWS for noise surrogates

''' INITIALISE '''

PLN_var = []
PLN_var_pv = []
PLN_ac = []
PLN_ac_pv = []

''' EWS LOOP FOR PLN SURROGATES '''

for column in standardised_PLN.columns[]:
    data = standardised_PLN[column]
    
    # time series and trend
    popt31, cov = curve_fit(funcfit3, time[time > time[0]], data[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
    
    # compute running variance
    p0, p1 = np.polyfit(time[bound : -bound], runstd(data, ws)[bound : -bound]**2, 1)
    
    # extract variance slope and p-value
    PLN_var.append(p0)
    pv = kendall_tau_test(data, tt_samples, ws, p0, "var", bound) #correct significance testing as in Ben-Yami et al 
    PLN_var_pv.append(pv)
    
    # compute running AR1
    p0, p1 = np.polyfit(time[bound : -bound], runac(data, ws)[bound : -bound], 1)
    
    # extract AR1 slope and p-value
    PLN_ac.append(p0)
    pv = kendall_tau_test(data, tt_samples, ws, p0, "ar", bound) #correct significance testing as in Ben-Yami et al 
    PLN_ac_pv.append(pv)

PLN = pd.DataFrame(list(zip(PLN_var, PLN_var_pv, PLN_ac, PLN_ac_pv)), columns =['var', 'var_pv', 'ac', 'ac_pv'])


### calculate EWS for noise surrogates

''' INITIALISE '''

WN_var = []
WN_var_pv = []
WN_ac = []
WN_ac_pv = []

''' EWS LOOP FOR WN SURROGATES '''

for column in standardised_WN.columns[]:
    data = standardised_WN[column]
    
    # time series and trend
    popt31, cov = curve_fit(funcfit3, time[time > time[0]], data[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)
    
    # compute running variance
    p0, p1 = np.polyfit(time[bound : -bound], runstd(data, ws)[bound : -bound]**2, 1)
    
    # extract variance slope and p-value
    WN_var.append(p0)
    pv = kendall_tau_test(data, tt_samples, ws, p0, "var", bound) #correct significance testing as in Ben-Yami et al 
    WN_var_pv.append(pv)
    
    # compute running AR1
    p0, p1 = np.polyfit(time[bound : -bound], runac(data, ws)[bound : -bound], 1)
    
    # extract AR1 slope and p-value
    WN_ac.append(p0)
    pv = kendall_tau_test(data, tt_samples, ws, p0, "ar", bound) #correct significance testing as in Ben-Yami et al 
    WN_ac_pv.append(pv)
    
WN = pd.DataFrame(list(zip(WN_var, WN_var_pv, WN_ac, WN_ac_pv)), columns =['var', 'var_pv', 'ac', 'ac_pv'])



#--------------------------------------------------------#
#------------------------- PLOT--------------------------#
#--------------------------------------------------------#

fig = plt.figure(figsize = (13,13))
fig.subplots_adjust(wspace=0.25)

ax = fig.add_subplot(321)
ax.set_ylabel('Slope of lag-1 ac fit (ws = %d yr)'%ws)
ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(PLN['ac'], '.b', label = 'Power Law Noise surrogates')
ax.plot(np.arange(0,len(WN))*1.3, WN['ac'], 'x', color='cornflowerblue', label = 'White Noise surrogates')
ax.axhline(y=0, color='grey', linestyle='dashed')
ax.plot(0, AMV_p[1], 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')
plt.legend(loc = 1)
plt.setp(ax, xticklabels=[], xticks=[])

ax = fig.add_subplot(322)
ax.set_ylabel('Slope of variance fit (ws = %d yr)'%ws)
ax.plot(PLN['var'], '.r', label = 'Power Law Noise surrogates')
ax.plot(np.arange(0,len(WN))*1.3, WN['var'], 'x', color='orange', label = 'White Noise surrogates')
ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.axhline(y=0, color='grey', linestyle='dashed')
ax.plot(0, AMV_p[0], 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')
plt.legend(loc = 1)
plt.setp(ax, xticklabels=[], xticks=[])

ax = fig.add_subplot(323)
ax.set_xlabel('Slope of lag-1 ac fit')
ax.set_ylabel('Frequency')
#ax.title("Slope of ar-1")
ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.hist(PLN['ac'], 30, color='b')
ax.hist(WN['ac'], 30, color='cornflowerblue', alpha=.9)
ax.axvline(x=0, color='grey', linestyle='dashed')
ax.plot(AMV_p[1], 2, 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')

ax = fig.add_subplot(324)
ax.set_xlabel('Slope of variance fit')
ax.set_ylabel('Frequency')
ax.text(-.15, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
#ax.title("Slope of variance")
ax.hist(PLN['var'], 30, color='r')
ax.hist(WN['var'], 30, color='orange', alpha=.9)
ax.axvline(x=0, color='grey', linestyle='dashed')
ax.plot(AMV_p[0], 2, 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')

ax = fig.add_subplot(325)
ax.set_xlabel('Slope of lag-1 ac fit')
ax.set_ylabel('p-value lag-1 ac (ws = %d yr)'%ws)
ax.plot(PLN['ac'], PLN['ac_pv'], '.b')
ax.plot(WN['ac'], WN['ac_pv'], 'x', color='cornflowerblue')
ax.plot(AMV_p[1], AMV_pv[1], 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')
ax.text(-.15, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

ax = fig.add_subplot(326)
ax.set_xlabel('Slope of variance fit')
ax.set_ylabel('p-value variance (ws = %d yr)'%ws)
ax.text(-.15, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(PLN['var'], PLN['var_pv'], '.r')
ax.plot(WN['var'], WN['var_pv'], 'x', color='orange')
ax.plot(AMV_p[0], AMV_pv[0], 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')

fig.savefig('FIG2.pdf', bbox_inches = 'tight')