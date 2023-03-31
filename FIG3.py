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
#---------FIG. 3 - EWS and significance testing proxies--------#
################################################################

# calculate standardised AMV reconstruction's EWS slopes and p values

''' DATA PREP '''

time = np.arange(np.min(AMV[:,0]), np.min(AMV[:,0]) + len(AMV[:,0]))
time2 = np.arange(time[0], time[-1])
AMV_df = pd.DataFrame(AMV)
data = standardise(AMV_df[AMV_df.columns[1:]], time)[1]

''' PARAMETERS '''

ws = 50 # window size
bound = ws // 2
tt_samples = 100 # number of fourier surrogates for significance testing of the trend
AMV_p = []
AMV_pv = []

''' COMPUTATION '''

# compute running variance
p0, p1 = np.polyfit(time[bound : -bound], runstd(data, ws)[bound : -bound]**2, 1)

# extract variance slope and p-value
pv = kendall_tau_test(data, tt_samples, ws, p0, "var", bound) 
AMV_p.append(p0)
AMV_pv.append(pv)

# compute running AR1
p0, p1 = np.polyfit(time[bound : -bound], runac(data, ws)[bound : -bound], 1)

# extract AR1 slope and p-value
pv = kendall_tau_test(data, tt_samples, ws, p0, "ar", bound)
AMV_p.append(p0)
AMV_pv.append(pv)


# calculate EWS for proxies

''' DATA PREP '''

# extract the 55 proxies from larger proxy dataset that were used in reconstruction

proxies_rec = proxies[["Year", "Arc.Lamoureux1996", "Arc.Schneider.2015.PolarUrals", "Arc.Zhang.Jamtland.2016",
                      "Asia.AGHERI", "Asia.BAG1JU", "Asia.BAG5JU", "Asia.BAWSIC", "Asia.BUKPIG", "Asia.CGPCDD",
                      "Asia.CGPPIG", "Asia.CHEPCS", "Asia.CHP2JU", "Asia.CHP3JU", "Asia.DAOSIC", "Asia.ESPPAK",
                      "Asia.GEFKJT", "Asia.HENTLS", "Asia.HYGJUP", "Asia.JOTPIG", "Asia.JUTPCS", 
                      "Asia.Kur2.Magda.2011", "Asia.MCCHFH", "Asia.MOR2JU", "Asia.MQAXJP", "Asia.MTASSP",
                      "Asia.NOGSAK", "Asia.QAMDJT", "Asia.TANSIC", "Asia.TSHEPS", "Asia.UULEWD",
                      "Asia.UULMND", "Asia.WL2", "Asia.YKA", "Asia.YKCOM", "Asia.ZACCDD", "Asia.ZMGLPS",
                      "Eur.Lotschental.Buentgen.2006", "Eur.SPyrenees.LiNaNn.2012", "Eur.Tallinn.Tarand.2001",
                      "Eur.Tat12", "NaNm.ak046", "NaNm.ak058", "NaNm.ca609", "NaNm.caNaN231", "NaNm.caNaN238",
                      "NaNm.caNaN382", "NaNm.caNaN439", "NaNm.caNaN446", "NaNm.caNaN449", "NaNm.caNaN453",
                      "NaNm.mt116", "NaNm.mt120", "Ocean2kHR.PacificMaiaNaNUrban2000",
                      "PotomacRiver", "Quebec.x"]]


# standardise + detrend

proxies_standardised = []

for column in proxies_rec.columns[1:]:
    data = proxies_rec[column].dropna()
    L = len(data)
    time = np.arange(0, L)
    proxies_standardised.append(standardise2(data, time))
    
standardised_proxies = pd.DataFrame(proxies_standardised).transpose()

prox_var = []
prox_ac = []
prox_var_pv = []
prox_ac_pv = []
length = []
name = []


''' COMPUTE EWS '''

for column in standardised_proxies:
    data = standardised_proxies[column].dropna()
    L = len(data)
    if L % 2 == 1:
        data = data[1:]
        L = L-1
    time = np.arange(np.min(data), np.min(data) + L)
    time2 = np.arange(time[0], time[-1])
    name.append(column)
    length.append(L)
    
    popt31, cov = curve_fit(funcfit3, time[time > time[0]], data[time > time[0]], p0 = [-1,  .1,  2030], maxfev = 1000000000, jac = funcfit3_jac)

    # variance fit
    p0, p1 = np.polyfit(time[bound : -bound], runstd(data, ws)[bound : -bound]**2, 1)
    
    # p value significance test
    pv = kendall_tau_test((data), tt_samples, ws, p0, "var", bound)
    prox_var.append(p0)
    prox_var_pv.append(pv)
    
    # autocorrelation fit
    p0, p1 = np.polyfit(time[bound : -bound], runac(data - funcfit3(time, *popt31), ws)[bound : -bound], 1)
    
    # p value significance test
    pv = kendall_tau_test((data), tt_samples, ws, p0, "ar", bound)
    prox_ac.append(p0)    
    prox_ac_pv.append(pv)


''' PLOT '''

fig = plt.figure(figsize = (13,16))
plt.subplots_adjust(wspace=0.25, hspace=0.25)

ax = fig.add_subplot(421)
ax.set_ylabel('Slope of lag-1 ac fit (ws = %d yr)'%ws)
ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(prox_ac, '.b', label = 'Proxies')
ax.axhline(y=0, color='grey', linestyle='dashed')
ax.plot(0, AMV_p[1], 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')
plt.legend(loc = 3)

ax = fig.add_subplot(422)
ax.set_ylabel('Slope of variance fit (ws = %d yr)'%ws)
ax.plot(prox_var, '.r', label = 'Proxies')
ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.axhline(y=0, color='grey', linestyle='dashed')
#ax.set_ylim(-1e-3,1e-3)
ax.plot(0, AMV_p[0], 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')
plt.legend(loc = 3)

ax = fig.add_subplot(423)
ax.set_xlabel('Slope of lag-1 ac fit')
ax.set_ylabel('Frequency')
ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.hist(prox_ac, 30, label = 'Proxies', color='blue')
ax.axvline(x=0, color='grey', linestyle='dashed')
ax.plot(AMV_p[1], 0.1,
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')

ax = fig.add_subplot(424)
ax.set_xlabel('Slope of variance fit')
ax.set_ylabel('Frequency')
ax.text(-.15, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.hist(prox_var, 30, label = "Proxies", color='red')
ax.axvline(x=0, color='grey', linestyle='dashed')
ax.plot(AMV_p[0], 0.1,
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='AMV reconstruction')

ax = fig.add_subplot(425)
ax.set_xlabel('Slope of lag-1 ac fit')
ax.set_ylabel('Length of record')
ax.plot(prox_ac, length, '.b', label = 'Proxies')
ax.axvline(x=0, color='grey', linestyle='dashed')
ax.text(-.15, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

ax = fig.add_subplot(426)
ax.set_xlabel('Slope of variance fit')
ax.set_ylabel('Length of record')
ax.plot(prox_var, length, '.r', label = 'Proxies')
ax.axvline(x=0, color='grey', linestyle='dashed')
ax.text(-.15, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

ax = fig.add_subplot(427)
ax.set_xlabel('Slope of lag-1 ac fit')
ax.set_ylabel('p-value lag-1 ac (ws = %d yr)'%ws)
ax.plot(prox_ac, prox_ac_pv, '.b')
ax.plot(AMV_p[1], AMV_pv[1], 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='Proxies')
ax.text(-.15, 1, s = 'g', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

ax = fig.add_subplot(428)
ax.set_xlabel('Slope of variance fit')
ax.set_ylabel('p-value variance (ws = %d yr)'%ws)
ax.text(-.15, 1, s = 'h', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(prox_var, prox_var_pv, '.r')
ax.plot(AMV_p[0], AMV_pv[0], 
        'v', markerfacecolor='w', markersize=10, markeredgewidth=1.5, markeredgecolor='lime', 
        label='Proxies')

fig.align_ylabels()
fig.savefig('FIG3.pdf', bbox_inches = 'tight')
