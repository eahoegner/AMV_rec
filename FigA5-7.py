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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# local application imports
from EWS_functions_AMV import *
from recurrence import *
from recurrence_quantification import *

''' DATA '''
# import
AMV = np.loadtxt("Micheletal2022_AMVreconstruction.txt", comments='#',encoding='utf-8', dtype=float)
proxies = pd.read_csv('all_proxies3.csv')

################################################################
#----------FIG. 5 - Recurrence Plot AMV reconstruction---------#
################################################################

''' DATA PREP '''

time = np.arange(np.min(AMV[:,0]), np.min(AMV[:,0]) + len(AMV[:,0]))
time2 = np.arange(time[0], time[-1])
AMV_df = pd.DataFrame(AMV)
data = standardise(AMV_df[AMV_df.columns[1:]], time)[1]


''' PARAMETERS '''
ws = 250
bound = ws // 2
lmin = 5
RR = 0.3
time = np.arange(np.min(AMV[:,0]), np.min(AMV[:,0]) + len(AMV[:,0]))
time2 = np.arange(time[0], time[-1])



################################################################
#--------------FIG. A5 - RQA sensitivity analysis--------------#
################################################################

''' FUNCTIONS '''

def D_test(ts, m, tau, e):
    D = []
    for i in np.arange(1,15):
        D.append(det(rp(ts, m, tau, e, norm="euclidean", threshold_by="frr"), lmin = i, verb=False))
    return D

def L_test(ts, m, tau, e):
    L = []
    for i in np.arange(1,15):
        L.append(lam(rp(ts, m, tau, e, norm="euclidean", threshold_by="frr"), lmin = i, verb=False))
    return L

''' PLOT '''

fig = plt.figure(figsize=(13,13))
fig.subplots_adjust(wspace=0.25, hspace=0.35)

ax = fig.add_subplot(321)
ax.text(-.15, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(np.arange(1,15), D_test(data, 1, 0, 0.1), 'o-k', label="$\epsilon=0.1$")
ax.plot(np.arange(1,15), D_test(data, 1, 0, 0.2), 'x-k', label="$\epsilon=0.2$")
ax.plot(np.arange(1,15), D_test(data, 1, 0, 0.3), 'v-k', label="$\epsilon=0.3$")
ax.plot(np.arange(1,15), D_test(data, 1, 0, 0.4), '*-k', label="$\epsilon=0.4$")
ax.set_xlabel("lmin")
ax.set_ylabel("DET")
ax.set_title("DET sensitivity to recurrence threshold, no embedding")
plt.legend()

ax = fig.add_subplot(322)
ax.text(-.15, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(np.arange(1,15), L_test(data, 1, 0, 0.1), 'o-k', label="$\epsilon=0.1$")
ax.plot(np.arange(1,15), L_test(data, 1, 0, 0.2), 'x-k', label="$\epsilon=0.2$")
ax.plot(np.arange(1,15), L_test(data, 1, 0, 0.3), 'v-k', label="$\epsilon=0.3$")
ax.plot(np.arange(1,15), L_test(data, 1, 0, 0.4), '*-k', label="$\epsilon=0.4$")
ax.set_xlabel("lmin")
ax.set_ylabel("LAM")
ax.set_title("LAM sensitivity to recurrence threshold, no embedding")
plt.legend()

ax = fig.add_subplot(323)
ax.text(-.15, 1, s = 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(np.arange(1,15), D_test(data, 2, 0, 0.3), 'o-k', label="$\u03C4=0$")
ax.plot(np.arange(1,15), D_test(data, 2, 1, 0.3), 'x-k', label="$\u03C4=1$")
ax.plot(np.arange(1,15), D_test(data, 2, 2, 0.3), 'v-k', label="$\u03C4=2$")
ax.plot(np.arange(1,15), D_test(data, 2, 4, 0.3), '*-k', label="$\u03C4=4$")
ax.set_xlabel("lmin")
ax.set_ylabel("DET")
ax.set_title("DET sensitivity to delay $\u03C4$ with $m = 2$ and $\epsilon = 0.3$")
plt.legend()

ax = fig.add_subplot(324)
ax.text(-.15, 1, s = 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(np.arange(1,15), L_test(data, 2, 0, 0.3), 'o-k', label="$\u03C4=0$")
ax.plot(np.arange(1,15), L_test(data, 2, 1, 0.3), 'x-k', label="$\u03C4=1$")
ax.plot(np.arange(1,15), L_test(data, 2, 2, 0.3), 'v-k', label="$\u03C4=2$")
ax.plot(np.arange(1,15), L_test(data, 2, 4, 0.3), '*-k', label="$\u03C4=4$")
ax.set_xlabel("lmin")
ax.set_ylabel("LAM")
ax.set_title("LAM sensitivity to delay $\u03C4$ with $m = 2$ and $\epsilon = 0.3$")
plt.legend()

ax = fig.add_subplot(325)
ax.text(-.15, 1, s = 'e', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(np.arange(1,15), D_test(data, 1, 4, 0.3), 'o-k', label="m=1")
ax.plot(np.arange(1,15), D_test(data, 2, 4, 0.3), 'x-k', label="m=2")
ax.plot(np.arange(1,15), D_test(data, 3, 4, 0.3), 'v-k', label="m=3")
ax.plot(np.arange(1,15), D_test(data, 4, 4, 0.3), '*-k', label="m=4")
ax.set_xlabel("lmin")
ax.set_ylabel("DET")
ax.set_title("DET sensitivity to embedding dimension m with $\u03C4=4$, $\epsilon = 0.3$")
plt.legend()

ax = fig.add_subplot(326)
ax.text(-.15, 1, s = 'f', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(np.arange(1,15), L_test(data, 1, 4, 0.3), 'o-k', label="m=1")
ax.plot(np.arange(1,15), L_test(data, 2, 4, 0.3), 'x-k', label="m=2")
ax.plot(np.arange(1,15), L_test(data, 3, 4, 0.3), 'v-k', label="m=3")
ax.plot(np.arange(1,15), L_test(data, 4, 4, 0.3), '*-k', label="m=4")
ax.set_xlabel("lmin")
ax.set_ylabel("LAM")
ax.set_title("LAM sensitivity to embedding dimension m with $\u03C4=4$, $\epsilon = 0.3$")
plt.legend()

fig.savefig('FigA5.pdf', bbox_inches = 'tight')



################################################################
#----------FIG. A6 - running RQA sensitivity analysis----------#
################################################################


''' COMPUTATION '''
# test sensitivity to window size

DD = []
LL = []

for ws in np.arange(100,800,100):
    print(ws)
    bound = ws // 2
    runL = runlam(ws, data, 5, 0.3)
    runD = rundet(ws, data, 5, 0.3)
    p0, p1 = np.polyfit(time[bound : -bound], runL[bound : -bound], 1)
    LL.append(p0)
    p0, p1 = np.polyfit(time[bound : -bound], runD[bound : -bound], 1)
    DD.append(p0)
    
# test sensitivity to lmin in range from 1 to max length (12 in this case)
LSD = []
LSL = []

for i in np.arange(1,13):
    print(i)
    ws = 250
    bound = ws // 2
    runL = runlam(ws, data, i, 0.3)
    runD = rundet(ws, data, i, 0.3)
    p0, p1 = np.polyfit(time[bound : -bound], runL[bound : -bound], 1)
    LSL.append(p0)
    p0, p1 = np.polyfit(time[bound : -bound], runD[bound : -bound], 1)
    LSD.append(p0)

    
''' PLOT '''

fig = plt.figure(figsize = (15, 4.5))
fig.subplots_adjust(wspace=0.1)

ax = fig.add_subplot(121)
ax.text(-.05, 1, s = 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(np.arange(100,800,100), DD, '.-', color='mediumorchid', label="DET, lmin=5, RR=0.3")
ax.plot(np.arange(100,800,100), LL, '.-', color='aqua', label="LAM, lmin=5, RR=0.3")
ax.set_xlabel("window size")
ax.set_title("AMV reconstruction RQA sensitivity to window size")
plt.legend()

ax = fig.add_subplot(122)
ax.text(-.05, 1, s = 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.plot(np.arange(1,13), LSD, '.-', color='mediumorchid', label="DET, ws=250, RR=0.3")
ax.plot(np.arange(1,13), LSL, '.-', color='aqua', label="LAM, ws=250, RR=0.3")
ax.set_xlabel("minimum length")
ax.set_title("AMV reconstruction RQA sensitivity to minimum length")

# inset plot histograms
ax_ins = inset_axes(ax, width="45%", height=1.3, loc=1)
ax_ins.hist(diagonal_lines_hist(rp_no_emb(data, 0.3),verb=False)[2],
         diagonal_lines_hist(rp_no_emb(data, 0.3),verb=False)[1],
         facecolor="m", edgecolor='magenta', lw=2, label="diagonal lines")
ax_ins.hist(vertical_lines_hist(rp_no_emb(data, 0.3),verb=False)[2],
         vertical_lines_hist(rp_no_emb(data, 0.3),verb=False)[1],
         facecolor="c", edgecolor='cyan', lw=2, label="vertical lines", alpha=.4)

plt.legend()
plt.xlim(0,13)
plt.yticks([])
ax.legend(loc=3)

fig.savefig('FigA6.pdf', bbox_inches = 'tight')



################################################################
#---------FIG. A7 - Recurrence Plot AMV reconstruction---------#
################################################################

''' DATA PREP '''

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

proxies_standardised = []

for column in proxies_rec.columns[1:]:
    data = proxies_rec[column].dropna()
    L = len(data)
    time = np.arange(0, L)
    proxies_standardised.append(standardise2(data, time))
    
standardised_proxies = pd.DataFrame(proxies_standardised).transpose()

''' PARAMETERS '''
ws = 250
lmin = 5
RR = 0.3
bound = ws // 2
prox_D = []
prox_L = []

''' PLOT '''

fig = plt.figure(figsize=(12,16))
plt.subplots_adjust(wspace=0.25, hspace=0.2)

for i in np.arange(0,18):
    column = standardised_proxies.columns[i]
    ax = fig.add_subplot(6,3,i+1)
    data = standardised_proxies[column].dropna()
    L = len(data)
    if L % 2 == 1:
        data = data[1:]
        L = L-1
    time = np.arange(np.min(data), np.min(data) + L)
    time2 = np.arange(time[0], time[-1])
    runL = runlam(ws, data, lmin, RR)
    runD = rundet(ws, data, lmin, RR)
    p0, p1 = np.polyfit(time[bound : -bound], runD[bound : -bound], 1)
    prox_D.append(p0)
    
    ax.plot(time[bound : -bound], runD[bound : -bound], color='mediumorchid')
    ax.plot(time2, p0 * time2 + p1, color = 'mediumorchid', ls = '--')
    ax.set_xlim((time[0], time[-1]))
    ax.set_xlabel("Time [yr CE]")
    ax.set_ylabel("DET", color='mediumorchid')
    plt.setp(ax, yticklabels=[], yticks=[])
    ax.axvspan(time[0], time[bound], facecolor = 'none', edgecolor = 'k', hatch = '/')
    ax.axvspan(time[:-bound][-1], time[-1], facecolor = 'none', edgecolor = 'k', hatch = '/')

    p0, p1 = np.polyfit(time[bound : -bound], runL[bound : -bound], 1)
    prox_L.append(p0)
    
    ax2 = ax.twinx()
    ax2.plot(time[bound : -bound], runL[bound : -bound], color='aqua')
    ax2.plot(time2, p0 * time2 + p1, color = 'aqua', ls = '--')
    ax2.set_xlabel("Time [yr CE]")
    ax2.set_ylabel("LAM", color='aqua')
    plt.setp(ax, yticklabels=[], yticks=[])


fig.savefig('FigA7.pdf', bbox_inches = 'tight')