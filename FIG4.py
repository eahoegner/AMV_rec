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
surrogate_WN = surrogate_WN.drop(surrogate_WN.columns[0], axis=1)


################################################################
#----------FIG. 4 - Recurrence Plot AMV reconstruction---------#
################################################################

''' DATA PREP '''

time = np.arange(np.min(AMV[:,0]), np.min(AMV[:,0]) + len(AMV[:,0]))
AMV_df = pd.DataFrame(AMV)
standardised_AMV = standardise(AMV_df[AMV_df.columns[1:]], time)
data_AMV = standardised_AMV[1]

''' PLOT ''' 

fig = plt.figure(figsize=(9,9))

plt.imshow(rp_no_emb(data_AMV, 0.3),
           cmap="binary", origin="lower", 
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
plt.title("Recurrence plot AMV reconstruction, RR=0.3")
plt.xlabel("Time [yr CE]")
plt.ylabel("Time [yr CE]")

fig.savefig('FIG4.pdf', bbox_inches = 'tight')


################################################################
#-----------FIG. A4 - Recurrence Plot AMV + smoothed-----------#
################################################################

# compute 10 year running mean
data_AMV_smoothed = data_AMV.ewm(span = 10).mean()

''' PLOT '''

fig = plt.figure(figsize=(11,11))
fig.subplots_adjust(wspace=0.2)

ax1 = plt.subplot2grid((4,5), (0,0), colspan=4)
ax2 = plt.subplot2grid((4,5), (1,0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((4,5), (1, 2), colspan=2, rowspan=2)

ax1.plot(time, data_AMV, 'k')
ax1.plot(time, data_AMV.ewm(span = 10).mean(), color='lightgray')
ax1.set_title("detrended AMV reconstruction and 10-year moving average")

ax2.imshow(rp_no_emb(data_AMV, 0.3),
           cmap="binary", origin="lower", 
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
ax2.set_title("AMV reconstruction RP, RR=0.3")
ax2.set_xlabel("Time [yr CE]")
ax2.set_ylabel("Time [yr CE]")

ax3.imshow(rp_no_emb(data_AMV_smoothed, 0.3),
           cmap="binary", origin="lower",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
ax3.set_title("smoothed AMV reconstruction RP, RR=0.3")
plt.setp(ax3, yticklabels=[], yticks=[])
ax3.set_xlabel("Time [yr CE]")

fig.savefig('FigA3.pdf', bbox_inches = 'tight')



################################################################
#---------FIG. A5 - Recurrence Plot sample surrogates----------#
################################################################


''' PLOT '''

fig = plt.figure(figsize=(6,12.1), constrained_layout=True)
subfigs = fig.subfigures(2, 1)
#fig.subplots_adjust(wspace=.1, hspace=0)

ax1 = subfigs[0].add_subplot(221)
ax1.imshow(rp_no_emb(standardise2(surrogate_PLN["V4"], time), 0.3),
           cmap="binary", origin="lower",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
ax1.set_ylabel("Time [yr CE]")
plt.setp(ax1, xticklabels=[], xticks=[])

ax2 = subfigs[0].add_subplot(222)
ax2.imshow(rp_no_emb(standardise2(surrogate_PLN["V14"], time), 0.3),
           cmap="binary", origin="lower",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
plt.setp(ax2, xticklabels=[], xticks=[])
plt.setp(ax2, yticklabels=[], yticks=[])

ax3 = subfigs[0].add_subplot(223)
ax3.imshow(rp_no_emb(standardise2(surrogate_PLN["V24"], time), 0.3),
           cmap="binary", origin="lower",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
ax3.set_xlabel("Time [yr CE]")
ax3.set_ylabel("Time [yr CE]")

ax4 = subfigs[0].add_subplot(224)
ax4.imshow(rp_no_emb(standardise2(surrogate_PLN["V54"], time), 0.3),
           cmap="binary", origin="lower",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
plt.setp(ax4, yticklabels=[], yticks=[])
ax4.set_xlabel("Time [yr CE]")

subfigs[0].suptitle('Power Law Noise surrogates', size=14)



ax1 = subfigs[1].add_subplot(221)
ax1.imshow(rp_no_emb(standardise2(surrogate_WN["V4"], time), 0.3),
           cmap="binary", origin="lower",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
ax1.set_ylabel("Time [yr CE]")
plt.setp(ax1, xticklabels=[], xticks=[])

ax2 = subfigs[1].add_subplot(222)
ax2.imshow(rp_no_emb(standardise2(surrogate_WN["V14"], time), 0.3),
           cmap="binary", origin="lower",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
plt.setp(ax2, xticklabels=[], xticks=[])
plt.setp(ax2, yticklabels=[], yticks=[])

ax3 = subfigs[1].add_subplot(223)
ax3.imshow(rp_no_emb(standardise2(surrogate_WN["V24"], time), 0.3),
           cmap="binary",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None,
           origin="lower")
ax3.set_xlabel("Time [yr CE]")
ax3.set_ylabel("Time [yr CE]")

ax4 = subfigs[1].add_subplot(224)
ax4.imshow(rp_no_emb(standardise2(surrogate_WN["V54"], time), 0.3),
           cmap="binary", origin="lower",
           extent=[np.min(time),np.max(time),np.min(time),np.max(time)], interpolation=None)
plt.setp(ax4, yticklabels=[], yticks=[])
ax4.set_xlabel("Time [yr CE]")

subfigs[1].suptitle('White Noise surrogates', size=14)

fig.savefig('FigA4.pdf', bbox_inches = 'tight')