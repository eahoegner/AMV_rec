This repository provides code and data accompanying the report
### *E.A. Högner: Early Warning Signals and Recurrence Quantification Analysis in Atlantic Multidecadal Variability reconstruction (2023)*

the report was produced as part of the course "Nonlinear Data Analysis Concepts" by Norbert Marwan, University of Potsdam WS22/23

## Files
_______________

### CODE
*  FIG1.py:				Python code to replicate Fig. 1 and Fig.s A1 and A2 of the manuscript, executable.
*  FIG2.py:				Python code to replicate Fig. 2 of the manuscript, executable.
*  FIG3.py:				Python code to replicate Fig. 3 of the manuscript, executable.
*  FIG4.py:				Python code to replicate Fig. 4 and Fig.s A3 and A4 of the manuscript, executable.
*  FIG5.py:				Python code to replicate Fig. 5 of the manuscript, executable.
*  FigA5-7.py:				Python code to replicate Fig.s A5-A7 of the manuscript, executable.
*  EWS_functions_AMV.py:				Python code with subroutines for the EWS analysis.
*  recurrence.py:		Python code with subroutines for the recurrence analysis.
*  recurrence_quantification.py: Python code with subroutines for the recurrence quantification analysis.

## DATA
### INPUT data:
*  Micheletal2022_AMVreconstruction.txt:					                            AMV reconstruction from Michel et al. 2022*
*  all_proxies3.csv:															                            Proxy database used in the reconstruction
*  SurrogateReconstructions_PowerLawNoise_RandomForest_850_1987_nasstF.csv:		PLN surrogates
*  SurrogateReconstructions_WhiteNoise_RandomForest_850_1987_nasstF.csv: 		  WN surrogates
		
    (*Michel, S. L. L. et al.: Early warning signal for a tipping point suggested by a millennial Atlantic Multidecadal Variability reconstruction, Nat Commun, 13, 5176, 2022.*)


## Required modules
_______________

* numpy
* pandas
* matplotlib
* scipy
* statsmodels
* itertools
* mpl_toolkits


## Description
_______________

Each executable script (FIG*.py) can be run independently.
This code was implemented in Python 3.9. For each script, it is advised to first check dependencies. 
To ensure that potential errors are trackable, scripts should be executed more or less linewise (or 'chunkwise').

_________________________
E.A. Högner, 30/03/2023
