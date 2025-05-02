# FICUS-with-Modifications
FICUS as used in Huberty et al. 2025 (see https://ui.adsabs.harvard.edu/abs/2025arXiv250113899H/abstract)
Based on Saldana-Lopez et al. 2023 (see https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.6295S/abstract) 
Please be sure to cite the Saldana-Lopez paper too. 
Alberto gives a great overview of the base FICUS code in this github: https://github.com/asalda/FiCUS (please look through this one first)

Further Modifications Made in Huberty et al. 2025:
1. Addition of damping wing system, characterized by a column density, N.
--> Both with and with out an ionized bubble. 
2. Addition of IGM absorption, originally based on from of Miralda-Escude 1998 characterized by x_HI (the neutral Hydrogen Fraction).
--> Can comment out this part of the code as needed.
3. Conversion to MCMC method of fitting from gradient descent. 

The ficus.py and ficus_scripts.py code should be able to run in the same manner as in the Saldana-Lopez github inventory. 
Running it on this version will create a file called "outputchains"+str(spec_name)+".txt" that saves your MCMC chains for you to analyze. 

Please email huber458@umn.edu with questions!



