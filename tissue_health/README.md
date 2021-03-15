# infarct_estimate_treatment.py

This Python script can predict the infarction based on the perfusion results in the whole brain model (healthy, stroke and treatment). This process consists of solving hypoxia-based cell death model twice, one is for the patient before treatment and another one is after treatment until the outcome imaging. 

The relationship between hypoxia and perfusion is a sigmoidal function based on statistical capillary networks (El-Bouri and Payne, 2015) and Green's function simulations (Secomb et al., 2004).

The required inputs and model parameters are included in config_tissue_damage.yaml. It should be noted that the VVUQ of these parameter values has not been finished yet.

###
Serial computation:
python3 infarct_estimate_treatment.py

###
Parallel compuation (under developing):
mpirun -n #number_of_processors python3 infarct_estimate_treatment.py

Yidan Xue - 03/2021
