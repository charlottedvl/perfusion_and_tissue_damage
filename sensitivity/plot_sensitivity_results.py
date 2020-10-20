import numpy as np
import yaml
import os

import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib import rc
import matplotlib


rc('text', usetex = True)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.dashed_pattern'] = [5, 5]
matplotlib.rcParams['lines.dotted_pattern'] = [1, 3]
matplotlib.rcParams['lines.dashdot_pattern'] = [6.5, 2.5, 1.0, 2.5]

plt.close('all')

n_sample = 11

K1gm_ref = []
K2gm_ref = []

IV_K1gm_ref = []
IV_K2gm_ref = []

for i in range(n_sample):
    
    occluded_config_file = './config_files/config_RMCA_occl' +'{:02d}'.format(i)+'.yml'
    with open(occluded_config_file, "r") as configfile:
        occluded_configs = yaml.load(configfile, yaml.SafeLoader)
    K1gm_ref.append(occluded_configs['physical']['K1gm_ref'])
    
    IV_K1gm_ref.append( np.loadtxt(occluded_configs['output']['res_fldr']+'vol_infarct_values.csv',delimiter=',')[-1,-1] )


for i in range(n_sample):
    
    occluded_config_file = './config_files/config_RMCA_occl' +'{:02d}'.format(n_sample+i)+'.yml'
    with open(occluded_config_file, "r") as configfile:
        occluded_configs = yaml.load(configfile, yaml.SafeLoader)
    K2gm_ref.append(occluded_configs['physical']['K2gm_ref'])
    
    IV_K2gm_ref.append( np.loadtxt(occluded_configs['output']['res_fldr']+'vol_infarct_values.csv',delimiter=',')[-1,-1] )


K1gm_ref = np.array(K1gm_ref)
IV_K1gm_ref = np.array(IV_K1gm_ref)

K2gm_ref = np.array(K2gm_ref)
IV_K2gm_ref = np.array(IV_K2gm_ref)

fsx = 17
fsy = 8

fig1 = plt.figure(num=1, figsize=(fsx/2.54, fsy/2.54))
gs1 = plt.GridSpec(1, 2)
gs1.update(left=0.1, right=0.97, bottom=0.15, top=0.925, wspace=0.3, hspace=0.225)

ax1=plt.subplot(gs1[0,0])
ax1.plot(K1gm_ref,IV_K1gm_ref,'k.-')
ax1.set_xlim([K1gm_ref.min(),K1gm_ref.max()])
# ax1.set_ylim([-1,0])
ax1.set_xlabel(r'$k_a$',labelpad=5)
ax1.set_ylabel(r'$\mathrm{Infarcted~volume~[ml]}$',labelpad=5)
ax1.set_xscale('log')

ax2=plt.subplot(gs1[0,1])
ax2.plot(K2gm_ref,IV_K2gm_ref,'k.-')
ax2.set_xlim([K2gm_ref.min(),K2gm_ref.max()])
ax2.set_ylim([238,238.2])
ax2.set_xlabel(r'$k_c$',labelpad=5)
# ax2.set_ylabel(r'$\mathrm{Infarcted~volume~[ml]}$',labelpad=5)
ax2.set_xscale('log')

fig1.savefig('permeability_sensitivity.png',dpi=300)