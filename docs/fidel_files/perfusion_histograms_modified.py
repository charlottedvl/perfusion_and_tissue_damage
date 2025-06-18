import os
import numpy as np
import nibabel as nib
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import rc

# Configure matplotlib
plt.close('all')
rc('text', usetex=False)
plt.rcParams.update({
    'font.size': 12,
    'lines.linewidth': 1,
    'lines.dashed_pattern': [5, 5],
    'lines.dotted_pattern': [1, 3],
    'lines.dashdot_pattern': [6.5, 2.5, 1.0, 2.5]
})

#%% Set working directory
res_fldr = "/mnt/beegfs/home/s441846/figure07/"
wdir = os.getcwd()
os.chdir(res_fldr)

#%% Load MRI data
img_mri = nib.load('040EPAD00002_CBF_masked.nii.gz').get_fdata()
data_mri = img_mri[img_mri != -1024]
data_mri = data_mri[~np.isnan(data_mri)]

#%% Load simulated data
img_sim = nib.load('perfusion.nii').get_fdata()
data_sim = img_sim[img_sim != -1024]

#%% Create plot
fsx = 9  # width in cm
fsy = 9  # height in cm
fig, ax = plt.subplots(figsize=(fsx / 2.54, fsy / 2.54))

# KDE plots
sns.kdeplot(list(data_mri), color="green", fill=True,
            label="healthy reference's ASL MRI", ax=ax)
sns.kdeplot(list(data_sim), color="blue", fill=True, linestyle='-.',
            label='virtual brain simulation', ax=ax)

# Gaussian curve
normalx = np.arange(-150, 150, 0.1)
normaly = scipy.stats.norm(34, 24)
ax.plot(normalx, normaly.pdf(normalx), '--', linewidth=1.5,
        label='Gaussian distribution', color="red")

# Axes
ax.set_xlim([0, 100])
ax.set_ylim([0, 0.06])
ax.set_xlabel('CBF [ml/min/100g]', labelpad=2)
ax.set_ylabel('Probability density function', labelpad=4)

# Legend outside the plot (above)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=1, frameon=False)

# Manual adjustment of margins to prevent label clipping
plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.15)

# Save figure
os.chdir(wdir)
fig.savefig('fig7.pdf', bbox_inches='tight')