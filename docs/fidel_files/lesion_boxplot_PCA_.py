import numpy as np
import pandas as pd
import seaborn as sns
import scipy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib import rc
from matplotlib.patches import Patch

# === PLOT SETTINGS ===
mpl.rcParams.update({
    'font.size': 10,
    'lines.linewidth': 1,
    'lines.dashed_pattern': [5, 5],
    'lines.dotted_pattern': [1, 3],
    'lines.dashdot_pattern': [6.5, 2.5, 1.0, 2.5],
    'font.family': 'serif'
})
plt.close('all')

# === USER INPUT ===
stroke_type = "PCA"         # or "ACA", "PCA"
use_TLC = False             # Set to False for non-TLC case
suffix = "_TLC" if use_TLC else ""

# === LOAD DATA ===
IVr_df = pd.read_csv(f'./IVr_{stroke_type}{suffix}.csv')
IVr = np.concatenate((IVr_df.iloc[:, 2], IVr_df.iloc[:, 3]))

IVa_df = pd.read_csv(f'./IVa_{stroke_type}{suffix}.csv')
IVa = np.concatenate((IVa_df.iloc[:, 2], IVa_df.iloc[:, 3]))

IV_phan = np.loadtxt('./RPCA_Reference_Data.csv', delimiter=',', skiprows=1)[:, 1]

df = pd.DataFrame({
    'Infarct volume [ml]': np.concatenate([IVr, IVa, IV_phan]),
    'Dataset': (
        ['IVr'] * len(IVr) +
        ['IVa'] * len(IVa) +
        ['Reference data'] * len(IV_phan)
    )
})

# === FIGURE SETUP ===
fsx, fsy = 9, 7  # cm
fig = plt.figure(figsize=(fsx/2.54, fsy/2.54))
gs = plt.GridSpec(1, 1)
gs.update(left=0.17, right=0.97, bottom=0.12, top=0.97, wspace=0.2)
ax = plt.subplot(gs[0, 0])
ax.set_ylim([0, 150])
ax.set_yticks(np.arange(0, 200, 50))

# === BOXPLOT ===
sns.boxplot(
    x='Dataset',
    y='Infarct volume [ml]',
    data=df,
    palette=['C0', 'C1', 'C2'],
    ax=ax
)

# === STRIPPLOT ===
sns.stripplot(
    x='Dataset', 
    y='Infarct volume [ml]', 
    data=df,
    color="black", 
    jitter=True, 
    size=2,
    dodge=False,
    ax=ax
)

# === LEGEND ===
legend_patches = [
    Patch(color='C0', label='IVr'),
    Patch(color='C1', label='IVa'),
    Patch(color='C2', label='Reference data')
]
ax.legend(handles=legend_patches, loc='lower left', frameon=True)

# === LABELS ===
ax.set_xlabel('')
ax.set_ylabel('Infarct volume [ml]')

# === SAVE FIGURE ===
plt.savefig(f'plot_{stroke_type}{suffix}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'figure_{stroke_type}{suffix}.pdf', bbox_inches='tight')
