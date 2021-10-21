import numpy as np
import yaml
import argparse
import dolfin
import sys
sys.path.append('../')
import IO_fcts
import finite_element_fcts as fe_mod
import os

from matplotlib.pyplot import *
from matplotlib import colors, ticker, cm
from matplotlib import rc
from matplotlib import cm


rc('text', usetex = True)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.dashed_pattern'] = [5, 5]
matplotlib.rcParams['lines.dotted_pattern'] = [1, 3]
matplotlib.rcParams['lines.dashdot_pattern'] = [6.5, 2.5, 1.0, 2.5]

close('all')

#%% obtain settings and resuls
parser = argparse.ArgumentParser(description="compare analytical and numerical solutions for verification")
parser.add_argument("--config_analyt", help="path to analytical configuration file",
                    type=str, default='config_decoupled_analyt.yaml')
parser.add_argument("--config_numeric", help="path to numerical configuration file",
                    type=str, default='config_basic_flow_solver_verification.yaml')

configf_analyt = parser.parse_args().config_analyt
configf_numeric = parser.parse_args().config_numeric

with open(configf_analyt, "r") as myconfigfile:
        config_analyt = yaml.load(myconfigfile, yaml.SafeLoader)
with open(configf_numeric, "r") as myconfigfile:
        config_numeric = yaml.load(myconfigfile, yaml.SafeLoader)

# analytical pressure field
con_data = np.loadtxt(config_analyt['res_path'] + 'con_data.csv',delimiter=',')
x = con_data[:,0]
p = con_data[:,1]
vel = con_data[:,2]

#%% numerical pressure field
Ly = np.sqrt(config_analyt['continuum']['area'])*1000
Lz = np.sqrt(config_analyt['continuum']['area'])*1000
points = [(x_, Ly/2, Lz/2) for x_ in 1000*x] # 1D points

# read mesh
os.chdir('../')
mesh, subdomains, boundaries = IO_fcts.mesh_reader(config_numeric['input']['mesh_file'])

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p_numeric, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, config_numeric['simulation']['fe_degr'], \
                            model_type = config_numeric['simulation']['model_type'], \
                            vel_order = config_numeric['simulation']['vel_order'])

p_numeric = dolfin.Function(Vp)
f_in = dolfin.XDMFFile(config_numeric['output']['res_fldr']+'press1.xdmf')
f_in.read_checkpoint(p_numeric, 'press1', 0)
f_in.close()

p_num = x*0
for i in range(len(points)):
    try: p_num[i] = p_numeric(points[i])
    except: p_num[i] = np.NAN

Pa2mmHg = 0.00750062
nx = len(x)
skp = int(nx/30)
fig0 = figure(0)
plot(x*1000,p*Pa2mmHg,label='analytical')
plot(x[::skp]*1000,p_num[::skp]*Pa2mmHg,'rx',label='numerical')
xlim([0,12])
ylim([0,80])
xlabel('x [mm]')
ylabel('p [mmHg]')
legend()
fig0.savefig('./verification/pressure_comparison.png',dpi=450)
