"""
Estimate the tissue health (infarct fraction) based on perfusion in each element
and the Green's function simulation results

This code considers the treatment (perfusion recovery) and its outcome

p - perfusion [mL/100mL/min]
t - time [hour]

The script can be ran with and without arguments. When no arguments are given,
the default configuration file at `./config_tissue_damage.yaml` is considered
and no `tissue_health_outcome.yml` is written (the latter is a insist-pipeline
specific outcome.

When passing two arguments, the first arguments contains the path to an
alternative YAML file to be used as the configuration file and the second
argument the path where to write an outcome summary as YAML to.

Yidan Xue - 2021/11
"""
from dolfin import *
import scipy.interpolate
from scipy.integrate import odeint
import numpy as np
import yaml
import time
import sys

# added module
import IO_fcts
import finite_element_fcts as fe_mod

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()

start0 = time.time()

# %% READ INPUT
if rank == 0:
    print('Step 1: Reading the input')

# read the .yaml file
path = './config_tissue_damage.yaml' if len(sys.argv) == 0 else sys.argv[1]
with open(path, "r") as configfile:
    configs = yaml.load(configfile, yaml.SafeLoader)

# read the mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])
fe_order = 1

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, fe_order)

# distinguish white and grey matter
dV = dx(subdomain_data=subdomains)
wm_idx = subdomains.where_equal(11)# white matter cell indices
gm_idx = subdomains.where_equal(12)# gray matter cell indices
num_wm_idx = len(wm_idx)
num_gm_idx = len(gm_idx)
num = len(wm_idx)+len(gm_idx) # total number of elements

# read perfusion results folders - the folder name should match the case
healthyfile = configs['input']['healthyfile']
strokefile = configs['input']['strokefile']
treatmentfile = configs['input']['treatmentfile']

# read the time parameters - both in hours
arrival_time, recovery_time = configs['input']['arrival_time'], configs['input']['recovery_time']

# sigmoidal function parameters for hypoxia estimation
ks1, ks2 = configs['parameter']['ks1'], configs['parameter']['ks2']

# cell death model parameters
# kf - forward rate constants
# kt - toxic production constant
# kb - toxic recycle constant
kf, kt, kb = configs['parameter']['kf'], configs['parameter']['kt'], configs['parameter']['kb']

# scale ratio between grey and white matters
perfusion_scale = configs['parameter']['perfusion_gm_wm']

# cell death threshold for core
core_threshold = configs['parameter']['core_threshold']

# define the relationship between hypoxic fraction and perfusion - based on Green's function simulations
def hypoxia_estimate(perfusion):
    return 1-1/(1+np.exp(-(ks1*perfusion+ks2)))

# define the ODE for cell death, h[0] - dead, h[1] - toxic, h[2] - hypoxic fraction
def cell_death(h, t):
    a = 1 - h[0]
    hypo = h[2]
    return [a*kf*h[1], a*kt*hypo-kb*h[1]*(1-hypo)*a, 0]

if rank == 0:
    print('Step 2: Reading perfusion files')
# load previous results

# read the healthy perfusion
perfusion_healthy = Function(K2_space)
f_in = XDMFFile(healthyfile)
f_in.read_checkpoint(perfusion_healthy,'perfusion', 0)
f_in.close()

# read the perfusion after stroke before treatment
perfusion_stroke = Function(K2_space)
f_in = XDMFFile(strokefile)
f_in.read_checkpoint(perfusion_stroke,'perfusion', 0)
f_in.close()

# read the perfusion after treatment
perfusion_treatment = Function(K2_space)
f_in = XDMFFile(treatmentfile)
f_in.read_checkpoint(perfusion_treatment,'perfusion', 0)
f_in.close()

if rank == 0:
    print('Step 3: Calculating the infarct fraction')

perfusion_healthy_vec = perfusion_healthy.vector().get_local()
perfusion_stroke_vec = perfusion_stroke.vector().get_local()
perfusion_treatment_vec = perfusion_treatment.vector().get_local()

# define the dead fraction and toxic state in each element

dead = Function(K2_space)
dead_vec = dead.vector().get_local()
toxic = Function(K2_space)
toxic_vec = toxic.vector().get_local()

# initialise the ODEs
# before treatment - time in seconds now
t_b = np.linspace(0,arrival_time*3600,2)
# after treatment - time in seconds now
t_a = np.linspace(0,recovery_time*3600,2)

start1 = time.time()
core = 0
# for grey matter
for i in range(num_gm_idx):
    # if the change in perfusion is smaller than 5% - no cell death
    if (perfusion_healthy_vec[gm_idx[i]]-perfusion_stroke_vec[gm_idx[i]])/perfusion_healthy_vec[gm_idx[i]] < 0.05:
        dead_vec[gm_idx[i]] = 0
    # if the change is larger
    else:
        hi1 = [0,0,hypoxia_estimate(perfusion_stroke_vec[gm_idx[i]])] # first input: after onset
        hs = odeint(cell_death, hi1, t_b)
        Dead = hs[-1,0]
        Toxic = hs[-1,1]
        hi2 = [Dead,Toxic,hypoxia_estimate(perfusion_treatment_vec[gm_idx[i]])] # second input: after treatment
        hs = odeint(cell_death, hi2, t_a)
        dead_vec[gm_idx[i]] = hs[-1,0]
        # core volume
        if dead_vec[gm_idx[i]] > core_threshold:
            ID = int(gm_idx[i])
            core = core + Cell(mesh, ID).volume()/1000

# for white matter
for i in range(num_wm_idx):
    # if the change in perfusion is smaller than 5% - no cell death
    if (perfusion_healthy_vec[wm_idx[i]]-perfusion_stroke_vec[wm_idx[i]])/perfusion_healthy_vec[wm_idx[i]] < 0.05:
        dead_vec[wm_idx[i]] = 0
    # if the change is larger
    else:
        hi1 = [0,0,hypoxia_estimate(perfusion_stroke_vec[wm_idx[i]]*perfusion_scale)] # first input: after onset
        hs = odeint(cell_death, hi1, t_b)
        Dead = hs[-1,0]
        Toxic = hs[-1,1]
        hi2 = [Dead,Toxic,hypoxia_estimate(perfusion_treatment_vec[wm_idx[i]]*perfusion_scale)] # second input: after treatment
        hs = odeint(cell_death, hi2, t_a)
        dead_vec[wm_idx[i]] = hs[-1,0]
        # core volume
        if dead_vec[wm_idx[i]] > core_threshold:
            ID = int(wm_idx[i])
            core = core + Cell(mesh, ID).volume()/1000

end1 = time.time()
dead.vector().set_local(dead_vec)

# vtkfile = File(configs['output']['res_fldr']+'infarct_'+str(arrival_time)+'_'+str(recovery_time)+'.xdmf')
# vtkfile << dead

with XDMFFile(configs['output']['res_fldr']+'infarct_'+str(arrival_time)+'_'+str(recovery_time)+'.xdmf') as myfile:
    myfile.write_checkpoint(dead,"dead", 0, XDMFFile.Encoding.HDF5, False)

if len(sys.argv) >= 2:
    # The second argument indicates the path where to write a summary of
    # outcome parameters too, this now considers only the infarct core volume.
    with open(sys.argv[2], 'w') as outfile:
        yaml.safe_dump(
            {'core-volume': core},
            outfile
        )

end0 = time.time()

if rank == 0:
    print('The core volume is '+str(core)+' mL')
    print('Infarct computation time [s]; \t\t\t', end1 - start1)
    print('Simulation finished - Total execution time [s]; \t\t\t', end0 - start0)
