"""
Estimate the tissue health (infarct fraction) based on perfusion in each element 
and the Green's function simulation results

p - perfusion [mL/100mL/min]
t - time [day]

Yidan Xue - 2021/01
"""
from dolfin import *
import scipy.interpolate
import numpy as np

# added module
import IO_fcts
import finite_element_fcts as fe_mod

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()

# healthy_perfusion

perfusion_gm = 56.04585125316684
perfusion_wm = 20.759413218846582
perfusion_scale = perfusion_gm/perfusion_wm

# %% READ INPUT
if rank == 0: print('Step 1: Reading and interpolating the cell death simulation results')

# fit the 2D plane - this might be optimised in the future
perfusion,time,dead = np.loadtxt('./dead_fraction.csv', unpack=True)
dead_estimate = scipy.interpolate.interp2d(perfusion,time,dead,kind='quintic')

# read the mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader('../brain_meshes/b0000/clustered.xdmf')
fe_order = 1

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, fe_order)

# read simulation results - the folder name should match the case
healthyfile = '../VP_results/p0000/mild_perfusion_occlusion/perfusion.xdmf'
# strokefile = configs.output.res_fldr + 'perfusion_stroke.xdmf'

# distinguish white and grey matter
dV = dx(subdomain_data=subdomains)
wm_idx = subdomains.where_equal(11)# white matter cell indices
gm_idx = subdomains.where_equal(12)# gray matter cell indices
num_wm_idx = len(wm_idx)
num_gm_idx = len(gm_idx)

if rank == 0:
    print('Step 2: Reading perfusion files')
# Load previous results
perfusion = Function(K2_space)
f_in = XDMFFile(healthyfile)
f_in.read_checkpoint(perfusion,'perfusion', 0)
f_in.close()

if rank == 0:
    print('Step 3: Calculating the infarct fraction')

# set the time [hour] - time points of interest
time_after = [2.0,4.0,6.0]

dead = Function(K2_space)
dead.vector()
dead.vector()[:]
dead_vec = dead.vector().get_local()
perfusion.vector()
perfusion.vector()[:]
perfusion_vec = perfusion.vector().get_local()

# scale the perfusion in white matter - this might be improved in the future
for n in range(len(time_after)):
    print('Simulating time point '+ str(n+1))
    # if this cell is in grey matter, then do nothing
    for i in range(num_gm_idx):
        if perfusion_vec[gm_idx[i]] > 60:
            dead_vec[gm_idx[i]] = 0
        else:
            dead_vec[gm_idx[i]] = dead_estimate(perfusion_vec[gm_idx[i]], time_after[n])
    # if this cell is in white matter, then scale the perfusion
    for i in range(num_wm_idx):
        perfusion_scaled = perfusion_vec[wm_idx[i]]*perfusion_scale
        if perfusion_scaled > 60:
            dead_vec[wm_idx[i]] = 0
        else:
            dead_vec[wm_idx[i]] = dead_estimate(perfusion_scaled, time_after[n])		
    
    dead.vector().set_local(dead_vec)
    
    vtkfile = File('/Users/xueyidan/Desktop/infarct_'+f'{(int(time_after[n]*100)):03}'+'.pvd')
    vtkfile << dead

print('Simulation finished')
