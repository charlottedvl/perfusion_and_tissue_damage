"""
This file reads the brain mesh and creates boundary conditions by distributing
the average volumetric flow rate of blood to the brain on the boundaries based
on their surface area

@author: Tamas Istvan Jozsa
"""

from dolfin import *
import numpy as np
from perfusion.src.Legacy_version.io import IO_fcts
import argparse

import os

#%% settings

parser = argparse.ArgumentParser(description="Impose boundary conditions without considering large arteries")
parser.add_argument("--healthy", help="used only in healthy cases (optional)",
                    type=bool, default=True)
parser.add_argument('--occluded', help="must be used in occluded cases", dest='healthy', action='store_false')
parser.add_argument("--occl_ID", help="a list of integers containing occluded major cerebral artery IDs (e.g. 24 25 26)",
                    nargs='+', type=int, default=[25])
parser.add_argument("--folder", help="folder of output file (string ended with /)",
                    type=str, default=None)
parser.add_argument("--config_file", help="path to configuration file (string)",
                    type=str, default='./config_basic_flow_solver.yaml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                type=str, default=None)
parser.add_argument("--mesh_file", help="path to mesh_file",
                    type=str, default=None)

args = parser.parse_args()

cmd_flags = [args.healthy, args.occl_ID, args.folder, args.config_file]

[healthy, occl_ID, out_folder, config_file] = cmd_flags

#%%
configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)

if out_folder == None: res_file = configs['input']['inlet_boundary_file']
else: res_file = out_folder + 'BCs.csv'

if not os.path.exists(res_file.rsplit('/', 1)[0]):
    os.makedirs(res_file.rsplit('/', 1)[0])

# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

# volumetric flow rate to the brain [ml / s]
Q_brain = 10.0

boundary_labels = list(set(boundaries.array()))
n_labels = len(boundary_labels)
# 0: interior face, 1: brain stem cut plane,
# 2: ventricular surface, 2+: brain surface

#count superficial regions
mask = np.array(boundary_labels) > 2

# compute surface area for each boundary region
surf_area = []
dS = ds(subdomain_data=boundaries)

for i in range(n_labels):
    ID = int(boundary_labels[i])
    area_value = assemble( Constant(1)*dS(ID,domain=mesh))
    surf_area.append( area_value )
    # print(ID,area_value)

total_surface_area = sum(mask*surf_area)

# compute volumetric flow rates proportional to sperficial surface areas
Q = mask*surf_area*Q_brain/total_surface_area

# define pressure
p = mask * configs['physical']['p_arterial']

if configs['input']['mesh_file'].rsplit('/', 1)[-1] == 'clustered.xdmf':
    boundary_mapper = np.loadtxt(configs['input']['mesh_file'].rsplit('/', 1)[0]+'/boundary_mapper.csv',skiprows=1,delimiter=',')
    boundary_values = np.array(list(set( (boundary_mapper[:,1]>2)*boundary_mapper[:,1] ))[1::],dtype=int)
    boundary_map = np.zeros(len(boundary_values))
    cter = 0
    for i in list(boundary_values):
        boundary_map[cter] = int(boundary_mapper[np.argwhere(boundary_mapper[:,1]==int(i))[0],0])
        cter = cter + 1



boundary_matrix = []
for i in range(len(mask)):
    if mask[i]>0:
        if configs['input']['mesh_file'].rsplit('/', 1)[-1] == 'clustered.xdmf':
            boundary_matrix.append([boundary_labels[i],Q[i],p[i],boundary_map[np.argwhere(boundary_values==boundary_labels[i])],0])
        else:
            boundary_matrix.append([boundary_labels[i],Q[i],p[i],0,0])

boundary_matrix = np.array(boundary_matrix)

# handle occluded scenario
if healthy == False:
    for i in range(boundary_matrix.shape[0]):
        for j in range(len(occl_ID)):
            if boundary_matrix[i,3]==occl_ID[j]:
                boundary_matrix[i,1:3] = np.array([0,configs['physical']['p_venous']])
                boundary_matrix[i,4] = 1

# save file
fheader = 'cluster ID,Q [ml/s],p [Pa],feeding artery,BC: p->0 or Q->1'
np.savetxt(res_file, boundary_matrix,"%d,%f,%f,%d,%d",header=fheader)
