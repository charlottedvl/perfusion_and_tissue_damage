"""
This script creates a quasi-patient specific brain mesh using a readily
available mesh and an affine transformation matrix obtained from the EPAD
database (https://ep-ad.org/)

The generated mesh depends on the age and sex of the virtual patient and it
will be placed in the folder of the input mesh folder

33 male patients,   age range [47.71; 86.36]
42 female patients, age range [56.77; 81.43]

example:
python3 VP_mesh_prep --bsl_msh_fldr ./brain_meshes/b0000/ --age 75.1 --sex 1

@author: Tamas Istvan Jozsa
"""

#%% IMPORT MODULES
import sys
import argparse
import numpy as np
import yaml
import os, shutil
from scipy.interpolate import NearestNDInterpolator
import tables

#%% READ INPUT


parser = argparse.ArgumentParser(description="script generating quasi-patient specific brain mesh based on age and sex")
parser.add_argument("--bsl_msh_fldr", help="path of the baseline mesh folder, the included files are subjects to affine transformation",
                    type=str, default='./brain_meshes/b0000/')
parser.add_argument("--age", help="age of the virtual patient as a float",
                type=float, default=75)
parser.add_argument("--sex", help="sex of the virtual patient as an integer (1-male; 2-female)",
                type=int, default=1)
parser.add_argument('--forced', help="must be used to ensure that result folders are overwritten", dest='forced', action='store_true')

if parser.parse_args().bsl_msh_fldr[-1] != '/':
    bsl_msh_fldr = parser.parse_args().bsl_msh_fldr+'/'
else:
    bsl_msh_fldr = parser.parse_args().bsl_msh_fldr

patient_data_file = bsl_msh_fldr+'affine_matrices.yaml'
with open(patient_data_file, "r") as myfile:
    patient_data = yaml.load(myfile, yaml.SafeLoader)
age = np.array(patient_data['age'])
# correction to avoid same age twice (mistake in original data?)
age[45] = age[45]+np.random.rand()
age[46] = age[46]+np.random.rand()

sex = np.array(patient_data['sex'],dtype=np.int8)
# array of affine transformation matrices
Maffs = np.array(patient_data['affine_matrix'])

#%% find suitable affine transformation matrix based on nearest neighbour
interpolator = NearestNDInterpolator(list(zip(age, sex)), Maffs)
Maff = interpolator(parser.parse_args().age,parser.parse_args().sex)

#%% copy and modify files
res_fldr = bsl_msh_fldr[:-1] + '_age' + '{:04.2f}'.format(parser.parse_args().age) + '_sex' + '{:1d}'.format(parser.parse_args().sex) + '/'
fldr_exist = os.path.exists(res_fldr)
if fldr_exist and parser.parse_args().forced!=True:
    print('result folder already exists - script terminating')
else:
    if fldr_exist:
        shutil.rmtree(res_fldr)
    shutil.copytree(bsl_msh_fldr, res_fldr)

    # carry out affine transformation
    grid_data = tables.open_file(res_fldr+'clustered.h5',mode='r+')
    vertices_orig = grid_data.root.Mesh.mesh.geometry[:,:]
    vertices_modi = vertices_orig.copy()
    npoint = np.shape(vertices_orig)[0]
    for i in range(npoint):
        vertices_modi[i,:] = np.matmul(Maff,vertices_orig[i,:])
    grid_data.root.Mesh.mesh.geometry[:,:] = vertices_modi[:,:]
    grid_data.close()
    
    grid_data = tables.open_file(res_fldr+'clustered_facet_region.h5',mode='r+')
    grid_data.root.MeshFunction.__getattr__('0').mesh.geometry[:,:] = vertices_modi[:,:]
    grid_data.close()
    
    grid_data = tables.open_file(res_fldr+'clustered_physical_region.h5',mode='r+')
    grid_data.root.MeshFunction.__getattr__('0').mesh.geometry[:,:] = vertices_modi[:,:]
    grid_data.close()