# installed python3 modules
import dolfin
import time
import sys
import argparse
import numpy
import nibabel as nib
numpy.set_printoptions(linewidth=200)

# ghost mode options: 'none', 'shared_facet', 'shared_vertex'
dolfin.parameters['ghost_mode'] = 'none'

# added module
import IO_fcts
import suppl_fcts
import finite_element_fcts as fe_mod

# solver runs is "silent" mode
dolfin.set_log_level(50)


#%% READ INPUT
parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                type=str, default='../VP_results/p0000/perfusion_healthy/')
parser.add_argument("--variable", help="e.g. press1, vel1, perfusion, K1, etc.",
                type=str, default='perfusion')
config_file = parser.parse_args().res_fldr + 'settings.yaml'

configs = IO_fcts.basic_flow_config_reader_yml(config_file,parser)
# physical parameters
p_arterial, p_venous = configs['physical']['p_arterial'], configs['physical']['p_venous']
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
    configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], \
    configs['physical']['K3gm_ref'], configs['physical']['gmowm_perm_rat']
beta12gm, beta23gm, gmowm_beta_rat = \
    configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat']

try:
    compartmental_model = configs['simulation']['model_type'].lower().strip()
except KeyError:
    compartmental_model = 'acv'

try:
    velocity_order = configs['simulation']['vel_order']
except KeyError:
    velocity_order = configs['simulation']['fe_degr'] - 1

# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, configs['simulation']['fe_degr'], \
                            model_type = compartmental_model, vel_order = velocity_order)


#%% READ FE RESULT
my_variable = parser.parse_args().variable.strip().lower()

variable_list = ['press1', 'press2', 'press3', 'vel1', 'vel2', 'vel3',
                 'k1', 'k2', 'k3', 'beta12', 'beta23', 'perfusion']

# check if variable is in the list
var_check = []
for i in variable_list:
    var_check.append( my_variable == i )
if not any(var_check):
    sys.exit("variable specified by '--variable' is not available")

if my_variable[:3] == 'per':
    myvar = dolfin.Function(K2_space)
    vartype = 'scalar'
elif my_variable[:3] == 'pre':
    Vp = dolfin.FunctionSpace(mesh, "Lagrange", configs['simulation']['fe_degr'])
    myvar = dolfin.Function(Vp)
    vartype = 'scalar'
elif my_variable[:3] == 'vel':
    myvar = dolfin.Function(Vvel)
    vartype = 'vector'
elif my_variable[0] == 'k':
    my_variable = my_variable.upper()
    myvar = dolfin.Function(K1_space)
    vartype = 'tensor'
elif my_variable[:3] == 'bet':
    myvar = dolfin.Function(K2_space)
    vartype = 'scalar'

try:
    f_in = dolfin.XDMFFile(configs['output']['res_fldr'] + my_variable + '.xdmf')
    f_in.read_checkpoint(myvar, my_variable, 0)
    f_in.close()
except ValueError:
    print(my_variable + '.xdmf file not available!')

# check read data
# file = dolfin.File("check.pvd")
# file << myvar


#%% CONVERT FE DATA TO IMAGE

img_coord_min = numpy.int32(numpy.floor(numpy.min(mesh.coordinates(),axis=0)))-1
img_coord_max = numpy.int32(numpy.ceil( numpy.max(mesh.coordinates(),axis=0)))+1

x = numpy.arange(img_coord_min[0], img_coord_max[0])
y = numpy.arange(img_coord_min[1], img_coord_max[1])
z = numpy.arange(img_coord_min[2], img_coord_max[2])
nx,ny,nz = len(x), len(y), len(z)

# TODO: speed up image recovery
if vartype == 'scalar':
    img_data = numpy.zeros([nx,ny,nz])
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                my_point = (x[i],y[j],z[k])
                try: img_data[i,j,k] = myvar(my_point)
                except: img_data[i,j,k] = 0
elif vartype == 'vector':
    img_data = numpy.zeros([nx,ny,nz,3])
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                my_point = (x[i],y[j],z[k])
                try: img_data[i,j,k,:] = myvar(my_point)
                except: img_data[i,j,k,:] = numpy.zeros(3)
elif vartype == 'tensor':
    img_data = numpy.zeros([nx,ny,nz,9])
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                my_point = (x[i],y[j],z[k])
                try: img_data[i,j,k,:] = myvar(my_point)
                except: img_data[i,j,k,:] = numpy.zeros(9)

affine_matrix = numpy.eye(4)
affine_matrix[:3,-1] = img_coord_min+1
img = nib.Nifti1Image(img_data, affine_matrix)
nib.save(img, configs['output']['res_fldr'] + my_variable +'.nii.gz')