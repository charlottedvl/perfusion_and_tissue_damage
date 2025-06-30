"""
This file converts a .msh file into a .nii.gz format.

@author: Charlotte Devillé
"""

# Python imports
import dolfin
import sys
import argparse
import numpy
import nibabel as nib
import matplotlib.pyplot as plt
import os

# Local imports
from src.Legacy_version.io import IO_fcts
from src.Legacy_version.utils import finite_element_fcts as fe_mod

def create_parser():
    parser = argparse.ArgumentParser(
        description="Convert finite elements results (*.h5 and *.xdmf) into an image (*.nii.gz)")
    parser.add_argument("--config_file", help="Path to configuration file",
                        type=str, default='./configs/config_basic_flow_solver.yaml')
    parser.add_argument("--res_fldr", help="Path to results folder (string ended with /)", type=str, default=None)
    parser.add_argument("--variable", help="One of the following: press1, press2, press3, vel1, vel2, vel3, \
                     k1, k2, k3, beta12, beta23, perfusion", type=str, default='perfusion')
    parser.add_argument("--voxel_size", help="Voxel edge size in [mm]", type=int, default=2)
    parser.add_argument("--background_value", help="Value used for background voxels", type=int, default=-1024)
    parser.add_argument('--save_figure', action='store_true', help="Save figure showing image along midline slices", \
                        default=False)
    return parser

def prepare_voxel_size(arguments):
    voxel_size = numpy.array(arguments.voxel_size)
    try:
        if len(voxel_size) != 3:
            voxel_size = numpy.array([voxel_size[0], voxel_size[0], voxel_size[0]], dtype=float)
    except:
        voxel_size = numpy.array([voxel_size, voxel_size, voxel_size], dtype=float)
    return voxel_size

def allocate_field(variable_name, mesh, K1_space, K2_space, Vvel, simulation_configs):
    """
    Allocates the appropriate dolfin.Function and determines variable type
    based on the variable name (e.g., 'k1', 'vel1', 'press1', etc.)
    """
    prefix = variable_name[:3]
    dolfin_function, variable_type = None, ""

    if prefix == 'per':
        dolfin_function = dolfin.Function(K2_space)
        variable_type = 'scalar'
    elif prefix == 'pre':
        Vp = dolfin.FunctionSpace(mesh, "Lagrange", simulation_configs.get('fe_degr'))
        dolfin_function = dolfin.Function(Vp)
        variable_type = 'scalar'
    elif prefix == 'vel':
        dolfin_function = dolfin.Function(Vvel)
        variable_type = 'vector'
    elif variable_name[0] == 'k':
        variable_name = variable_name.upper()
        dolfin_function = dolfin.Function(K1_space)
        variable_type = 'tensor'
    elif prefix == 'bet':
        dolfin_function = dolfin.Function(K2_space)
        variable_type = 'scalar'
    else:
        raise ValueError(f"Unknown variable type: {variable_name}")
    return dolfin_function, variable_type, variable_name

def load_dolfin_data(variable, dolfin_variable, results_folder):
    try:
        f_in = dolfin.XDMFFile(results_folder + variable + '.xdmf')
        f_in.read_checkpoint(dolfin_variable, variable, 0)
        f_in.close()
    except ValueError:
        print(variable + '.xdmf file not available!')
    return

def create_image_grid(mesh, voxel_size):
    image_coord_min = numpy.int32(numpy.floor(numpy.min(mesh.coordinates(), axis=0) - voxel_size))
    image_coord_max = numpy.int32(numpy.ceil(numpy.max(mesh.coordinates(), axis=0) + voxel_size))

    x = numpy.arange(image_coord_min[0], image_coord_max[0], voxel_size[0])
    y = numpy.arange(image_coord_min[1], image_coord_max[1], voxel_size[1])
    z = numpy.arange(image_coord_min[2], image_coord_max[2], voxel_size[2])
    return image_coord_min, x, y, z

# TODO: speed up image recovery
def finite_element_to_image_data(var, variable_type, x, y, z, length_x, length_y, length_z, bg_value):
    if variable_type == 'scalar':
        image_data = numpy.ones((length_x, length_y, length_z)) * bg_value
    elif variable_type == 'vector':
        image_data = numpy.ones((length_x, length_y, length_z, 3)) * bg_value
    elif variable_type == 'tensor':
        image_data = numpy.ones((length_x, length_y, length_z, 9)) * bg_value
    else:
        raise ValueError(f"Unknown var_type: {variable_type}")

    for i in range(length_x):
        for j in range(length_y):
            for k in range(length_z):
                point = (x[i], y[j], z[k])
                try:
                    value = var(point)
                    if variable_type == 'scalar':
                        image_data[i, j, k] = value
                    else:
                        image_data[i, j, k, :] = value
                except Exception:
                    pass # image_date have been instantiated to the default value so no need to update again
    return image_data

def save_nifti(voxel_size, image_coord_min, image_data, results_folder, variable):
    affine_matrix = numpy.eye(4)
    for i in range(3): affine_matrix[i, i] = voxel_size[i]
    affine_matrix[:3, -1] = image_coord_min + 1
    img = nib.Nifti1Image(image_data, affine_matrix)
    nib.save(img, results_folder + variable + '.nii.gz')
    return

def save_image(image_data, results_directory, args_variable, length_x, length_y, length_z):
    dimensions = len(list(image_data.shape))
    if dimensions == 4:
        image_data = numpy.linalg.norm(image_data, axis=3)
    if dimensions != 3:
        print('Saving figure is not available for tensor spaces!')
        return
    slices = [image_data[int(length_x / 2), :, :],
              image_data[:, int(length_y / 2), :],
              image_data[:, :, int(length_z / 2)]]

    fsx = 17
    fsy = 8
    fig1 = plt.figure(num=1, figsize=(fsx / 2.54, fsy / 2.54))
    gs1 = plt.GridSpec(1, 2)
    gs1.update(left=0.05, right=0.99, bottom=0.01, top=0.99, wspace=0.2)

    for index in [1, 2]:
        ax = plt.subplot(gs1[0, index - 1])
        ax.imshow(numpy.flip(numpy.rot90(slices[index]), axis=1), cmap='gist_gray', vmin=0, vmax=image_data.max())
    fig1.savefig(results_directory + args_variable.strip().lower() + '.png', transparent=True, dpi=450)

def main():
    # DOLFIN settings
    dolfin.parameters['ghost_mode'] = 'none' # ghost mode options: 'none', 'shared_facet', 'shared_vertex'
    # solver runs is "silent" mode
    dolfin.set_log_level(50)

    # Read input
    parser = create_parser()
    args = parser.parse_args()

    # Read config file
    config_file = args.config_file
    if not os.path.isfile(config_file):
        config_file = args.res_fldr + 'settings.yaml'

    voxel_size = prepare_voxel_size(args)

    configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
    results_folder = configs['output']['res_fldr']

    # Simulation parameters
    simulation = configs.get('simulation', {})
    compartmental_model = simulation.get('model_type', 'acv').lower().strip()
    velocity_order = simulation.get('vel_order', simulation.get('fe_degr', 2) - 1)

    # Read mesh
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

    # Determine functions space
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_mod.alloc_fct_spaces(mesh, simulation.get('fe_degr'), model_type=compartmental_model,
                                vel_order=velocity_order)

    # Check if variable is a valid variable
    variable = args.variable.strip().lower()
    valid_variables = ['press1', 'press2', 'press3', 'vel1', 'vel2', 'vel3',
                       'k1', 'k2', 'k3', 'beta12', 'beta23', 'perfusion']

    if variable not in valid_variables:
        sys.exit("Variable specified by '--variable' is not available")

    # Allocate field according to the variable studied
    dolfin_variable, var_type, variable = allocate_field(variable, mesh, K1_space, K2_space, Vvel, simulation)

    load_dolfin_data(variable, dolfin_variable, results_folder)

    # Create the coordinates for the image
    img_coord_min, x, y, z = create_image_grid(mesh, voxel_size)
    nx, ny, nz = len(x), len(y), len(z)

    # Convert finite element data to image
    img_data = finite_element_to_image_data(variable, var_type, x, y, z, nx, ny, nz, args.background_value)

    # Save image to nifti format
    save_nifti(voxel_size, img_coord_min, img_data, results_folder, variable)

    # Save image slices if possible
    if args.save_figure:
        save_image(img_data, results_folder, args.variable, nx, ny, nz)

if __name__ == "__main__":
    main()
