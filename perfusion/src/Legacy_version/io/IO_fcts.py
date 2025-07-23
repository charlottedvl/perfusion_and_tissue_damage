from dolfin import *
import numpy as np
import untangle


#%%
def mesh_reader(mesh_file):
    comm = MPI.comm_world
    
    mesh = Mesh()
    with XDMFFile(comm,mesh_file) as myfile: myfile.read(mesh)
    subdomains = MeshFunction("size_t", mesh, 3)
    with XDMFFile(comm,mesh_file[:-5]+'_physical_region.xdmf') as myfile: myfile.read(subdomains)
    boundaries = MeshFunction("size_t", mesh, 2)
    with XDMFFile(comm,mesh_file[:-5]+'_facet_region.xdmf') as myfile: myfile.read(boundaries)

    return mesh, subdomains, boundaries


#%%
def argument_reader(parser):
    parser.add_option("--darcy_file", dest="darcy_file",
                      help="path to file storing defaults", type="str",default="config.xml")
    parser.add_option("--mesh_file", dest="mesh_file",
                      help="path to file storing volume mesh", type="str")
    parser.add_option("--res_fldr", dest="res_fldr",
                      help="path to folder to store results", type="str")
    parser.add_option("--pial_surf_file", dest="pial_surf_file",
                      help="path to file storing pial surface mesh", type="str")
    parser.add_option("--inflow_file", dest="inflow_file",
                      help="path to file storing volumetric flow rate values and surface boundary IDs", type="str")
    parser.add_option("--press_file", dest="press_file",
                      help="first part of the file name storing pressure values (results)", type="str")
    parser.add_option("--vel_file", dest="vel_file",
                      help="first part of the file name storing velocity values (results)", type="str")
    parser.add_option("--fe_degr", dest="fe_degr",
                      help="polynomial degree of finite elements", type="int")
    parser.add_option("--perm_tens_file", dest="perm_tens_file",
                      help="file containing permeability tensors", type="str")
    return parser


def input_file_reader(input_file_path):
    configs = untangle.parse(input_file_path).porobrain
    
    mesh_file = configs.files_and_folders.mesh_file.cdata.strip()
    pial_surf_file = configs.files_and_folders.pial_surf_file.cdata.strip()
    inflow_file = configs.files_and_folders.inflow_file.cdata.strip()
    res_fldr = configs.files_and_folders.res_fldr.cdata.strip()
    
    p_arterial = float(configs.physical_variables.p_arterial.cdata)
    p_venous = float(configs.physical_variables.p_venous.cdata)
    e_ref = np.array(list(map(float,configs.physical_variables.e_ref.cdata.split(','))))
    K1_ref = np.array(list(map(float,configs.physical_variables.K1_ref.cdata.split(',')))).reshape((3, 3))
    K2_ref = np.array(list(map(float,configs.physical_variables.K2_ref.cdata.split(',')))).reshape((3, 3))
    K3_ref = np.array(list(map(float,configs.physical_variables.K3_ref.cdata.split(',')))).reshape((3, 3))
    beta = np.array(list(map(float,configs.physical_variables.beta.cdata.split(',')))).reshape((3, 3))
    
    fe_degr = int(configs.simulation_settings.fe_degr.cdata)
    
    return mesh_file, p_arterial, p_venous, e_ref, \
        K1_ref, K2_ref, K3_ref, beta, fe_degr, res_fldr, pial_surf_file, inflow_file



#%%
def inlet_file_reader(inlet_boundary_file):
    # ID, Q [ml/s], p [Pa]
    boundary_data = np.loadtxt(inflow_file,skiprows=1)
    boundary_data[:,1] = 1000 * boundary_data[:,1]
    return boundary_data


#%%
def pvd_saver(variable,folder,name):
    variable.rename(name, "1")
    vtkfile = File(folder+name+'.pvd')
    vtkfile << variable


#%%
def hdf5_saver(mesh,variable,folder,file_name,variable_name):
    hdf = HDF5File(mesh.mpi_comm(), folder+file_name, "w")
    hdf.write(variable, "/"+variable_name)
    hdf.close()

#%%
def hdf5_reader(mesh,variable,folder,file_name,variable_name):
    hdf = HDF5File(mesh.mpi_comm(), folder+file_name, "r")
    hdf.read(variable, "/"+variable_name)
    hdf.close()


def xdmf_reader(file, function, checkpoint_function):
    """
    Load a FEniCS Function from an XDMF file checkpoint.

    Reads a previously stored function from disk and returns it in the given function space.

    Args:
        file (str): Path to the XDMF file containing the checkpoint.
        function (FunctionSpace): FEniCS function space into which the data will be loaded.
        checkpoint_function (str): Name of the checkpoint variable stored in the file.

    Returns:
        Function: The loaded FEniCS function containing the stored simulation data.
    """
    variable = Function(function)
    f_in = XDMFFile(file)
    f_in.read_checkpoint(variable, checkpoint_function, 0)
    f_in.close()
    return variable


#%%
