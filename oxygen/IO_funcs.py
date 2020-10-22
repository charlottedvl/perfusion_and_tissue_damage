from dolfin import *
import numpy as np
import untangle
import yaml

#%%
def oxygen_config_reader(input_file_path, parser):
    config_format = input_file_path[-3::] # xml or yaml(aml)
    
    if config_format == 'aml':
        with open(input_file_path, "r") as configfile:
            configs = yaml.load(configfile, yaml.SafeLoader)
        configs = dict2obj(configs)
    else:
        raise Exception("unknown input file format: " + config_format)
    
    if parser.parse_args().rslt != None:
        configs.output.rslt = parser.parse_args().reslt
    
    return configs

#%%
def mesh_reader_xdmf(mesh_file):
    mesh=Mesh()
    with XDMFFile(mesh.mpi_comm(),mesh_file) as myfile: myfile.read(mesh)
    subdomains = MeshFunction("size_t", mesh, 3)
    with XDMFFile(mesh.mpi_comm(),mesh_file[:-5]+'_physical_region.xdmf') as myfile: myfile.read(subdomains)
    boundaries = MeshFunction("size_t", mesh, 2)
    with XDMFFile(mesh.mpi_comm(),mesh_file[:-5]+'_facet_region.xdmf') as myfile: myfile.read(boundaries)
    
    return mesh, subdomains, boundaries

def mesh_reader_h5(mesh_file):
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_file, "r")
    hdf.read(mesh, "/mesh", False)
    subdomains = MeshFunction("size_t", mesh, 3)
    hdf.read(subdomains, "/subdomains")
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    hdf.close()
    
    return mesh, subdomains, boundaries

#%%
def xdmf_reader(variable, variable_name, folder):
    with XDMFFile(folder+variable_name+'.xdmf') as myfile:
        myfile.read_checkpoint(variable, variable_name)
    
    return variable

def xdmf_h5_saver(variable, variable_name, rslt):
    with XDMFFile(rslt+variable_name+'.xdmf') as myfile:
        myfile.write_checkpoint(variable, variable_name, 0, XDMFFile.Encoding.HDF5, False)
        
#%%
def hdf5_reader(mesh,variable,variable_name,folder):
    hdf = HDF5File(mesh.mpi_comm(), folder+variable_name+'.h5', "r")
    hdf.read(variable, "/"+variable_name)
    hdf.close()
    return variable

def hdf5_saver(mesh,variable,variable_name,folder):
    hdf = HDF5File(mesh.mpi_comm(), folder+variable_name+'.h5', "w")
    hdf.write(variable, "/"+variable_name)
    hdf.close()

#%%
def pvd_saver(variable,folder,name):
    variable.rename(name, "1")
    vtkfile = File(folder+name+'.pvd')
    vtkfile << variable

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

#%%
class dict2obj(dict):
    def __init__(self, my_dict):
        for a, b in my_dict.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)
