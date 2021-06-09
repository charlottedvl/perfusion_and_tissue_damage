from dolfin import *
import numpy as np
import untangle
import yaml


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


#%%
def perm_init_config_reader(input_file_path):
    configs = untangle.parse(input_file_path).permeability_initialiser
    
    mesh_file = configs.input_files_and_folders.mesh_file.cdata.strip()
    
    e_ref = np.array(list(map(float,configs.physical_variables.e_ref.cdata.split(','))))
    K1_form = np.array(list(map(float,configs.physical_variables.K1_form.cdata.split(',')))).reshape((3, 3))
    
    res_fldr = configs.output_files_and_folders.res_fldr.cdata.strip()
    save_subres = 'True'==configs.output_files_and_folders.save_subres.cdata.strip()
    
    return mesh_file, e_ref, K1_form, res_fldr, save_subres

#%%
def perm_init_config_reader_yml(input_file_path):
    with open(input_file_path, "r") as configfile:
        configs = yaml.load(configfile, yaml.SafeLoader)
    configs['physical']['K1_form'] = np.array( configs['physical']['K1_form'] ).reshape((3, 3))
    configs['physical']['e_ref'] =   np.array( configs['physical']['e_ref'] )
    
    return configs


#%%
def basic_flow_config_reader(input_file_path):
    configs = untangle.parse(input_file_path).basic_flow_solver
    
    mesh_file = configs.input_files_and_folders.mesh_file.cdata.strip()
    read_inlet_boundary = 'True'==configs.input_files_and_folders.read_inlet_boundary.cdata.strip()
    inlet_boundary_file = configs.input_files_and_folders.inlet_boundary_file.cdata.strip()
    inlet_BC_type = configs.input_files_and_folders.inlet_BC_type.cdata.strip()
    permeability_folder = configs.input_files_and_folders.permeability_folder.cdata.strip()
    
    p_arterial = float(configs.physical_variables.p_arterial.cdata)
    p_venous = float(configs.physical_variables.p_venous.cdata)
    
    K1gm_ref = float(configs.physical_variables.K1gm_ref.cdata)
    K2gm_ref = float(configs.physical_variables.K2gm_ref.cdata)
    K3gm_ref = float(configs.physical_variables.K3gm_ref.cdata)
    gmowm_perm_rat = float(configs.physical_variables.gmowm_perm_rat.cdata)
    
    beta12gm = float(configs.physical_variables.beta12gm.cdata)
    beta23gm = float(configs.physical_variables.beta23gm.cdata)
    gmowm_beta_rat = float(configs.physical_variables.gmowm_beta_rat.cdata)
    
    fe_degr = int(configs.simulation_settings.fe_degr.cdata)
    
    res_fldr = configs.output_files_and_folders.res_fldr.cdata.strip()
    save_pvd = 'True'==configs.output_files_and_folders.save_pvd.cdata.strip()
    comp_ave = 'True'==configs.output_files_and_folders.comp_ave.cdata.strip()
    
    
    if inlet_BC_type == 'NBC' and read_inlet_boundary == False:
        comm = MPI.comm_world
        rank = comm.Get_rank()
        raise Exception('NBC is not available without reading inlet boundary conditions')
    
    return mesh_file, read_inlet_boundary, inlet_boundary_file, inlet_BC_type, permeability_folder, \
           p_arterial, p_venous, K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, \
           beta12gm, beta23gm, gmowm_beta_rat, fe_degr, res_fldr, save_pvd, comp_ave


#%%
def basic_flow_config_reader2(input_file_path,parser):
    if input_file_path.endswith('xml'):
        configs = untangle.parse(input_file_path).basic_flow_solver_settings       
        from pydoc import locate
        
        for mydir in dir(configs):
            for mysubdir in dir(getattr(configs,mydir)):
                    mydata = getattr(getattr(configs,mydir),mysubdir)
                    mydata_type = mydata['type']
                    if mydata_type == None:
                        setattr(getattr(configs,mydir),mysubdir, mydata.cdata.strip())
                    elif mydata_type != 'bool':
                        converter = locate(mydata_type)
                        setattr(getattr(configs,mydir),mysubdir, converter(mydata.cdata.strip()))
                    else:
                        setattr(getattr(configs,mydir),mysubdir, mydata.cdata.strip()=='True')
    elif input_file_path.endswith('yaml'):
        with open(input_file_path, "r") as configfile:
            configs = yaml.load(configfile, yaml.SafeLoader)
        configs = dict2obj(configs)
    else:
        raise Exception("unknown input file format: " + config_format)
    
    if parser.parse_args().res_fldr != None:
        configs.output.res_fldr = parser.parse_args().res_fldr

    if parser.parse_args().mesh_file != None:
        configs.input.mesh_file = parser.parse_args().mesh_file

    if parser.parse_args().inlet_boundary_file != None:
        configs.input.inlet_boundary_file = parser.parse_args().inlet_boundary_file
    return configs


#%%
def basic_flow_config_reader_yml(input_file_path,parser):
    if input_file_path.endswith('yaml'):
        with open(input_file_path, "r") as configfile:
            configs = yaml.load(configfile, yaml.SafeLoader)
    else:
        raise Exception("unknown input file format: " + config_format)
    
    if parser.parse_args().res_fldr != None:
        configs['output']['res_fldr'] = parser.parse_args().res_fldr

    if parser.parse_args().mesh_file != None:
        configs['input']['mesh_file'] = parser.parse_args().mesh_file

    if parser.parse_args().inlet_boundary_file != None:
        configs['input']['inlet_boundary_file'] = parser.parse_args().inlet_boundary_file
    return configs


#%%
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
def initialise_permeabilities(K1_space,K2_space,mesh, permeability_folder,**kwarg):
    if 'model_type' in kwarg:
        model_type = kwarg.get('model_type')
    else:
        model_type = 'acv'
    
    comm = MPI.comm_world
    
    if model_type == 'acv':
        K1 = Function(K1_space)
        K2 = Function(K2_space)
        
        with XDMFFile(comm,permeability_folder+"K1_form.xdmf") as myfile:
            myfile.read_checkpoint(K1, "K1_form")
        
        K3 = K1.copy(deepcopy=True)
    elif model_type == 'a':
        K1 = Function(K1_space)
        K2, K3 = [], []
        
        with XDMFFile(comm,permeability_folder+"K1_form.xdmf") as myfile:
            myfile.read_checkpoint(K1, "K1_form")
    else:
        raise Exception("unknown model type: " + model_type)
    
    return K1, K2, K3


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

#%%
class dict2obj(dict):
    def __init__(self, my_dict):
        for a, b in my_dict.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)