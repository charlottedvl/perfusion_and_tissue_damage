import os
from dolfin import *
import untangle
import yaml


# TODO: determine why it is never used
def basic_flow_config_reader(input_file_path):
    """
    Parse and extract configuration parameters for the basic flow solver from an XML input file.

    This function reads simulation parameters, mesh and file paths, physical constants, and
    output preferences from a configuration file formatted in XML (using the `untangle` library).

    Args:
        input_file_path (str): Path to the XML configuration file.

    Returns:
        tuple:
            mesh_file (str): Path to the mesh file.
            read_inlet_boundary (bool): Whether to read inlet boundary conditions from a file.
            inlet_boundary_file (str): Path to the inlet boundary condition file.
            inlet_BC_type (str): Type of inlet boundary condition ("NBC" or other).
            permeability_folder (str): Path to the folder containing permeability field files.
            p_arterial (float): Arterial pressure in Pascals.
            p_venous (float): Venous pressure in Pascals.
            K1gm_ref (float): Reference permeability for K1 in gray matter.
            K2gm_ref (float): Reference permeability for K2 in gray matter.
            K3gm_ref (float): Reference permeability for K3 in gray matter.
            gmowm_perm_rat (float): Gray-to-white matter permeability ratio.
            beta12gm (float): Coupling coefficient between compartments 1 and 2 in gray matter.
            beta23gm (float): Coupling coefficient between compartments 2 and 3 in gray matter.
            gmowm_beta_rat (float): Gray-to-white matter coupling ratio.
            fe_degr (int): Finite element degree used in the simulation.
            res_fldr (str): Path to the folder for simulation results.
            save_pvd (bool): Whether to save PVD output files.
            comp_ave (bool): Whether to compute average quantities.

    Raises:
        Exception: If inlet_BC_type is "NBC" but `read_inlet_boundary` is False.
    """
    configs = untangle.parse(input_file_path).basic_flow_solver

    mesh_file = configs.input_files_and_folders.mesh_file.cdata.strip()
    read_inlet_boundary = 'True' == configs.input_files_and_folders.read_inlet_boundary.cdata.strip()
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
    save_pvd = 'True' == configs.output_files_and_folders.save_pvd.cdata.strip()
    comp_ave = 'True' == configs.output_files_and_folders.comp_ave.cdata.strip()

    if inlet_BC_type == 'NBC' and read_inlet_boundary == False:
        raise Exception('NBC is not available without reading inlet boundary conditions')

    return mesh_file, read_inlet_boundary, inlet_boundary_file, inlet_BC_type, permeability_folder, \
        p_arterial, p_venous, K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, \
        beta12gm, beta23gm, gmowm_beta_rat, fe_degr, res_fldr, save_pvd, comp_ave


def basic_flow_config_reader2(input_file_path, parser):
    """
    Read and process simulation configuration from either XML or YAML file formats.

    This function loads configuration settings for the basic flow solver from the specified
    input file (`.xml` or `.yaml`). It converts the data into Python-native types using
    either `untangle` (for XML) or `yaml` (for YAML). It also applies command-line argument
    overrides using the provided `argparse.ArgumentParser` parser instance.

    Supported override fields:
        - `res_fldr`: Output results folder
        - `mesh_file`: Path to mesh file
        - `inlet_boundary_file`: Path to inlet boundary condition file

    Args:
        input_file_path (str): Path to the input configuration file (.xml or .yaml).
        parser (argparse.ArgumentParser): Argument parser that may provide override values.

    Returns:
        object: Configuration object with accessible attributes (via dot notation).

    Raises:
        Exception: If the input file format is unsupported.
    """
    if input_file_path.endswith('xml'):
        configs = untangle.parse(input_file_path).basic_flow_solver_settings
        from pydoc import locate

        for mydir in dir(configs):
            for mysubdir in dir(getattr(configs, mydir)):
                mydata = getattr(getattr(configs, mydir), mysubdir)
                mydata_type = mydata['type']
                if mydata_type == None:
                    setattr(getattr(configs, mydir), mysubdir, mydata.cdata.strip())
                elif mydata_type != 'bool':
                    converter = locate(mydata_type)
                    setattr(getattr(configs, mydir), mysubdir, converter(mydata.cdata.strip()))
                else:
                    setattr(getattr(configs, mydir), mysubdir, mydata.cdata.strip() == 'True')
    elif input_file_path.endswith('yaml'):
        with open(input_file_path, "r") as config_file:
            configs = yaml.load(config_file, yaml.SafeLoader)
        configs = dict2obj(configs)
    else:
        config_format = os.path.splitext(input_file_path)[-1]
        raise Exception("unknown input file format: " + config_format)

    if parser.parse_args().res_fldr != None:
        configs.output.res_fldr = parser.parse_args().res_fldr

    if parser.parse_args().mesh_file != None:
        configs.input.mesh_file = parser.parse_args().mesh_file

    if parser.parse_args().inlet_boundary_file != None:
        configs.input.inlet_boundary_file = parser.parse_args().inlet_boundary_file
    return configs


def basic_flow_config_reader_yml(input_file_path, parser):
    """
    Load and update simulation configuration from a YAML file and command-line parser overrides.

    This function reads simulation parameters from a `.yaml` configuration file and overrides
    specific fields (if provided) using arguments parsed from an `argparse.ArgumentParser` instance.
    It also ensures the results folder exists and saves the final configuration into it.

    Args:
        input_file_path (str): Path to the YAML configuration file.
        parser (argparse.ArgumentParser): Argument parser containing optional overrides.

    Returns:
        dict: Parsed and possibly updated configuration dictionary.

    Raises:
        Exception: If the input file format is unsupported (non-YAML).
    """
    if input_file_path.endswith('yaml'):
        with open(input_file_path, "r") as config_file:
            configs = yaml.load(config_file, yaml.SafeLoader)
    else:
        config_format = os.path.splitext(input_file_path)[-1]
        raise Exception("Unknown input file format: " + config_format)

    if hasattr(parser.parse_args(), 'res_fldr'):
        if parser.parse_args().res_fldr is not None:
            configs['output']['res_fldr'] = parser.parse_args().res_fldr

    if hasattr(parser.parse_args(), 'mesh_file'):
        if parser.parse_args().mesh_file is not None:
            configs['input']['mesh_file'] = parser.parse_args().mesh_file

    if hasattr(parser.parse_args(), 'inlet_boundary_file'):
        if parser.parse_args().inlet_boundary_file is not None:
            configs['input']['inlet_boundary_file'] = parser.parse_args().inlet_boundary_file

    comm = MPI.comm_world
    rank = comm.Get_rank()
    if rank == 0:
        if not os.path.exists(configs['output']['res_fldr']):
            os.makedirs(configs['output']['res_fldr'])
        with open(configs['output']['res_fldr'] + 'settings.yaml', 'w') as outfile:
            yaml.dump(configs, outfile, default_flow_style=False)

    return configs


class dict2obj(dict):
    def __init__(self, my_dict):
        for a, b in my_dict.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)