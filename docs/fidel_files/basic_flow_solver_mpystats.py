# basic_flow_solver_mpystats.py

# IMPORT MODULES
import dolfinx
from dolfinx import fem, io, mesh
import numpy as np
import time
import sys
import os
import argparse
import mpi4py.MPI as MPI

# Added for profiling (updated to track memory as well)
from mbspystats import stats
stats.enable()

np.set_printoptions(linewidth=200)
from dolfinx.log import set_log_level, LogLevel
set_log_level(LogLevel.WARNING)

# added modules (assumed to be in the PYTHONPATH)
import IO_fcts
import suppl_fcts
import finite_element_fcts as fe_mod

# Define MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("Step 1: Initialisation...")

start0 = time.time()

# Helper function to write XDMF
def write_xdmf(mesh_obj, function, filename):
    print(f"Writing {filename}")
    with io.XDMFFile(mesh_obj.comm, filename, "w", encoding=io.XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write_mesh(mesh_obj)
        xdmf.write_function(function)
    print(f"Finished writing {filename}")

# READ INPUT
stats.timer_start("read_input")
if rank == 0:
    print("Step 1.1: Reading input files, initialising functions and parameters")

parser = argparse.ArgumentParser(description="Perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="Path to configuration file", type=str, default='./config_basic_flow_solver.yaml')
parser.add_argument("--res_fldr", help="Path to results folder (string ended with /)", type=str, default=None)
config_file = parser.parse_args().config_file

configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)

p_arterial, p_venous = configs['physical']['p_arterial'], configs['physical']['p_venous']
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], configs['physical']['gmowm_perm_rat']
beta12gm, beta23gm, gmowm_beta_rat = configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat']

try:
    compartmental_model = configs['simulation']['model_type'].lower().strip()
except KeyError:
    compartmental_model = 'acv'

try:
    velocity_order = configs['simulation']['vel_order']
except KeyError:
    velocity_order = configs['simulation']['fe_degr'] - 1

mesh_obj, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = fe_mod.alloc_fct_spaces(
    mesh_obj, configs['simulation']['fe_degr'],
    model_type=compartmental_model, vel_order=velocity_order
)

K1, K2, K3 = IO_fcts.initialise_permeabilities(
    K1_space, K2_space, mesh_obj,
    configs['input']['permeability_folder'], model_type=compartmental_model
)

stats.timer_end("read_input")

# SCALING
stats.timer_start("scale_permeabilities")
if rank == 0:
    print("\tScaling coupling coefficients and permeability tensors")

beta12, beta23 = suppl_fcts.scale_coupling_coefficients(
    subdomains, beta12gm, beta23gm, gmowm_beta_rat,
    K2_space, configs['output']['res_fldr'],
    model_type=compartmental_model
)

K1, K2, K3 = suppl_fcts.scale_permeabilities(
    subdomains, K1, K2, K3,
    K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat,
    configs['output']['res_fldr'], model_type=compartmental_model
)
stats.timer_end("scale_permeabilities")

# DEAD TISSUE HANDLING
stats.timer_start("read_dead_tissue")
tissue_health_file = configs['output']['res_fldr'] + '../feedback/infarct.xdmf'

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

if is_non_zero_file(tissue_health_file):
    dead_tissue = fem.Function(K2_space)
    with io.XDMFFile(mesh_obj.comm, tissue_health_file, "r") as xdmf_in:
        xdmf_in.read_mesh(mesh_obj)
        xdmf_in.read_function(dead_tissue)
stats.timer_end("read_dead_tissue")

if is_non_zero_file(tissue_health_file):
    dt_array = dead_tissue.x.array
    scale_factor = ((1 - dt_array) * (1 - configs['simulation']['feedback_limit']) + configs['simulation']['feedback_limit'])
    K2.x.array[:] *= scale_factor
    beta12.x.array[:] *= scale_factor
    beta23.x.array[:] *= scale_factor

# SOLVING SYSTEM
stats.timer_start("setup_solver")
LHS, RHS, sigma1, sigma2, sigma3, BCs = fe_mod.set_up_fe_solver2(
    mesh_obj, subdomains, boundaries, Vp, v_1, v_2, v_3,
    p, p1, p2, p3, K1, K2, K3, beta12, beta23,
    p_arterial, p_venous,
    configs['input']['read_inlet_boundary'],
    configs['input']['inlet_boundary_file'],
    configs['input']['inlet_BC_type'],
    model_type=compartmental_model
)
stats.timer_end("setup_solver")

stats.timer_start("solve_system")
if rank == 0:
    print("\tPressure computation")
p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, 'bicgstab', 'petsc_amg', False, False, False,
                         model_type=compartmental_model)
stats.timer_end("solve_system")

# POSTPROCESSING
stats.timer_start("postprocessing")
if rank == 0:
    print("Step 3: Computing velocity fields and saving results")

myResults = {}
suppl_fcts.compute_my_variables(p, K1, K2, K3, beta12, beta23, p_venous, Vp, Vvel, K2_space,
                                configs, myResults, compartmental_model, rank)

if 'perfusion' not in configs['output']['res_vars']:
    configs['output']['res_vars'].append('perfusion')

# Write requested variables
res_fldr = configs['output']['res_fldr']
for var in configs['output']['res_vars']:
    if var in myResults:
        filename = os.path.join(res_fldr, f"{var}.xdmf")
        write_xdmf(mesh_obj, myResults[var], filename)
    else:
        print(f"Warning: {var} not found in results.")

my_integr_vars = {}
surf_int_values, surf_int_header, volu_int_values, volu_int_header = \
    suppl_fcts.compute_integral_quantities(configs, myResults, my_integr_vars, mesh_obj, subdomains, boundaries, rank)

stats.timer_end("postprocessing")

# --- Final stats report and memory usage donut ---
if rank == 0:
    report = stats.report()
    print("\n--- Performance Stats ---\n", report)

    with open("performance_stats.txt", "w") as f:
        f.write(report)

    stats.save_pdf("mbs_performance_stats.pdf")
    stats.save_memory_donut_chart("memory_usage_donut.png")
