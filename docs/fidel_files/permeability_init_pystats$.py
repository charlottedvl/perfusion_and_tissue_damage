import argparse
import time
import numpy as np
import yaml

from mpi4py import MPI
import basix.ufl
from dolfinx import fem, mesh, log
from dolfinx.io import XDMFFile

import IO_fcts
import suppl_fcts
import finite_element_fcts as fe_mod  # if needed
from pystats import stats


def main():
    stats.enable()  # Start stats tracking
    log.set_log_level(log.LogLevel.WARNING)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # --- Parse Config ---
    parser = argparse.ArgumentParser(description="Compute permeability tensor from vessel structure.")
    parser.add_argument("--config_file", type=str, default="./config_permeability_initialiser.yaml")
    config_file = parser.parse_args().config_file

    IO_fcts.print0("Step 1: Reading input files")
    configs = IO_fcts.perm_init_config_reader_yml(config_file)

    # --- Load Mesh ---
    stats.timer_start("mesh_reader")
    mesh_data, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])
    stats.timer_end("mesh_reader")

    IO_fcts.print0("Step 2: Computing permeability tensor")
    gdim = mesh_data.geometry.dim

    # --- Define Function Space ---
    stats.timer_start("element_creation")
    element = basix.ufl.element("DG", "tetrahedron", 0, shape=(gdim, gdim))
    K_space = fem.functionspace(mesh_data, element)
    stats.timer_end("element_creation")
    IO_fcts.print0("Function space created")

    # --- Compute vessel directions ---
    IO_fcts.print0("Step 2.1: Computing vessel orientation and main direction...")
    stats.timer_start("comp_vessel_orientation")
    e_loc, main_direction = suppl_fcts.comp_vessel_orientation(
        subdomains, boundaries, mesh_data,
        configs['output']['res_fldr'], configs['output']['save_subres']
    )
    stats.timer_end("comp_vessel_orientation")

    # --- Compute permeability tensor ---
    IO_fcts.print0("Step 2.2: Computing permeability tensor...")
    stats.timer_start("perm_tens_comp")
    K1 = suppl_fcts.perm_tens_comp(
        K_space, subdomains, mesh_data,
        configs['physical']['e_ref'], e_loc,
        configs['physical']['K1_form']
    )
    stats.timer_end("perm_tens_comp")

    # --- Write Outputs ---
    IO_fcts.print0("Step 3: Saving output files")
    stats.timer_start("writing_outputs")

    write_xdmf(mesh_data, K1, configs['output']['res_fldr'] + "K1_form.xdmf")
    write_xdmf(mesh_data, e_loc, configs['output']['res_fldr'] + "e_loc.xdmf")
    write_xdmf(mesh_data, main_direction, configs['output']['res_fldr'] + "main_direction.xdmf")

    myResults = {
        "K1_form": K1,
        "e_loc": e_loc,
        "main_direction": main_direction
    }
    for var in configs['output']['res_vars']:
        if var in myResults:
            write_xdmf(mesh_data, myResults[var], configs['output']['res_fldr'] + f"{var}.xdmf")
        else:
            print(f"Warning: {var} not found in results.")

    stats.timer_end("writing_outputs")

    # --- Final stats report ---
    if rank == 0:
        report = stats.report()
        print("\n--- Performance Stats ---\n", report)

        with open("performance_stats.txt", "w") as f:
            f.write(report)

        stats.save_pdf("performance_stats.pdf")


def write_xdmf(mesh, function, filename):
    print(f"Writing {filename}")
    with XDMFFile(mesh.comm, filename, "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(function)
    print(f"Finished writing {filename}")


if __name__ == "__main__":
    main()
