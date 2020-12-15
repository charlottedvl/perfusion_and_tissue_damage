"""
Multi-compartment Darcy flow model with mixed Dirichlet and Neumann
boundary conditions

System of equations (no summation notation)
Div ( Ki Grad(pi) ) - Sum_j=1^3 beta_ij (pi-pj) = sigma_i

Ki - permeability tensor [mm^3 s / g]
pi & pj - volume averaged pressure in the ith & jth comparments [Pa]
beta_ij - coupling coefficient between the ith & jth compartments [Pa / s]
sigma_i - source term in the ith compartment [1 / s]

@author: Tamas Istvan Jozsa
"""

import argparse
import sys
import time

import numpy
# %% IMPORT MODULES
# installed python3 modules
from dolfin import *

numpy.set_printoptions(linewidth=200)

# ghost mode options: 'none', 'shared_facet', 'shared_vertex'
parameters['ghost_mode'] = 'none'

# added module
import IO_fcts
import suppl_fcts
import finite_element_fcts as fe_mod

# location of the 1-D blood flow model
sys.path.insert(0, "../../1d-blood-flow/")
from Blood_Flow_1D import Patient, Results
import contextlib
import copy
import scipy.optimize

# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

start0 = time.time()

# %% READ INPUT
if rank == 0: print('Step 1: Reading input files, initialising functions and parameters')
start1 = time.time()

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="path to configuration file (string ended with /)",
                    type=str, default='./config_coupled_flow_solver.xml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                    type=str, default=None)
config_file = parser.parse_args().config_file

configs = IO_fcts.basic_flow_config_reader2(config_file, parser)
# physical parameters
p_arterial, p_venous = configs.physical.p_arterial, configs.physical.p_venous
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
    configs.physical.K1gm_ref, configs.physical.K2gm_ref, configs.physical.K3gm_ref, configs.physical.gmowm_perm_rat
beta12gm, beta23gm, gmowm_beta_rat = \
    configs.physical.beta12gm, configs.physical.beta23gm, configs.physical.gmowm_beta_rat

# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs.input.mesh_file)

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, configs.simulation.fe_degr)

# initialise permeability tensors
K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space, K2_space, mesh, configs.input.permeability_folder)

if rank == 0: print('\t Scaling coupling coefficients and permeability tensors')

# set coupling coefficients
beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains,
                                                        beta12gm, beta23gm, gmowm_beta_rat,
                                                        K2_space, configs.output.res_fldr, configs.output.save_pvd)

K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1, K2, K3,
                                             K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat,
                                             configs.output.res_fldr, configs.output.save_pvd)
end1 = time.time()

# %% SET UP FINITE ELEMENT SOLVER AND SOLVE GOVERNING EQUATIONS
if rank == 0: print('Step 2: Defining and solving governing equations')
start2 = time.time()

# 1-D blood flow model
# get patient folder location
patient_folder = "/".join(
    configs.input.inlet_boundary_file.split("/")[:-2]) + "/"  # assume boundary file is in bf_sim folder

# run 1-D blood flow model and update boundary file
coarseCollaterals = False
clotactive = False

if rank == 0:
    Patient = Patient.Patient(patient_folder)
    Patient.LoadBFSimFiles()
    Patient.LoadModelParameters("Model_parameters.txt")
    Patient.LoadClusteringMapping(Patient.Folders.ModellingFolder + "Clusters.csv")
    Patient.LoadPositions()

    frictionconstant = Patient.ModelParameters["FRICTION_C"]  # 8 = laminar, 22 = blunt

    Patient.Initiate1DSteadyStateModel()  # run with original wk elements
    Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                  coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                  scale_resistance=False)
    # save old flowrates
    for index, node in enumerate(Patient.Topology.OutletNodes):
        # node.OldFlow = node.FlowRate
        node.OldFlow = node.WKNode.AccumulatedFlowRate
    Patient.UpdatePressureCouplingPoints(p_arterial)
    # update boundary file
    for outlet in Patient.Topology.OutletNodes:
        outlet.Pressure = outlet.OutPressure
    Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)
comm.Barrier()

# set up finite element solver
# TODO: handle Neuman/dirichlet boundary conditions
LHS, RHS, sigma1, sigma2, sigma3, BCs = \
    fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3,
                             p, p1, p2, p3, K1, K2, K3, beta12, beta23,
                             p_arterial, p_venous,
                             configs.input.read_inlet_boundary, configs.input.inlet_boundary_file,
                             configs.input.inlet_BC_type)

lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

# tested iterative solvers for first order elements: gmres, cg, bicgstab
# linear_solver_methods()
# krylov_solver_preconditioners()
if rank == 0: print('\t pressure computation')
p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol)
end2 = time.time()

# %% COMPUTE VELOCITY FIELDS, SAVE SOLUTION, EXTRACT FIELD VARIABLES
if rank == 0: print('Step 3: Computing velocity fields, saving results, and extracting some field variables')
start3 = time.time()

p1, p2, p3 = p.split()

perfusion = project(beta12 * (p1 - p2) * 6000, K2_space, solver_type='bicgstab', preconditioner_type='amg')

# compute velocities
vel1 = project(-K1 * grad(p1), Vvel, solver_type='bicgstab', preconditioner_type='amg')
vel2 = project(-K2 * grad(p2), Vvel, solver_type='bicgstab', preconditioner_type='amg')
vel3 = project(-K3 * grad(p3), Vvel, solver_type='bicgstab', preconditioner_type='amg')

ps = [p1, p2, p3]
vels = [vel1, vel2, vel3]
Ks = [K1, K2, K3]

# get surface values
fluxes, surf_p_values = suppl_fcts.surface_ave(mesh, boundaries, vels, ps)

FlowRateAtBoundary = fluxes[:, 2][2:] * -1  # Flow rate from the perfusion model (sign to match 1-d bf model, positive flow towards the brain)
PressureAtBoundary = surf_p_values[:, 2][2:]  # Pressure from the perfusion model
sys.stdout.flush()

# %% OPTIMIZE 1-D BLOOD FLOW MODEL
if rank == 0:
    start_r = time.time()

    perfusion_target = 600
    total_surface_flow = 600

    # without other outlets, set inlet flow rate if not optimized for the same value!
    #FlowRateAtBoundary in mm^3/s
    if len(Patient.Perfusion.CouplingPoints) == len(Patient.Topology.OutletNodes):
        print(f"\033[91mDetected only brain outlets, updating inlet flow\033[m")
        sys.stdout.flush()
        print(f"\tTotal inflow: {60*FlowRateAtBoundary[-1]/1000} mL/min")
        total_surface_flow = sum([FlowRateAtBoundary[index] * 1e-3 for index, _ in enumerate(Patient.Perfusion.CouplingPoints)]) * 60
        print(f"\ttotal surface flow: {total_surface_flow} mL/min")
        Patient.Topology.InletNodes[0][0].InletFlowRate = total_surface_flow *1e-6/60 # m^3
        perfusion_target = total_surface_flow #600

    print(f"\tTotal cerebral flow:{sum([cp.Node.WKNode.AccumulatedFlowRate for cp in Patient.Perfusion.CouplingPoints]) * 60} mL/min")
    print(f"\tInlet pressure:{Patient.Topology.InletNodes[0][0].Pressure} Pa")
    print(f"\tInlet flow:{Patient.Topology.InletNodes[0][0].FlowRate} mL/s")
    for index, node in enumerate(Patient.Topology.OutletNodes):
        node.OldFlow = node.WKNode.AccumulatedFlowRate

    print("Optimize 1-D blood flow model to match perfusion model.")
    sys.stdout.flush()
    correction = total_surface_flow / (sum([FlowRateAtBoundary[index] * 1e-3 for index, _ in enumerate(Patient.Perfusion.CouplingPoints)]) * 60)
    print(f"\033[91m\tTotal Surface flux:{total_surface_flow/correction}, Correction factor:{correction}\033[m")
    # update boundary condition
    for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
        cp.Node.TargetFlow = FlowRateAtBoundary[index] * 1e-3 #* correction  # todo scaling to compensate for low bc resolution
        # print(f"\tCurrent flow:{cp.Node.WKNode.AccumulatedFlowRate}, Target flow:{cp.Node.TargetFlow}")

    def coupling_resistance(R):
        # NOTE: using this function seems to lead to a coupled model that does not converge
        for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
            cp.Node.R1 = 0
            cp.Node.R2 = R[index]
        with contextlib.redirect_stdout(None):
            Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                          coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                          scale_resistance=False)
        residuals = [cp.Node.WKNode.AccumulatedFlowRate - cp.Node.TargetFlow for cp in Patient.Perfusion.CouplingPoints]
        sys.stdout.flush()
        return residuals

    def estimate_resistance(patient):
        patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                      coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                      scale_resistance=False)
        rel_tol = 1e-9
        relative_residual = 1
        while relative_residual > rel_tol:
            oldR = numpy.array([node.Node.R1 + node.Node.R2 for node in patient.Perfusion.CouplingPoints])
            for index, cp in enumerate(patient.Perfusion.CouplingPoints):
                cp.Node.R2 = (cp.Node.Pressure - cp.Node.OutPressure) / (cp.Node.TargetFlow * 1e-6)
                cp.Node.R1 = 0
            with contextlib.redirect_stdout(None):
                patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                              coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                              scale_resistance=False)
            relative_residual = max(
                [abs(node.Node.R1 + node.Node.R2 - oldR[index]) / (node.Node.R1 + node.Node.R2) for index, node in
                 enumerate(patient.Perfusion.CouplingPoints)])
            print(f'\tMax relative residual1: {relative_residual}')
            sys.stdout.flush()

    # update boundary conditions (original value)
    Patient.UpdatePressureCouplingPoints(Patient.ModelParameters["OUT_PRESSURE"])
    estimate_resistance(Patient)
    # # # print([node.Node.R2 for node in Patient.Perfusion.CouplingPoints])
    # #
    # # # # x_scale = numpy.array([1e9 for node in Patient.Perfusion.CouplingPoints])
    # # # # Patient.UpdatePressureCouplingPoints(p_arterial)
    # # # guess_resistance = numpy.array([1e10 for node in Patient.Perfusion.CouplingPoints])  # healthy scenario
    # # # # guess_resistance = numpy.array([node.Node.R2 for node in Patient.Perfusion.CouplingPoints])  # healthy scenario
    # # # sol = scipy.optimize.root(coupling_resistance, guess_resistance, method='krylov',
    # # #                           options={'disp': True, 'maxiter': 50, 'fatol': 1e-12})
    # # # for index, node in enumerate(Patient.Perfusion.CouplingPoints):
    # # #     node.Node.R2 = sol.x[index]
    # # # print(sol.x)
    # #
    print(f"\tTotal cerebral flow:{sum([cp.Node.WKNode.AccumulatedFlowRate for cp in Patient.Perfusion.CouplingPoints]) * 60} mL/min")
    print(f"\tInlet pressure:{Patient.Topology.InletNodes[0][0].Pressure} Pa")
    print(f"\tInlet flow:{Patient.Topology.InletNodes[0][0].FlowRate} mL/s")
    pressure_minimal = min([cp.Node.Pressure for cp in Patient.Perfusion.CouplingPoints])
    print(f"\tMinimal pressure found:{pressure_minimal}")
    if pressure_minimal < p_arterial:
        print("\033[91mSurface pressure is higher than expected! Optimizing inlet pressure \033[m")
        while pressure_minimal < p_arterial:
            difference_surface_pressure = p_arterial-pressure_minimal
            #increase inlet pressure
            Patient.Topology.InletNodes[0][0].InletPressure += difference_surface_pressure
            with contextlib.redirect_stdout(None):
                Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                                  coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                                  scale_resistance=True)
            pressure_minimal = min([cp.Node.Pressure for cp in Patient.Perfusion.CouplingPoints])
        estimate_resistance(Patient)
        # sol = scipy.optimize.root(coupling_resistance, guess_resistance, method='krylov',
        #                           options={'disp': True, 'maxiter': 50, 'fatol': 1e-12})
        # for index, node in enumerate(Patient.Perfusion.CouplingPoints):
        #     node.Node.R2 = sol.x[index]
        print(f"\033[91mNew inlet Pressure:{Patient.Topology.InletNodes[0][0].InletPressure} \033[m")
        print(f"\tTotal cerebral flow:{sum([cp.Node.WKNode.AccumulatedFlowRate for cp in Patient.Perfusion.CouplingPoints]) * 60} mL/min")
        print(f"\tInlet pressure:{Patient.Topology.InletNodes[0][0].Pressure} Pa")
        print(f"\tInlet flow:{Patient.Topology.InletNodes[0][0].FlowRate} mL/s")

    print(f"Updating boundary conditions")
    sys.stdout.flush()
    Patient.UpdatePressureCouplingPoints(p_arterial)
    estimate_resistance(Patient)
    # guess_resistance = numpy.array([node.Node.R2 - (p_arterial - node.Node.OutPressure)/(node.Node.TargetFlow*1e-6)
    #                                 for node in Patient.Perfusion.CouplingPoints])  # healthy scenario
    # # guess_resistance = numpy.array([1e9 for node in Patient.Perfusion.CouplingPoints])  # healthy scenario
    # sol = scipy.optimize.root(coupling_resistance, guess_resistance, method='krylov',
    #                           options={'disp': True, 'maxiter': 50, 'fatol': 1e-12})
    # # sol = scipy.optimize.least_squares(coupling_resistance, guess_resistance, gtol=None,x_scale=x_scale, verbose=2)
    # for index, node in enumerate(Patient.Perfusion.CouplingPoints):
    #     node.Node.R2 = sol.x[index]

    pressure_diff = Patient.Topology.InletNodes[0][0].Pressure / Patient.Topology.InletNodes[0][0].InletPressure
    flow_rate_diff = 1e-6 * Patient.Topology.InletNodes[0][0].FlowRate / Patient.Topology.InletNodes[0][0].InletFlowRate
    while abs(pressure_diff-1) > 1e-6 or abs(flow_rate_diff-1) > 1e-6:
        # update total resistance to maintain inlet boundary conditions.
        with contextlib.redirect_stdout(None):
            Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                          coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                          scale_resistance=True)

        # update cerebral resistance to obtain target flow rates.
        estimate_resistance(Patient)

        # guess_resistance = numpy.array([node.Node.R2 for node in Patient.Perfusion.CouplingPoints])  # healthy scenario
        # sol = scipy.optimize.root(coupling_resistance, guess_resistance, method='krylov',
        #                               options={'disp': True, 'maxiter': 50, 'fatol': 1e-6})
        # # sol = scipy.optimize.least_squares(coupling_resistance, guess_resistance, gtol=None,x_scale=x_scale, verbose=2)
        # for index, node in enumerate(Patient.Perfusion.CouplingPoints):
        #     node.Node.R2 = sol.x[index]

        pressure_diff = Patient.Topology.InletNodes[0][0].Pressure / Patient.Topology.InletNodes[0][0].InletPressure
        flow_rate_diff = 1e-6 * Patient.Topology.InletNodes[0][0].FlowRate / Patient.Topology.InletNodes[0][0].InletFlowRate
        print(f"\tTotal cerebral flow:{sum([cp.Node.WKNode.AccumulatedFlowRate for cp in Patient.Perfusion.CouplingPoints])*60} mL/min")
        print(f"\tInlet pressure:{Patient.Topology.InletNodes[0][0].Pressure} Pa")
        print(f"\tInlet flow:{Patient.Topology.InletNodes[0][0].FlowRate} mL/s")
        print(f"\tPressure difference:{pressure_diff}")
        print(f"\tFlow rate difference:{flow_rate_diff}")

    end_r = time.time()
    print("Optimization complete.")
    print('Execution time: \t', end_r - start_r, '[s]')

    # save optimization results and model parameters
    with open(patient_folder + 'Model_values_Healthy.csv', "w") as f:
        f.write(
            "Region,Resistance,Outlet Pressure(pa),WK Pressure, Perfusion Surface Pressure(pa),Old Flow Rate,Flow Rate(mL/s),Perfusion Flow Rate(mL/s)\n")
        for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
            f.write("%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n" % (
                fluxes[:, 0][2:][index],
                cp.Node.R1 + cp.Node.R2,
                cp.Node.Pressure,
                cp.Node.WKNode.Pressure,
                PressureAtBoundary[index],
                cp.Node.OldFlow,
                cp.Node.WKNode.AccumulatedFlowRate,
                cp.Node.TargetFlow))

    CouplingResistance = [node.Node.R1 + node.Node.R2 for node in Patient.Perfusion.CouplingPoints]
    for r in CouplingResistance:
        if r < 1e6:
            print('\033[93m' + "Warning: Low coupling resistance found. R=%f \033[m" % r)

    # update boundary conditions
    for index, node in enumerate(Patient.Topology.OutletNodes):
        node.OutletFlowRate = node.WKNode.AccumulatedFlowRate * -1e-6
    Patient.UpdateFlowRateCouplingPoints(-1e-9 * FlowRateAtBoundary)
    Patient.Results1DSteadyStateModel()
    Patient.ExportMeanResults(file="ResultsPerVesselHealthy.csv")

    for outlet in Patient.Topology.OutletNodes:
        outlet.Pressure = outlet.WKNode.Pressure
    Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)

comm.Barrier()
# todo save optimized model settings to speed up multiple consecutive runs.

# %% RUN COUPLED MODEL
def coupledmodel(P, stopp):
    stopp[0] = comm.bcast(stopp[0], root=0)
    # update boundary file and vessel outlet
    if rank == 0:
        for index, node in enumerate(Patient.Perfusion.CouplingPoints):
            node.Node.OutPressure = P[index]  # set pressure at the coupling point
            node.Node.Pressure = P[index]  # for updating boundary file
        Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)
    P = comm.bcast(P, root=0)
    # Run perfusion model
    with contextlib.redirect_stdout(None):
        Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
            fe_mod.alloc_fct_spaces(mesh, configs.simulation.fe_degr)  # do we need to allocate them again?

        LHS, RHS, sigma1, sigma2, sigma3, BCs = \
            fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, p, p1, p2, p3, K1, K2, K3, beta12,
                                     beta23,
                                     p_arterial, p_venous,
                                     configs.input.read_inlet_boundary, configs.input.inlet_boundary_file,
                                     configs.input.inlet_BC_type)

        p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol)
        p1, p2, p3 = p.split()
        # compute velocities
        vel1 = project(-K1 * grad(p1), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        vel2 = project(-K2 * grad(p2), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        vel3 = project(-K3 * grad(p3), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        ps = [p1, p2, p3]
        vels = [vel1, vel2, vel3]
        # get surface values
        fluxes, surf_p_values = suppl_fcts.surface_ave(mesh, boundaries, vels, ps)

        FlowRateAtBoundary = fluxes[:, 2][2:] * -1
        # PressureAtBoundary = surf_p_values[:, 2][2:]

        # Run 1-D bf model
        residualFlowrate = 0
        if rank == 0:
            Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                          FlowRateOutlets=False, coarseCollaterals=coarseCollaterals,
                                          frictionconstant=frictionconstant, scale_resistance=False)

            # return residuals
            flowrate1d = [Node.Node.WKNode.AccumulatedFlowRate for index, Node in enumerate(Patient.Perfusion.CouplingPoints)]
            # pressure1d = [Node.Node.WKNode.Pressure for index, Node in enumerate(Patient.Perfusion.CouplingPoints)]
            residualFlowrate = [(i * 1e-3 - j) for i, j in zip(FlowRateAtBoundary, flowrate1d)]
        residualFlowrate = comm.bcast(residualFlowrate, root=0)
    return residualFlowrate

clotactive = True
 # Find the pressure at coupling points (identical to the surface regions) such that flowrate of the models are equal.
if rank == 0:
    print("\033[96mRunning two-way coupling\033[m")
    sys.stdout.flush()
    guessPressure = numpy.array([node.Node.WKNode.Pressure for node in Patient.Perfusion.CouplingPoints])  # healthy scenario
    stop = [0]
    sol = scipy.optimize.root(coupledmodel, guessPressure, args=(stop,), method='krylov',
                              options={'disp': True, 'maxiter': 50, 'ftol': 1e-9})
    stop = [1]
    coupledmodel(sol.x, stop)
    print(sol)
    sys.stdout.flush()
    # for index, node in enumerate(Patient.Perfusion.CouplingPoints):
    #     node.Node.OutPressure = sol.x[index]
    #     node.Node.Pressure = sol.x[index]
    # Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)
else:
    stop = [0]
    guessPressure = numpy.zeros(len(FlowRateAtBoundary))
    while stop[0] == 0:
        coupledmodel(guessPressure, stop)

# update perfusion model (1d bf results stored in object)
comm.Barrier()
with contextlib.redirect_stdout(None):
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_mod.alloc_fct_spaces(mesh, configs.simulation.fe_degr)  # do we need to allocate them again?

    LHS, RHS, sigma1, sigma2, sigma3, BCs = \
        fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, p, p1, p2, p3, K1, K2, K3, beta12,
                                 beta23,
                                 p_arterial, p_venous,
                                 configs.input.read_inlet_boundary, configs.input.inlet_boundary_file,
                                 configs.input.inlet_BC_type)

    p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol)
    p1, p2, p3 = p.split()
    # compute velocities
    vel1 = project(-K1 * grad(p1), Vvel, solver_type='bicgstab', preconditioner_type='amg')
    vel2 = project(-K2 * grad(p2), Vvel, solver_type='bicgstab', preconditioner_type='amg')
    vel3 = project(-K3 * grad(p3), Vvel, solver_type='bicgstab', preconditioner_type='amg')
    ps = [p1, p2, p3]
    vels = [vel1, vel2, vel3]
    # get surface values
    fluxes, surf_p_values = suppl_fcts.surface_ave(mesh, boundaries, vels, ps)
    FlowRateAtBoundary = fluxes[:, 2][2:] * -1
    PressureAtBoundary = surf_p_values[:, 2][2:]  # Pressure from the perfusion model
comm.Barrier()

if rank == 0:
    # export some results
    Patient.Results1DSteadyStateModel()
    # export data in same format at the 1-D pulsatile model
    # start point t=0
    TimePoint = Results.TimePoint(0)
    TimePoint.Flow = [node.FlowRate for node in Patient.Topology.Nodes]
    TimePoint.Pressure = [node.Pressure for node in Patient.Topology.Nodes]
    TimePoint.Radius = [node.Radius for node in Patient.Topology.Nodes]
    # end point, t=duration of a single heart beat
    TimePoint2 = Results.TimePoint(Patient.ModelParameters['Beat_Duration'])
    TimePoint2.Flow = TimePoint.Flow
    TimePoint2.Pressure = TimePoint.Pressure
    TimePoint2.Radius = TimePoint.Radius
    Patient.Results.TimePoints = [TimePoint, TimePoint2]
    Patient.Results.ExportResults(Patient.Folders.ModellingFolder + "Results.dyn")
    # Patient.LoadResults("Results.dyn")
    # Patient.GetMeanResults()

    Patient.ExportMeanResults(file="ResultsPerVesselStroke.csv")
    # Patient.DistributeFlowTriangles()
    # Patient.ExportTriangleFlowData()
    Patient.Results.AddResultsPerNodeToFile(Patient.Folders.ModellingFolder + "Topology.vtp")
    Patient.Results.AddResultsPerVesselToFile(Patient.Folders.ModellingFolder + "Topology.vtp")

    # save optimization results and model parameters
    with open(patient_folder + 'Model_values_Stroke.csv', "w") as f:
        f.write(
            "Region,Resistance,Outlet Pressure(pa),WK Pressure, Perfusion Surface Pressure(pa),Old Flow Rate,Flow Rate(mL/s),Perfusion Flow Rate(mL/s)\n")
        for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
            f.write("%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n" % (
                fluxes[:, 0][2:][index],
                cp.Node.R1 + cp.Node.R2,
                cp.Node.Pressure,
                cp.Node.WKNode.Pressure,
                PressureAtBoundary[index],
                cp.Node.OldFlow,
                cp.Node.WKNode.AccumulatedFlowRate,
                FlowRateAtBoundary[index] * 1e-3))

## save results (6000 comes from minute*100mL)
perfusion_stroke = project(abs(beta12 * (p1 - p2) * 6000), K2_space, solver_type='bicgstab', preconditioner_type='amg')
perfusion_change = project(((perfusion - perfusion_stroke) / perfusion) * -100, K2_space, solver_type='bicgstab',
                           preconditioner_type='amg')
infarct = project(conditional(gt(perfusion_change, Constant(-70)), Constant(0.0), Constant(1.0)), K2_space,
                  solver_type='bicgstab', preconditioner_type='amg')

vars2save = [ps, vels, Ks]
fnames = ['press', 'vel', 'K']
for idx, fname in enumerate(fnames):
    for i in range(3):
        with XDMFFile(configs.output.res_fldr + fname + str(i + 1) + '.xdmf') as myfile:
            myfile.write_checkpoint(vars2save[idx][i], fname + str(i + 1), 0, XDMFFile.Encoding.HDF5, False)

with XDMFFile(configs.output.res_fldr + 'beta12.xdmf') as myfile:
    myfile.write_checkpoint(beta12, "beta12", 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(configs.output.res_fldr + 'beta23.xdmf') as myfile:
    myfile.write_checkpoint(beta23, "beta23", 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(configs.output.res_fldr + 'perfusion.xdmf') as myfile:
    myfile.write_checkpoint(perfusion, 'perfusion', 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(configs.output.res_fldr + 'perfusion_stroke.xdmf') as myfile:
    myfile.write_checkpoint(perfusion_stroke, 'perfusion', 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(configs.output.res_fldr + 'perfusion_change.xdmf') as myfile:
    myfile.write_checkpoint(perfusion_change, 'perfusion_change', 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(configs.output.res_fldr + 'infarct.xdmf') as myfile:
    myfile.write_checkpoint(infarct, 'infarct', 0, XDMFFile.Encoding.HDF5, False)

fheader = 'FE degree, K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, beta12gm, beta23gm, gmowm_beta_rat'
dom_props = numpy.array(
    [configs.simulation.fe_degr, K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, beta12gm, beta23gm, gmowm_beta_rat])
numpy.savetxt(configs.output.res_fldr + 'dom_props.csv', [dom_props], "%d,%e,%e,%e,%e,%e,%e,%e", header=fheader)

# %%

if configs.output.comp_ave == True:
    # obtain fluxes (ID, surface area, flux1, flux2, flux3)
    # fluxes, surf_p_values = suppl_fcts.surface_ave(mesh, boundaries, vels, ps)

    # calculate perfusion in white and grey matter (and total brain)
    vol_perfusion_values = suppl_fcts.perfusion_vol(mesh, subdomains, perfusion)
    vol_perfusion_stroke_values = suppl_fcts.perfusion_vol(mesh, subdomains, perfusion_stroke)
    vol_infarct_values = suppl_fcts.infarct_vol(mesh, subdomains, infarct)

    # obtain some characteristic values within the domain (ID, volume, average, min, max)
    vol_p_values, vol_vel_values = suppl_fcts.vol_ave(mesh, subdomains, ps, vels)

    if rank == 0:
        print(fluxes, '\n')
        print(surf_p_values, '\n')
        print(vol_p_values, '\n')
        print(vol_vel_values, '\n')

        fheader = 'surface ID, Area [mm^2], Qa [mm^3/s], Qc [mm^3/s], Qv [mm^3/s]'
        numpy.savetxt(configs.output.res_fldr + 'fluxes.csv', fluxes, "%d,%e,%e,%e,%e", header=fheader)

        fheader = 'surface ID, Area [mm^2], pa [Pa], pc [Pa], pv [Pa]'
        numpy.savetxt(configs.output.res_fldr + 'surf_p_values.csv', surf_p_values, "%d,%e,%e,%e,%e", header=fheader)

        fheader = 'volume ID, Volume [mm^3], pa [Pa], pc [Pa], pv [Pa]'
        numpy.savetxt(configs.output.res_fldr + 'vol_p_values.csv', vol_p_values, "%e,%e,%e,%e,%e", header=fheader)

        fheader = 'volume ID, Volume [mm^3], ua [m/s], uc [m/s], uv [m/s]'
        numpy.savetxt(configs.output.res_fldr + 'vol_vel_values.csv', vol_vel_values, "%d,%e,%e,%e,%e", header=fheader)

        # perfusion
        fheader = 'volume ID,Volume [mm^3],perfusion [(ml blood)/min/(100 ml tissue)]'
        numpy.savetxt(configs.output.res_fldr + 'vol_perfusion_values.csv', vol_perfusion_values, "%d,%e,%e",
                      header=fheader)
        numpy.savetxt(configs.output.res_fldr + 'vol_perfusion_stroke_values.csv', vol_perfusion_stroke_values,
                      "%d,%e,%e", header=fheader)

        fheader = 'volume ID,Volume [mm^3],infarct volume [mL]'
        numpy.savetxt(configs.output.res_fldr + 'vol_infarct_values.csv', vol_infarct_values, "%d,%e,%e",
                      header=fheader)

end3 = time.time()
end0 = time.time()

# %% REPORT EXECUTION TIME
if rank == 0:
    oldstdout = sys.stdout
    logfile = open(configs.output.res_fldr + "time_info.log", 'w')
    sys.stdout = logfile
    print('Total execution time [s]; \t\t\t', end0 - start0)
    print('Step 1: Reading input files [s]; \t\t', end1 - start1)
    print('Step 2: Solving governing equations [s]; \t\t', end2 - start2)
    print('Step 3: Preparing and saving output [s]; \t\t', end3 - start3)
    logfile.close()
    sys.stdout = oldstdout
    print('Execution time: \t', end0 - start0, '[s]')
    print('Step 1: \t\t', end1 - start1, '[s]')
    print('Step 2: \t\t', end2 - start2, '[s]')
    print('Step 3: \t\t', end3 - start3, '[s]')

# # export visual of collaterals
# Patient.Topology.addCollateralsToTopology()
# Patient.TopologyToVTP(filename="TopologyCollaterals.vtp")

#####
# # infarct growth toy model
# from dolfin import *
#
# u_n = interpolate(Constant(1.0), K2_space)
# v = TestFunction(K2_space)
# u = TrialFunction(K2_space)
#
# t = 0
# num_steps = 100
# dt = 0.01
#
# # death rate is non-zero in possible infarction zones given by infarct variable
# deathrate = project(infarct * pow(perfusion_change, 2) * 1e-4, K2_space, solver_type='bicgstab',
#                     preconditioner_type='amg')
#
# F = (u - u_n) * v * dx + dt * deathrate * u * v * dx
# a, L = lhs(F), rhs(F)
# import os
#
# infarctpath = Patient.Folders.PatientFolder + "/infarct/"
# try:
#     os.mkdir(infarctpath)
#     # copy_tree("/home/raymond/Desktop/1d-blood-flow/DataFiles/DefaultPatient/", patient_folder)
# except OSError:
#     print("Creation of the directory %s failed" % infarctpath)
# else:
#     print("Successfully created the directory %s " % infarctpath)
#
# vtkfile = File(infarctpath + '/solution.pvd')
# u = Function(K2_space, name="Tissue State")
#
# u_n.rename("Tissue State", "label")
# vtkfile << (u_n, t)
#
# for n in range(num_steps):
#     t += dt
#     solve(a == L, u)
#     vtkfile << (u, t)
#     u_n.assign(u)
