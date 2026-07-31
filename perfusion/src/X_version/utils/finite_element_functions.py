import time
import basix
import numpy as np
from . import suppl_fcts
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
from ufl import (
    Constant,
    #MixedElement,
    TestFunction,
    TestFunctions,
    TrialFunction,
    dx,
    #FiniteElement,
    grad,
    inner,
    split,
)




def allocation_functions_space(mesh_obj, fe_degr, **kwarg):
    """Allocates finite element function spaces for pressure and velocity

    in FEniCSx.
    """
    model_type = kwarg.get("model_type", "acv")
    vel_order = kwarg.get("vel_order", fe_degr - 1)

    # Get the cell type of the mesh (e.g., tetrahedron, triangle)
    cell_type = mesh_obj.basix_cell()
    #print(f"Mesh cell type: {cell_type}")
    if model_type == "acv":
        # Standard Lagrange element 
        P_el = basix.ufl.element("Lagrange", cell_type, fe_degr)
        # Mixed element combining the 3 compartments
        mixed_el = basix.ufl.mixed_element([P_el, P_el, P_el])
        Vp = fem.functionspace(mesh_obj, mixed_el)

        v_1, v_2, v_3 = TestFunctions(Vp)
        p_1, p_2, p_3 = ufl.TrialFunctions(Vp)
        p = TrialFunction(Vp)

    elif model_type == "a":
        P_el = basix.ufl.element("Lagrange", cell_type, fe_degr)
        Vp = fem.functionspace(mesh_obj, P_el)

        v_1, v_2, v_3 = TestFunction(Vp), [], []
        p = TrialFunction(Vp)
        p_1, p_2, p_3 = [], [], []
    else:
        raise Exception("Unknown model type: " + model_type)


    # Discontinuous spaces (DG) for permeability tensors
    gdim = mesh_obj.geometry.dim
    K1_el = basix.ufl.element("DG", cell_type, 0, shape=(gdim, gdim))
    K2_el = basix.ufl.element("DG", cell_type, 0)

    K1_space = fem.functionspace(mesh_obj, K1_el)
    K2_space = fem.functionspace(mesh_obj, K2_el)

    # Vectoriel spaces for velocity
    if vel_order == 0:
        Vvel_el = basix.ufl.element("DG", cell_type, 0, shape=(gdim,))
    else:
        Vvel_el = basix.ufl.element("Lagrange", cell_type, vel_order, shape=(gdim,))

    Vvel = fem.functionspace(mesh_obj, Vvel_el)

    return Vp, Vvel, v_1, v_2, v_3, p, p_1, p_2, p_3, K1_space, K2_space


def read_boundary_conditions(file_path):
    """
    Reads boundary conditions from a CSV file (same as original version)."""
    data = np.loadtxt(file_path, skiprows=1, delimiter=",")
    data_dimension_greater_than_1 = data.ndim > 1
    if data_dimension_greater_than_1:
        b1 = 1000 * data[:, 1]
        labels = list(data[:, 0])
    else:
        b1 = [1000 * data[1]]
        labels = [data[0]]
    return data, b1, labels, data_dimension_greater_than_1


def apply_dirichlet_BC(
    V_space,
    BC_data,
    BCs,
    boundaries,
    data_dimension_greater_than_1,
    index,
    boundary_id,
):
    # We get the mesh
    domain = V_space.mesh
    
    # Determine dimension of facets (boundary entities) for locating dofs
    fdim = domain.topology.dim - 1
    
    # We find the facets (boundary entities) corresponding to this boundary_id
    facets = boundaries.find(boundary_id)

    # We locate the degrees of freedom (dofs) on these facets 
    dofs = fem.locate_dofs_topological(V_space, fdim, facets)
  
    # We prepare the constant value for the Dirichlet BC, using the appropriate column from BC_data based on its dimension.
    if data_dimension_greater_than_1:
        bc_value = fem.Constant(domain, BC_data[index, 2])
    else:
        bc_value = fem.Constant(domain, BC_data[2])
    
    # We create the Dirichlet boundary condition (Syntax: value, dofs, Space)
    new_bc = fem.dirichletbc(bc_value, dofs, V_space)
    BCs.append(new_bc)

    return BCs

def apply_neumann_BC(mesh_obj, b1, integrals_N, index, v_1, dS, boundary_id):

    # We compile with fem.form()
    form_area = fem.form(fem.Constant(mesh_obj, 1.0) *dS(boundary_id, domain=mesh_obj))

    # Assemble simplified
    area = fem.assemble_scalar(form_area)
    area = mesh_obj.comm.allreduce(area, op=mesh._MPI.SUM)
    b1[index] = b1[index] / area
    # b1 includes the average surface-normal velocity for the perfusion regions
    # computed from the volumetric flow rate [mm^3/s] and the surface area [mm^2]

    if b1[index] > 0:
        integrals_N.append(b1[index] * v_1 * dS(boundary_id))

    return integrals_N


def apply_mixed_BC(
    mesh_obj,
    boundaries,
    integrals_N,
    BCs,
    BC_data,
    V_space,
    b1,
    dS,
    v_1,
    index,
    data_dimension_greater_than_1,
    boundary_id,
):
    bc_type_flag = BC_data[index, 4] if data_dimension_greater_than_1 else BC_data[4]

    if bc_type_flag == 0:
        BCs = apply_dirichlet_BC(
            V_space,
            BC_data,
            BCs,
            boundaries,
            data_dimension_greater_than_1,
            index,
            boundary_id,
        )
    elif bc_type_flag == 1:
        integrals_N = apply_neumann_BC(
            mesh_obj, b1, integrals_N, index, v_1, dS, boundary_id
        )
    else:
        raise Exception(
            "Boundary condition in the BCs.csv file must be 0 (Dirichlet) or 1 (Neumann)"
        )
    return integrals_N, BCs


def set_up_fe_solver(mesh, boundaries, V, v_1, v_2, v_3, \
                         p, p_1, p_2, p_3, K1, K2, K3, beta12, beta23, \
                         pa, pv, read_inlet_boundary, inlet_boundary_file, inlet_BC_type, **kwarg):
    """
    Sets up the finite element variational problem (LHS and RHS) and applies boundary conditions
    for the perfusion solver, based on the specified model type and boundary conditions.

    Args:
        mesh (dolfin.cpp.mesh.Mesh): The mesh object.
        boundaries (dolfin.cpp.mesh.MeshFunction): MeshFunction storing boundary markers.
        V (dolfin.fem.functionspace.FunctionSpace): The main function space for pressure (mixed for 'acv', scalar for 'a').
        v_1 (dolfin.fem.form.TestFunction): Test function for the first pressure component.
        v_2 (dolfin.fem.form.TestFunction or list): Test function for the second pressure component (or empty list).
        v_3 (dolfin.fem.form.TestFunction or list): Test function for the third pressure component (or empty list).
        p (dolfin.fem.form.TrialFunction): Trial function for pressure(s).
        p_1 (dolfin.fem.form.Argument or list): Split component of the trial function for the first pressure.
        p_2 (dolfin.fem.form.Argument or list): Split component of the trial function for the second pressure (or empty list).
        p_3 (dolfin.fem.form.Argument or list): Split component of the trial function for the third pressure (or empty list).
        K1 (dolfin.fem.function.Constant or dolfin.fem.function.Function): Permeability tensor/scalar for the first compartment.
        K2 (dolfin.fem.function.Constant or dolfin.fem.function.Function): Permeability tensor/scalar for the second compartment.
        K3 (dolfin.fem.function.Constant or dolfin.fem.function.Function): Permeability tensor/scalar for the third compartment.
        beta12 (dolfin.fem.function.Constant): Exchange coefficient between compartment 1 and 2.
        beta23 (dolfin.fem.function.Constant): Exchange coefficient between compartment 2 and 3.
        pa (float): Constant value for arterial pressure (Dirichlet boundary condition).
        pv (float): Constant value for venous pressure (Dirichlet boundary condition).
        read_inlet_boundary (bool): If True, boundary conditions are read from `inlet_boundary_file`.
        inlet_boundary_file (str): Path to the CSV file containing inlet boundary conditions (if `read_inlet_boundary` is True).
        inlet_BC_type (str): Type of inlet boundary condition ('DBC' for Dirichlet, 'NBC' for Neumann, 'mixed' for mixed).
        **kwarg:
            model_type (str, optional): The type of model to use ('acv' or 'a'). Defaults to 'acv'.

    Returns:
        tuple: A tuple containing:
            - LHS (dolfin.fem.form.Form): The left-hand side of the variational problem.
            - RHS (dolfin.fem.form.Form): The right-hand side of the variational problem.
            - sigma1 (dolfin.fem.function.Constant): Source term for the first compartment (typically 0.0).
            - sigma2 (dolfin.fem.function.Constant): Source term for the second compartment (typically 0.0).
            - sigma3 (dolfin.fem.function.Constant): Source term for the third compartment (typically 0.0).
            - BCs (list): A list of dolfin.fem.dirichletbc.DirichletBC objects applied to the problem.

    Raises:
        Exception: If `inlet_BC_type` is unknown or `model_type` is unknown.
    """
    model_type = kwarg.get('model_type', 'acv')

    # Source terms are equal to zero
    sigma1 = fem.Constant(mesh, 0.0)
    sigma2 = fem.Constant(mesh, 0.0)
    sigma3 = fem.Constant(mesh, 0.0)

    BCs = []
    integrals_N = []
    V_space = V.sub(0) if model_type == 'acv' else V
    
    # Based on inlet boundary file
    if read_inlet_boundary:
        BC_data, b1, boundary_labels, data_dimension_greater_than_1 = read_boundary_conditions(inlet_boundary_file)
        n_labels = len(boundary_labels)

        if model_type == 'acv':
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                facet_indices = np.where(boundaries.values == boundary_id)[0]
                boundary_facets = boundaries.indices[facet_indices]
                # CORRECTION : Localisation of DOFs for compartment 3 (venous) 
                dofs_v3 = fem.locate_dofs_topological(V.sub(2), mesh.topology.dim - 1, boundary_facets)
                BCs.append(fem.dirichletbc(fem.Constant(mesh, float(pv)), dofs_v3, V.sub(2)))

        dS = ufl.ds(subdomain_data=boundaries)
        if inlet_BC_type == 'DBC': # Dirichlet BC
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                BCs = apply_dirichlet_BC(V_space, BC_data, BCs, boundaries, data_dimension_greater_than_1, i, boundary_id)

        elif inlet_BC_type == 'NBC': # Neumann BC
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                integrals_N = apply_neumann_BC(mesh, b1, integrals_N, i, v_1, dS, boundary_id)

        elif inlet_BC_type == 'mixed': # Mixed BC
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                integrals_N, BCs = apply_mixed_BC(mesh, boundaries, integrals_N, BCs, BC_data, V_space, b1, dS, v_1, i,
                                                   data_dimension_greater_than_1, boundary_id)
        else:
            raise Exception("inlet_BC_type must be Neumann, Dirichlet or mixed ('NBC', 'DBC' or 'mixed)")

    else:
        boundary_labels, n_labels = suppl_fcts.region_label_assembler(boundaries)
        for i in range(n_labels):
            if boundary_labels[i] > 2:
                # Brain surface boundary
                boundary_id = boundary_labels[i]
                facet_indices = np.where(boundaries.values == boundary_id)[0]
                boundary_facets = boundaries.indices[facet_indices]
                
                if inlet_BC_type == 'DBC':
                    # CORRECTION : We give V_space (V.sub(0)) instead of V
                    dofs_p = fem.locate_dofs_topological(V_space, mesh.topology.dim - 1, boundary_facets)
                    BCs.append(fem.dirichletbc(fem.Constant(mesh, float(pa)), dofs_p, V_space))
                    
                if model_type == "acv":
                    dofs_v = fem.locate_dofs_topological(V.sub(2), mesh.topology.dim - 1, boundary_facets)
                    BCs.append(fem.dirichletbc(fem.Constant(mesh, float(pv)), dofs_v, V.sub(2)))

    if model_type == 'acv':
        # Define variational problem
        LHS = \
        inner(K1 * grad(p_1), grad(v_1)) * dx + beta12 * (p_1 - p_2) * v_1 * dx \
        + inner(K2 * grad(p_2), grad(v_2)) * dx + beta12 * (p_2 - p_1) * v_2 * dx + beta23 * (p_2-p_3) * v_2 * dx \
        + inner(K3 * grad(p_3), grad(v_3)) * dx + beta23 * (p_3 - p_2) * v_3 * dx
        RHS = sigma1 * v_1 * dx + sigma2 * v_2 * dx + sigma3 * v_3 * dx + sum(integrals_N)

    elif model_type == 'a':
        # Set constant venous pressure
        p_venous =  fem.Constant(mesh, float(pv))

        # Define variational problem
        beta_total = 1 / (1/beta12+1/beta23)
        LHS = \
        inner(K1 * grad(p), grad(v_1)) * dx + beta_total * p * v_1 * dx
        RHS = sigma1 * v_1 * dx + sum(integrals_N) + beta_total * p_venous * v_1 * dx

    else:
        raise Exception("unknown model type: " + model_type)

    return LHS, RHS, sigma1, sigma2, sigma3, BCs

def solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol, **kwarg):
    """
    Solves the linear system defined by the variational problem using a specified solver and preconditioner.

    Args:
        Vp (dolfin.fem.functionspace.FunctionSpace): The function space for pressure.
        LHS (dolfin.fem.form.Form): The left-hand side of the variational problem.
        RHS (dolfin.fem.form.Form): The right-hand side of the variational problem.
        BCs (list): A list of dolfin.fem.dirichletbc.DirichletBC objects applied to the problem.
        lin_solver (str): The type of linear solver to use (e.g., 'gmres', 'lu').
        precond (str or bool): The type of preconditioner to use (e.g., 'ilu', 'jacobi'), or False if no preconditioner.
        rtol (float or bool): The relative tolerance for the Krylov solver, or False if default is used.
        mon_conv (bool): If True, monitor the convergence of the Krylov solver.
        init_sol (bool): If True, use a non-zero initial guess for the solver.
        **kwarg:
            timer (bool, optional): If True, prints the computation time for solving the system. Defaults to True if not specified.

    Returns:
        dolfin.fem.function.Function: The solved pressure field as a DOLFIN Function object.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Define functions for pressures
    p = fem.Function(Vp)
    form_LHS = fem.form(LHS)
    form_RHS = fem.form(RHS)
    mesh_obj = Vp.mesh

    # 1. Initialization of PETSc structures for assembly
    A = petsc.create_matrix(form_LHS)
    b = petsc.create_vector(Vp)

    # 2. Configuration of the KSP (Krylov) solver
    ksp = PETSc.KSP().create(mesh_obj.comm)
    ksp.setType(lin_solver)

    if precond:
        ksp.getPC().setType(precond)
    else:
        ksp.getPC().setType("none")

    # Tolerance options
    if rtol:
        ksp.setTolerances(rtol=float(rtol))
    
    opts = PETSc.Options()
    if mon_conv:
        opts["ksp_monitor"] = None
    if init_sol:
        ksp.setInitialGuessNonzero(True)
    ksp.setFromOptions()

    start = time.time()

    # 3. Assembly of the matrix and vector with BC handling
    A.zeroEntries()
    petsc.assemble_matrix(A, form_LHS, bcs=BCs)
    A.assemble()

    with b.localForm() as b_local:
        b_local.set(0.0)
    petsc.assemble_vector(b, form_RHS)
    petsc.apply_lifting(b, [form_LHS], bcs=[BCs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, BCs)

    # 4. Actual solving of the linear system
    ksp.setOperators(A)
    ksp.solve(b, p.x.petsc_vec)
    p.x.scatter_forward()  # Ensure the solution is updated across all processes

    end = time.time()

    # 5. Clean up PETSc objects to free memory
    A.destroy()
    b.destroy()
    ksp.destroy()
    if rank == 0:
        if 'timer' in kwarg:
            if kwarg.get('timer')==True:
                print ('\t\t pressure computation took', end - start, '[s]')
        else:
            print ('\t\t pressure computation took', end - start, '[s]')
    
    return p