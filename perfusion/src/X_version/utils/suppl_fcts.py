from symtable import Function

import dolfinx
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
import basix
import numpy as np
from typing import Optional
import pandas as pd
from mpi4py import MPI
import warnings
import os


# Local import
from src.X_version.io.IO_functions import print0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

def set_coupling_coeff(beta):
    beta12 = beta[0,1]
    beta13 = beta[0,2]
    beta23 = beta[1,2]
    
    beta21 = beta12
    beta31 = beta13
    beta32 = beta23
    
    return beta12, beta13, beta21, beta23, beta31, beta32

def compute_vessel_orientation(subdomains, boundaries, mesh, res_fldr, save_subres):
    """
    Orientation is computed based on a flow field originating from the cortical
    surface and running towards the ventricles.

    Args:
        subdomains (dolfinx.mesh.MeshTags or None): MeshTags for cell regions (not directly used in this function, but passed for consistency).
        boundaries (dolfinx.mesh.MeshTags or None): MeshTags for facet regions (boundary markers).
        mesh_obj (dolfinx.mesh.Mesh): The Dolfinx mesh object.
        res_fldr (str): Folder to save results.
        save_subres (bool): Whether to save intermediate results (pe field).

    Returns:
        tuple: A tuple containing:
            - e (dolfinx.fem.Function): Normalised penetrating vessel axis direction.
            - main_direction (dolfinx.fem.Function): Main direction of penetrating vessel axes.
    """
    fe_degr = 2

    # Scalar function space for the thickness values.
    Vpe = fem.functionspace(mesh, ("Lagrange", fe_degr))

    pe_in = 1.0  # Pial surface
    pe_out = 0.0  # Ventricular surface

    # NOTATION:
    # 0: Interior surfaces, 1: Brainstem-cut plane,
    # 2: Ventricular surface, 2+: Brain surface (pial)

    # Ventricular surface
    facet_indices = np.array(boundaries.indices[np.where(boundaries.values == 2)], dtype=np.int32)
    boundary_dofs = fem.locate_dofs_topological(Vpe, entity_dim=2, entities=facet_indices)
    BCs = [fem.dirichletbc(pe_out, boundary_dofs, Vpe)]

    # Pial surface
    boundary_labels, n_labels = region_label_assembler(boundaries)

    for i in range(n_labels):
        if boundary_labels[i] > 2:
            facet_indices = np.array(boundaries.indices[np.where(boundaries.values == boundary_labels[i])],
                                     dtype=np.int32)
            boundary_dofs = fem.locate_dofs_topological(Vpe, entity_dim=2, entities=facet_indices)
            BCs.append(fem.dirichletbc(pe_in, boundary_dofs, Vpe))

    # Solve for scalar thickness field
    pe_trial = ufl.TrialFunction(Vpe)
    ve_test = ufl.TestFunction(Vpe)
    f = fem.Constant(mesh, 0.0)
    LHS = ufl.inner(ufl.grad(pe_trial), ufl.grad(ve_test)) * ufl.dx  # Bilinear form
    RHS = f * ve_test * ufl.dx  # Linear form
    # Initialise the function pe in the predefined function space
    pe = fem.Function(Vpe)
    problem = LinearProblem(LHS, RHS, bcs=BCs, petsc_options={"ksp_type": "bcgs", "pc_type": "hypre"}, petsc_options_prefix="pe_")
    pe = problem.solve()
    # Lower the order to save or visualise the solution
    V_write = fem.functionspace(mesh, ("Lagrange", 1))
    pe_interpolated = fem.Function(V_write)
    pe_interpolated.interpolate(pe)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, res_fldr + 'pe.xdmf', "w") as myfile:
        myfile.write_mesh(mesh)  # NOTE: needed?
        myfile.write_function(pe_interpolated)

    # Define function spaces based on fe_degr
    if fe_degr == 1:
        # Vector element for DG degree 0
        # A function space for piecewise constant vectors over tetrahedra
        element_vec = basix.ufl.element("DG", "tetrahedron", 0, shape=(3,))
        Ve = fem.functionspace(mesh, element_vec)
    else:
        # Vector element for Lagrange and DG degree (fe_degr-1)
        element_vec_lagrange = basix.ufl.element("Lagrange", "tetrahedron", fe_degr - 1, shape=(3,))
        Ve = fem.functionspace(mesh, element_vec_lagrange)

        # DG degree 0 element for Ve_DG
        element_vec_DG = basix.ufl.element("DG", "tetrahedron", 0, shape=(3,))
        Ve_DG = fem.functionspace(mesh, element_vec_DG)

    # Solve finite element problem for projection
    u_proj = ufl.TrialFunction(Ve)
    v_proj = ufl.TestFunction(Ve)
    f_expr = -ufl.grad(pe)
    a_proj = ufl.inner(u_proj, v_proj) * ufl.dx(mesh)
    L_proj = ufl.inner(f_expr, v_proj) * ufl.dx(mesh)
    problem_proj = LinearProblem(a_proj, L_proj, bcs=[], petsc_options={"ksp_type": "bcgs"}, petsc_options_prefix="proj_")
    E = problem_proj.solve()
    E.name = "e_loc" 

    # Solve finite element problem for interpolation
    if fe_degr > 1:
        # Cell-averaging evaluation
        # Negligible difference with Legacy after normalisation
        u_dg = ufl.TrialFunction(Ve_DG)
        v_dg = ufl.TestFunction(Ve_DG)
        a_proj_dg = ufl.inner(u_dg, v_dg) * ufl.dx(mesh)
        L_proj_dg = ufl.inner(E, v_dg) * ufl.dx(mesh)
        problem_proj_dg = LinearProblem(a_proj_dg, L_proj_dg, bcs=[], petsc_options={"ksp_type": "bcgs"}, petsc_options_prefix="proj_dg_")
        e = problem_proj_dg.solve()
    else:
        e = E

    # Normalise e
    e_array = e.x.array
    for i in range(int(len(e_array) / 3)):
        norm_val = np.linalg.norm(e_array[i * 3:(i + 1) * 3])
        if norm_val > 0:
            e_array[i * 3:(i + 1) * 3] /= norm_val
    e.x.array[:] = e_array

    # Define Vdir (using a DG0 space for scalar values)
    Vdir = fem.functionspace(mesh, ("DG", 0))
    # Compute main direction of the vessels
    print0('step 2.3.1: COMPUTING MAIN DIRECTION')
    main_direction = fem.Function(Vdir)
    main_direction.name = "main_direction"
    main_direction_array = main_direction.x.array
    for i in range(int(len(e_array) / 3)):
        indices = np.where(abs(e_array[i * 3:(i + 1) * 3]) >= np.sqrt(1 / 3))[0]
        if indices.size > 0:
            main_direction_array[i] = indices[0]
        else:
            main_direction_array[i] = -1

    main_direction.x.array[:] = main_direction_array

    return e, main_direction

def permeability_tensor_computation(
        K_space: Optional[dolfinx.fem.function.FunctionSpace],
        subdomains: Optional[dolfinx.mesh.MeshTags],
        mesh: Optional[dolfinx.mesh.Mesh],
        e_ref,
        e_loc,
        K1_form
):
    """
    Computes a function which gives the rotated permeability tensor
    at every point in the brain, as it follows the local direction of
    blood vessels.

    Parameters
    ----------
    K_space : dolfinx.fem.function.FunctionSpace
        The function space defined for the permeabilities.
    subdomains : dolfinx.mesh.MeshTags
        Subdomains from the mesh.
    mesh : dolfinx.mesh.Mesh
        Information about the mesh.
    e_ref : type
        Direction vector representing vessel orientation at the reference point.
    e_loc : type
        The local direction vector.
    K1_form : type
        The normalised form of the permeability tensor at the reference point,
        to be rotated and scaled.

    Returns
    -------
    type
        Description of the return value.
    """
    # Get the mesh's topological dimension
    dim   = mesh.topology.dim
    # Get an index map of facets
    imap  = mesh.topology.index_map(dim)
    # Get the local ownership rank
    start, end = imap.local_range

    # Extract array from the function
    e_loc_arr = e_loc.x.array

    # Initialise a new function in K_space
    K1 = fem.Function(K_space)
    K1.name = "K1_form"
    K1_arr = np.zeros(K1.x.array.shape, dtype=K1.x.array.dtype)
    print(f"[Rank {rank}] Initialized K1_arr dtype={K1_arr.dtype}, shape={K1_arr.shape}")

    # Loop over owned cells only (local indices)
    for local_cell in range(start, end):
        off = local_cell - start
        # Extract local direction vector segment
        segment_start = off*3
        segment_end = (off+1)*3
        e1 = e_loc_arr[segment_start:segment_end]
        # Debug: e1 shape
        if e1.shape != (3,):
            print(f"[Rank {rank}] WARNING: e1 slice shape mismatch at local_cell={local_cell}, got {e1.shape}, expected (3,)")

        # Compute transformation matrix
        T = comp_transf_mat(e_ref, e1)
        # Debug: T shape and finite check
        if T.shape != (3,3):
            print(f"[Rank {rank}] WARNING: T shape mismatch at local_cell={local_cell}, got {T.shape}, expected (3,3)")

        # Rotate permeability tensor
        K1_rot = T @ K1_form @ T.T
        # Debug: rotated tensor shape and values
        if K1_rot.shape != (3,3):
            print(f"[Rank {rank}] WARNING: K1_rot shape mismatch at local_cell={local_cell}, got {K1_rot.shape}, expected (3,3)")
        if not np.all(np.isfinite(K1_rot)):
            print(f"[Rank {rank}] WARNING: Non-finite values in K1_rot at local_cell={local_cell}")

        flat = K1_rot.reshape(9)
        # Debug: flat shape
        if flat.shape != (9,):
            print(f"[Rank {rank}] WARNING: flat reshape mismatch at local_cell={local_cell}, got {flat.shape}, expected (9,)")
        # Assign to output vector
        start_idx = off*9
        end_idx = (off+1)*9
        K1_arr[start_idx:end_idx] = flat

    # Zero‐tolerance cleanup
    tol = 1e-9
    mask = np.abs(K1_arr) < tol
    cleanup_count = np.count_nonzero(mask)
    K1_arr[mask] = 0.0

    # Assign to Function and sync ghost values
    K1.x.array[:] = K1_arr
    K1.x.scatter_forward()

    return K1

def scale_permeabilities(subdomains, K1, K2, K3, \
                         K1_ref_gm, K2_ref_gm, K3_ref_gm, gmowm_perm_rat,res_fldr,**kwarg):

    loc1 = subdomains.indices[np.where(subdomains.values == 11)] # white matter cell indices
    loc2 = subdomains.indices[np.where(subdomains.values == 12)] # gray matter cell indices
        
    K1_array = K1.x.array
    K2_array = K2.x.array
    K3_array = K3.x.array
    
    # obtain reference values    
    K1_ref_wm = K1_ref_gm/gmowm_perm_rat
    K2_ref_wm = K2_ref_gm/gmowm_perm_rat
    K3_ref_wm = K3_ref_gm/gmowm_perm_rat
    
    location = 0
    for loc in [loc1,loc2]:
        for i in range(len(loc)):
            idx1 = int(loc[i])*9
            idx2 = int(loc[i])*9+9
            if location == 0: #WM
                K1_array[idx1:idx2] *= K1_ref_wm
                K3_array[idx1:idx2] *= K3_ref_wm
                K2_array[loc[i]] = K2_ref_wm
            else: #GM
                K1_array[idx1:idx2] *= K1_ref_gm
                K3_array[idx1:idx2] *= K3_ref_gm
                K2_array[loc[i]] = K2_ref_gm
        location = location + 1
    
    K1.x.array[:] = K1_array
    K2.x.array[:] = K2_array
    K3.x.array[:] = K3_array
    
    return K1, K2, K3

def scale_coupling_coefficients(subdomains, beta12gm, beta23gm, gmowm_beta_rat, \
                                K2_space, res_fldr,**kwarg): 
    
    loc1 = subdomains.indices[np.where(subdomains.values == 11)] # white matter cell indices
    loc2 = subdomains.indices[np.where(subdomains.values == 12)] # gray matter cell indices
    
    beta12 = fem.Function(K2_space)
    beta23 = fem.Function(K2_space)
    beta12_array = beta12.x.array
    beta23_array = beta23.x.array
    
    beta12_array[loc2] = beta12gm
    beta12_array[loc1] = beta12gm/gmowm_beta_rat
    beta23_array[loc2] = beta23gm
    beta23_array[loc1] = beta23gm/gmowm_beta_rat
    
    beta12.x.array[:] = beta12_array
    beta12.x.scatter_forward()

    beta23.x.array[:] = beta23_array
    beta23.x.scatter_forward()
    
    return beta12, beta23


def region_label_assembler(region: Optional[dolfinx.mesh.MeshTags]) -> tuple[np.ndarray, int]:
    """
    Returns a list of unique region tags and how many unique regions there are.

    The 2,164,216 faces from the mesh are split between each process, and the
    attached meshtags are noted. The integer tags are recast. A list of the number
    of faces that each process has managed is gathered in an array on the root
    using comm.Gather. This is summed, and an empty receive array is created of
    the proper size. Then comm.Gatherv receives each process' list of tags and
    concatenates them into a single list on the root. The root calculates the
    list of unique labels and the number of unique labels. These are then
    broadcasted to the processes as a tuple containing a NumPy array and an integer.

    Parameters:
        region: dolfinx.mesh.Meshtags
            The boundary facet meshtags from mesh_reader

    Returns:
        region_labels : np.array.int64
            A NumPy array of unique labels; now identical on all processes.

        n_labels : int
            An integer count of unique labels; now identical on all processes.

    Notes
    -----
    - For this particular clustered.xdmf mesh:
    - 1,042,301 cells
    - 2,164,216 facets (in the whole mesh, with no repeats)
    - 159,228 facets on a boundary

    """
    # Access tags from facets counted by process
    region_labels = region.values

    # Cast to int64
    region_labels = np.array(region_labels, dtype=np.int64)
    # Force into positive int32 range
    region_labels = (region_labels % (2 ** 31)).astype(np.int64)

    # Package data
    sendbuf = region_labels
    # Gathers the length of the locally computed facet tags
    sendcounts = np.array(comm.gather(len(sendbuf), root=root))

    # Creates an empty NumPy array on the root process for storing all region labels
    if rank == root:
        recvbuf = np.empty(sum(sendcounts), dtype=np.int64)
    else:
        recvbuf = None

    # Collects all sendbuf arrays from each process and concatenates onto recvbuf on root
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)

    if rank == root:
        # Calculates the distinct labels
        unique_labels = np.array(list(set(recvbuf)))
        # Calculates the number of distinct labels
        n_labels = np.array([len(unique_labels)], dtype=np.int64)
    else:
        n_labels = np.array([0], dtype=np.int64)

    # Broadcast n_labels to all processes
    comm.Bcast(n_labels, root=root)

    # Change name unique_labels to region_labels
    if rank == root:
        region_labels = unique_labels
    else:
        region_labels = np.zeros(n_labels[0], dtype=np.int64)

    # Broadcast region_labels to all processes
    comm.Bcast(region_labels, root=root)

    # Convert n_labels to scalar
    n_labels = int(n_labels[0])

    # Return tuple (region_labels, n_labels) or e.g. ([1, 2, ...], 2)
    return region_labels, n_labels




def comp_transf_mat(e0, e1):
    v = np.cross(e0, e1)
    s = np.linalg.norm(v)  # sine
    c = np.dot(e0, e1)  # cosine
    I = np.identity(3)  # identity matrix
    u = v / s
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    T = c * I + s * ux + (1 - c) * np.tensordot(u, u, axes=0)

    return T

def compute_boundary_area(mesh, boundaries, labels, n_labels):
    """
    Compute the area of each boundary region.

    Args:
        mesh (dolfinx.mesh.Mesh): The mesh.
        boundaries (dolfinx.mesh.MeshTags): Facet tags.
        labels (np.ndarray): Array of boundary label IDs.
        n_labels (int): Number of labels.

    Returns:
        np.ndarray: Area for each label.
    """
    area = []
    for i in range(n_labels):
        tag = int(labels[i])
        dS = ufl.Measure('ds', domain=mesh, subdomain_data=boundaries)
        form = fem.form(fem.Constant(mesh, 1.0) * dS(tag))
        area.append(fem.assemble_scalar(form))
    return np.array(area)


# ---------------------------------------------------------------------------
# compute_subdm_vol
# ---------------------------------------------------------------------------
def compute_subdm_vol(mesh, subdomains, labels, n_labels):
    """
    Compute the volume of each subdomain region.

    Args:
        mesh (dolfinx.mesh.Mesh): The mesh.
        subdomains (dolfinx.mesh.MeshTags): Cell tags.
        labels (np.ndarray): Array of subdomain label IDs.
        n_labels (int): Number of labels.

    Returns:
        np.ndarray: Volume for each label.
    """
    volume = []
    for i in range(n_labels):
        tag = int(labels[i])
        dV = ufl.Measure('dx', domain=mesh, subdomain_data=subdomains)
        form = fem.form(fem.Constant(mesh, 1.0) * dV(tag))
        volume.append(fem.assemble_scalar(form))
    return np.array(volume)


# ---------------------------------------------------------------------------
# surface_integrate
# ---------------------------------------------------------------------------
def surface_integrate(variable, mesh, boundaries, labels, n_labels, magn):
    """
    Integrate a field over each boundary surface region.

    Args:
        variable (dolfinx.fem.Function): Field to integrate.
        mesh (dolfinx.mesh.Mesh): The mesh.
        boundaries (dolfinx.mesh.MeshTags): Facet tags.
        labels (np.ndarray): Boundary label IDs.
        n_labels (int): Number of labels.
        magn (bool): If True, integrate magnitude (for vector fields).

    Returns:
        np.ndarray: Integral value per label.
    """
    dS = ufl.Measure('ds', domain=mesh, subdomain_data=boundaries)
    surface_integrals = []

    # value_size returns an int — safer than value_shape tuple comparison
    is_scalar = (len(variable.function_space.element.value_shape) == 0)

    for i in range(n_labels):
        tag = int(labels[i])
        if is_scalar:
            expr = variable * dS(tag)
        elif magn:
            expr = ufl.sqrt(ufl.inner(variable, variable)) * dS(tag)
        else:
            n = ufl.FacetNormal(mesh)
            expr = ufl.dot(variable, n) * dS(tag)
        surface_integrals.append(fem.assemble_scalar(fem.form(expr)))

    return np.array(surface_integrals)


# ---------------------------------------------------------------------------
# volume_integrate
# ---------------------------------------------------------------------------
def volume_integrate(variable, mesh, subdomains, labels, n_labels, magn):
    """
    Integrate a field over each subdomain volume region.

    Args:
        variable (dolfinx.fem.Function): Field to integrate.
        mesh (dolfinx.mesh.Mesh): The mesh.
        subdomains (dolfinx.mesh.MeshTags): Cell tags.
        labels (np.ndarray): Subdomain label IDs.
        n_labels (int): Number of labels.
        magn (bool): If True, integrate magnitude (for vector fields).

    Returns:
        np.ndarray: Integral value per label, or empty list for non-scalar/non-magn vectors.
    """
    dV = ufl.Measure('dx', domain=mesh, subdomain_data=subdomains)
    volume_integrals = []
    is_scalar = (len(variable.function_space.element.value_shape) == 0)
    if is_scalar:
        for i in range(n_labels):
            tag = int(labels[i])
            form = fem.form(variable * dV(tag))
            volume_integrals.append(fem.assemble_scalar(form))
    elif magn:
        for i in range(n_labels):
            tag = int(labels[i])
            form = fem.form(ufl.sqrt(ufl.inner(variable, variable)) * dV(tag))
            volume_integrals.append(fem.assemble_scalar(form))
    else:
        print0("warning: volumetric integration of non-scalar variables has not been implemented!")
        return volume_integrals

    return np.array(volume_integrals)

def _project(expression, target_space):
    """Project an UFL expression onto a given FunctionSpace using DOLFINx.
    This function assembles the necessary linear system and solves it to obtain
    the projected function. It handles both scalar and vector expressions, and
    can be used for any compatible UFL expression and target function space.
    This function replaces the project() from Legacy"""
      
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting
    from petsc4py import PETSc

    u = ufl.TrialFunction(target_space)
    v = ufl.TestFunction(target_space)

    a_form = fem.form(ufl.inner(u, v) * ufl.dx)
    L_form = fem.form(ufl.inner(expression, v) * ufl.dx)

    # Assemble matrix and vector
    A = assemble_matrix(a_form, bcs=[])
    A.assemble()
    b = assemble_vector(L_form)
    apply_lifting(b, [a_form], bcs=[[]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Configuration of KSP solver (BiCGSTAB + Jacobi preconditioner)
    ksp = PETSc.KSP().create(target_space.mesh.comm)
    ksp.setOperators(A)
    ksp.setType("bcgs")        # Bicgstab
    ksp.getPC().setType("jacobi") 
    ksp.setFromOptions()
    ksp.setUp()

    result = fem.Function(target_space)
    ksp.solve(b, result.x.petsc_vec)
    result.x.scatter_forward()
    
    # Clean up PETSc objects
    A.destroy()
    ksp.destroy()

    return result

def compute_my_variables(p, K1, K2, K3, beta12, beta23, p_venous,
                         Vp, Vvel, K2_space,
                         configs, myResults, compartmental_model, rank, comm, **kwarg):
    """
    Compute derived variables (pressures, velocities, perfusion) and save them in a FEniCSx-friendly format.
    """
    save_data = kwarg.get('save_data', True)
    out_vars = configs['output']['res_vars']
    mesh = K2_space.mesh

    if len(out_vars) > 0:
        if compartmental_model == 'acv':
            # IMPORTANT FEniCSx : We extract and COLLAPSE (.collapse) the sub-functions 
            # so that DOLFINx can save them separately in XDMF files!
            p1 = p.sub(0).collapse()
            p2 = p.sub(1).collapse()
            p3 = p.sub(2).collapse()

            # For mathematical operations/UFL (gradients, Darcy's law), we use ufl.split 
            p1_ufl, p2_ufl, p3_ufl = ufl.split(p)

            if 'perfusion' in out_vars:
                myResults['perfusion'] = _project(beta12 * (p1_ufl - p2_ufl), K2_space)

        elif compartmental_model == 'a':
            # Artériel : p1 is the main solved field
            p1 = p  
            p1_ufl = p

            # Construxtion of constant venous pressure p3
            p3 = fem.Function(Vp)
            p3.x.array[:] = float(p_venous)
            p3.x.scatter_forward()
            p3_ufl = p3

            # Capillary pressure p2 (Algebraic weighted average)
            p2 = _project((beta12 * p1 + beta23 * p3) / (beta12 + beta23), Vp)
            p2_ufl = p2

            # Total exchange coefficient (harmonic mean of beta12 and beta23)
            beta_total = _project(1.0 / (1.0 / beta12 + 1.0 / beta23), K2_space)

            if 'perfusion' in out_vars:
                p_venous_const = fem.Constant(mesh, float(p_venous))
                myResults['perfusion'] = _project(beta_total * (p1 - p_venous_const), K2_space)

        else:
            raise Exception("Unknown compartmental model: " + compartmental_model)

        # Name assignment is important for DOLFINx to save the functions with the correct names in XDMF files!
        p1.name = "press1"
        p2.name = "press2"
        p3.name = "press3"
        
        # Storing in the results dictionary
        myResults['press1'] = p1
        myResults['press2'] = p2
        myResults['press3'] = p3
        myResults['K1']     = K1
        myResults['K2']     = K2
        myResults['K3']     = K3
        myResults['beta12'] = beta12
        myResults['beta23'] = beta23
        
        # Gestion of permeability tensors for UFL (Darcy's law)
        K1_tensor = K1
        K3_tensor = K3
        K2_scalar = K2  # Capillary permeability is treated as a scalar for Darcy's law

        # Computation of flow velocities using Darcy's law (-K * grad(p))
        if 'vel1' in out_vars:
            myResults['vel1'] = _project(-K1_tensor * ufl.grad(p1_ufl), Vvel)
            myResults['vel1'].name = "vel1"
        if 'vel2' in out_vars:
            myResults['vel2'] = _project(-K2_scalar * ufl.grad(p2_ufl), Vvel)
            myResults['vel2'].name = "vel2"
        if 'vel3' in out_vars:
            myResults['vel3'] = _project(-K3_tensor * ufl.grad(p3_ufl), Vvel)
            myResults['vel3'].name = "vel3"

    else:
        if rank == 0:
            print("No variable has been requested for saving in the configuration.")

    if save_data:
        res_keys = set(myResults.keys())
        for myvar in out_vars:
            if myvar in res_keys:
                filepath = configs['output']['res_fldr'] + myvar + '.xdmf'
                field = myResults[myvar]

                if myvar == 'perfusion':
                    perf_scaled = fem.Function(field.function_space)
                    perf_scaled.name = "perfusion"
                    perf_scaled.x.array[:] = field.x.array[:] * 6000.0
                    perf_scaled.x.scatter_forward()
                    
                    with io.XDMFFile(comm, filepath, "w") as myfile:
                        myfile.write_mesh(mesh)
                        myfile.write_function(perf_scaled)
                else:
                    with io.XDMFFile(comm, filepath, "w") as myfile:
                        myfile.write_mesh(mesh)
                        myfile.write_function(field)
            else:
                if rank == 0:
                    print(f"Warning: la variable '{myvar}' n'a pas pu être sauvée car elle n'est pas définie.")

def compute_integral_quantities(configs, myResults, my_integr_vars,
                                mesh, subdomains, boundaries, rank, **kwarg):
    """
    Compute surface and volume integrals of result fields and write to CSV.

    Integral types supported (suffix on variable name in config):
        _surfint  : surface integral
        _voluint  : volume integral
        _surfave  : surface average (integral / area)
        _voluave  : volume average (integral / volume)

    Args:
        configs (dict): Configuration dictionary.
        myResults (dict): Computed field variables.
        my_integr_vars (dict): Dictionary to store integral results (modified in place).
        mesh (dolfinx.mesh.Mesh): The mesh.
        subdomains (dolfinx.mesh.MeshTags): Cell tags.
        boundaries (dolfinx.mesh.MeshTags): Facet tags.
        rank (int): MPI rank.
        **kwarg: Optional 'save_data' (bool, default True).

    Returns:
        tuple: (surf_int_values, surf_int_header, volu_int_values, volu_int_header)
    """
    save_data = kwarg.get('save_data', True)

    surf_int_values = []; surf_int_header = ''; surf_int_dat_struct = ''
    volu_int_values = []; volu_int_header = ''; volu_int_dat_struct = ''
    res_keys = set(myResults.keys())

    int_vars = configs['output']['integral_vars']
    if len(int_vars) == 0:
        if rank == 0:
            print('No variables have been defined for integration!')
        return [], [], [], []

    int_types = {intvar.split('_')[-1] for intvar in int_vars}

    # Prepare boundary info if needed
    if 'surfave' in int_types or 'surfint' in int_types:
        bound_label, n_bound_label = region_label_assembler(boundaries)
        bound_label = bound_label[bound_label > 0]
        n_bound_label = len(bound_label)

    if 'surfave' in int_types:
        bound_areas = compute_boundary_area(mesh, boundaries, bound_label, n_bound_label)
        surf_int_values.append(bound_label)
        surf_int_values.append(bound_areas)
        surf_int_header += 'surf ID,area,'
        surf_int_dat_struct += '%d,%e,'
    elif 'surfint' in int_types:
        surf_int_values.append(bound_label)
        surf_int_header += 'surf ID,'
        surf_int_dat_struct += '%d,'

    # Prepare subdomain info if needed
    if 'voluave' in int_types or 'voluint' in int_types:
        subdom_label, n_subdom_label = region_label_assembler(subdomains)

    if 'voluave' in int_types:
        subdom_vols = compute_subdm_vol(mesh, subdomains, subdom_label, n_subdom_label)
        volu_int_values.append(subdom_label)
        volu_int_values.append(subdom_vols)
        volu_int_header += 'volu ID,volu,'
        volu_int_dat_struct += '%d,%e,'
    elif 'voluint' in int_types:
        volu_int_values.append(subdom_label)
        volu_int_header += 'volu ID,'
        volu_int_dat_struct += '%d,'

    # Compute each requested integral
    for intvar in int_vars:
        intvar_parts = intvar.split('_')
        var2int = intvar_parts[0]
        magn_indicator = intvar_parts[1] == 'magn'
        int_type = intvar_parts[-1]

        if var2int not in res_keys:
            if rank == 0:
                print('warning: ' + var2int + ' variable cannot be integrated - variable undefined!')
            continue

        if int_type == 'surfint':
            my_integr_vars[intvar] = surface_integrate(
                myResults[var2int], mesh, boundaries, bound_label, n_bound_label, magn_indicator
            )
        elif int_type == 'voluint':
            result = volume_integrate(
                myResults[var2int], mesh, subdomains, subdom_label, n_subdom_label, magn_indicator
            )
            if len(result) > 0:
                my_integr_vars[intvar] = result
        elif int_type == 'surfave':
            integrals = surface_integrate(
                myResults[var2int], mesh, boundaries, bound_label, n_bound_label, magn_indicator
            )
            my_integr_vars[intvar] = integrals / bound_areas
        elif int_type == 'voluave':
            result = volume_integrate(
                myResults[var2int], mesh, subdomains, subdom_label, n_subdom_label, magn_indicator
            )
            if len(result) > 0:
                my_integr_vars[intvar] = result / subdom_vols
        else:
            if rank == 0:
                print('warning: ' + int_type + ' is not recognised!')

    # Collect into flat arrays for CSV export
    for intvar in list(my_integr_vars.keys()):
        suffix = intvar.split('_')[-1]
        if suffix[:4] == 'surf':
            surf_int_values.append(my_integr_vars[intvar])
            surf_int_header += intvar + ','
            surf_int_dat_struct += '%e,'
        else:
            volu_int_values.append(my_integr_vars[intvar])
            volu_int_header += intvar + ','
            volu_int_dat_struct += '%e,'

    surf_int_values = np.array(surf_int_values).T if surf_int_values else np.array([])
    volu_int_values = np.array(volu_int_values).T if volu_int_values else np.array([])

    if save_data and rank == 0:
        results_folder = configs['output']['res_fldr'].strip()
        os.makedirs(results_folder, exist_ok=True)

        if surf_int_values.size > 0:
            surf_cols = [c.strip() for c in surf_int_header.rstrip(',').split(',')]
            pd.DataFrame(surf_int_values, columns=surf_cols).to_csv(
                os.path.join(results_folder, 'surface_integrals.csv'), index=False
            )

        if volu_int_values.size > 0:
            volu_cols = [c.strip() for c in volu_int_header.rstrip(',').split(',')]
            pd.DataFrame(volu_int_values, columns=volu_cols).to_csv(
                os.path.join(results_folder, 'volume_integrals.csv'), index=False
            )

    return surf_int_values, surf_int_header, volu_int_values, volu_int_header
