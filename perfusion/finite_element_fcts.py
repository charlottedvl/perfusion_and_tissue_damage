from dolfin import *
import numpy as np
import time
import suppl_fcts

#%%
def mesh_reader(mesh_file):
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
def alloc_fct_spaces(mesh,fe_degr):
    # element type and degree
    P = FiniteElement('Lagrange', tetrahedron, fe_degr)
    # define mixed element (3D vector)
    element = MixedElement([P, P, P])
    # define continous function space for pressure
    Vp = FunctionSpace(mesh, element)
    
    # Define test functions
    v_1, v_2, v_3 = TestFunctions(Vp)
    
    # Define trial function for pressures
    p = TrialFunction(Vp)
    # Split pressure function to access components
    p_1, p_2, p_3 = split(p)
    
    # define discontinous function space for permeability tensors
    K1_space = TensorFunctionSpace(mesh, "DG", 0)
    K2_space = FunctionSpace(mesh, "DG", 0)
    
    # define function space for velocity vectors
    if fe_degr == 1:
        Vvel = VectorFunctionSpace(mesh, "DG", 0)
    else:
        Vvel = VectorFunctionSpace(mesh, "Lagrange", fe_degr-1)
    
    return Vp, Vvel, v_1, v_2, v_3, p, p_1, p_2, p_3, K1_space, K2_space

#%%
def set_up_fe_solver(mesh, V, v_1, v_2, v_3, p, p_1, p_2, p_3, \
                     K1,K2,K3,beta12,beta13,beta21,beta23,beta31,beta32, \
                     boundaries,pa,pv,subdomains,inflow_file):
    # source terms are equal to zero
    sigma1 = Constant(0.0)
    sigma2 = Constant(0.0)
    sigma3 = Constant(0.0)

    neumann_bc_data = np.loadtxt(inflow_file,skiprows=1)
    boundaryids = [int(i) for i in neumann_bc_data[:,0]]


    # Define boundary conditions
    # DIRICHLET BOUNDARY CONDITIONS
    # compartment 1 - arterioles
    arteriolesboudnary = [DirichletBC(V.sub(0),  Constant(pa), boundaries, i) for i in boundaryids]

    # a1_4 = DirichletBC(V.sub(1),  Constant(pa), boundaries, 4)
    # a1_21 = DirichletBC(V.sub(1), Constant(pa), boundaries, 21)
    # a1_22 = DirichletBC(V.sub(1), Constant(pa), boundaries, 22)
    # a1_23 = DirichletBC(V.sub(1), Constant(pa), boundaries, 23)
    # a1_24 = DirichletBC(V.sub(1), Constant(pa), boundaries, 24)
    # a1_25 = DirichletBC(V.sub(1), Constant(pa), boundaries, 25)
    # a1_26 = DirichletBC(V.sub(1), Constant(pa), boundaries, 26)
    # compartment 3 - venules
    venulesboundary =[DirichletBC(V.sub(2), Constant(pv), boundaries, i) for i in boundaryids]
    # a3_4 = DirichletBC(V.sub(2), Constant(pv), boundaries, 4)
    # a3_21 = DirichletBC(V.sub(2), Constant(pv), boundaries, 21)
    # a3_22 = DirichletBC(V.sub(2), Constant(pv), boundaries, 22)
    # a3_23 = DirichletBC(V.sub(2), Constant(pv), boundaries, 23)
    # a3_24 = DirichletBC(V.sub(2), Constant(pv), boundaries, 24)
    # a3_25 = DirichletBC(V.sub(2), Constant(pv), boundaries, 25)
    # a3_26 = DirichletBC(V.sub(2), Constant(pv), boundaries, 26)
    
    b1 = Constant(0.0)
    b2 = Constant(0.0)
    b3 = Constant(0.0)
    
    
    """ START - VOLUME FLOW RATE READING/COMPUTATION """
    # bc ID, Q [ml/s], n tri, area [mm^2]
    # neumann_bc_data = np.loadtxt(inflow_file,skiprows=1)
    neumann_bc_data[:,1] = 1000 * neumann_bc_data[:,1]
    b1 = neumann_bc_data[:,1]/neumann_bc_data[:,3]
    integrals_N = []
    dS = ds(subdomain_data=boundaries)
    
    # b1 includes the average surface-normal velocity for the perfusion regions
    # computed from the volumentric flow rate [mm^3/s] and the surface area [mm^2]
    for i in range(len(neumann_bc_data)):
        integrals_N.append(b1[i]*v_1*dS(int(neumann_bc_data[i,0])))
    """ END - VOLUME FLOW RATE READING/COMPUTATION """
    
    
    # Define variational problem
    LHS = \
    -inner(K1*grad(p_1), grad(v_1))*dx - beta12*(p_1-p_2)*v_1*dx \
    -inner(K2*grad(p_2), grad(v_2))*dx - beta21*(p_2-p_1)*v_2*dx - beta23*(p_2-p_3)*v_2*dx \
    -inner(K3*grad(p_3), grad(v_3))*dx - beta32*(p_3-p_2)*v_3*dx
    
    RHS = - sigma1*v_1*dx - sigma2*v_2*dx - sigma3*v_3*dx - sum(integrals_N)

    # return LHS, RHS, sigma1, sigma2, sigma3, b1, b2, b3, \
    #     a1_4, a1_21, a1_22, a1_23, a1_24, a1_25, a1_26, \
    #     a3_4, a3_21, a3_22, a3_23, a3_24, a3_25, a3_26
    return LHS, RHS, sigma1, sigma2, sigma3, b1, b2, b3, \
           arteriolesboudnary, venulesboundary, boundaryids
           

#%%
def set_up_fe_solver2(mesh, subdomains, boundaries, V, v_1, v_2, v_3, \
                         p, p_1, p_2, p_3, K1, K2, K3, beta12, beta23, \
                         pa, pv, read_inlet_boundary, inlet_boundary_file, inlet_BC_type):
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0
    
    
    # source terms are equal to zero
    sigma1 = Constant(0.0)
    sigma2 = Constant(0.0)
    sigma3 = Constant(0.0)
    
    # Boundary treatment
    BCs = []
    integrals_N = []
    # based on inlet boundary file
    if read_inlet_boundary == True:
        BC_data = np.loadtxt(inlet_boundary_file,skiprows=1,delimiter=',')
        if BC_data.ndim>1:
            b1 = 1000 * BC_data[:,1]
            boundary_labels = list(BC_data[:,0])
            n_labels = len(boundary_labels)
        else:
            b1 = [1000 * BC_data[1]]
            boundary_labels = [BC_data[0]]
            n_labels = len(boundary_labels)
        # print(boundary_labels)
        # set constant venous pressure
        for i in range(n_labels):
            BCs.append( DirichletBC(V.sub(2), Constant(pv), boundaries, int(boundary_labels[i])) )
            
        # set dirichlet boundary conditions
        if inlet_BC_type == 'DBC':
            for i in range(n_labels):
                if BC_data.ndim>1:
                    BCs.append( DirichletBC(V.sub(0), Constant(BC_data[i,2]), boundaries, int(boundary_labels[i])) )
                else:
                    BCs.append( DirichletBC(V.sub(0), Constant(BC_data[2]), boundaries, int(boundary_labels[i])) )
        elif inlet_BC_type == 'NBC':
            # Neumann boundary conditions
            dS = ds(subdomain_data=boundaries)
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                area = assemble( Constant(1)*dS(boundary_id,domain=mesh) )
                b1[i] = b1[i]/area
            # b1 includes the average surface-normal velocity for the perfusion regions
            # computed from the volumentric flow rate [mm^3/s] and the surface area [mm^2]
            for i in range(n_labels):
                if b1[i]>0:
                    integrals_N.append(b1[i]*v_1*dS( int(boundary_labels[i]) ))
        elif inlet_BC_type == 'mixed':
            # Neumann boundary conditions
            dS = ds(subdomain_data=boundaries)
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                if BC_data[i,4] == 0: # Dirichlet (pressure) boundary condition
                    if BC_data.ndim>1:
                        BCs.append( DirichletBC(V.sub(0), Constant(BC_data[i,2]), boundaries, boundary_id) )
                    else:
                        BCs.append( DirichletBC(V.sub(0), Constant(BC_data[2]), boundaries, boundary_id) )
                elif BC_data[i,4] == 1: # Neumann (pressure gradient ~ flux) boundary condition
                    area = assemble( Constant(1)*dS(boundary_id,domain=mesh) )
                    b1[i] = b1[i]/area
                else:
                    raise Exception("Boundary condition in the BCs.csv file must be 0 (Dirichlet) or 1 (Neumann)")

            # b1 includes the average surface-normal velocity for the perfusion regions
            # computed from the volumentric flow rate [mm^3/s] and the surface area [mm^2]
            for i in range(n_labels):
                if BC_data[i,4] == 1: # Neumann (pressure gradient ~ flux) boundary condition
                    if b1[i]>0:
                        integrals_N.append(b1[i]*v_1*dS( int(boundary_labels[i]) ))
        else:
            raise Exception("inlet_BC_type must be Neumann or Dirichlet ('NBC' or 'DBC')")
        
    # based on predefined arterial and venous pressure
    # TODO: uniform DBC description does not work in parallel
    # implemented based on the following link:
    # https://stackoverflow.com/questions/37999866/how-to-gather-arrays-of-unequal-length-with-mpi4py
    else:
        # boundary_labels = boundaries.array()
        # # 0: interior face, 1: brain stem cut plane,
        # # 2: ventricular surface, 2+: brain surface
        
        # # STEP 0: collect total array of boundary labels on root
        # sendbuf = boundary_labels
        # # Collect local array sizes using the high-level mpi4py gather
        # sendcounts = np.array(comm.gather(len(sendbuf), root))
        
        # if rank == root:
        #     # print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        #     recvbuf = np.empty(sum(sendcounts), dtype=int)
        # else:
        #     recvbuf = None
        # comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
        
        # # STEP 1: broadcast total array of boundary labels from root
        # if rank == root:
        #     # print("Gathered array: {}".format(recvbuf))
        #     # print(len(recvbuf),len(set(recvbuf)))
        #     recvbuf = np.array(list(set(recvbuf)))
        #     n_labels = np.array([len(recvbuf)])
        # else:
        #     n_labels = np.array([0])
        # comm.Bcast(n_labels, root=root)
        # # print(rank,n_labels)
        
        # if rank == root:
        #     boundary_labels=recvbuf
        # else:
        #     boundary_labels=np.zeros(n_labels[0],dtype=int)
        # comm.Bcast(boundary_labels, root=0)
        
        # # if rank==root:
        # #     print('\n\n TEST ARRAY \n\n')
        # # print(rank,boundary_labels)

        # n_labels = n_labels[0]
        boundary_labels, n_labels = suppl_fcts.region_label_assembler(boundaries)
        for i in range(n_labels):
            if boundary_labels[i]>2:
                # brain surface boundary
                if inlet_BC_type == 'DBC':
                    BCs.append( DirichletBC(V.sub(0), Constant(pa), boundaries, boundary_labels[i]) )
                BCs.append( DirichletBC(V.sub(2), Constant(pv), boundaries, boundary_labels[i]) )
        
        # TODO: write Neumann boundary condition treatment

    """ END - VOLUME FLOW RATE READING/COMPUTATION """
    
    # Define variational problem
    LHS = \
    inner(K1*grad(p_1), grad(v_1))*dx + beta12*(p_1-p_2)*v_1*dx \
    + inner(K2*grad(p_2), grad(v_2))*dx + beta12*(p_2-p_1)*v_2*dx + beta23*(p_2-p_3)*v_2*dx \
    + inner(K3*grad(p_3), grad(v_3))*dx + beta23*(p_3-p_2)*v_3*dx
    
    RHS = sigma1*v_1*dx + sigma2*v_2*dx + sigma3*v_3*dx + sum(integrals_N)

    return LHS, RHS, sigma1, sigma2, sigma3, BCs


#%%s
def solve_lin_sys(Vp,LHS,RHS,BCs,lin_solver,precond,rtol,mon_conv,init_sol):
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Define functions for pressures
    p = Function(Vp)
    
    # define weak form
    problem = LinearVariationalProblem(LHS, RHS, p, BCs)
        
    # TODO: set up initialisation with first order:
    # https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/?show=1124#q1124
    
    # solver settings
    solver = LinearVariationalSolver(problem)
    prm = solver.parameters
    #prm.keys()
    prm['linear_solver'] = lin_solver
    if precond != False:
        prm['preconditioner'] = precond
    if rtol != False:
        PETScOptions.set('ksp_rtol', str(rtol))
        prm['krylov_solver']['relative_tolerance']=rtol
    prm['krylov_solver']["monitor_convergence"] = mon_conv
    prm['krylov_solver']["nonzero_initial_guess"] = init_sol
    
    # solve equation system
    start = time.time()
    solver.solve()
    end = time.time()
    if rank == 0: print ('\t\t pressure computation took', end - start, '[s]')
    
    # syntax for adaptive solver (does not work in parallel)
    #p_1, p_2, p_3 = split(p)
    #M = p_2*dx()
    #tol = 1.e-1
    #problem = LinearVariationalProblem(LHS, RHS, p, BCs)
    #solver = AdaptiveLinearVariationalSolver(problem, M)
    #solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "bicgstab"
    #solver.solve(tol)
    #solver.summary()
    
    # sytanx for mesh refinement
    #new_mesh = refine(mesh)
    #File("new_mesh.pvd") << new_mesh
    return p