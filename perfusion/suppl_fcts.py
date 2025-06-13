from dolfin import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import IO_fcts


#%%
def set_coupling_coeff(beta):
    beta12 = beta[0,1]
    beta13 = beta[0,2]
    beta23 = beta[1,2]
    
    beta21 = beta12
    beta31 = beta13
    beta32 = beta23
    
    return beta12, beta13, beta21, beta23, beta31, beta32


#%%
def comp_vessel_orientation(subdomains,boundaries,mesh,res_fldr,save_subres):
    """
    orientation is computed based on a flow field originating from the cortical
    surface and running towards the ventricles
    """
    
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0
    
    fe_degr = 2
    Vpe = FunctionSpace(mesh, "Lagrange", fe_degr)
    
    pe_in  = 1.0
    pe_out = 0.0
    
    # list of boundary conditions --> ventricle boundary
    BCs  = [DirichletBC(Vpe, Constant(pe_out), boundaries, 2)]
    
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
    boundary_labels, n_labels = region_label_assembler(boundaries)
    for i in range(n_labels):
        if boundary_labels[i]>2:
            # brain surface boundary
            BCs.append( DirichletBC(Vpe, Constant(pe_in), boundaries, boundary_labels[i]) )
    
    pe = TrialFunction(Vpe)
    ve = TestFunction(Vpe)
    
    f = Constant(0.0)
    
    LHS = inner(grad(pe), grad(ve))*dx
    RHS = f*ve*dx
    
        
    pe = Function(Vpe)
    solve(LHS == RHS, pe, BCs, solver_parameters={'linear_solver':'bicgstab'})
    pe.rename("pe","distorted normalised thickness scalar field")
    
    if save_subres == True:
        with XDMFFile(res_fldr+'pe.xdmf') as myfile:
            myfile.write_checkpoint(pe,"pe", 0, XDMFFile.Encoding.HDF5, False)
    
    if fe_degr == 1:
        Ve = VectorFunctionSpace(mesh, "DG", 0)
    else:
        Ve = VectorFunctionSpace(mesh, "Lagrange", fe_degr-1)
        Ve_DG = VectorFunctionSpace(mesh, "DG", 0)
        
    Vdir = FunctionSpace(mesh, "DG", 0) # function space to store major direction
        
    e = project(-grad(pe),Ve, solver_type='bicgstab')
    if fe_degr > 1:
        e = interpolate(e,Ve_DG)
    
    e_array = e.vector().get_local()
    for i in range(int(len(e_array)/3)):
        e_array[i*3:(i+1)*3] = e_array[i*3:(i+1)*3]/np.linalg.norm(e_array[i*3:(i+1)*3])
    e.vector().set_local(e_array)
    
    main_direction = Function(Vdir)
    main_direction_array = main_direction.vector().get_local()
    for i in range(int(len(e_array)/3)):
        main_direction_array[i] = np.where(abs(e_array[i*3:(i+1)*3])>=np.sqrt(1/3))[0][0]
    main_direction.vector().set_local(main_direction_array)
    
    e.rename("e","normalised penetrating vessel axis direction")
    main_direction.rename("main_direction","main direction of penetrating vessel axes")
    
    return e, main_direction


#%%
def perm_tens_comp(K_space,subdomains,mesh,e_ref,e_loc,K1_form):
    # function spaces for permeability tensors
    K1 = Function(K_space)    
    K1_array = K1.vector().get_local()
    
    e_loc_array = e_loc.vector().get_local()
    
    for i in range(mesh.num_cells()):
        e1 = e_loc_array[i*3:i*3+3]
        T = comp_transf_mat(e_ref,e1)
        
        # rotate local permeability tensor
        K1_loc = np.reshape( np.matmul(np.matmul(T,K1_form),T.transpose()),9 )
        
        # set local permeability tensor
        K1_array[i*9:i*9+9] = K1_loc
    
    tolerance = 1e-9
    mask = abs(K1_array) < tolerance
    K1_array[mask] = 0
    
    K1.vector().set_local(K1_array)
    
    return K1


#%%
def scale_permeabilities(subdomains, K1, K2, K3, \
                         K1_ref_gm, K2_ref_gm, K3_ref_gm, gmowm_perm_rat,res_fldr,**kwarg):
    
    loc1 = subdomains.where_equal(11)# white matter cell indices
    loc2 = subdomains.where_equal(12)# gray matter cell indices
    
    K1_array = K1.vector().get_local()
    K2_array = K2.vector().get_local()
    K3_array = K3.vector().get_local()
    
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
    
    K1.vector().set_local(K1_array)
    K2.vector().set_local(K2_array)
    K3.vector().set_local(K3_array)
    
    return K1, K2, K3
    
#%%
def scale_coupling_coefficients(subdomains, beta12gm, beta23gm, gmowm_beta_rat, \
                                K2_space, res_fldr,**kwarg): 
    
    loc1 = subdomains.where_equal(11)# white matter cell indices
    loc2 = subdomains.where_equal(12)# gray matter cell indices
    
    beta12 = Function(K2_space)
    beta23 = Function(K2_space)
    beta12_array = beta12.vector().get_local()
    beta23_array = beta23.vector().get_local()
    
    beta12_array[loc2] = beta12gm
    beta12_array[loc1] = beta12gm/gmowm_beta_rat
    beta23_array[loc2] = beta23gm
    beta23_array[loc1] = beta23gm/gmowm_beta_rat
    
    beta12.vector().set_local(beta12_array)
    beta23.vector().set_local(beta23_array)
    
    return beta12, beta23
    
#%%
def comp_transf_mat(e0,e1):
#    # formulation #1
#    # (rotation in the plane with a normal vector formed by e1 x e2)
    v = np.cross(e0,e1)
    s = np.linalg.norm(v) # sine
    c = np.dot(e0,e1) # cosine
    I = np.identity(3) # identity matrix
    u = v/s
    ux = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
#    
#    # transformation matrix
#    # based on https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    T = c*I + s*ux + (1-c)*np.tensordot(u,u,axes=0)

#    # formulation #2    
#    T = np.array([
#            [c+u[0]**2*(1-c),u[0]*u[1]*(1-c)-u[2]*s,u[0]*u[2]*(1-c)+u[1]*s],
#            [u[1]*u[0]*(1-c)+u[2]*s,c+u[1]**2*(1-c),u[1]*u[2]*(1-c)+u[0]*s],
#            [u[0]*u[2]*(1-c)+u[1]*s,u[2]*u[1]*(1-c)+u[0]*s,c+u[2]**2*(1-c)]])
    
#    # formulation #3
#    v = np.cross(e0,e1)
#    s = np.linalg.norm(v) # sine
#    rot_vec = v/s*np.arcsin(s)
#    transformation_matrix = R.from_rotvec(rot_vec)
#    T = transformation_matrix.as_dcm()
    return T


#%%
def perm_tens_comp_old(K_space,subdomains,mesh,e0,K1_ref,K2_ref,K3_ref,pial_surf_file):
    # function spaces for permeability tensors
    K1 = Function(K_space)
    K2 = Function(K_space)
    K3 = Function(K_space)
    
    # tetrahedral cell centre coordinates
    K_dof_coord = K_space.tabulate_dof_coordinates()
    K_dof_coord = K_dof_coord.reshape((-1, mesh.geometry().dim()))[::9,:]    
    
    loc1 = subdomains.where_equal(11)# white matter cell indices
    loc2 = subdomains.where_equal(12)# gray matter cell indices
    
    
    # pial surface triangle centres and normal vectors
    import tables
    pial_surf_data = tables.open_file(pial_surf_file)
    pial_mean_tri_coord = pial_surf_data.root.mean_tri_coord[:,:]
    pial_norm_vec = pial_surf_data.root.norm_vec[:,:] 
    pial_surf_data.close()
    
    # WM perfusion is approx. third of GM perfusion
    # GM permeability = 3 * WM permeability
    K1_array = K1.vector().get_local()
    K2_array = K2.vector().get_local()
    K3_array = K3.vector().get_local()
    
    cter = 0
    for loc in [loc1,loc2]:
        for i in range(len(loc)):
            dist = np.linalg.norm( pial_mean_tri_coord - K_dof_coord[loc[i]], axis = 1 )
            min_idx = np.argmin(dist)
            # local coordinate direction is the normal vector of the nearest surface triangle
            e1 = pial_norm_vec[min_idx,:]
            # local coordinate direction is based on the vector connecting point to nearest surface triangle
#            e1 = (pial_mean_tri_coord[min_idx] - K_dof_coord[loc[i]])/dist[min_idx]
            
            T = comp_transf_mat(e0,e1)
            
            K1_loc = abs(np.reshape( np.matmul(np.matmul(T,K1_ref),T.transpose()),9 ))
            K2_loc = abs(np.reshape( np.matmul(np.matmul(T,K2_ref),T.transpose()),9 ))
            K3_loc = abs(np.reshape( np.matmul(np.matmul(T,K3_ref),T.transpose()),9 ))
            
            if cter == 0: # WM permeability is third of GM permeability
                K1_array[int(loc[i])*9:int(loc[i])*9+9] = K1_loc/3
                K2_array[int(loc[i])*9:int(loc[i])*9+9] = K2_loc/3
                K3_array[int(loc[i])*9:int(loc[i])*9+9] = K3_loc/3
            else:
                K1_array[int(loc[i])*9:int(loc[i])*9+9] = K1_loc
                K2_array[int(loc[i])*9:int(loc[i])*9+9] = K2_loc
                K3_array[int(loc[i])*9:int(loc[i])*9+9] = K3_loc
        cter = cter + 1
    
    K1.vector().set_local(K1_array)
    K2.vector().set_local(K2_array)
    K3.vector().set_local(K3_array)
    
    return K1, K2, K3


#%%
def surface_ave(mesh,boundaries,vels,ps):
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    boundary_labels, n_labels = region_label_assembler(boundaries)
    boundary_labels = boundary_labels[1::]
    n_labels = n_labels - 1
    
    n = FacetNormal(mesh)
    dS = ds(subdomain_data=boundaries)
    
    fluxes = []
    surf_p_values = []
    p_pial_ave = [0,0,0]
    
    
    total_area = assemble( Constant(1.0)*ds(domain=mesh) )
    pial_weighting_area = 0
    pressure_weight = 0
    # compute fluxes and surface area for each boundary region
    for i in range(n_labels):
        ID = int(boundary_labels[i])
        area = assemble( Constant(1.0)*dS(ID,domain=mesh) )
        
        fluxes_ID = [ ID, area ]
        surf_p = [ ID, area ]
        for j in range(3):
            fluxes_ID.append( assemble(dot(vels[j],n)*dS(ID)) )
            p_ave = assemble( Constant(1.0)*ps[j]*dS(ID) )
            surf_p.append( p_ave/area )
            if ID > 2:
                p_pial_ave[j] = p_pial_ave[j] + p_ave
        fluxes.append(fluxes_ID)
        surf_p_values.append(surf_p)
        if ID > 2:
            pial_weighting_area = pial_weighting_area + area
    
    # compute the net surface area and fluxes of compartments
    fluxes_ID = [ sum(boundary_labels), total_area ]
    surf_p = [ sum(boundary_labels), total_area ]
    for j in range(3):
        fluxes_ID.append( assemble(dot(vels[j],n)*ds) )
        surf_p.append( assemble(ps[j]*ds)/total_area )
    fluxes.append( fluxes_ID )
    surf_p_values.append(surf_p)
    
    surf_p = [ sum((np.array(boundary_labels)>2)*boundary_labels), pial_weighting_area, p_pial_ave[0]/pial_weighting_area, p_pial_ave[1]/pial_weighting_area, p_pial_ave[2]/pial_weighting_area]
    surf_p_values.append(surf_p)
    return np.array(fluxes), np.array(surf_p_values)

# infarct calculation
def infarct_vol(mesh,subdomains,infarct):
    comm = MPI.comm_world
    rank = comm.Get_rank()

    subdom_labels, n_labels = region_label_assembler(subdomains)

    dV = dx(subdomain_data=subdomains)
    vol_p_values = []

    # compute volume and characteristic values for each region
    for i in range(n_labels):
        ID = int(subdom_labels[i])
        vol = assemble(Constant(1.0) * dV(ID, domain=mesh))
        char_p_ID = [ID, vol]

        # volume averaged quantities
        char_p_ID.append(assemble(infarct * dV(ID))/1000)
        vol_p_values.append(char_p_ID)

    # compute the net volume and average
    ID = int(sum(subdom_labels))
    vol = assemble(Constant(1.0) * dx(domain=mesh))
    char_p_ID = [ID, vol]

    char_p_ID.append(assemble(infarct * dx)/1000)
    vol_p_values.append(char_p_ID)

    return np.array(vol_p_values)



# perfusion calculation
def perfusion_vol(mesh,subdomains,perfusion):
    comm = MPI.comm_world
    rank = comm.Get_rank()

    subdom_labels, n_labels = region_label_assembler(subdomains)

    dV = dx(subdomain_data=subdomains)
    vol_p_values = []

    # compute volume and characteristic values for each region
    for i in range(n_labels):
        ID = int(subdom_labels[i])
        vol = assemble(Constant(1.0) * dV(ID, domain=mesh))
        char_p_ID = [ID, vol]

        # volume averaged quantities
        char_p_ID.append(assemble(perfusion * dV(ID)) / vol)
        vol_p_values.append(char_p_ID)

    # compute the net volume and average
    ID = int(sum(subdom_labels))
    vol = assemble(Constant(1.0) * dx(domain=mesh))
    char_p_ID = [ID, vol]

    char_p_ID.append(assemble(perfusion * dx) / vol)
    vol_p_values.append(char_p_ID)

    return np.array(vol_p_values)

#%%
def vol_ave(mesh,subdomains,ps,vels):
    comm = MPI.comm_world
    rank = comm.Get_rank()
    
    subdom_labels, n_labels = region_label_assembler(subdomains)
    
    dV = dx(subdomain_data=subdomains)
    
    vel_mag = []
    vol_p_values = []
    vol_vel_values = []
    
    
    # compute volume and characteristic values for each region
    for i in range(n_labels):
        ID = int(subdom_labels[i])
        vol = assemble( Constant(1.0)*dV(ID,domain=mesh) )
        char_p_ID = [ ID, vol ]
        char_vel_ID = [ ID, vol ]
        
        # volume averaged quantities
        for j in range(3):
            char_p_ID.append( assemble( ps[j]*dV(ID) )/vol )
            char_vel_ID.append( assemble( sqrt(inner(vels[j], vels[j]))*dV(ID) )/vol )
        
        # TODO: add min and max
        vol_p_values.append(char_p_ID)
        vol_vel_values.append(char_vel_ID)
    
    # compute the net volume and average
    ID = int(sum(subdom_labels))
    vol = assemble( Constant(1.0)*dx(domain=mesh) )
    char_p_ID = [ ID, vol ]
    char_vel_ID = [ ID, vol ]
    for j in range(3):
        char_p_ID.append( assemble(ps[j]*dx)/vol )
        char_vel_ID.append( assemble( sqrt(inner(vels[j], vels[j]))*dx )/vol )
    
    vol_p_values.append(char_p_ID)
    vol_vel_values.append(char_vel_ID)
    
    return np.array(vol_p_values), np.array(vol_vel_values)


#%%
def region_label_assembler(region):
    # in parallel region labels might be distributed between cores
    # this function assembles labels and distributes the corresponding array to each processor
    
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0
    
    region_labels = region.array()
    # 0: interior face, 1: brain stem cut plane,
    # 2: ventricular surface, 2+: brain surface
    
    # STEP 0: collect total array of region labels on root
    sendbuf = region_labels
    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(sendbuf), root))
    
    if rank == root:
        # print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=int)
    else:
        recvbuf = None
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
    
    # STEP 1: broadcast total array of region labels from root
    if rank == root:
        # print("Gathered array: {}".format(recvbuf))
        # print(len(recvbuf),len(set(recvbuf)))
        recvbuf = np.array(list(set(recvbuf)))
        n_labels = np.array([len(recvbuf)])
    else:
        n_labels = np.array([0])
    comm.Bcast(n_labels, root=root)
    # print(rank,n_labels)
    
    if rank == root:
        region_labels=recvbuf
    else:
        region_labels=np.zeros(n_labels[0],dtype=int)
    comm.Bcast(region_labels, root=0)
    
    # if rank==root:
    #     print('\n\n TEST ARRAY \n\n')
    # print(rank,region_labels)

    n_labels = n_labels[0]
    
    return region_labels, n_labels
    
    
#%%
def compute_boundary_area(mesh,boundaries,labels,n_labels):
    area = []
    
    n = FacetNormal(mesh)
    dS = ds(subdomain_data=boundaries)
    for i in range(n_labels):
        ID = int(labels[i])
        area.append( assemble( Constant(1.0)*dS(ID,domain=mesh) ) )
    return np.array(area)


#%%
def compute_subdm_vol(mesh,subdomains,labels,n_labels):
    volume = []
    
    dV = dx(subdomain_data=subdomains)
    for i in range(n_labels):
        ID = int(labels[i])
        volume.append(  assemble( Constant(1.0)*dV(ID,domain=mesh) ) )
    return np.array(volume)
    

#%%
def surface_integrate(variable,mesh,boundaries,labels,n_labels,magn):
    dS = ds(subdomain_data=boundaries)
    surface_integrals = []
    
    if variable.value_rank()==0:
        for i in range(n_labels):
            ID = int(labels[i])
            surface_integrals.append( assemble( variable*dS(ID) ) )
    elif magn:
        for i in range(n_labels):
            ID = int(labels[i])
            surface_integrals.append( assemble( sqrt(inner(variable, variable))*dS(ID) ) )
    else:
        n = FacetNormal(mesh)
        for i in range(n_labels):
            ID = int(labels[i])
            surface_integrals.append( assemble(dot(variable,n)*dS(ID)) )
    return np.array(surface_integrals)


#%%
def volume_integrate(variable,mesh,subdomains,labels,n_labels,magn):
    dV = dx(subdomain_data=subdomains)
    volume_integrals = []
    
    if variable.value_rank()==0:
        for i in range(n_labels):
            ID = int(labels[i])
            volume_integrals.append( assemble( variable*dV(ID) ) )
    elif magn:
        for i in range(n_labels):
            ID = int(labels[i])
            volume_integrals.append( assemble( sqrt(inner(variable, variable))*dV(ID) ) )
    else:
        print("warning: volumetric integration of non-scalar variables has not been implemented!")
        return volume_integrals
    return np.array(volume_integrals)


#%%
def compute_my_variables(p,K1,K2,K3,beta12,beta23,p_venous,Vp,Vvel,K2_space, \
                         configs,myResults,compartmental_model,rank,**kwarg):
    if 'save_data' in kwarg:
        save_data = kwarg.get('save_data')
    else:
        save_data = True
    
    out_vars = configs['output']['res_vars']
    if len(out_vars)>0:
        if compartmental_model == 'acv':
            p1, p2, p3 = p.split()
            if 'perfusion' in out_vars: myResults['perfusion'] = project(beta12 * (p1-p2),K2_space,\
                                                                          solver_type='bicgstab', preconditioner_type='petsc_amg')
        elif compartmental_model == 'a':
            p1, p3 = p.copy(deepcopy=False), p.copy(deepcopy=True)
            p3vec = p3.vector().get_local()
            p3vec[:] = p_venous
            p3.vector().set_local(p3vec)
            p2 = project( (beta12*p1 + beta23*p3)/(beta12+beta23), Vp, solver_type='bicgstab', preconditioner_type='petsc_amg')
            beta_total = project( 1 / (1/beta12+1/beta23), K2_space, solver_type='bicgstab', preconditioner_type='petsc_amg')
            if 'perfusion' in out_vars: myResults['perfusion'] = project( beta_total * (p-Constant(p_venous)),K2_space,\
                                                                          solver_type='bicgstab', preconditioner_type='petsc_amg')
        else:
            raise Exception("unknown model type: " + compartmental_model)
        myResults['press1'], myResults['press2'], myResults['press3'] = p1, p2, p3
        myResults['K1'], myResults['K2'], myResults['K3'] = K1, K2, K3
        myResults['beta12'], myResults['beta23'] = beta12, beta23
        # compute velocities and perfusion
        if 'vel1' in out_vars: myResults['vel1'] = project(-K1*grad(p1),Vvel, solver_type='bicgstab', preconditioner_type='petsc_amg')
        if 'vel2' in out_vars: myResults['vel2'] = project(-K2*grad(p2),Vvel, solver_type='bicgstab', preconditioner_type='petsc_amg')
        if 'vel3' in out_vars: myResults['vel3'] = project(-K3*grad(p3),Vvel, solver_type='bicgstab', preconditioner_type='petsc_amg')
    else:
        if rank==0: print('No variables have been defined for saving!')
    
    # save variables
    res_keys = set(myResults.keys())
    if save_data:
        for myvar in out_vars:
            if myvar in res_keys:
                with XDMFFile(configs['output']['res_fldr']+myvar+'.xdmf') as myfile:
                    if myvar!='perfusion':
                        myfile.write_checkpoint(myResults[myvar], myvar, 0, XDMFFile.Encoding.HDF5, False)
                    else:
                        perf_scaled = myResults[myvar].copy(deepcopy=True)
                        perf_scaled.vector()[:] = perf_scaled.vector()[:]*6000
                        myfile.write_checkpoint(perf_scaled, myvar, 0, XDMFFile.Encoding.HDF5, False)
            else:
                if rank==0: print('warning: '+myvar+' variable cannot be saved - variable undefined!')


#%%
def compute_integral_quantities(configs,myResults,my_integr_vars,mesh,subdomains,boundaries,rank,**kwarg):
    if 'save_data' in kwarg:
        save_data = kwarg.get('save_data')
    else:
        save_data = True
    
    surf_int_values = []; surf_int_header = ''; surf_int_dat_struct = ''
    volu_int_values = []; volu_int_header = ''; volu_int_dat_struct = ''
    res_keys = set(myResults.keys())
    
    int_vars = configs['output']['integral_vars']
    if len(int_vars)>0:
        int_types = set()
        for intvar in int_vars:
            int_types.add( intvar.split('_')[-1] )
        if 'surfave' in int_types:
            bound_label, n_bound_label = region_label_assembler(boundaries)
            bound_label = bound_label[bound_label>0]
            n_bound_label = len(bound_label)
            bound_areas = compute_boundary_area(mesh,boundaries,bound_label,n_bound_label)
            surf_int_values.append(bound_label); surf_int_values.append(bound_areas)
            surf_int_header += 'surf ID,area,'; surf_int_dat_struct += '%d,%e,'
        elif 'surfint' in int_types:
            bound_label, n_bound_label = region_label_assembler(boundaries)
            bound_label = bound_label[bound_label>0]
            n_bound_label = len(bound_label)
            surf_int_values.append(bound_label)
            surf_int_header += 'surf ID,'; surf_int_dat_struct += '%d,'
        if 'voluave' in int_types:
            subdom_label, n_subdom_label = region_label_assembler(subdomains)
            subdom_vols  = compute_subdm_vol(mesh,subdomains,subdom_label,n_subdom_label)
            volu_int_values.append(subdom_label); volu_int_values.append(subdom_vols)
            volu_int_header += 'volu ID,volu,'; volu_int_dat_struct += '%d,%e,'
        elif 'voluint' in int_types:
            subdom_label, n_subdom_label = region_label_assembler(subdomains)
            volu_int_values.append(subdom_label)
            volu_int_header += 'volu ID,'; volu_int_dat_struct += '%d,'
        
        for intvar in int_vars:
            intvar_parts = intvar.split('_')
            var2int = intvar_parts[0]
            magn_indicator = intvar.split('_')[1] == 'magn'
            int_type = intvar_parts[-1]
            if var2int in res_keys:
                if int_type == 'surfint':
                    my_integr_vars[intvar] = surface_integrate(myResults[var2int],mesh,boundaries,\
                                                                          bound_label,n_bound_label,magn_indicator)
                elif int_type == 'voluint':
                    my_integr_vars[intvar] = volume_integrate(myResults[var2int],mesh,subdomains,\
                                                                          subdom_label,n_subdom_label,magn_indicator)
                    if len(my_integr_vars[intvar])==0: del my_integr_vars[intvar]
                elif int_type == 'surfave':
                    my_integr_vars[intvar] = surface_integrate(myResults[var2int],mesh,boundaries,\
                                                                          bound_label,n_bound_label,magn_indicator)
                    my_integr_vars[intvar] = my_integr_vars[intvar]/bound_areas
                elif int_type == 'voluave':
                    my_integr_vars[intvar] = volume_integrate(myResults[var2int],mesh,subdomains,\
                                                                          subdom_label,n_subdom_label,magn_indicator)
                    if len(my_integr_vars[intvar])==0: del my_integr_vars[intvar]
                else:
                    if rank==0: print('warning: ' + int_type + ' is not recognised!')
            else:
                if rank==0: print('warning: '+var2int+' variable cannot be integrated - variable undefined!')
        
        for intvar in list(my_integr_vars.keys()):
            int_types = ( intvar.split('_')[-1] )
            if int_types[:4] == 'surf':
                surf_int_values.append(my_integr_vars[intvar])
                surf_int_header += intvar+','; surf_int_dat_struct += '%e,'
            else:
                volu_int_values.append(my_integr_vars[intvar])
                volu_int_header += intvar+','; volu_int_dat_struct += '%e,'
        surf_int_values = np.array(surf_int_values)
        surf_int_values = surf_int_values.transpose()
        volu_int_values = np.array(volu_int_values)
        volu_int_values = volu_int_values.transpose()
        
        
    
        if save_data:
            if len(surf_int_values)>0:
                np.savetxt(configs['output']['res_fldr']+'surface_integrals.csv',\
                              surf_int_values,surf_int_dat_struct[:-1],header=surf_int_header[:-1])
            if len(volu_int_values)>0:
                np.savetxt(configs['output']['res_fldr']+'volume_integrals.csv',\
                              volu_int_values,volu_int_dat_struct,header=volu_int_header[:-1])
        return surf_int_values, surf_int_header, volu_int_values, volu_int_header
    else:
        if rank==0:
            print('No variables have been defined for integration!')
        return [], [], [], []
