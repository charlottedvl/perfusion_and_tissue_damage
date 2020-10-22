from dolfin import *
import numpy as np
import IO_funcs

#%%
def func_space(mesh, eleD, configs):
    # fenite element type and solution function space
    P=FiniteElement('CG', tetrahedron, eleD)
    ele=MixedElement([P, P, P])
    Vc=FunctionSpace(mesh, ele)
    
    # function spaces
    DGSpace=FunctionSpace(mesh,'DG',0)
    CGSpace=FunctionSpace(mesh,'CG',eleD)
    if eleD == 1:
        uSpace = VectorFunctionSpace(mesh, "DG", 0)
    else:
        uSpace = VectorFunctionSpace(mesh, "Lagrange", eleD-1)
    
    # beta
    beta_ac, beta_cv=Function(DGSpace), Function(DGSpace)
    beta_ac=IO_funcs.xdmf_reader(beta_ac, configs.input.beta_ac, configs.input.para_path)
    beta_cv=IO_funcs.xdmf_reader(beta_cv, configs.input.beta_cv, configs.input.para_path)
    # pressure
    pa, pc, pv=Function(CGSpace), Function(CGSpace), Function(CGSpace)
    pa=IO_funcs.xdmf_reader(pa, configs.input.pa, configs.input.para_path)
    pc=IO_funcs.xdmf_reader(pc, configs.input.pc, configs.input.para_path)
    pv=IO_funcs.xdmf_reader(pv, configs.input.pv, configs.input.para_path)
    # velocity
    ua, uc=Function(uSpace), Function(uSpace)
    ua=IO_funcs.xdmf_reader(ua, configs.input.ua, configs.input.para_path)
    uc=IO_funcs.xdmf_reader(uc, configs.input.uc, configs.input.para_path)
    
    # # h5
    # beta_ac, beta_cv=Function(DGSpace), Function(DGSpace)
    # beta_ac=IO_funcs.hdf5_reader(mesh, beta_ac, configs.input.beta_ac, configs.input.para_path)
    # beta_cv=IO_funcs.hdf5_reader(mesh, beta_cv, configs.input.beta_cv, configs.input.para_path)
    # pa, pc, pv=Function(CGSpace), Function(CGSpace), Function(CGSpace)
    # pa=IO_funcs.hdf5_reader(mesh, pa, configs.input.pa, configs.input.para_path)
    # pc=IO_funcs.hdf5_reader(mesh, pc, configs.input.pc, configs.input.para_path)
    # pv=IO_funcs.hdf5_reader(mesh, pv, configs.input.pv, configs.input.para_path)
    # ua, uc=Function(uSpace), Function(uSpace)
    # ua=IO_funcs.hdf5_reader(mesh, ua, configs.input.ua, configs.input.para_path)
    # uc=IO_funcs.hdf5_reader(mesh, uc, configs.input.uc, configs.input.para_path)
    
    # brain depth expression
    depth=Function(DGSpace)
    depth=IO_funcs.hdf5_reader(mesh, depth, configs.input.depth, './')
    
    return Vc, DGSpace, CGSpace, uSpace, beta_ac, beta_cv, pa, pc, pv, ua, uc, depth

#%% Calculation of artifitial diffusion
def art_diff(mesh, ua, D_a, DGSpace, depth, configs):
    uaMag=sqrt(dot(ua,ua))
    h=CellDiameter(mesh)
    
    Peh=uaMag*h/(2*D_a)
    PehVal=project(Peh,DGSpace)
    
    if configs.simulation.Pehdepth == True:
        Pehlim=Expression(PehExp,degree=2,depth=depth)
    else:
        Pehlim=interpolate(Constant(configs.parameter.PehCon),DGSpace)
    
    alpha=np.zeros(mesh.num_cells())
    for i in range(mesh.num_cells()):
        if PehVal.vector()[i]!=0:
            alpha[i]=(1/np.tanh(PehVal.vector()[i])-1/PehVal.vector()[i])/Pehlim.vector()[i]
    alphaMesh=Function(DGSpace)
    alphaMesh.vector()[:]=alpha
    dalta=alphaMesh*uaMag*h/2+D_a
    
    return dalta

#%% Compute DBC in arteriole compartment
# boundary labels: 0 interior face, 1 brain stem cut plane, 2 ventricular surface, 2+ (two digits) brain surface 
def BC(boundaries, Vc, configs):
    # BCa=DirichletBC(Vc.sub(0), configs.simulation.BCa, boundaries,3)
    
    BCa = []
    # based on inlet boundary file
    if configs.input.read_inlet_boundary == True:
        BC_data = np.loadtxt(configs.input.pialBC_file,skiprows=1,delimiter=',')
        if BC_data.ndim>1:
            boundary_labels = list(BC_data[:,0])
            n_labels = len(boundary_labels)
        else:
            boundary_labels = [BC_data[0]]
            n_labels = len(boundary_labels)
        # set constant arteriole concentration
        for i in range(n_labels):
            BCa.append( DirichletBC(Vc.sub(0), configs.simulation.BCa, boundaries, int(boundary_labels[i])) )
            
    return BCa

#%%
def O2_Linear(beta12,beta23,mesh,Vc,pa,pc,pv,ua,uc,phiA,phiC,phiT,D_a,D_c,D_t,SaVa,ScVc,gammaA,gammaC,tau,M,BCa):
    # solve concentration
    v_a, v_c, v_t=TestFunctions(Vc)
    C=TrialFunction(Vc)
    C_a, C_c, C_t=split(C)
    
    LHS = dot(ua, grad(C_a))*v_a*dx + phiA*D_a*dot(grad(C_a),grad(v_a))*dx \
        + beta12*(pa-pc)*C_a*v_a*dx + SaVa*phiA*gammaA*(tau*C_a-C_t)*v_a*dx \
        + dot(uc, grad(C_c))*v_c*dx + phiC*D_c*dot(grad(C_c),grad(v_c))*dx \
        - beta12*(pa-pc)*C_a*v_c*dx + beta23*(pc-pv)*C_c*v_c*dx + ScVc*phiC*gammaC*(tau*C_c-C_t)*v_c*dx \
        + phiT*D_t*dot(grad(C_t),grad(v_t))*dx - SaVa*phiA*gammaA*(tau*C_a-C_t)*v_t*dx - ScVc*phiC*gammaC*(tau*C_c-C_t)*v_t*dx \
        + phiT*M*C_t*v_t*dx
        
    RHS = Constant(0.0)*v_a*dx + Constant(0.0)*v_c*dx + Constant(0.0)*v_t*dx
    
    C=Function(Vc)
    problem=LinearVariationalProblem(LHS, RHS, C, BCa)
    solver=LinearVariationalSolver(problem)
    slprm=solver.parameters
    slprm['linear_solver']='mumps'
    
    solver.solve()
    Ca, Cc, Ct=C.split(deepcopy=True)
    
    return Ca, Cc, Ct

#%%
def O2_nonLinear(beta12,beta23,mesh,ele,pa,pc,pv,ua,uc,phiA,phiC,phiT,D_a,D_c,D_t,SaVa,ScVc,gammaA,gammaC,tau,G,C50,BCa):
    v_a, v_c, v_t=TestFunctions(Vc)
    C=Function(Vc)
    C_a, C_c, C_t=split(C)
    
    LHS = dot(ua, grad(C_a))*v_a*dx + phiA*D_a*dot(grad(C_a),grad(v_a))*dx \
        + beta12*(pa-pc)*C_a*v_a*dx + SaVa*phiA*gammaA*(tau*C_a-C_t)*v_a*dx \
        + dot(uc, grad(C_c))*v_c*dx + phiC*D_c*dot(grad(C_c),grad(v_c))*dx \
        - beta12*(pa-pc)*C_a*v_c*dx + beta23*(pc-pv)*C_c*v_c*dx + ScVc*phiC*gammaC*(tau*C_c-C_t)*v_c*dx \
        + phiT*D_t*dot(grad(C_t),grad(v_t))*dx - SaVa*phiA*gammaA*(tau*C_a-C_t)*v_t*dx - ScVc*phiC*gammaC*(tau*C_c-C_t)*v_t*dx \
        + phiT*G*C_t/(C50-C_t)*v_t*dx
        
    J=derivative(LHS, C)
    problem=NonlinearVariationalProblem(LHS, C, BCa, J)
    solver=NonlinearVariationalSolver(problem)
    slprm=solver.parameters
    slprm['newton_solver']['linear_solver']='mumps'
    
    solver.solve()
    Ca, Cc, Ct=C.split(deepcopy=True)
    
    return Ca, Cc, Ct
    

