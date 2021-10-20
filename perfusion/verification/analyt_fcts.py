import numpy as np

#%%
def set_up_network(configs):
    # diameter corresponding to each node [m]
    D = np.array(configs['network']['D'])
    
    # number of nodes
    Nn = len(D)
    
    # connectivity based on length [m]
    L = np.zeros([Nn,Nn])
    L_data = configs['network']['L_data']
    for i in range(len(L_data)):
        L[L_data[i][0],L_data[i][1]] = L_data[i][2]
    L = (L + L.T)
    
    #% rigid or flexible large vessels
    block_loc = configs['network']['block_loc']
    
    # define boundary nodes
    BC_ID_ntw = np.array(configs['network']['BC_ID_ntw'])
    
    # define boundary type
    BC_type_ntw = np.array(configs['network']['BC_type_ntw'])
    
    # define bondary values
    BC_val_ntw = np.array(configs['network']['BC_val_ntw'])
    for i in range(len(BC_type_ntw)):
        if BC_type_ntw[i]=='CBC' and BC_val_ntw[i]==1:
            BC_val_ntw[i] = np.sum(np.array(configs['continuum']['l_subdom']))
    
    # average diameter corresponding to each branch [m]
    D_ave = np.zeros([Nn,Nn])
    for i in range(Nn):
        for j in range(Nn):
            D_ave[i,j] = 0.5*(D[i] + D[j])
    D_ave[L==0] = 0
    
    for i in range(len(block_loc)):
        if block_loc[i] != []:
            D_ave[block_loc[i][0],block_loc[i][1]] = 0
            D_ave[block_loc[i][1],block_loc[i][0]] = 0
    
    # resistance and conductivity matrices
    G = np.zeros([Nn,Nn])
    for i in range(Nn):
        for j in range(Nn):
            if L[i,j]>0:
                G[i,j] = np.pi * D_ave[i,j]**4 / (32 * configs['network']['xi'] * L[i,j] * configs['network']['mu'])
    
    return D, D_ave, G, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw


#%%
def set_up_continuum(configs):
    # coupling coefficients
    beta = np.array(configs['continuum']['beta'])
    
    # number of subdomains
    Nc = len(beta)
    
    
    # subdomain_lengths [m]
    l_subdom = np.array(configs['continuum']['l_subdom'])
    
    nx = 1001
    x = np.linspace(0,np.sum(l_subdom),nx)
    beta_sub = 0*x
    subdom_id = np.zeros(len(x),dtype=np.int32)
    for i in range(Nc):
        if i==0:
            subdom_id[ x <= l_subdom[0] ] = i
            beta_sub[ x <= l_subdom[0] ] = beta[i]
        elif i==Nc-1:
            subdom_id[ x > np.sum(l_subdom[:-1]) ] = Nc-1
            beta_sub[ x > np.sum(l_subdom[:-1]) ] = beta[Nc-1]
        else:
            subdom_id[ (x > np.sum(l_subdom[:i])) * (x <= np.sum(l_subdom[:i+1]))] = i
            beta_sub[ (x > np.sum(l_subdom[:i])) * (x <= np.sum(l_subdom[:i+1]))] = beta[i]
    
    # define boundary type
    BC_type_con = configs['continuum']['BC_type_con']
    
    # define bondary values
    BC_val_con = configs['continuum']['BC_val_con']
    
    return beta, Nc, l_subdom, x, beta_sub, subdom_id, BC_type_con, BC_val_con


#%%
def define_network_eq(configs, A, b, D, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw, beta, x, Nc, subdom_id, D_ave, G):
    # network model boundary treatment
    for i in range(Nn):
        if i in(BC_ID_ntw):
            if BC_type_ntw[np.where(BC_ID_ntw==i)[0][0]] == 'DBC':
                A[i,i] = 1
                b[i] = BC_val_ntw[np.where(BC_ID_ntw==i)[0][0]]
            elif BC_type_ntw[np.where(BC_ID_ntw==i)[0][0]] == 'NBC':
                A[i,i] = np.sum(G[i,:])
                for j in range(Nn):
                    if G[i,j] !=0:
                        A[i,j] = -G[i,j]
                b[i] = BC_val_ntw[np.where(BC_ID_ntw==i)[0][0]]
            elif BC_type_ntw[np.where(BC_ID_ntw==i)[0][0]] == 'CBC':
                xloc = BC_val_ntw[np.where(BC_ID_ntw==i)[0][0]]
                my_subdom = int( np.interp(xloc,x,subdom_id) )
                alpha = beta[my_subdom]/configs['continuum']['K']
                
                A[i,i] = 1
                A[i,Nn+my_subdom] = -np.exp(np.sqrt(alpha)*xloc)
                A[i,Nn+Nc+my_subdom] = -np.exp(-np.sqrt(alpha)*xloc)
                b[i] = configs['continuum']['pv']
        else:
            A[i,i] = np.sum(G[i,:])
            for j in range(Nn):
                if G[i,j] !=0:
                    A[i,j] = -G[i,j]
    
    return


#%%
def define_continuum_eq(configs, A, b, beta, Nn, Nc, BC_type_con, BC_val_con, G, l_subdom, x):
    # continuum model boundary treatment
    alpha = beta[0]/configs['continuum']['K']
    if BC_type_con[0] == 'DBC':
        A[Nn,Nn] = 1
        A[Nn,Nn+Nc] = 1
        b[Nn] = BC_val_con[0]
    elif BC_type_con[0] == 'NBC':
        A[Nn,Nn] = np.sqrt(alpha)
        A[Nn,Nn+Nc] = -np.sqrt(alpha)
        b[Nn] = -BC_val_con[0]/configs['continuum']['area']/configs['continuum']['K']
    elif BC_type_con[0] == 'CBC':
        # continuum side
        A[Nn,Nn] = np.sqrt(alpha)
        A[Nn,Nn+Nc] = -np.sqrt(alpha)
        # network side
        A[Nn,BC_val_con[0]] = -np.sum( G[BC_val_con[0],:] )/configs['continuum']['area']/configs['continuum']['K']
        for j in range(Nn):
            if G[BC_val_con[0],j] !=0:
                A[Nn,j] = G[BC_val_con[0],j]/configs['continuum']['area']/configs['continuum']['K']
    
    xloc = np.sum(l_subdom)
    alpha = beta[-1]/configs['continuum']['K']
    A[Nn+Nc,Nn+Nc] = 0
    if BC_type_con[1] == 'DBC':
        A[Nn+Nc,Nn+Nc-1] = np.exp(np.sqrt(alpha)*xloc)
        A[Nn+Nc,Nn+2*Nc-1] = np.exp(-np.sqrt(alpha)*xloc)
        b[Nn+Nc] = BC_val_con[1]
    elif BC_type_con[1] == 'NBC':
        A[Nn+Nc,Nn+Nc-1] = np.sqrt(alpha) * np.exp(np.sqrt(alpha)*x[-1])
        A[Nn+Nc,Nn+2*Nc-1] = -np.sqrt(alpha) * np.exp(-np.sqrt(alpha)*x[-1])
        b[Nn+Nc] = BC_val_con[1]/configs['continuum']['area']/configs['continuum']['K']
    elif BC_type_con[1] == 'CBC':
        # continuum side
        A[Nn+Nc,Nn+Nc-1] = -np.sqrt(alpha) * np.exp(np.sqrt(alpha)*x[-1])
        A[Nn+Nc,Nn+2*Nc-1] = np.sqrt(alpha) * np.exp(-np.sqrt(alpha)*x[-1])
        # network side
        A[Nn+Nc,BC_val_con[1]] = -np.sum( G[BC_val_con[1],:] )/configs['continuum']['area']/configs['continuum']['K']
        for j in range(Nn):
            if G[BC_val_con[1],j] !=0:
                A[Nn+Nc,j] = G[BC_val_con[1],j]/configs['continuum']['area']/configs['continuum']['K']
    
    # # continuum model interface treatment
    for i in range(Nc-1):
        xloc = np.sum(l_subdom[:i+1])
        alpha1 = beta[i]/configs['continuum']['K']
        alpha2 = beta[i+1]/configs['continuum']['K']
        # continous function
        A[Nn+i+1,Nn+i] = np.exp(np.sqrt(alpha1)*xloc)
        A[Nn+i+1,Nn+i+1] = -np.exp(np.sqrt(alpha2)*xloc)
        A[Nn+i+1,Nn+Nc+i] = np.exp(-np.sqrt(alpha1)*xloc)
        A[Nn+i+1,Nn+Nc+i+1] = -np.exp(-np.sqrt(alpha2)*xloc)
        # continous derivative
        A[Nn+Nc+i+1,Nn+i] = np.sqrt(alpha1) * np.exp(np.sqrt(alpha1)*xloc)
        A[Nn+Nc+i+1,Nn+i+1] = -np.sqrt(alpha2) * np.exp(np.sqrt(alpha2)*xloc)
        A[Nn+Nc+i+1,Nn+Nc+i] = -np.sqrt(alpha1) * np.exp(-np.sqrt(alpha1)*xloc)
        A[Nn+Nc+i+1,Nn+Nc+i+1] = +np.sqrt(alpha2) * np.exp(-np.sqrt(alpha2)*xloc)


#%%
def comp_res(configs,beta,subdom_id,xvec,x,G,Nn,Nc,):
    P = xvec[:Nn]
    Q = 0.0*G
    for i in range(Nn):
        for j in range(Nn):
            Q[i,j] = G[i,j] * (P[i]-P[j])
    
    # continuum solution
    Acoeff = xvec[Nn:Nn+Nc]
    Bcoeff = xvec[Nn+Nc:]
    p = 0*x
    vel = 0*x
    for i in range(Nc):
        Term1 = Acoeff[i]*np.exp(np.sqrt(beta[i]/configs['continuum']['K'])*x)
        Term2 = Bcoeff[i]*np.exp(-np.sqrt(beta[i]/configs['continuum']['K'])*x)
        p += (subdom_id==i) * ( configs['continuum']['pv'] + Term1 + Term2 )
        
        Term1 =  Acoeff[i]*np.sqrt(beta[i]/configs['continuum']['K']) * np.exp( np.sqrt(beta[i]/configs['continuum']['K'])*x)
        Term2 = -Bcoeff[i]*np.sqrt(beta[i]/configs['continuum']['K']) * np.exp(-np.sqrt(beta[i]/configs['continuum']['K'])*x)
        vel += (subdom_id==i) * configs['continuum']['K']*( Term1 + Term2 ) * (-1)
    
    return P, Q, p, vel