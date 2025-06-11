import numpy as np
import yaml
import argparse

import analyt_fcts

from matplotlib.pyplot import *
from matplotlib import colors, ticker, cm
from matplotlib import rc
from matplotlib import cm


rc('text', usetex = True)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.dashed_pattern'] = [5, 5]
matplotlib.rcParams['lines.dotted_pattern'] = [1, 3]
matplotlib.rcParams['lines.dashdot_pattern'] = [6.5, 2.5, 1.0, 2.5]

close('all')

#%% obtain settings and resuls
parser = argparse.ArgumentParser(description="coupled 1D perfusion and network simulator based on analytical solutions")
parser.add_argument("--config_file", help="path to configuration file",
                    type=str, default='./config_coupled_analyt.yaml')
config_file = parser.parse_args().config_file

with open(config_file, "r") as myconfigfile:
        configs = yaml.load(myconfigfile, yaml.SafeLoader)

# specify 1D network and continuum problems
D, D_ave, G, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw = analyt_fcts.set_up_network(configs)
beta, Nc, l_subdom, x, beta_sub, subdom_id, BC_type_con, BC_val_con = analyt_fcts.set_up_continuum(configs)

# load results
con_data = np.loadtxt(configs['res_path'] + 'con_data.csv',delimiter=',')
P = np.loadtxt(configs['res_path'] + 'P_ntw.csv',delimiter=',')
Q = np.loadtxt(configs['res_path'] + 'Q_ntw.csv',delimiter=',')
x = con_data[:,0]
p = con_data[:,1]
vel = con_data[:,2]


#%%
# flow through the boundary surfaces of the continuum
Q0_cont = vel[0] * configs['continuum']['area']
Q1_cont = -vel[-1] * configs['continuum']['area']

# perfusion
F = beta_sub*(p-configs['continuum']['pv'])

# total flow based on perfusion
Q_total_con = np.trapz(F,x)*configs['continuum']['area']

# total flow based on network inlet and outlet
Q_ntw_in = Q[0,1]
Q_ntw_out = Q[1,2] + Q[1,3]

# average perfusion
F_total = Q_total_con/x.max()/configs['continuum']['area']*6000

# total volume
V_total = x.max()*configs['continuum']['area']

print(Q[1,2],Q0_cont)
print(P[2],p[0],'\n')

print(Q[1,3],Q1_cont)
print(P[3],p[-1],'\n')

print(6000*F.min(),6000*F.max())


#%% generate plots

L = np.triu(L, 1)
nodal_coord = np.zeros([Nn,2])
for i in range(Nn-1):
    if np.sum(L[i,:] != 0) == 1:
        nodal_coord[i+1,0] = nodal_coord[i,0]
        nodal_coord[i+1,1] = nodal_coord[i,1] + L[i,np.where(L[i,:] != 0)[0][0]]
    elif np.sum(L[i,:] != 0) > 1:
        idx = np.where(L[1,:] != 0)[0]
        idx = idx[idx>i]
        angle = np.pi/(len(idx)+1)
        for j in range(len(idx)):
            nodal_coord[idx[j],0] = nodal_coord[i,0] - L[i,idx[j]] * np.cos(angle+j*angle)
            nodal_coord[idx[j],1] = nodal_coord[i,1] + L[i,idx[j]] * np.sin(angle+j*angle)

all_press = np.concatenate((p,P))
pmin = 0# all_press.min()
pmax = all_press.max()

# pressure plot in network
fsx = 20
fsy = 14

fig0 = figure(num=0, figsize=(fsx/2.54, fsy/2.54))
gs0 = GridSpec(1, 1)
gs0.update(left=-0.65, right=0.95, bottom=0.2, top=1, wspace=0.1, hspace=0.225)
ax0=subplot(gs0[0,0])

x_ntw_total = np.max(nodal_coord[:,0]) - np.min(nodal_coord[:,0])
scl4lim = 0.07
ax0.set(xlim=(np.min(nodal_coord[:,0])-scl4lim*x_ntw_total, np.max(nodal_coord[:,0])+scl4lim*x_ntw_total),
    ylim=(np.min(nodal_coord[:,1])-scl4lim*x_ntw_total, np.max(nodal_coord[:,1])+scl4lim*x_ntw_total))
ax0.set_aspect('equal', adjustable='box')

for i in range(Nn):
    ax0.plot(nodal_coord[i,0],nodal_coord[i,1],'ko')
    ax0.text(nodal_coord[i,0]+0.05*x_ntw_total,nodal_coord[i,1],str(i)+'; P='+str(int(P[i]))+' Pa')

diam_scale = 15/D_ave.max()
npoints = 201
for i in range(Nn-1):
    if np.sum(L[i,:] != 0) == 1:
        nidx = np.where(L[i,:] != 0)[0][0]
        xl = np.linspace(nodal_coord[i,0], nodal_coord[nidx,0],npoints)
        yl = np.linspace(nodal_coord[i,1], nodal_coord[nidx,1],npoints)
        cl = (np.linspace(P[i],P[nidx],npoints)-pmin)/(pmax-pmin)
        # ax0.plot(xl, yl, lw=3, c=cm.hot(cl))
        ax0.scatter(xl,yl,c=cm.viridis(cl), edgecolor='none',marker='o',s=D_ave[i,nidx]*diam_scale)
    elif np.sum(L[i,:] != 0) > 1:
        idx = np.where(L[1,:] != 0)[0]
        idx = idx[idx>i]
        for j in range(len(idx)):
            nidx = idx[j]
            xl = np.linspace(nodal_coord[i,0], nodal_coord[nidx,0],npoints)
            yl = np.linspace(nodal_coord[i,1], nodal_coord[nidx,1],npoints)
            cl = (np.linspace(P[i],P[nidx],npoints)-pmin)/(pmax-pmin)
            if D_ave[i,nidx] == 0:
                mysize = 4
                ax0.text(np.mean(xl)+0.15*(xl.max()-xl.min()),np.mean(yl)-0.15*(xl.max()-xl.min()),'blocked \n vessel')
            else:
                mysize = D_ave[i,nidx]*diam_scale
            ax0.scatter(xl,yl,c=cm.viridis(cl), edgecolor='none',marker='o',s=mysize)
axis('off')

cbaxes = fig0.add_axes([0.01, 0.1, 0.35, 0.02])
norm = colors.Normalize(vmin=pmin,vmax=pmax)
sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])
cbar = colorbar(sm, ticks=np.linspace(pmin,pmax,6), orientation='horizontal',cax = cbaxes)
cbaxes.text(0.5,-3.5,'P [Pa]', transform=cbaxes.transAxes)


# pressure and perfusion plot in porous medium
ax1=fig0.add_axes([0.45,0.1,0.48,0.8])
ax2 = ax1.twinx()
ax1.plot(x, p, '-g',label='pressure')
ax1.plot([-1,-2], [-1,-2], '--r',label='perfusion')
ax2.plot(x, 6000*F, '--r',label='perfusion')

ax1.set_xlabel('x [m]')
ax1.set_ylabel('pressure [Pa]', color='g')
ax2.set_ylabel('perfusion [ml/min/100 g]', color='r')

ax1.set(xlim=[x.min(),x.max()])
ax1.set(ylim=[0,1e4])
ax2.set(ylim=[0,80])
ax1.legend()

ax1.text(0.2,1.01,'Porous medium between nodes 2 and 3', transform=ax1.transAxes)

fig0.savefig(configs['res_path'] + 'res.png',dpi=450)
