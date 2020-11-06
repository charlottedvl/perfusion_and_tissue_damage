#%% LOAD MODULES
from numpy import *
from scipy import interpolate
from matplotlib.pyplot import *
from matplotlib import colors, ticker, cm
from matplotlib import rc

rc('text', usetex = True)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.dashed_pattern'] = [4, 3]
matplotlib.rcParams['lines.dotted_pattern'] = [1, 3]
matplotlib.rcParams['lines.dashdot_pattern'] = [6.5, 2.5, 1.0, 2.5]

close('all')

fn_start = 'opt_res_v10_DBC_'
opt_method = ['Nelder-Mead','BFGS']
sim_ID = [1,0,1]

opt_data = []

smbl = ['-k','--r','-.g']
lbl =  ['$\mathrm{Nelder}$-$\mathrm{Mead}$','$\mathrm{BFGS}$','$\mathrm{SLSQP}$']

fsx = 17
fsy = 12
fig1 = figure(num=1, figsize=(fsx/2.54, fsy/2.54))
gs1 = GridSpec(1, 1)
gs1.update(left=0.1, right=0.98, bottom=0.705, top=0.98, wspace=0.05, hspace=0.08)

gs1a = GridSpec(2, 1)
gs1a.update(left=0.1, right=0.98, bottom=0.1, top=0.65, wspace=0.05, hspace=0.08)

ax1 = subplot(gs1[0, 0])
yscale('log')
xlim(0,100)
ylim(0.01,10000)
setp(ax1, xticklabels=[])
ylabel(r'$\mathrm{Cost~function~}(J)$')

ax2 = subplot(gs1a[0, 0])
xlim(0,100)
ylim(-5,100)
setp(ax2, xticklabels=[])
xlabel(r'$\mathrm{Number~of~simulations}$')
#ylabel(r'$\mathrm{Perfusion~\left[\frac{ml}{min}\frac{1}{100ml}\right]}$')

ax3 = subplot(gs1a[1, 0])
xlim(0,100)
ylim(0,65)
xlabel(r'$\mathrm{Number~of~simulations}$')
#ylabel(r'$\mathrm{Perfusion~\left[\frac{ml}{min}\frac{1}{100ml}\right]}$')


#fsx = 17
#fsy = 9
#fig2 = figure(num=2, figsize=(fsx/2.54, fsy/2.54))
#gs2 = GridSpec(2, 1)
#gs2.update(left=0.1, right=0.98, bottom=0.125, top=0.98, wspace=0.05, hspace=0.08)
#
#ax4 = subplot(gs2[0, 0])
#yscale('log')
##ylim(1,2000)
#xlim(0,70)
#setp(ax4, xticklabels=[])
#ylabel(r'$k_a/k_c$')
#
#ax5 = subplot(gs2[1, 0])
#xlim(0,70)
##ylim(-10,65)
#xlabel(r'$\mathrm{Number~of~simulations}$')
#ylabel(r'$\beta^G/\beta^W$')
#
for i in range(1):
    fname = 'opt_res_Nelder-Mead.csv'
    opt_data = loadtxt(fname,skiprows=1,delimiter=',')
    ax1.plot(arange(len(opt_data))+1,opt_data[:,-1],'-k')
    ax2.plot(arange(len(opt_data))+1,opt_data[:,2],'--k')
    ax2.plot(arange(len(opt_data))+1,opt_data[:,3],'-k')
    ax3.plot(arange(len(opt_data))+1,opt_data[:,5],'-k')
    ax3.plot(arange(len(opt_data))+1,opt_data[:,4],'--k')
#    ax4.plot(arange(len(opt_data))+1,opt_data[:,0],smbl[i])
#    ax5.plot(arange(len(opt_data))+1,opt_data[:,1],smbl[i],label=lbl[i])
#
ax3.text(0.8,0.78,r'$\mathrm{Grey~matter}$',horizontalalignment='center',verticalalignment='center',transform=ax3.transAxes)
ax3.text(0.8,0.22,r'$\mathrm{White~matter}$',horizontalalignment='center',verticalalignment='center',transform=ax3.transAxes)

ax2.text(0.8,0.72,r'$\mathrm{max}$',horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
ax2.text(0.8,0.2,r'$\mathrm{min}$',horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)

ax2.text(-0.075,-0.05,r'$\mathrm{Perfusion~[ml~blood/min/(100~ml~tissue)]}$',horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes,rotation=90)

ax1.text(0.97,0.75,r'$\mathrm{(a)}$',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)
ax2.text(0.97,0.75,r'$\mathrm{(b)}$',horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
ax3.text(0.97,0.75,r'$\mathrm{(c)}$',horizontalalignment='center',verticalalignment='center',transform=ax3.transAxes)
#ax4.text(0.97,0.9,r'$\mathrm{(a)}$',horizontalalignment='center',verticalalignment='center',transform=ax4.transAxes)
#ax5.text(0.97,0.9,r'$\mathrm{(b)}$',horizontalalignment='center',verticalalignment='center',transform=ax5.transAxes)


ax1.legend(loc=[0.6,0.5],ncol=1,frameon=False,fontsize=11,columnspacing=0.7,handletextpad=0.5,labelspacing=0.2,handlelength=1.7)
#ax5.legend(loc=[0.5,0.5],ncol=1,frameon=False,fontsize=11,columnspacing=0.7,handletextpad=0.5,labelspacing=0.2,handlelength=1.7)

#fig1.savefig('opt1_v2.eps')
#fig1.savefig('opt1_v2.pdf')
fig1.savefig('opt1_v2.png',dpi=450)
#
##fig2.savefig('opt2_v2.eps')
##fig2.savefig('opt2_v2.pdf')
#fig2.savefig('opt2_v2.png',dpi=450)