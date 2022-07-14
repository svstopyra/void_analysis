# Correlation length:
import numpy as np
import h5py as h5
import seaborn as sns
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
import matplotlib.pylab as plt
import pickle
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from matplotlib.ticker import NullFormatter
import emcee

#[mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
#    std_field,hmc_Elh,hmc_Eprior,hades_accept_count,hades_attempt_count] = \
#    pickle.load(open("chain_properties.p","rb"))


[mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
    std_field,hmc_Elh,hmc_Eprior,hades_accept_count,hades_attempt_count] = \
    pickle.load(open("chain_properties_combined.p","rb"))

[mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
    std_field,hmc_Elh,hmc_Eprior,hades_accept_count,hades_attempt_count] = \
    pickle.load(open("borg_groups/chain_properties_combined.p","rb"))

#subfolder = "supporting_plots/"
subfolder = "borg_groups/"

mcmcArray = np.arange(6214,8286)
NCAT = 16
no_bias_params = 4
num = len(mcmcArray)

bias_matrix = np.array(np.full((num,NCAT,no_bias_params+1),0),dtype=np.float64)
#num - files
#NCAT - catalogs
#no_bias_params = number of bias parameters
#df = pd.DataFrame()
"""
# If you have an array of a sampled parameter (how to get this array, see next section),
# then you can add it to the correlation matrix like below:
df['Name_of_cosmo_param'] = sampled_parameter_array
"""
for i in tqdm(mcmcArray):
    mcmc_file = h5.File("mcmc_%d.h5" % i,'r')
    for j in np.arange(NCAT):
        for k in np.arange(no_bias_params+1):
            if k == 0:
                bias_value = mcmc_file['scalars/galaxy_nmean_%d' % j][0]
            else:
                bias_value = mcmc_file['scalars/galaxy_bias_%d' % j][k-1]
            bias_matrix[i,j,k] = bias_value
    mcmc_file.close()

def correlation_length(array_of_sampling_parameter,\
        savename = None,show=True,label=None,showZero=True,\
        linestyle=None,color=None,guideColour='grey',ax=None):
    # COMPUTES THE CORRELATION LENGTH
    autocorr = np.fft.irfft( (
        np.abs(np.fft.rfft(
            array_of_sampling_parameter - np.mean(array_of_sampling_parameter))) )**2 )
    zero_line = np.zeros((autocorr/autocorr[0]).shape)
    # PLOT THE CORRELATION LENGTH
    #fig = plt.figure(figsize = (15,10))
    if ax is None:
        fig, ax = plt.subplots()
    hand = ax.plot(autocorr/autocorr[0],color = color,label=label,\
        linestyle=linestyle)
    if showZero:
        ax.plot(zero_line, 'r--',color = guideColour)
    Fmax=len(array_of_sampling_parameter)
    mcDelta=1
    ax.set_xlim(0,Fmax/(2*mcDelta))
    ax.set_ylabel(r'$\mathrm{Correlation}$')
    ax.set_xlabel(r'$\mathrm{n \ (Step \ of \ mcmc \ chain)}$')
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    return hand

# Runs the function on one of the bias-parameters
# -> adjust this call as in the trace-histogram field!
correlation_length(bias_matrix[:,1,2])

# Showing all of the bias function correlations:
parameterLabels = ['$\\beta$','$\\epsilon$','$\\rho$']
styles = ['-','--',':']
handles = []
for k in range(0,16):
    for l in range(0,3):
        if l == 0:
            handles.append(\
                correlation_length(bias_matrix[:,k,l+2],show=False,\
                label = "Catalogue " + str(k+1),linestyle=styles[l])[0])
        else:
            correlation_length(bias_matrix[:,k,l+2],show=False,\
                linestyle=styles[l])

for l in range(0,3):
    handles.append(mlines.Line2D([],[],linestyle=styles[l],color='k',
        label=parameterLabels[l]))

plt.legend(handles=handles)
plt.savefig(subfolder + "bias_correlation_bins.pdf")
plt.show()


# Estimates of the correlation length:
nStartList = np.array([0,1000,2000,3000,4000,5000,6000])
taufArr = np.array([[[emcee.autocorr.integrated_time(\
    bias_matrix[nStart:,k,l+2],quiet=True) \
    for k in range(0,16)] for l in range(0,3)]\
    for nStart in nStartList]).reshape((len(nStartList),3,16))

taufArr7000 = np.array([[emcee.autocorr.integrated_time(\
    bias_matrix[(7000-1260):,k,l+2],quiet=True) \
    for k in range(0,16)] for l in range(0,3)])

autoCorrEst = [[emcee.autocorr.function_1d(bias_matrix[:,k,l+2]) \
    for k in range(0,16)] for l in range(0,3)]

# Panel figure:
MabsList = np.linspace(-21,-25,9)
MabsLabels = ["$" + str(MabsList[m]) + " \\leq M < " + \
        str(MabsList[m+1]) + "$" for m in range(0,8)]
handles = []
nRange = np.arange(0,bias_matrix.shape[0])
#nRange = np.arange(7000-1260,bias_matrix.shape[0])
useEmcee = True
fig, ax = plt.subplots(3,2,figsize = (10,15))
for l in range(0,3):
    for m in range(0,8):
        if useEmcee:
            est1 = emcee.autocorr.function_1d(bias_matrix[nRange,2*m,l+2])
            est2 = emcee.autocorr.function_1d(bias_matrix[nRange,2*m+1,l+2])
            hand1 = ax[l,0].plot(est1,linestyle='-',color=seabornColormap[m],\
                label = MabsLabels[m])[0]
            hand2 = ax[l,1].plot(est2,linestyle='-',color=seabornColormap[m],\
                label = MabsLabels[m])[0]
            ax[l,0].plot([0,len(nRange)],[0,0],linestyle='--',color='grey')
            ax[l,1].plot([0,len(nRange)],[0,0],linestyle='--',color='grey')
            ax[l,0].set_ylabel(r'$\mathrm{Correlation}$')
            ax[l,0].set_xlabel(r'$\mathrm{n \ (Step \ of \ mcmc \ chain)}$')
            ax[l,1].set_ylabel(r'$\mathrm{Correlation}$')
            ax[l,1].set_xlabel(r'$\mathrm{n \ (Step \ of \ mcmc \ chain)}$')
        else:
            hand1 = correlation_length(bias_matrix[nRange,2*m,l+2],show=False,\
                    label = MabsLabels[m],linestyle='-',\
                    ax = ax[l,0],color=seabornColormap[m])[0]
            hand2 = correlation_length(bias_matrix[nRange,2*m+1,l+2],show=False,\
                    label = MabsLabels[m],linestyle='-',\
                    ax = ax[l,1],color=seabornColormap[m])[0]
        ax[l,0].set_title(parameterLabels[l] + \
            ", bright Catalogue ($m \\leq 11.5$)")
        ax[l,1].set_title(parameterLabels[l] + \
            ", dim Catalogue ($11.5 < m \\leq 12.5$)")
        if l == 0:
            handles.append(hand1)
        ax[l,1].yaxis.label.set_visible(False)
        ax[l,1].yaxis.set_major_formatter(NullFormatter())
        ax[l,1].yaxis.set_minor_formatter(NullFormatter())
        if l != 2:
            ax[l,0].xaxis.label.set_visible(False)
            ax[l,0].xaxis.set_major_formatter(NullFormatter())
            ax[l,0].xaxis.set_minor_formatter(NullFormatter())
            ax[l,1].xaxis.label.set_visible(False)
            ax[l,1].xaxis.set_major_formatter(NullFormatter())
            ax[l,1].xaxis.set_minor_formatter(NullFormatter())

plt.legend(handles=handles)
#plt.tight_layout()
plt.subplots_adjust(top=0.974,bottom=0.067,left=0.082,right=0.98,\
    hspace=0.101,wspace=0.0)
plt.savefig(subfolder + "bias_correlation_bins.pdf")
plt.show()


# Likelihood plot:
nzFilter = np.where(hmc_Elh != 0)[0]
convergenceFilter = np.where((mcmcArray > 7000) & (mcmcArray < 7500))[0]
meanL = np.mean(hmc_Elh[convergenceFilter])
#plt.plot(mcmcArray[nzFilter],-np.log10(hmc_Elh[nzFilter]/hmc_Elh[0])/1000)
plt.plot(mcmcArray[nzFilter],np.log(hmc_Elh[nzFilter]/hmc_Elh[0]))
#plt.axvline(6214,linestyle='--',color='grey',label='20 Steps start')
#plt.axhline(np.log(meanL/hmc_Elh[0]),linestyle=':',color='k',\
#    label='Mean, 7000-7500')
plt.xlabel('MCMC Sample')
plt.ylabel('$\\log(\\mathcal{L}_S/\\mathcal{L}_0)$')
plt.legend()
plt.tight_layout()
plt.savefig(subfolder + "likelihood_plot.pdf")
plt.show()

# Bias parameter plots:
handles=[]
fig, ax = plt.subplots(3,2,figsize = (10,15))
#mcmcRange = np.where((mcmcArray >= 7000))[0]
mcmcRange = np.where((mcmcArray >= 0))[0]
for l in range(0,3):
    for m in range(0,8):
        hand1 = ax[l,0].plot(mcmcArray[mcmcRange],\
            bias_matrix[mcmcRange,2*m,l+2],\
            linestyle='-',color=seabornColormap[m],\
            label = MabsLabels[m])[0]
        hand2 = ax[l,1].plot(mcmcArray[mcmcRange],\
            bias_matrix[mcmcRange,2*m+1,l+2],\
            linestyle='-',color=seabornColormap[m],\
            label = MabsLabels[m])[0]
        ax[l,0].plot(mcmcArray[mcmcRange],np.zeros(mcmcRange.shape),\
            linestyle='--',color='grey')
        ax[l,1].plot(mcmcArray[mcmcRange],np.zeros(mcmcRange.shape),\
            linestyle='--',color='grey')
        ax[l,0].set_ylabel(parameterLabels[l])
        ax[l,0].set_xlabel(r'$\mathrm{n \ (Step \ of \ mcmc \ chain)}$')
        ax[l,1].set_ylabel(parameterLabels[l])
        ax[l,1].set_xlabel(r'$\mathrm{n \ (Step \ of \ mcmc \ chain)}$')
        ax[l,0].set_title(parameterLabels[l] + \
            ", bright Catalogue ($m \\leq 11.5$)")
        ax[l,1].set_title(parameterLabels[l] + \
            ", dim Catalogue ($11.5 < m \\leq 12.5$)")
        if l == 0:
            handles.append(hand1)
        ax[l,1].yaxis.label.set_visible(False)
        ax[l,1].yaxis.set_major_formatter(NullFormatter())
        ax[l,1].yaxis.set_minor_formatter(NullFormatter())
        if l != 2:
            ax[l,0].xaxis.label.set_visible(False)
            ax[l,0].xaxis.set_major_formatter(NullFormatter())
            ax[l,0].xaxis.set_minor_formatter(NullFormatter())
            ax[l,1].xaxis.label.set_visible(False)
            ax[l,1].xaxis.set_major_formatter(NullFormatter())
            ax[l,1].xaxis.set_minor_formatter(NullFormatter())

plt.legend(handles=handles)
#plt.tight_layout()
plt.subplots_adjust(top=0.974,bottom=0.067,left=0.082,right=0.98,\
    hspace=0.101,wspace=0.0)
plt.savefig(subfolder + "bias_parameters_bins.pdf")
plt.show()


# Plots of the bias functional form:
def biasFunctionalForm(delta,b,rho,eps,N=1,S=1,A=1,numericalOffset = 1e-6):
    prefactor = S*N*A
    logresult = np.log(prefactor) + \
        b*np.log(1.0 + delta + numericalOffset) - \
        ((1.0 + delta + numericalOffset)/rho)**(-eps)
    return np.exp(logresult)

# Bias form plots:
handles=[]
fig, ax = plt.subplots(3,2,figsize = (10,15))
#mcmcRange = np.where((mcmcArray >= 6214))[0]
mcmcRange = np.where((mcmcArray >= 0))[0]
deltaTest = np.array([-0.2,0,600])
for l in range(0,len(deltaTest)):
    for m in range(0,8):
        hand1 = ax[l,0].plot(mcmcArray[mcmcRange],\
            biasFunctionalForm(deltaTest[l],bias_matrix[mcmcRange,2*m,2],\
            bias_matrix[mcmcRange,2*m,4],bias_matrix[mcmcRange,2*m,3]),\
            linestyle='-',color=seabornColormap[m],\
            label = MabsLabels[m])[0]
        hand2 = ax[l,1].plot(mcmcArray[mcmcRange],\
            biasFunctionalForm(deltaTest[l],bias_matrix[mcmcRange,2*m+1,2],\
            bias_matrix[mcmcRange,2*m+1,4],bias_matrix[mcmcRange,2*m+1,3]),\
            linestyle='-',color=seabornColormap[m],\
            label = MabsLabels[m])[0]
        ax[l,0].set_ylabel('Bias function')
        ax[l,0].set_xlabel(r'$\mathrm{n \ (Step \ of \ mcmc \ chain)}$')
        ax[l,1].set_ylabel('Bias function')
        ax[l,1].set_xlabel(r'$\mathrm{n \ (Step \ of \ mcmc \ chain)}$')
        ax[l,0].set_title("$\\delta = " + str(deltaTest[l]) + "$" + \
            ", bright Catalogue ($m \\leq 11.5$)")
        ax[l,1].set_title("$\\delta = " + str(deltaTest[l]) + "$" + \
            ", dim Catalogue ($11.5 < m \\leq 12.5$)")
        ax[l,0].set_yscale('log')
        ax[l,1].set_yscale('log')
        if l == 0:
            handles.append(hand1)
        ax[l,1].yaxis.label.set_visible(False)
        ax[l,1].yaxis.set_major_formatter(NullFormatter())
        ax[l,1].yaxis.set_minor_formatter(NullFormatter())
        if l != 2:
            ax[l,0].xaxis.label.set_visible(False)
            ax[l,0].xaxis.set_major_formatter(NullFormatter())
            ax[l,0].xaxis.set_minor_formatter(NullFormatter())
            ax[l,1].xaxis.label.set_visible(False)
            ax[l,1].xaxis.set_major_formatter(NullFormatter())
            ax[l,1].xaxis.set_minor_formatter(NullFormatter())

plt.legend(handles=handles)
#plt.tight_layout()
plt.subplots_adjust(top=0.974,bottom=0.067,left=0.082,right=0.98,\
    hspace=0.101,wspace=0.0)
plt.savefig(subfolder + "bias_functional_form_bins.pdf")
plt.show()


np.save("bias_matrix.npy",bias_matrix)


