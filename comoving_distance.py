import numpy as np
import scipy
import astropy
from astropy import cosmology
from scipy import integrate
import matplotlib.pylab as plt
#-------------------------------------------------------------------------------
# VERIFY THAT UNITS ARE WHAT WE EXPECT
# Cosmology:
H0 = 70.5
Om0 = 0.307
Ol0 = 0.693
h = H0/100
cosmo = cosmology.LambdaCDM(H0,Om0,Ol0)

# Test range of redshifts
zrange = np.linspace(0,0.5,101)

c = 299792458 # Speed of light, m/s
ckm = c/1000 # km/s

# Hubble function, H(z)/H0:
def E(z,Om0,Ol0,Or0 = 0.0):
    Ok0 = 1.0 - Om0 - Ol0 - Or0
    return np.sqrt(Or0*(1+z)**4 + Om0*(1 + z)**3 + Ok0*(1 + z)**2 + Ol0)

# Comoving distance integrand, in units of Mpc (NOT Mpc/h):
def integrand(z,h,Om0,Ol0,Or0=0.0):
    return (ckm/(100*h))/E(z,Om0,Ol0,Or0=Or0)

dc = np.zeros(zrange.shape)
for k in range(0,len(zrange)):
    dc[k] = integrate.quad(lambda z: integrand(z,h,Om0,Ol0),\
        0,zrange[k])[0]

# Using astropy, to verify that we get the same result:
dcAstropy = cosmo.comoving_distance(zrange).value
diff = dcAstropy - dc
print(np.mean(diff))

#-------------------------------------------------------------------------------
# IMPORT 2M++ DATA

tmpp = np.loadtxt("2mpp_data/2MPP.txt")
z = tmpp[:,3] # Redshift
d = cosmo.comoving_distance(tmpp[:,3]).value*h # Comoving distance (Mpc/h)
dLh = (1.0 + z)*d # Luminosity distance in Mpc/h
dL = dLh/h # Luminosity distance in Mpc

# Get absolute and apparent magnitudes:
M2mpp = tmpp[:,5] # Absolute magnitude
m2mpp = tmpp[:,4] # Apparent magnitude

# Compute what the absolute magnitude should be (not accounting for 
# K-corrections, etc):
filt = np.where(d > 0)[0] # Remove spurious negative distances
Mtheory = m2mpp[filt] - 5*np.log10(dL[filt]) - 25 + 2.9*z[filt]


# Fit the theory vs recorded absolute Magnitudes:
fit = np.polyfit(M2mpp[filt],Mtheory,deg=1)



def plotMagnitudeComparison(M2mpp,Mtheory,fit,\
        title = 'Calculated vs 2M++ file Absolute magnitudes',\
        xlabel = '$M_{\\mathrm{2m++}}$, 2M++ Claimed Absolute Magnitude',\
        ylabel = '$M_{\\mathrm{calc}}$, Calculated Absolute Magnitude',\
        titleFontSize = 10):
    # Plot these against each other:
    plt.scatter(M2mpp,Mtheory,marker='x',label="2M++ data")
    # Line showing linear fit:
    lineLabel = '$M_{\\mathrm{calc}} = ' + "%.3g" % fit[0] +\
         'M_{\\mathrm{2m++}} '
    if fit[1] < 0:
        lineLabel += "%.3g" % fit[1]
    else:
        lineLabel += '+' + "%.3g" % fit[1]
    xPoints = np.array([np.min(M2mpp),np.max(M2mpp)])
    plt.plot(xPoints,fit[0]*xPoints + fit[1],'g--',\
        label='Linear fit, ' + lineLabel + '$')
    # Line showing case of equal magnitudes:
    plt.plot(xPoints,xPoints,'r:',\
        label="$M_{\\mathrm{calc}} = M_{\\mathrm{2m++}}$")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title,fontsize=titleFontSize)
    plt.legend()
    plt.show()

# Magnitude comparison when the 'correct' method is used (Luminosity distance
# in Mpc, not Mpc/h)
plotMagnitudeComparison(M2mpp[filt],Mtheory,fit,\
    title = 'Calculated vs 2M++ file Absolute magnitudes' + \
    ' (Luminosity distance in Mpc)')

# Now show the same, using Luminosity distances in Mpc/h as the input (which
# isn't correct to do):
Mtheoryh = m2mpp[filt] - 5*np.log10(dLh[filt]) - 25 + 2.9*z[filt]
fith = np.polyfit(M2mpp[filt],Mtheoryh,deg=1)

plotMagnitudeComparison(M2mpp[filt],Mtheoryh,fith,\
    title = 'Calculated vs 2M++ file Absolute magnitudes' + \
    ' (Luminosity distance in Mpc/h)')







