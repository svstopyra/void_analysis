#
from . import plot_utilities, context, snapedit, stacking, tools, cosmology
import pynbody
import numpy as np
import imageio
import os
import gc
import matplotlib.pylab as plt
from matplotlib import cm
from . import cosmology
from scipy import integrate
#import pandas
#import seaborn as sns
from . plot_utilities import *
from . cosmology import TMF_from_hmf
import alphashape
import healpy
from matplotlib import patches
import matplotlib.lines as mlines
import matplotlib.colors as colors
import scipy
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import stats
from descartes import PolygonPatch
import seaborn as sns
from matplotlib.ticker import NullFormatter
import matplotlib
# Use seaborn colours:
seabornColormap = sns.color_palette("colorblind",as_cmap=True)


# Halves the RGB values of a specified color
def half_color(color):
    if(len(color) != 3):
        raise Exception("Colour must be a three element tuple.")
    return (color[0]/2,color[1]/2,color[2]/2)

# Construct a set of ncolors even spaced colours in rgb space.
def construct_color_list(n,ncolors):
    ncbrt = np.ceil(np.cbrt(ncolors))
    if(n > ncolors):
        raise Exception("Requested colour exceeds maximum" + \
        " specified number of colours. Specify more colours.")
    k = np.floor(n/(ncbrt**2))
    j = np.floor((n - k*ncbrt**2)/ncbrt)
    i = (n - k*ncbrt**2 - j*ncbrt)
    return (i/(ncbrt-1),j/(ncbrt-1),k/(ncbrt-1))

# Returns the specified number in scientific notation:
def scientificNotation(x,latex=False,s=3,powerRange = 0):
    log10x = np.log10(np.abs(x))
    z = np.floor(log10x).astype(int)
    y = 10.0**(log10x - z)
    if np.abs(z) > powerRange:
        resultString = "10^{" + "{0:0d}".format(z) + "}"
        if y != 1.0:
            resultString = ("{0:." + str(s) + "g}").format(y) + \
            "\\times " + resultString
        if latex:
            resultString = "$" + resultString + "$"
    else:
        resultString = ("{0:." + str(s) + "g}").format(y*(10.0**z))
    if x < 0:
        resultString = "-" + resultString
    return resultString

# Plot binned halo densities as a function of redshift
def plot_halo_void_densities(z,rhoVnav,rhoVnsd,rhoVrav,rhoVrsd,bins):
    binNo = len(bins) - 1
    legendList = []
    # Want to format bins in scientific notation:

    for k in range(0,binNo):
        plt.semilogy(z,rhoVnav[:,k],color=construct_color_list(k+1,2*binNo))
        plt.fill_between(z,rhoVnav[:,k] - rhoVnsd[:,k],
            rhoVnav[:,k] + rhoVnsd[:,k],
            color=half_color(construct_color_list(k+1,2*binNo)))
        legendList.append('Halo Density, $' + scientificNotation(bins[k]) + \
        '$ - $' + scientificNotation(bins[k+1]) + ' M_{sol}/h$')
    for k in range(0,binNo):
        plt.semilogy(z,rhoVrav[:,k],color=construct_color_list(binNo + k+1,2*binNo))
        plt.fill_between(z,rhoVrav[:,k] - rhoVrsd[:,k],
            rhoVrav[:,k] + rhoVrsd[:,k],
            color=half_color(construct_color_list(binNo + k+1,2*binNo)))
        legendList.append('Anti-halo Density, $' + scientificNotation(bins[k]) + \
            '$ - $' + scientificNotation(bins[k+1]) + ' M_{sol}/h$')
    plt.xlabel('z')
    plt.ylabel('(Local Density)/(Background Density)')
    plt.legend(legendList)
    plt.show()

def computeHistogram(x,bins,z=1.0,count = False,density = False,useGaussianError=False,
        alpha = 0.68):
    noInBins = np.zeros(len(bins)-1,dtype=int)
    N = len(x)
    prob = np.zeros(len(bins)-1)
    sigma = np.zeros(len(bins)-1)
    inBins = []
    if N != 0:
        for k in range(0,len(bins)-1):
            inBins.append(np.where((x > bins[k]) & (x < bins[k+1]))[0])
            noInBins[k] = len(inBins[k])
            # Estimate of the probability density for this bin:
            p = len(inBins[k])/N
            if count:
                prob[k] = len(inBins[k])
            else:
                prob[k] = p
            if density:
                prob[k] /= (bins[k+1] - bins[k])
            # Normal distribution approximation of the 
            # confidence interval on this density:
            if useGaussianError:
                if count:
                    sigma[k] = z*np.sqrt(p*(1.0-p)*N)
                else:
                    sigma[k] = z*np.sqrt(p*(1.0-p)/N)
                if density:
                    sigma[k] /= (bins[k+1] - bins[k])
        if not useGaussianError:
            bounds = stats.binom_conf_interval(noInBins,N*np.ones(noInBins.shape))
            sigma = np.vstack((noInBins/N - bounds[1,:],bounds[0,:] - noInBins/N))
    return [prob,sigma,noInBins,inBins]

# Create bins for a list of values:
def createBins(values,nBins,log=False):
    if log:
        # Logarithmically spaced bins:
        return 10**np.linspace(np.log10(np.min(values)),\
            np.log10(np.max(values)),nBins+1)
    else:
        # linearly spaced bins:
        return np.linspace(np.min(values),np.max(values),nBins+1)

# Plot a histogram, but include error bars for the confidence interval of the uncertainty
def histWithErrors(p,sigma,bins,ax = None,label="Bin probabilities",\
        color=None,alpha = 0.5):
    x = (bins[1:len(bins)] + bins[0:(len(bins)-1)])/2
    width = bins[1:len(bins)] - bins[0:(len(bins)-1)]
    if ax is None:
        return plt.bar(x,p,width=width,yerr=sigma,alpha=alpha,\
            label=label,color=color)
    else:
        return ax.bar(x,p,width=width,yerr=sigma,alpha=alpha,\
            label=label,color=color,error_kw=error_kw)

# Histogram of halo densities
def haloHistogram(logrho,logrhoBins,masses,massBins,massBinList = None,
        massBinsToPlot = None,density=True,logMassBase = None,subplots=True,
        subplot_shape=None):
    # Plot all the mass bins unless otherwise specified:
    if massBinsToPlot is None:
        massBinsToPlot = range(0,len(massBins)-1)
    # Bin the masses of the halos if this has not already been supplied.
    if massBinList is None:
        [massBinList,noInBins] = plot_utilities.binValues(masses,massBins)
    legendList = []
    if subplots:
        if subplot_shape is not None:
            # Check that the requested shape makes sense:
            if len(subplot_shape) != 2:
                raise Exception("Sub-plots must be arranged on a 2d grid.")
            if subplot_shape[0]*subplot_shape[1] < len(massBinsToPlot):
                raise Exception("Not enough room in requested" + \
                    " sub-plot arrangement to fit all plots.")
            a = subplot_shape[0]
            b = subplot_shape[1]
        else:
            nearestSquareRoot = np.ceil(np.sqrt(len(massBinsToPlot))).astype(int)
            a = b = nearestSquareRoot
        fig, ax = plt.subplots(nrows=a,ncols=b)
        counter = 0
    for k in massBinsToPlot:
        [p,sigma,noInBins,inBins] = computeHistogram(logrho[massBinList[k]],logrhoBins)
        if subplots:
            # Plot axes on a square grid:
            i = np.floor(counter/b).astype(int)
            j = np.mod(counter,b).astype(int)
            histWithErrors(p,sigma,logrhoBins,ax[i,j])
            if logMassBase is None:
                ax[i,j].legend(['$' + scientificNotation(massBins[k]) + \
                    ' < M < ' + scientificNotation(massBins[k+1]) + \
                    ' M_{sol}/h$'])
            else:
                ax[i,j].legend(['$' + \
                    scientificNotation(logMassBase**massBins[k]) + \
                    '$ < M < $' + \
                    scientificNotation(logMassBase**massBins[k+1]) + \
                    ' M_{sol}/h$'])
            if i == a - 1:
                ax[i,j].set_xlabel('$log(\\langle\\rho\\rangle_V/\\bar{\\rho})$')
            if j == 0:
                ax[i,j].set_ylabel('Probability Density')
            counter = counter + 1
        else:
            # Plot everything on one axis:
            histWithErrors(p,sigma,logrhoBins)
        if logMassBase is None:
            legendList.append('$' + scientificNotation(massBins[k]) + \
                ' < M < ' + scientificNotation(massBins[k+1]) + ' M_{sol}/h$')
        else:
            # Exponentiate the masses if they were supplied in log space:
            legendList.append('$' + scientificNotation(logMassBase**massBins[k]) + \
            ' < M < ' + scientificNotation(logMassBase**massBins[k+1]) + \
            ' M_{sol}/h$')
    if not subplots:
        plt.xlabel('$log(\\langle\\rho\\rangle_V/\\bar{\\rho})$')
        plt.ylabel('Probability Density')
        plt.legend(legendList)
    plt.show()

# Plot Fraction of halos in a set of mass bins that are underdense:
def plotUnderdenseFraction(frac,sigma,logMassBins):
    # frac and sigma should be computed by halo_analysis.getExpansionFraction
    # Assuming given in log10 mass bins:
    massCentres = 10**((logMassBins[1:len(logMassBins)] + \
        logMassBins[0:(len(logMassBins)-1)])/2)
    fig, ax = plt.subplots()
    ax.errorbar(massCentres,frac,yerr=sigma)
    ax.set_xscale('log')
    plt.xlabel('Mass bin$/M_{sol}/h$')
    plt.ylabel('Underdense fraction at z = 0')
    plt.show()

# Plot average density in each of the supplies mass bins:
def plotMassBinDensity(rhoV,binList,logMassBins):
    massCentres = 10**((logMassBins[1:len(logMassBins)] + \
        logMassBins[0:(len(logMassBins)-1)])/2)
    fig, ax = plt.subplots()
    rhoVav = np.zeros(len(massCentres))
    rhoVsd = np.zeros(len(massCentres))
    for k in range(0,len(rhoVav)):
        if len(binList[k] != 0):
            rhoVav[k] = np.mean(rhoV[binList[k]])
            rhoVsd[k] = np.sqrt(np.var(rhoV[binList[k]])/len(binList[k]))
    ax.errorbar(massCentres,rhoVav,yerr=rhoVsd)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('Mass bin$/M_{sol}/h$')
    plt.ylabel('$\\langle\\rho\\rangle/\\bar{\\rho}$')
    plt.show()

# Linear regression, together with various statistics
def linearRegression(x,y,full=False,errors=False):
    z = np.polyfit(x,y,deg=1)
    a = z[0]
    b = z[1]
    xbar = np.mean(x)
    ybar = np.mean(y)
    n = len(x)
    ssxx = np.sum((x - xbar)**2)
    ssyy = np.sum((np.log10(y) - ybar)**2)
    ssxy = np.sum((np.log10(y) - ybar)*(x - xbar))
    ssxy = np.sum((np.log10(y) - ybar)*(x - xbar))
    r2 = (ssxy**2)/(ssxx*ssyy)
    s = np.sqrt((ssyy - (ssxy**2)/ssxx)/(n-2))
    sea = s*np.sqrt((1/n) + (xbar**2)/ssxx)
    seb = s*np.sqrt(ssxx)
    if (full and errors):
        return [a,b,sea,seb,ssxx,ssyy,ssyy,r2,s]
    elif full:
        return [a,b,ssxx,ssyy,ssyy,r2,s]
    elif errors:
        return [a,b,sea,seb]
    else:
        return [a,b]



# Generate colours on the fly:
def linearColour(n,nmax,colourMap=cm.jet):
    return colourMap(np.int32(np.round((n/nmax)*256)))[0:3]


# Plot a halo relative to the centre of mass:
def plotPointsRelative(pos,boxsize,centre = None,
        weights = None,color_spec=(1,1,1),type='2dvertex',scale=1.0):
    if centre is None:
        if weights is None:
            weights = np.ones(len(pos))
        centre = context.computePeriodicCentreWeighted(pos,weights,boxsize)
    posAdjusted = snapedit.unwrap(snapedit.wrap(snapedit.unwrap(pos,boxsize) -\
        snapedit.unwrap(centre,boxsize),boxsize),boxsize)
    point_scatter(posAdjusted,color_spec=color_spec,type=type,scale=scale)


def plotHaloRelative(halo,centre = None,weights = None,
        color_spec=(1,1,1),type='2dvertex',scale=1.0):
    boxsize = halo.properties['boxsize'].ratio("Mpc a h**-1")
    plotPointsRelative(halo['pos'],boxsize,centre=centre,weights = weights,color_spec=color_spec,type=type,scale=scale)

# Violin plots
#def plotViolins(rho,radialBins,radiiFilter=None,ylim=1.4,ax = None,fontsize=14,fontname="serif",color=None,inner=None,linewidth=None,saturation=1.0,palette="colorblind"):
#	radii = binCentres(radialBins)
#	if radiiFilter is None:
#		radiiFilter = np.arange(0,len(radii))
#	if ax is None:
#		fig, ax = plt.subplots()
#	panData = pandas.DataFrame(data=rho[:,radiiFilter],columns=floatsToStrings(radii[radiiFilter]))
	#sns.violinplot(data=panData,ax=ax,color=color,inner=inner,linewidth=linewidth,saturation=saturation,palette=palette)
#	ax.set_xlabel('$R/R_{\\mathrm{eff}}$',fontsize=fontsize,fontfamily=fontname)
#	ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=fontsize,fontfamily=fontname)
#	ax.set_ylim([0,ylim])
#	xlim = ax.get_xlim()
#	ax.tick_params(axis='both',labelsize=fontsize)
#	ax.axhline(y = 1.0,color='0.75',linestyle=':')

class LinearMapper:
    def __init__(self,inMin,inMax,outMin=0,outMax=1):
        self.inMin = inMin
        self.inMax = inMax
        self.outMin = outMin
        self.outMax = outMax
    def __call__(self,x,clip=False):
        return self.outMin + \
        (self.outMax - self.outMin)*(x - self.inMin)/(self.inMax - self.inMin)
    def autoscale(self,A):
        self.inMin = np.min(A)
        self.inMax = np.max(A)
    def inverse(self,x):
        return self.inMin + \
        (self.inMax - self.inMin)*(x - self.outMin)/(self.outMax - self.outMin)


class LogMapper:
    def __init__(self,inMin,inMax,outMin=0,outMax=1,logMin = 1.0):
        self.inMin = np.log(logMin + inMin)
        self.inMax = np.log(logMin + inMax)
        self.outMin = outMin
        self.outMax = outMax
        self.logMin = logMin
    def __call__(self,x,clip=False):
        return self.outMin + \
        (self.outMax - self.outMin)*(np.log(self.logMin + x) - self.inMin)/\
        (self.inMax - self.inMin)
    def autoscale(self,A):
        self.inMin = np.min(A)
        self.inMax = np.max(A)


# Function to plot a slice from a simulation:
def plotslice(snap,zslice,thickness = 15,width=None,cmap="PuOr_r",
        logScale=True,qty='rho',units="Msol h**2 Mpc^-3",av_z=True,
        vmin=None,vmax=None,linthresh=None):
    slicez = np.where((snap['pos'][:,2] >= zslice - thickness/2) & \
        (snap['pos'][:,2] <= zslice + thickness/2))
    if width is None:
        width = snap.properties['boxsize'].ratio("Mpc a h**-1")
    im1 = sph.image(snap[slicez],qty=qty,units=units,width=width,
        cmap=cmap,av_z = av_z,log=logScale,vmin=vmin,vmax=vmax,linthresh=linthresh)

# Plot a circle at some position.
def plotCircle(centre,radius,fmt='r--',offset = np.array([0,0])):
    theta = np.linspace(0,2*np.pi,100)
    X = radius*np.cos(theta) + centre[0] + offset[0]
    Y = radius*np.sin(theta) + centre[1] + offset[1]
    plt.plot(X,Y,fmt)

# Plot a set of circles
def plotVoidCircles(voidCentres,voidRadii,voidsToPlot,offset = np.array([0,0]),fmt='r--'):
    for k in voidsToPlot:
        plotCircle(voidCentres[k,:],voidRadii[k],fmt,offset=offset)

# Plot the outline of voids as projected alpha-shapes on top of a density slice:
def plotVoidParticles(snap,hr,voidsToPlot,zslice = None,snapsort=None,
    offset = np.array([0,0]),fmt='r--',marker='.',color='r',thickness=15,
    s=1,differentColours=True,cmap='hsv',includeLabels=True,voidRadii=None,
    voidCentres = None,includeNumbers=True,alphashapeParam=None,alpha=0.5,
    useAlphaShapes = True,Xrange=None,Yrange=None,includeScatter=True):
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    # Hacky way to filter out edge cases which aren't polygons:
    sample = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    alpha_sample = alphashape.alphashape(sample,0.5)
    if snapsort is None:
        snapshort = np.argsort(snap['iord'])
    if zslice is not None:
        zpos = snapedit.unwrap(snap['pos'][:,2] - zslice,boxsize)
        slicez = np.where((zpos >= - thickness/2) & (zpos <= thickness/2))
    cmapFunc = cm.get_cmap(cmap)
    counter = 0
    if np.isscalar(includeNumbers):
        includeAnyNumbers = np.ones(len(voidsToPlot),dtype=np.bool) & includeNumbers
    else:
        includeAnyNumbers = includeNumbers
    for k in voidsToPlot:
        print(k)
        if zslice is not None:
            haloPos = snap['pos'][snapsort[hr[k+1]['iord']],:]
            haloPosToUse = np.ones(len(haloPos),dtype=np.bool)
            if Xrange is not None:
                haloPosToUse = haloPosToUse & \
                (haloPos[:,0] >= Xrange[0]) & (haloPos[:,0] < Xrange[1])
            if Yrange is not None:
                haloPosToUse = haloPosToUse & \
                (haloPos[:,1] >= Yrange[0]) & (haloPos[:,1] < Yrange[1])
            intersecting = np.intersect1d(
                slicez[0],snapsort[hr[k+1]['iord'][haloPosToUse]])
            pos = snapedit.unwrap(snap['pos'][intersecting],boxsize)
        else:	
            pos = snapedit.unwrap(snap['pos'][snapsort[hr[k+1]['iord']]],boxsize)
        if len(pos) == 0:
            continue
        if differentColours:
            if includeScatter:
                plt.scatter(pos[:,0] + offset[0],
                    pos[:,1] + offset[1],marker=marker,
                    color=cmapFunc(counter/len(voidsToPlot)),s=s)
        else:
            if includeScatter:
                plt.scatter(pos[:,0] + offset[0],
                    pos[:,1] + offset[1],marker=marker,color=color,s=s)
        if includeAnyNumbers[counter]:
            label = "$" + str(k) + "$"
            if voidRadii is not None:
                label += "\n$" + \
                plot.scientificNotation(voidRadii[k],powerRange=1) + \
                "\\mathrm{\\,Mpc}h^{-1}$"
            if voidCentres is not None:
                labelPos = voidCentres[k,:]
            else:
                allPartsToUse = np.ones(pos.shape[0],dtype=np.bool)
                if Xrange is not None:
                    allPartsToUse = allPartsToUse & \
                    (pos[:,0] >= Xrange[0]) & (pos[:,0] < Xrange[1])
                if Yrange is not None:
                    allPartsToUse = allPartsToUse & \
                    (pos[:,1] >= Yrange[0]) & (pos[:,1] < Yrange[1])
                if np.any(allPartsToUse):
                    labelPos = np.mean(pos[allPartsToUse,0:2],0)
            if differentColours:
                colorToUse = cmapFunc(counter/len(voidsToPlot))
            else:
                colorToUse = color
            plt.text(labelPos[0],labelPos[1],label,
                color='w',horizontalalignment='center',
                verticalalignment='center')
        if useAlphaShapes and (len(pos) > 2):
            if differentColours:
                colorToUse = cmapFunc(counter/len(voidsToPlot))
            else:
                colorToUse = color
            if alphashapeParam is None:
                alphaValToUse = alphashape.optimizealpha(pos[:,0:2])
            else:
                alphaValToUse = alphashapeParam
            alpha_shape = alphashape.alphashape(pos[:,0:2],alphaValToUse)
            if type(alpha_shape) == type(alpha_sample):
                ax = plt.gca()
                ax.add_patch(PolygonPatch(alpha_shape,
                    fc=colorToUse,ec='k',alpha=alpha))
        counter += 1

# Plot a halo mass function
def plotHMF(hmasses,snap,massLower=1e12,massUpper = 1e16,nBins=101,ylim=[1e-1,1e5],
    volSim = None,ax=None,ylabel='Number of Anti-halos',
    xlabel='Mass Bin Centre',label='Halos',labelLine='TMF prediction',
    plotTMF = True,dens_type='SOMean',Delta=200,marker='x',color=None,
    linestyle='',tmfcolor=None,tmfstyle=':',mass_function="Tinker"):
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if volSim is None:
        volSim = boxsize**3
    [dndm,m] = TMF_from_hmf(massLower,massUpper,h=snap.properties['h'],
        Om0=snap.properties['omegaM0'],Delta=Delta,
        delta_wrt=dens_type,mass_function=mass_function)
    massBins = 10**np.linspace(np.log10(massLower),np.log10(massUpper),nBins)
    n = cosmology.dndm_to_n(m,dndm,massBins)
    [binList,noInBins] = plot_utilities.binValues(hmasses,massBins)
    sigmaBins = np.sqrt(noInBins)
    massBinCentres = plot_utilities.binCentres(massBins)
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(massBinCentres,noInBins,yerr=sigmaBins,marker=marker,linestyle=linestyle,label=label,color=color)
    if plotTMF:
        ax.plot(massBinCentres,n*volSim,tmfstyle,label=labelLine,color=tmfcolor)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

# Plot distribution of void radii:
def plotVoidRadii(radii,rMin = 0, rMax = 22,nBins = 43):
    rBins = np.linspace(rMin,rMax,nBins)
    fig, ax = plt.subplots()
    [prob,sigma,noInBins,inBins] = plot.computeHistogram(radii,rBins,density=False)
    histWithErrors(prob,sigma,rBins,label='Void Radius Distribution',ax=ax)
    ax.set_xlabel('r $[\\mathrm{Mpc}h^{-1}]$')
    ax.set_ylabel('Number of Voids')
    ax.set_yscale('log')

# Plot the numbers associated to a particular halo:	
def numberHalos(hcentres,toNumber=None,circle=False,boxsize = None,color='r',
        offset = np.array([0,0]),radius = 3):
    if toNumber is None:
        toNumber = np.arange(0,len(h))
    for k in toNumber:
        if boxsize is None:
            centre = np.array([hcentres[k,0] + offset[0],hcentres[k,1] + offset[1]])
        else:
            centre = snapedit.unwrap(np.array([hcentres[k,0] + offset[0],
                hcentres[k,1] + offset[1]]),boxsize)
        plt.text(centre[0],centre[1],str(k+1),color=color)
        if circle:
            plotCircle(centre,radius)

def plotVoidsInSlice(zslice,width,thickness,snap,hr,hrcentres,
        plotVoids = None,includeNumbers=True,voidRadii=None,zranges=None,snapsort=None,
        useAlphaShapes=True,differentColours=True,alphashapeParam=0.54,
        alpha=0.2,includeScatter=False):
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if plotVoids is None:
        listToPlot = np.ones(len(hr),dtype=np.bool)
    else:
        listToPlot = plotVoids
    if snapsort is None:
        snapsort =  np.argsort(snap['iord'])
    if zranges is None:
        zranges = np.zeros((len(hr),2))
        for k in range(0,len(hr)):
            zranges[k,:] = getAntihaloExtent(snap,hr[k+1],
                centre=hrcentres[k,:],snapsort=snapsort)
    intersectsSlice = np.where(
        intersectsSliceWithWrapping(zranges,zslice,thickness,boxsize) &\
        pointsInBoundedPlaneWithWrap(hrcentres,[-width/2,width/2],
        [-width/2,width/2],boxsize=boxsize) & listToPlot)[0]
    plotslice(snap,zslice,width=width,thickness=thickness)
    plotVoidParticles(snap,hr,intersectsSlice,zslice=zslice,
        Xrange = [-width/2,width/2],Yrange = [-width/2,width/2],
        snapsort=snapsort,useAlphaShapes=useAlphaShapes,
        differentColours=differentColours,voidRadii=voidRadii,
        includeNumbers=includeNumbers,alphashapeParam=alphashapeParam,
        alpha=alpha,includeScatter=includeScatter)
    plt.xlim([-width/2,width/2])
    plt.ylim([-width/2,width/2])

def sphericalSlice(snap,radius,centre=np.array([0,0,0]),thickness=15,nside=64,fillZeros = 1e-3):
    annulus = pynbody.filt.Annulus(radius-thickness/2,radius+thickness/2,cen=centre)
    posXYZ = snap[annulus]['pos']
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    # Wrapped displacements:
    dispXYZ = snapedit.unwrap(posXYZ - centre,boxsize)
    # Angular coordinates
    r = np.sqrt(np.sum(dispXYZ**2,1))
    phi = np.arctan2(dispXYZ[:,1],dispXYZ[:,0])
    theta = np.pi/2 - np.arcsin(dispXYZ[:,2]/r)# Assuming input is a declination.
    # Healpix conversion:
    npix = healpy.nside2npix(nside)
    ind = healpy.ang2pix(nside,theta,phi)
    # Density:
    hpxMap = np.zeros(npix)
    voxelVolume = (4*np.pi*(radius+thickness/2)**3/3 - 4*np.pi*(radius-thickness/2)**3/3)/npix
    #np.add.at(hpxMap,ind,snap[annulus]['rho'])
    np.add.at(hpxMap,ind,snap[annulus]['mass'].in_units("Msol h**-1")/voxelVolume)
    if fillZeros is not None:
        hpxMap[np.where(hpxMap == 0.0)] += fillZeros
    return hpxMap

def filterPolarPointsToAnnulus(lonlat,r,radius,thickness = 15):
    return lonlat[np.where((r >= radius - thickness/2) & (r <= radius + thickness/2))[0],:]

def plotMollweide(hpxMap,radius=None,galaxyAngles=None,galaxyDistances=None,
        centre=np.array([0,0,0]),thickness=15,vmin=1e-2,vmax=1e2,cmap='PuOr_r',
        shrink=0.5,pad=0.05,nside=64,showGalaxies=True,ax=None,title=None,
        fontname='serif',fontsize=8,fig=None,guideColor='grey',boundaryOff=False,
        titleFontSize=8,margins = (0,0,0,0),figsize = (8,4),xsize=800,dpi=300,\
        cbarSize=[4.0,0.5],returnAx=False,doColorbar=True,\
        cbarLabel='$\\rho/\\bar{\\rho}$',sub=None,reuse_axes=False):
    if galaxyAngles is not None:
        if radius is None:
            radius = 135/2.0
        pointsToScatter = filterPolarPointsToAnnulus(galaxyAngles,\
            galaxyDistances,radius,thickness=thickness)
    if fig is None:
        #fig = plt.figure(figsize = figsize)
        #fig, ax = plt.subplots(1,1,figsize = figsize,dpi=dpi)
        fig = plt.gcf()
    if ax is not None:
        plt.axes(ax)
    else:
        ax = plt.gca()
    healpy.mollview(hpxMap,cmap=cmap,cbar=False,norm='log',
        min=vmin,max=vmax,hold=True,fig=fig,margins=margins,xsize=xsize,\
        sub=sub,reuse_axes=reuse_axes)
    ax = plt.gca()
    ax.set_autoscale_on(True)
    healpy.graticule(color=guideColor)
    # Hacky solution to make the boundary grey, since healpy hardcodes this:
    lines = ax.get_lines()
    for l in lines:
        if l.get_color() != guideColor:
            l.set_color(guideColor)
    if boundaryOff:
        # Very hacky, and liable to break if healpy changes, 
        # but not clear how else we would identify which line is the boundary...
        for l in range(20,len(lines)):
            lines[l].set_linestyle('None')
    if showGalaxies:
        mollweideScatter(pointsToScatter,ax=ax)
    if title is None:
        plt.title("Spherical slice, $R = " + ("%.2g" % radius) + \
            "\\mathrm{\\,Mpc}h^{-1}$, Thickness=$" + ("%.2g" % thickness) + \
            "\\mathrm{\\,Mpc}h^{-1}$",
            fontfamily=fontname,fontsize=titleFontSize)
    else:
        plt.title(title,fontfamily=fontname,fontsize=titleFontSize)
    if doColorbar:
        sm = cm.ScalarMappable(colors.LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
        plt.colorbar(sm,location='bottom',label=cbarLabel,\
            shrink=shrink,pad=pad)
        #cbax = fig.add_axes([figsize[0]/4,0.05,figsize[0]/2,figsize[0]/16])
        #cbar = plt.colorbar(sm, orientation="horizontal",
        #    pad=pad,label=cbarLabel,shrink=shrink,\
        #    aspect=cbarSize[0]/cbarSize[1],cax=cbax)
        #cbar.ax.tick_params(axis='both',labelsize=fontsize)
        #cbar.set_label(label = cbarLabel,fontsize = fontsize,\
        #    fontfamily = fontname)
    if returnAx:
        return fig, ax

# Scatter points at the specified angles in a Mollweide projection.
def mollweideScatter(angles,color='r',s=1,marker='.',angleCoord="ra_dec",
        angleUnit="deg",text=None,
        fontname='serif',fontsize=7,horizontalalignment='left',
        verticalalignment='bottom',ax=None,textPos=None,textcoords='data',
        arrowprops=None,arrowpad = 0,textColour='k',label=None):
    MW = healpy.projector.MollweideProj()
    if ax is None:
        fig, ax = plt.subplots()
    if angleUnit == "deg":
        angleFactor = np.pi/180
    elif angleUnit == "rad":
        angleFactor = 1.0
    else:
        raise Exception("Unrecognised angle unit (options = {'deg','rad'}).")
    if angleCoord == "ra_dec":
        sgMW = MW.ang2xy(theta = np.pi/2 - \
            angleFactor*angles[:,1],phi=angleFactor*angles[:,0],lonlat=False)
    elif angleCoord == "spherical":
        sgMW = MW.ang2xy(theta = angleFactor*angles[:,1],
            phi=angleFactor*angles[:,0],lonlat=False)
    else:
        raise Exception("Unrecognised angular coordinate " + \
            "system (options = {'ra_dec','spherical'}).")
    if marker == 'c':
        ax.scatter(sgMW[0],sgMW[1],marker='o',s=s,edgecolors=color,\
            facecolors=None,label=label)
    else:
        ax.scatter(sgMW[0],sgMW[1],marker=marker,s=s,color=color,label=label)
    if text is not None:
        if np.isscalar(sgMW[0]):
            lensgMW = 1
        else:
            lensgMW = len(sgMW[0])
        for k in range(0,lensgMW):
            if type(horizontalalignment) == type('string'):
                ha = horizontalalignment
            else:
                ha = horizontalalignment[k]
            if type(verticalalignment) == type('string'):
                va = verticalalignment
            else:
                va = verticalalignment[k]
            if textPos is None:
                xytext = None
                textCoordToUse = None
                arrow=None
            else:
                xytext = textPos[k]
                if xytext is None:
                    textCoordToUse = None
                    arrow=None
                else:
                    textCoordToUse = textcoords
                    arrow=arrowprops
            if np.isscalar(sgMW[0]):
                sgVal0 = sgMW[0]
                sgVal1 = sgMW[1]
            else:
                sgVal0 = sgMW[0][k]
                sgVal1 = sgMW[1][k]
            plt.annotate(text[k],np.array([sgVal0,
                sgVal1]),fontfamily=fontname,fontsize=fontsize,
                horizontalalignment=ha,verticalalignment=va,xytext=xytext,
                textcoords=textCoordToUse,arrowprops=arrow,
                bbox = dict(pad=arrowpad,fc='none',ec='none'),\
                color=textColour)

# Plot a slice from a snapshot along the z direction, with some additional formatting.
def plotZSlice(snap,posToPlot,zslice,width,thickness=15,av_z=True,
        useGalaxy="all",marker='.',s=1,color='r',circleRadius = 300,
        circleCentre = np.array([0,0,0])):
    if useGalaxy == "all":
        useGalaxy = np.ones(len(posToPlot),dtype=bool)
    plotslice(snap,zslice,width=width,
        units="Msol h**2 Mpc**-3",thickness=thickness,av_z=av_z)
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    catPartsInSlice = pointsInRangeWithWrap(posToPlot,[zslice - thickness/2,
        zslice + thickness/2],boxsize = boxsize,axis=2)
    catPartsInBoundedRegion = pointsInBoundedPlaneWithWrap(posToPlot,
        [-width/2,width/2],[-width/2,width/2],boxsize=boxsize)
    catPartsToPlot = np.where(catPartsInSlice & catPartsInBoundedRegion & useGalaxy)[0]
    plt.scatter(posToPlot[catPartsToPlot,0],
        posToPlot[catPartsToPlot,1],marker=marker,s=s,color=color)
    plt.xlim([-width/2,width/2])
    plt.ylim([-width/2,width/2])
    if circleRadius is not None:
        plotCircle(circleCentre,circleRadius)

# Radial profiles:
def plotVoidProfiles(antiHaloRadii,antiHaloCentres,pairCountsAH,volumesListAH,
        rBins,nbar,snap,ranges=None,valuesAH=None,fontsize=15,
        returnAx = False,rangesText="\\,\\mathrm{Mpc}h^{-1}",
        title="Stacked Void Profiles",method="poisson",errorType="Weighted",
        conditionAH=None,ax=None,includeLegend = True,legendFontSize=15,
        fmt='-',label = "Density profile",color=None,ylim=[0,1.5],
        rangeText="\\,M_{\\mathrm{sol}}h^{-1}",cycle='format',
        includeXLabel=True,includeYLabel=True,includeGuides=True,
        guideColor='grey',fontname="serif"):
    #Filter:
    filterListAH = []
    if conditionAH is None:
        conditionAH = np.ones(antiHaloRadii.shape,dtype=np.bool)
    if ranges is None:
        filterListAH.append(np.where(conditionAH)[0])
    else:
        if valuesAH is None:
            raise Exception("Specify value to filter by setting valuesAH.")
        for k in range(0,len(ranges)-1):
            filterListAH.append(np.where((valuesAH > ranges[k]) & \
            (valuesAH <= ranges[k+1]) & conditionAH)[0])
    if ax is None:
        fig, ax = plt.subplots()
    rBinStackCentres = plot.binCentres(rBins)
    for k in range(0,len(filterListAH)):
        if filterListAH[k] is None:
            filterToUse = slice(len(antiHaloRadii))
        else:
            filterToUse = filterListAH[k]
        [nbarj,sigma] = stacking.stackVoidsWithFilter(antiHaloCentres,
            antiHaloRadii,filterListAH[k],snap,rBins,nPairsList = pairCountsAH,
            volumesList=volumesListAH,method=method,errorType=errorType)
        if ranges is not None:
            rangeLabel = ', $' + plot.scientificNotation(ranges[k],powerRange=2) + \
                ' \\mathrm{-} ' + \
                plot.scientificNotation(ranges[k+1],powerRange=2) + \
                rangeText + '$'
        else:
            rangeLabel = ""
        ax.errorbar(rBinStackCentres,nbarj/nbar,yerr=sigma/nbar,
            label=label + rangeLabel,fmt=fmt,color=color)
    if includeGuides:
        ax.plot(rBinStackCentres,np.ones(rBinStackCentres.shape),
            linestyle='--',color=guideColor)
        ax.plot([1,1],ylim,linestyle='--',color=guideColor)
    if includeXLabel:
        ax.set_xlabel("$R/R_{\mathrm{eff}}$",fontsize=fontsize,fontfamily=fontname)
    if includeYLabel:
        ax.set_ylabel("$\\rho/\\bar{\\rho}$",fontsize=fontsize,fontfamily=fontname)
    ax.tick_params(axis='both',labelsize=fontsize)
    if includeLegend:
        ax.legend(prop={"size":legendFontSize,"family":fontname})
    ax.set_ylim(ylim)
    if returnAx:
        return ax

# Plot the signal to noise ratio for a set of BORG simulations:
def plotBORGSNR(snr,positions,nRadBins = 101,rmin=0,rmax=300,centre=None,
        label = 'SNR in radial bins',labelsize=15):
    # Radial bin of snr:
    rBins = np.linspace(rmin,rmax,nRadBins)
    rBinCentres = binCentres(rBins)
    snrRadial = np.zeros(nRadBins-1)
    snrRadialError = np.zeros(nRadBins-1)
    if centre is None:
        centre = np.array([0,0,0])
    dist = np.sqrt(np.sum((positions - centre)**2,1))
    for k in range(0,nRadBins-1):
        radialValues = snr[np.where((dist > rBins[k]) & (dist <= rBins[k+1]))]
        snrRadial[k] = np.mean(radialValues)
        snrRadialError[k] = np.sqrt(np.var(radialValues))/np.sqrt(len(radialValues))
    # Get transition point:
    #nTrans = np.where(snrRadial > 1)[0][-1]
    #rCross = (rBinCentres[nTrans] + rBinCentres[nTrans+1])/2

    # SNR plot:
    plt.errorbar(rBinCentres,snrRadial,yerr=snrRadialError,label=label)
    plt.plot([rBinCentres[0],rBinCentres[-1]],[1,1],'k:',label='SNR = 1')
    plt.legend(prop={"size":15,"family":"serif"})
    plt.xlabel("Radial bin centre, $[\\mathrm{Mpc}h^{-1}]$",fontsize=15,fontfamily="serif")
    plt.ylabel("Mean SNR in bin ($\\delta^2/\sigma_{\\delta}^2$)",fontsize=15,fontfamily="serif")
    ax = plt.gca()
    ax.tick_params(axis='both',labelsize=labelsize)
    plt.subplots_adjust(bottom = 0.15)
    plt.show()

# Function to compare different halo mass functions to their TMF predictions.
def compareHaloMassFunctions(massBins,noInBins1,sigma1,
        vol1,noInBins2=None,sigma2=None,vol2=None,ax=None,label1='Constrained region',label2 = 'Whole Simulation',
        linestyle1='',linestyle2='',marker1='.',marker2='.',tmffmt1 = ':',
        tmffmt2 = '-.',tmflabel1 = 'TMF prediction (constrained region)',
        tmflabel2 ='TMF prediction (whole simulation)',bottom=0.15,left=0.15,fontsize=15,
        font="serif",legendLoc='lower left',
        title="Halo mass function - average of 6 samples",
        xlabel="Mass bin centre [$M_{\odot}h^{-1}$]",
        ylabel="Number of Halos",labelRight=True,grid=True,gridcolor='grey',
        gridstyle=':',gridalpha=0.5,ylim=[1e-2,1e5],legendFontsize=10,
        showDiff=False,nsamples=6,diffstyle='',markerDiff=',',
        diffLabel="Unconstrained region.",bbox_to_anchor=None,interval=True,
        fill_color1='r',fill_color2='g',fill_color3='b',fill_color4='m',
        fill_alpha=0.5,scaleInterval=True,color1=None,color2=None,
        Tcmb0 = 2.725,Om0=0.307,Ob0 = 0.0486,Delta=200,delta_wrt="SOMean",
        h=0.705,sigma8 = 0.8288,showTMF=True,plotFirst=True,plotSecond=True,
        showLegend=True,mass_function1="Tinker",mass_function2="Bhattacharya",
        plot_both=False,Ol0=None):	
    if ax is None:
        fig, ax = plt.subplots()
    massBinCentres = binCentres(massBins)
    if plotFirst:
        ax.errorbar(massBinCentres,noInBins1,marker=marker1,
            yerr=sigma1,linestyle=linestyle1,label=label1,color=color1)
    if plotSecond and (noInBins2 is not None):
        ax.errorbar(massBinCentres,noInBins2,marker=marker2,yerr=sigma2,
            linestyle=linestyle2,label=label2,color=color2)
    if showDiff:
        noInBinsDiff = noInBins2 - noInBins1
        noInBinsSigmaDiff = np.sqrt(sigma2**2 - sigma1**2 + \
            noInBins1*nsamples/(nsamples-1))
        ax.errorbar(massBinCentres,noInBinsDiff,yerr=noInBinsSigmaDiff,
            linestyle=diffstyle,marker=markerDiff,label=diffLabel)
    if showTMF:
        if plot_both:
            [dndmT,mT] = TMF_from_hmf(np.min(massBins),
                np.max(massBins),h=h,Om0=Om0,Ob0=Ob0,
                Tcmb0 = Tcmb0,Delta=Delta,delta_wrt=delta_wrt,
                sigma8=sigma8,mass_function=mass_function1,Ol0=Ol0)
            nT = cosmology.dndm_to_n(mT,dndmT,massBins)
            [dndmB,mB] = TMF_from_hmf(np.min(massBins),np.max(massBins),
                h=h,Om0=Om0,Ob0=Ob0,Tcmb0 = Tcmb0,Delta=Delta,
                delta_wrt=delta_wrt,sigma8=sigma8,
                mass_function=mass_function2,Ol0=Ol0)
            nB = cosmology.dndm_to_n(mB,dndmB,massBins)
            if plotFirst:
                ax.plot(massBinCentres,nT*vol1,tmffmt1,
                    label=mass_function1 + " (constrained)",color=fill_color1)
                ax.plot(massBinCentres,nB*vol1,tmffmt1,
                    label=mass_function2 + " (constrained)",color=fill_color3)
            if plotSecond and (vol2 is not None):
                ax.plot(massBinCentres,nT*vol2,tmffmt2,
                    label=mass_function1 + " (whole sim.)",color=fill_color2)
                ax.plot(massBinCentres,nB*vol2,tmffmt2,
                    label=mass_function2 + " (whole sim.)",color=fill_color4)
        else:
            [dndm,m] = TMF_from_hmf(np.min(massBins),np.max(massBins),
                h=h,Om0=Om0,Ob0=Ob0,Tcmb0 = Tcmb0,
                Delta=Delta,delta_wrt=delta_wrt,sigma8=sigma8,
                mass_function=mass_function1,Ol0=Ol0)
            n = cosmology.dndm_to_n(m,dndm,massBins)
            if plotFirst:
                ax.plot(massBinCentres,n*vol1,tmffmt1,
                    label=tmflabel1,color=fill_color1)
            if plotSecond and (vol2 is not None):
                ax.plot(massBinCentres,n*vol2,tmffmt2,
                    label=tmflabel2,color=fill_color2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.subplots_adjust(bottom=bottom,left=left)
    #plt.legend(ncol=1,bbox_to_anchor=(1.05,1),prop={"size":10,"family":"serif"})
    if interval:
        if plot_both:
            if scaleInterval:
                bounds1T = scipy.stats.poisson(nT*vol1*nsamples).interval(0.95)
                bounds2T = scipy.stats.poisson(nT*vol2*nsamples).interval(0.95)
                bounds1B = scipy.stats.poisson(nB*vol1*nsamples).interval(0.95)
                bounds2B = scipy.stats.poisson(nB*vol2*nsamples).interval(0.95)
                if plotFirst:
                    ax.fill_between(massBinCentres,
                        bounds1T[0]/nsamples,bounds1T[1]/nsamples,
                        facecolor=fill_color1,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
                    ax.fill_between(massBinCentres,
                        bounds1B[0]/nsamples,bounds1B[1]/nsamples,
                        facecolor=fill_color3,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
                if plotSecond:
                    ax.fill_between(massBinCentres,
                        bounds2T[0]/nsamples,bounds2T[1]/nsamples,
                        facecolor=fill_color2,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
                    ax.fill_between(massBinCentres,
                        bounds2B[0]/nsamples,bounds2B[1]/nsamples,
                        facecolor=fill_color4,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
            else:
                bounds1T = scipy.stats.poisson(nT*vol1).interval(0.95)
                bounds2T = scipy.stats.poisson(nT*vol2).interval(0.95)
                bounds1B = scipy.stats.poisson(nB*vol1).interval(0.95)
                bounds2B = scipy.stats.poisson(nB*vol2).interval(0.95)
                if plotFirst:
                    ax.fill_between(massBinCentres,bounds1T[0],bounds1T[1],
                        facecolor=fill_color1,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
                    ax.fill_between(massBinCentres,bounds1B[0],bounds1B[1],
                        facecolor=fill_color3,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
                if plotSecond:
                    ax.fill_between(massBinCentres,bounds2T[0],bounds2T[1],
                        facecolor=fill_color2,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
                    ax.fill_between(massBinCentres,bounds2B[0],bounds2B[1],
                        facecolor=fill_color4,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
        else:
            if scaleInterval:
                bounds1 = scipy.stats.poisson(n*vol1*nsamples).interval(0.95)
                if vol2 is not None:
                    bounds2 = scipy.stats.poisson(
                        n*vol2*nsamples).interval(0.95)
                if plotFirst:
                    ax.fill_between(massBinCentres,
                        bounds1[0]/nsamples,bounds1[1]/nsamples,
                        facecolor=fill_color1,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
                if plotSecond and (vol2 is not None):
                    ax.fill_between(massBinCentres,
                        bounds2[0]/nsamples,bounds2[1]/nsamples,
                        facecolor=fill_color2,alpha=fill_alpha,
                        interpolate=True,label='95% Poisson interval')
            else:
                bounds1 = scipy.stats.poisson(n*vol1).interval(0.95)
                if vol2 is not None:
                    bounds2 = scipy.stats.poisson(n*vol2).interval(0.95)
                if plotFirst:
                    ax.fill_between(massBinCentres,bounds1[0],bounds1[1],
                    facecolor=fill_color1,alpha=fill_alpha,
                    interpolate=True,label='95% Poisson interval')
                if plotSecond and (vol2 is not None):
                    ax.fill_between(massBinCentres,bounds2[0],bounds2[1],
                    facecolor=fill_color2,alpha=fill_alpha,
                    interpolate=True,label='95% Poisson interval')
    if showLegend:
        ax.legend(prop={"size":legendFontsize,"family":font},
        loc=legendLoc,frameon=False,bbox_to_anchor=bbox_to_anchor)
    ax.set_title(title,fontsize=fontsize,fontfamily=font)
    ax.set_xlabel(xlabel,fontsize=fontsize,fontfamily=font)
    ax.set_ylabel(ylabel,fontsize=fontsize,fontfamily=font)
    ax.tick_params(axis='both',labelsize=fontsize,labelright=labelRight,right=labelRight)
    ax.tick_params(axis='both',which='minor',bottom=True,labelsize=fontsize)
    ax.tick_params(axis='y',which='minor')
    ax.yaxis.grid(color=gridcolor,linestyle=gridstyle,alpha=gridalpha)
    ax.set_ylim(ylim)

# Plot individual mass-functions, rather than the average of all of them:
def plotAllSampleHMFs(massBinCentres,n,noInSampleBins1,vol1,colorList,sampleNames,
    sigmaUpperSample1,noInSampleBins2=None,
    vol2=None,sigmaUpperSample2=None,sigmaLowerSample1=None,
    sigmaLowerSample2=None,ax=None,label1='Constrained region',
    label2 = 'Whole Simulation',linestyle1=':',linestyle2='-.',marker1='.',marker2='.',
    tmffmt1 = 'k:',tmffmt2 = 'k-.',tmflabel1 = 'TMF prediction (constrained region)',
    tmflabel2 ='TMF prediction (whole simulation)',bottom=0.15,left=0.15,fontsize=15,
    font="serif",legendLoc='lower left',
    title="Halo mass function - average of 6 samples",
    xlabel="Mass bin centre [$M_{\odot}h^{-1}$]",ylabel="Number of Halos",
    labelRight=True,grid=True,gridcolor='grey',gridstyle=':',gridalpha=0.5,
    ylim=[1e-2,1e5],legendFontsize=10,showDiff=False,nsamples=6,diffstyle='',
    markerDiff=',',diffLabel="Unconstrained region.",bbox_to_anchor=None,interval=True,
    fill_color1='r',fill_color2='g',fill_alpha=0.5,scaleInterval=False,
    plotFirst=True,plotSecond=True,showLegend=True,labelFirst=True,
    labelSecond=True,plotTMFLabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    if sigmaLowerSample1 is None:
        sigmaLowerSample1 = -sigmaUpperSample1
    if sigmaLowerSample2 is None:
        sigmaLowerSample2 = -sigmaLowerSample2
    for k in range(0,len(noInSampleBins1)):
        noInBins1 = noInSampleBins1[k]
        sigma1 = np.vstack((sigmaUpperSample1[k],sigmaLowerSample1[k]))
        if plotFirst:
            if labelFirst:
                if label1 == "":
                    label = sampleNames[k]
                else:
                    label = label1 + ", " + sampleNames[k]
            else:
                label = None
            ax.errorbar(massBinCentres,noInBins1,marker=marker1,yerr=sigma1,
                linestyle=linestyle1,label=label,color=colorList[k])
        if plotSecond and (noInSampleBins2 is not None):
            noInBins2 = noInSampleBins2[k]
            sigma2 = np.vstack((sigmaUpperSample2[k],sigmaLowerSample2[k]))
            if labelSecond:
                if label1 == "":
                    label = sampleNames[k]
                else:
                    label = label2 + ", " + sampleNames[k]
            else:
                label = None
            ax.errorbar(massBinCentres,noInBins2,marker=marker2,yerr=sigma2,
                linestyle=linestyle2,label=label,color=colorList[k])
        if showDiff and (noInSampleBins2 is not None):
            noInBins2 = noInSampleBins2[k]
            sigma2 = np.vstack((sigmaUpperSample2[k],sigmaLowerSample2[k]))
            noInBinsDiff = noInBins2 - noInBins1
            noInBinsSigmaDiff = np.sqrt(sigma2**2 - sigma1**2 + \
                noInBins1*nsamples/(nsamples-1))
            ax.errorbar(massBinCentres,noInBinsDiff,yerr=noInBinsSigmaDiff,
                linestyle=diffstyle,marker=markerDiff,label=diffLabel)
    if plotFirst:
        if plotTMFLabel:
            label = tmflabel1
        else:
            label = None
        ax.plot(massBinCentres,n*vol1,tmffmt1,label=label)
    if plotSecond and (vol2 is not None):
        if plotTMFLabel:
            label = tmflabel2
        else:
            label = None
        ax.plot(massBinCentres,n*vol2,tmffmt2,label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.subplots_adjust(bottom=bottom,left=left)
    #plt.legend(ncol=1,bbox_to_anchor=(1.05,1),prop={"size":10,"family":"serif"})
    if interval:
        if scaleInterval:
            bounds1 = scipy.stats.poisson(n*vol1*nsamples).interval(0.95)/nsamples
            if vol2 is not None:
                bounds2 = scipy.stats.poisson(
                    n*vol2*nsamples).interval(0.95)/nsamples
        else:
            bounds1 = scipy.stats.poisson(n*vol1).interval(0.95)
            if vol2 is not None:
                bounds2 = scipy.stats.poisson(n*vol2).interval(0.95)
        if plotFirst:
            ax.fill_between(massBinCentres,bounds1[0],bounds1[1],
                facecolor=fill_color1,alpha=fill_alpha,interpolate=True)
        if plotSecond and vol2 is not None:
            ax.fill_between(massBinCentres,bounds2[0],bounds2[1],
                facecolor=fill_color2,alpha=fill_alpha,interpolate=True)
    if showLegend:
        ax.legend(prop={"size":legendFontsize,"family":font},loc=legendLoc,
            frameon=False,bbox_to_anchor=bbox_to_anchor)
    ax.set_title(title,fontsize=fontsize)
    ax.set_xlabel(xlabel,fontsize=fontsize,fontfamily=font)
    ax.set_ylabel(ylabel,fontsize=fontsize,fontfamily=font)
    ax.tick_params(axis='both',labelsize=fontsize,labelright=labelRight,right=labelRight)
    ax.tick_params(axis='both',which='minor',bottom=True,labelsize=fontsize)
    ax.tick_params(axis='y',which='minor')
    ax.yaxis.grid(color=gridcolor,linestyle=gridstyle,alpha=gridalpha)
    ax.set_ylim(ylim)



# Plot an anlpha shape around points that have been mapped onto a Mollweide projection:
def plotMollweideAlphaShape(positions,alpha_shape=None,origin=None,posMW=None,
    centreMW = None,color='r',s=1,ec='k',marker='.',boxsize=None,weights=None,
    angleCoord="ra_dec",angleUnit="deg",text=None,fontname=None,fontsize=10,
    ax=None,includePoints=False,alphaVal=1.22,alpha=0.5,textPos=None,
    horizontalalignment='left',verticalalignment='bottom',h=0.705):
    if ax is None:
        fig, ax = plt.subplots()
    if posMW is None:
        posMW = computeMollweidePositions(positions,angleUnit=angleUnit,
            angleCoord=angleCoord,centre=origin,boxsize=boxsize,h=h)
    if centreMW is None:
        if boxsize is not None:
            if weights is None:
                weights = np.ones(positions.shape[0])
            centre = context.computePeriodicCentreWeighted(
                positions,weights,boxsize)
        else:
            if len(positions.shape) == 1:
                centre = np.array(positions)
            else:
                centre = np.mean(positions,0)
        centreMW = computeMollweidePositions(centre,angleUnit="deg",angleCoord="ra_dec",
            centre=origin,boxsize=boxsize,h=h)
    # Compute the associated alpha shape:
    if alphaVal is None:
        alphaVal = alphashape.optimizealpha(np.array([posMW[0],posMW[1]]).T)
    if alpha_shape is None:
        alpha_shape = alphashape.alphashape(np.array([posMW[0],posMW[1]]).T,alphaVal)
    if includePoints:
        ax.scatter(posMW[0],posMW[1],marker=marker,s=s,color=color)
    ax.add_patch(PolygonPatch(alpha_shape,fc=color,ec=ec,alpha=alpha))
    if text is not None:
        plt.text(centreMW[0],centreMW[1],text,horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            fontfamily=fontname,fontsize=fontsize)

# Plot the mass profile around a given point in a simulation:
def plotMassProfile(radii,mprof,rvir=None,logy=False,show=True):
    plt.plot(radii,mprof,label='Mass within radius R')
    plt.xlabel("Radius, R [$\\mathrm{Mpc}h^{-1}$]")
    plt.ylabel("Mass, M(r < R) [$M_{\\odot}h^{-1}$]")
    if logy:
        plt.yscale("log")
    if rvir is not None:
        ax = plt.gca()
        ylim = ax.get_ylim()
        plt.plot([rvir,rvir],[ylim[0],ylim[1]],linestyle=':',color='grey',label='Virial radius')
    plt.legend()
    if show:
        plt.show()

# Compare unconstrained and constrained void profiles
# Compare unconstrained and constrained void profiles
def plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjStack,\
        sigmaStack,nbarjRandStack,sigmaRandStack,nbar,rMin,mMin,mMax,\
        labelRand = "Unconstrained profiles (mean)",fmtRand = '-.',\
        colourRand = 'grey',labelCon = "Average of 6 samples (constrained)",\
        fmtCon = '-',colourCon = 'r',colorRandMean='k',\
        labelRandIndividual = "Unconstrained profiles (individual)",\
        fmtRandInd = ':',colourRandInd = 'grey',numRandsToPlot = 3,\
        ylim = [0,1.4],ax = None,guideColour = 'grey',guideStyle='--',\
        legendFontSize=12,fontname="serif",fontsize=12,frameon=False,\
        legendLoc = 'upper right',bottom=0.125,left=0.125,includeLegend=True,\
        show = True,title=None,hideYLabels = False,\
        plotIndividuals = False,\
        errorType = 'scatter',errorAlpha=0.5,meanType = 'standard',\
        plotIndividualsMean = False,savename=None,showTitle=True,\
        meanErrorLabel = 'Mean unconstrained \nprofile',\
        profileErrorLabel = 'Standard Deviation of \nunconstrained profiles',\
        nbarjUnconstrainedStacks = None,sigmajUnconstrainedStacks=None,\
        showMean = True,xlim=None):
    nbarjMean = stacking.weightedMean(nbarjStack,1.0/sigmaStack**2,axis=0)
    sigmaMean = np.sqrt(stacking.weightedVariance(nbarjStack,\
        1.0/sigmaStack**2,axis=0))
    if meanType == 'standard':
        nbarjRandMean = nbarjRandStack
    elif meanType == 'scatter':
        nbarjRandMean = stacking.weightedMean(nbarjUnconstrainedStacks,\
            1.0/sigmaIndividual**2,axis=0)
    else:
        raise Exception('Invalid meanType')
    if errorType == 'standard':
        sigmaRandMean = sigmaRandStack
    elif(errorType == 'scatter'):
        sigmaRandMean = np.sqrt(stacking.weightedVariance(\
            nbarjUnconstrainedStacks,1.0/sigmajUnconstrainedStacks**2,axis=0))
    else:
        raise Exception('Invalid errorType')
    # Plot mean profiles:
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(rBinStackCentres,nbarjMean/nbar,
        yerr=sigmaMean/nbar,label=labelCon,fmt=fmtCon,color=colourCon)
    #ax.errorbar(rBinStackCentres,nbarjRandMean/nbar,
    #	yerr=sigmaRandMean/nbar,label=labelRand,fmt=fmtRand,color=colourRand)
    #ax.plot(rBinStackCentres,nbarjRandMean/nbar,fmtRand,
    #    label=labelRand,color=colourRand)
    if showMean:
        ax.fill_between(rBinStackCentres,\
            y1 = (nbarjRandStack - sigmaRandStack)/nbar,\
            y2 = (nbarjRandStack + sigmaRandStack)/nbar,alpha=errorAlpha,\
            color = colorRandMean,label=meanErrorLabel)
    ax.fill_between(rBinStackCentres,\
        y1 = (nbarjRandMean - sigmaRandMean)/nbar,\
        y2 = (nbarjRandMean + sigmaRandMean)/nbar,alpha=errorAlpha,\
        color = colourRand,\
        label=profileErrorLabel)
    if title is None:
        title = 'Void Profiles, $R_{\\mathrm{eff}} > ' + \
            str(rMin) + '\\mathrm{\\,Mpc}h^{-1}$, $' + \
            scientificNotation(mMin) + ' < M/(M_{\\odot}h^{-1}) < ' + \
            scientificNotation(mMax) + '$'
    if showTitle:
        ax.set_title(title,fontsize=fontsize,fontfamily=fontname)
    ax.set_xlabel('$R/R_{\\mathrm{eff}}$',fontsize=fontsize,\
        fontfamily=fontname)
    if not hideYLabels:
        ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=fontsize,\
            fontfamily=fontname)
    ax.axhline(1.0,linestyle=guideStyle,color=guideColour)
    ax.axvline(1.0,linestyle=guideStyle,color=guideColour)
    if includeLegend:
        ax.legend(prop={"size":legendFontSize,"family":fontname},
            frameon=frameon,loc=legendLoc)
    ax.tick_params(axis='both',labelsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if hideYLabels:
        ax.set_yticklabels([])
    plt.subplots_adjust(bottom=bottom,left=left)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()


# Plot halo count histograms
def plotHaloCountHistogram(haloCounts,localCount=None,sigmaLocalCount=None,
        antiHaloLocalCount = None,sigmaAntiHaloLocalCount = None,
        haloCounts2=None,combined=True,countMax=10,
        textwidth=7.1014,label='All regions',rCut= 135,deltaLow = -0.065,
        deltaHigh = -0.055,label2 = 'Underdense regions',legLoc = 'lower left',
        marker = 'x',fitLabel = 'Poisson fit',localColour = 'grey',mThresh = 1e15,
        alpha = 0.5,localLabel= 'Local Universe (halos)',fontsize=12,fontname='serif',
        localColour2 = 'grey',localLabel2 = 'Local Universe (anti-halos)',
        includePoissonFit = True,nExpected=2,color1='b',color2 = 'r',ylim=[1e-5,1],
        legendFontSize = 10,title=None,figOut = None,showFig = True,
        includeMassFunctionFit = True,includeTheoryError = False,N = 1,
        alphaInterval = 0.95,hmfMarker = None,sigmaEstimate = 'volume',
        hmfErrors = True,histAlpha = 0.25,xlim = None,aspecty=0.5,aspectx=1,
        left = 0.1,right = 0.98,bottom = 0.135,top = 0.93,hmfLabel='HMF + Poisson',
        xlabel = "Number of halos or anti-halos",
        ylabel = "Fraction of Samples"):
    haloCountBins = np.arange(-0.5,countMax + 1.5)
    fig, ax = plt.subplots(figsize=(aspectx*textwidth,aspecty*textwidth))
    if combined:
        combinedCount = np.hstack(haloCounts)
        [meanPoisson,errorPoisson] = tools.getPoissonAndErrors(
            np.arange(0,countMax),combinedCount)
        [probHalos,sigmaHalos,noInBinsHalos,inBinsHalos] = computeHistogram(
            combinedCount,haloCountBins,alpha = alphaInterval)
        if sigmaEstimate == 'volume':
            kappa = np.sqrt(2)*scipy.special.erfinv(1 - alphaInterval)
            nSuccess = N*probHalos
            intervalLower = ((nSuccess + kappa**2/2)/(N + kappa**2)) - \
                kappa*np.sqrt(N)*np.sqrt(probHalos*(1 - probHalos) + \
                kappa**2/(4*N))/(N + kappa**2)
            intervalUpper = ((nSuccess + kappa**2/2)/(N + kappa**2)) + \
                kappa*np.sqrt(N)*np.sqrt(probHalos*(1 - probHalos) + \
                kappa**2/(4*N))/(N + kappa**2)
            sigmaHalos = np.vstack((probHalos - intervalLower,
                intervalUpper - probHalos))
        # Hide large errors, which are distracting since we are really extrapolating
        # in that regime:
        sigmaHalos[:,np.where(probHalos == 0)] = 0
        hist1 = histWithErrors(probHalos,sigmaHalos,haloCountBins,
            ax=ax,label='All regions',color=color1,alpha=histAlpha)
        if includePoissonFit:
            plt.errorbar(np.arange(0,countMax),meanPoisson,
                yerr = errorPoisson,marker='x',
                label='Poisson fit (all regions)',color=color1)
        if haloCounts2 is not None:
            combinedCount2 = np.hstack(haloCounts2)
            [meanPoisson2,errorPoisson2] = tools.getPoissonAndErrors(
                np.arange(0,countMax),combinedCount2)
            [probHalos2,sigmaHalos2,
                noInBinsHalos2,inBinsHalos2] = computeHistogram(
                combinedCount2,haloCountBins,alpha = alphaInterval)
            if sigmaEstimate == 'volume':
                kappa = np.sqrt(2)*scipy.special.erfinv(1 - alphaInterval)
                nSuccess = N*probHalos2
                intervalLower = ((nSuccess + kappa**2/2)/(N + kappa**2)) - \
                    kappa*np.sqrt(N)*np.sqrt(probHalos2*(1 - probHalos2) + \
                    kappa**2/(4*N))/(N + kappa**2)
                intervalUpper = ((nSuccess + kappa**2/2)/(N + kappa**2)) + \
                    kappa*np.sqrt(N)*np.sqrt(probHalos2*(1 - probHalos2) + \
                    kappa**2/(4*N))/(N + kappa**2)
                sigmaHalos2 = np.vstack((probHalos2 - intervalLower,
                    intervalUpper - probHalos2))
            hist2 = histWithErrors(probHalos2,sigmaHalos2,
                haloCountBins,ax=ax,label=label2,color = color2,alpha=histAlpha)
            if includePoissonFit:
                plt.errorbar(np.arange(0,countMax),meanPoisson2,
                    yerr = errorPoisson2,marker='x',
                    label="Poisson fit ($" + str(deltaLow) + \
                    " < \\delta <" + \
                    str(deltaHigh) + "$)",color=color2)
    else:
        for k in range(0,len(haloCounts)):
            [probHalos,sigmaHalos,
                noInBinsHalos,inBinsHalos] = computeHistogram(
                haloCountAll[k,:],haloCountBins,alpha = alphaInterval)
            histWithErrors(
                probHalos,sigmaHalos,haloCountBins,
                ax=ax,label='Sample ' + str(k+1),alpha=histAlpha)
    if localCount is not None:
        ax.axvspan(localCount - sigmaLocalCount,localCount + \
            sigmaLocalCount,alpha=alpha,color=localColour,label=localLabel,ec=None)
    if antiHaloLocalCount is not None:
        ax.axvspan(antiHaloLocalCount - sigmaAntiHaloLocalCount,
            antiHaloLocalCount + sigmaAntiHaloLocalCount,
            alpha=alpha,color=localColour2,label=localLabel2,ec=None)
    if includeMassFunctionFit:
        hmfMean = scipy.stats.poisson.pmf(np.arange(0,countMax),nExpected)
        if hmfErrors:
            kappa = np.sqrt(2)*scipy.special.erfinv(1 - alphaInterval)
            nSuccess = N*hmfMean
            intervalLower = ((nSuccess + kappa**2/2)/(N + kappa**2)) - \
                kappa*np.sqrt(N)*np.sqrt(hmfMean*(1 - hmfMean) + \
                kappa**2/(4*N))/(N + kappa**2)
            intervalUpper = ((nSuccess + kappa**2/2)/(N + kappa**2)) + \
                kappa*np.sqrt(N)*np.sqrt(hmfMean*(1 - hmfMean) + \
                kappa**2/(4*N))/(N + kappa**2)
            sigmaMean = np.vstack((hmfMean - intervalLower,intervalUpper - hmfMean))
            # Remove distacting errors in the extrapolated region:
            #sigmaMean[0,np.where(intervalLower < ylim[0])] = 0
            plt.errorbar(np.arange(0,countMax),hmfMean,
                yerr = sigmaMean,marker = hmfMarker,
                    label=hmfLabel,color=color1)
        else:
            plt.plot(np.arange(0,countMax),hmfMean,marker = hmfMarker,
                    label=hmfLabel,color=color1)
    if title is None:
        title = "$M \\geq " + scientificNotation(mThresh) + \
            "$ $M_{\\odot}h^{-1}$ within " + str(rCut) + \
            " $\\mathrm{Mpc}h^{-1}$ spheres, $" + str(deltaLow) + \
            " < \\delta <" + str(deltaHigh) + "$"
    ax.set_title(title,fontsize=fontsize,fontfamily=fontname)
    ax.set_xticks(range(0,countMax))
    ax.set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontname)
    ax.set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontname)
    ax.tick_params(axis='both',labelsize=fontsize)
    plt.legend(prop={"size":legendFontSize,"family":fontname},frameon=False,loc=legLoc)
    plt.subplots_adjust(left = left,right=right,top=top,bottom=bottom)
    ax.set_yscale('log')
    ax.set_ylim(ylim)
    if xlim is None:
        ax.set_xlim([haloCountBins[0],haloCountBins[-1]])
    else:
        ax.set_xlim(xlim)
    if figOut is not None:
        plt.savefig(figOut)
    if showFig:
        plt.show()

def plotZoA(ax=None,galacticCentreZOA = [-30,30],nPointsZOA=201,\
        bRangeCentre = [-10,10],bRange = [-5,5],nPointsEdgeZOA = 21,\
        fc='grey',ec=None,alpha=0.5,label='Zone of Avoidance'):
        # Zone of avoidance?
    if ax is None:
        fig, ax = plt.subplots()
    lZOA = np.linspace(-np.pi,np.pi,nPointsZOA)
    zoaCentral = np.where((lZOA*180/np.pi > galacticCentreZOA[0]) & \
        (lZOA*180/np.pi < galacticCentreZOA[1]))
    bZOAupp = (np.pi*bRange[1]/180)*np.ones(lZOA.shape)
    bZOAupp[zoaCentral] = (np.pi*bRangeCentre[1]/180)
    bZOAlow = (np.pi*bRange[0]/180)*np.ones(lZOA.shape)
    bZOAlow[zoaCentral] = (np.pi*bRangeCentre[0]/180)
    zoaUppCoord = SkyCoord(l=lZOA*u.rad,b=bZOAupp*u.rad,frame='galactic')
    zoaLowCoord = SkyCoord(l=lZOA*u.rad,b=bZOAlow*u.rad,frame='galactic')
    raZOAUpp = zoaUppCoord.icrs.ra.value
    decZOAUpp = zoaUppCoord.icrs.dec.value
    raZOALow = zoaLowCoord.icrs.ra.value
    decZOALow = zoaLowCoord.icrs.dec.value
    MW = healpy.projector.MollweideProj()
    angleFactor = np.pi/180.0
    XYUpp = MW.ang2xy(theta = np.pi/2 - angleFactor*decZOAUpp,
        phi=angleFactor*raZOAUpp,lonlat=False)
    XYLow = MW.ang2xy(theta = np.pi/2 - angleFactor*decZOALow,
        phi=angleFactor*raZOALow,lonlat=False)
    # Roll around to prevent sudden jumps in the lines:
    rollNumUpp = np.where(XYUpp[0][0:-1]*XYUpp[0][1:] < 0)[0][0] + 1
    rollNumLow = np.where(XYLow[0][0:-1]*XYLow[0][1:] < 0)[0][0] + 1
    # mollweide boundary part of ZOA:
    zoaBoundUpp = MW.xy2ang(XYUpp[0][(rollNumUpp-1):(rollNumUpp+1)],
        XYUpp[1][(rollNumUpp-1):(rollNumUpp+1)])
    zoaBoundLow = MW.xy2ang(XYLow[0][(rollNumLow-1):(rollNumLow+1)],
        XYLow[1][(rollNumLow-1):(rollNumLow+1)])
    zoaBoundLeft = np.linspace(zoaBoundUpp[0,0],zoaBoundLow[0,0],nPointsEdgeZOA)
    zoaBoundRight = np.linspace(zoaBoundLow[0,1],zoaBoundUpp[0,1],nPointsEdgeZOA)
    leftXY = MW.ang2xy(zoaBoundLeft,(-np.pi -1e-2)*np.ones(zoaBoundLeft.shape))
    rightXY = MW.ang2xy(zoaBoundRight,np.pi*np.ones(zoaBoundRight.shape))
    #Polygon defining ZOA:
    polyX = np.hstack((leftXY[0],np.flip(np.roll(XYLow[0],
        -rollNumLow)),rightXY[0],np.roll(XYUpp[0],-rollNumUpp)))
    polyY = np.hstack((leftXY[1],np.flip(np.roll(XYLow[1],
        -rollNumLow)),rightXY[1],np.roll(XYUpp[1],-rollNumUpp)))
    polyXY = np.vstack((polyX,polyY)).T
    polygon = patches.Polygon(polyXY,fc=fc,ec=ec,alpha=alpha,
        label=label)
    ax.add_patch(polygon)
    return polygon

# Plot Mollweide-view of the local universe, showing large halos, large antihalos, or both.
def plotLocalUniverseMollweide(rCut,snap,hpxMap=None,\
        ha=None,va=None,annotationPos=None,nameListLargeClusters=None,
        galaxyAngles = None,galaxyDistances = None,nside=64,
        alpha_shapes = None,coordAbell = None,abellListLocation=None,
        vmin=1e-2,vmax=1e2,title=None,showGalaxies=False,boundaryOff=True,
        haloColour = 'b',s=30,haloMarker='c',clusterMarker='x',
        labelFontSize = 8,arrowstyle='->',connectionstyle = "arc3,rad=0.",
        arrowcolour = 'k',shrinkArrowB = 5,shrinkArrow = 5,arrowpad = 1,
        largeAntihalos = None,hr=None,haloCentres=None,snapsort=None,alphaVal=7,
        cmap='hsv',voidAlpha = 0.2,includeZOA = True,nPointsZOA = 201,
        galacticCentreZOA = [-30,30],bRangeCentre = [-10,10],bRange = [-5,5],
        nPointsEdgeZOA = 21,bbox_to_anchor=(-0.1, -0.2),legLoc='lower left',
        legendFontSize = 8,antihaloCentres = None,figOut = None,showFig=True,
        titleFontSize = 8,fontname = 'serif',margins = (0,0,0,0),figsize = (8,4),
        xsize = 800,extent = None,bbox_inches = None,voidColour = None,
        antiHaloLabel = 'haloID',dpi=300,shrink=0.5,pad=0.05,\
        cbarLabel='$\\rho/\\bar{\\rho}$',ax=None,arrowAnnotations=True,\
        doColorbar=True,sub=None,showLegend=True,reuse_axes=False):
    if hpxMap is None:
        rhobar = (np.sum(snap['mass'])/\
            (snap.properties['boxsize']**3)).in_units("Msol h**2 Mpc**-3")
        hpxMap = sphericalSlice(snap,rCut/2,thickness=rCut,
            fillZeros=vmin*rhobar,centre=np.array([0,0,0]),\
            nside=nside)/rhobar
    fig, ax = plotMollweide(hpxMap,galaxyAngles=galaxyAngles,\
        galaxyDistances=galaxyDistances,ax=ax,\
        thickness=rCut,radius=rCut/2,nside=nside,\
        vmin=vmin,vmax=vmax,showGalaxies=showGalaxies,
        title=title,boundaryOff=boundaryOff,margins=margins,
        fontname=fontname,titleFontSize=titleFontSize,figsize=figsize,\
        xsize=xsize,dpi=dpi,returnAx=True,doColorbar=doColorbar,\
        cbarLabel=cbarLabel,sub=sub,reuse_axes=reuse_axes)
    if haloCentres is not None:
        haloAngles = context.equatorialXYZToSkyCoord(haloCentres)
        anglesToPlotHalos = np.vstack((haloAngles.icrs.ra.value,
            haloAngles.icrs.dec.value)).T
    if coordAbell is not None:
        anglesToPlotClusters = np.vstack((coordAbell.icrs.ra.value,
            coordAbell.icrs.dec.value)).T
        if arrowAnnotations:
            textPos = annotationPos
            arrowprops=dict(arrowstyle = arrowstyle,shrinkA=shrinkArrow,
                    color=arrowcolour,shrinkB = shrinkArrowB,
                    connectionstyle=connectionstyle)
        else:
            textPos = None
            arrowprops = None
        mollweideScatter(anglesToPlotClusters[abellListLocation,:],\
            color=haloColour,s=s,marker=haloMarker,
            text=nameListLargeClusters,fontsize=labelFontSize,
            horizontalalignment=ha,
            verticalalignment=va,ax=ax,textPos=textPos,
            arrowprops= arrowprops,
            arrowpad = arrowpad)
    if haloCentres is not None:
        mollweideScatter(anglesToPlotHalos,color=haloColour,s=s,marker=clusterMarker,
            fontsize=labelFontSize,ax=ax,arrowpad=arrowpad)
    if largeAntihalos is not None:
        if snapsort is None:
            snapsort = np.argsort(snap['iord'])
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        cmapFunc = cm.get_cmap(cmap)
        if alpha_shapes is None:
            ahMWPos = []
            alpha_shapes = []
            h = snap.properties['h']
            for k in range(0,len(largeAntihalos)):
                posXYZ = snapedit.unwrap(
                    snap['pos'][snapsort[hr[largeAntihalos[k]+1]['iord']],:],
                    boxsize)
                posMW = computeMollweidePositions(posXYZ,h=h)
                ahMWPos.append(posMW)
                alpha_shapes.append(
                    alphashape.alphashape(np.array([posMW[0],posMW[1]]).T,
                    alphaVal))
        for k in range(0,len(largeAntihalos)):
            if antihaloCentres is None:
                centreMW = None
            else:
                centreMW = (antihaloCentres[k][0],antihaloCentres[k][1])
            if voidColour is None:
                colourToUse = cmapFunc(k/len(largeAntihalos))
            elif type(voidColour) is list:
                colourToUse = voidColour[k]
            else:
                colourToUse = voidColour
            if antiHaloLabel == 'haloID':
                textToUse = str(largeAntihalos[k])
            elif antiHaloLabel == 'inPlot':
                textToUse = str(k + 1)
            elif type(antiHaloLabel) == list:
                textToUse = antiHaloLabel[k]
            else:
                raise Exception('Unrecognised antihalo label option.')
            plotMollweideAlphaShape(
                snapedit.unwrap(
                    snap['pos'][snapsort[hr[largeAntihalos[k]+1]['iord']],:],
                    boxsize),
                ax=ax,alphaVal = alphaVal,alpha_shape=alpha_shapes[k],
                alpha=voidAlpha,color=colourToUse,
                text=textToUse,includePoints=False,
                fontsize = labelFontSize,boxsize=boxsize,h=snap.properties['h'],
                centreMW = centreMW)
    if includeZOA:
        polygon = plotZoA(ax=ax,galacticCentreZOA = galacticCentreZOA,\
            nPointsZOA=nPointsZOA,bRangeCentre = bRangeCentre,bRange = bRange,\
            nPointsEdgeZOA = nPointsEdgeZOA,\
            fc='grey',ec=None,alpha=0.5,label='Zone of Avoidance')
    # Legend:
    handles = []
    if haloCentres is not None:
        haloMarkerHandle = mlines.Line2D([],[],color=haloColour,
            marker='x',linestyle='',label='BORG halo locations')
        handles.append(haloMarkerHandle)
    if coordAbell is not None:
        clusterMarkerHandle = mlines.Line2D([],[],linestyle='',marker='o',
            mec=haloColour,mfc=None,label='Observed locations')
        handles.append(clusterMarkerHandle)
    if includeZOA:
        handles.append(polygon)
    if voidColour is not None:
        if type(voidColour) is list:
            fakeVoid = patches.Polygon(np.array([[0,1,-1],[0,-1,-1]]).T,
                fc='b',ec='None',alpha=0.5,
                label='Large anti-halos')
        else:
            fakeVoid = patches.Polygon(np.array([[0,1,-1],[0,-1,-1]]).T,
                fc=voidColour,ec='None',alpha=0.5,
                label='Large anti-halos')
        handles.append(fakeVoid)
    if showLegend:
        ax.legend(handles=handles,frameon=False,
            prop={"size":legendFontSize,"family":"serif"},
            loc=legLoc,bbox_to_anchor=bbox_to_anchor)
    # As the last step,add a colorbar:
    #sm = cm.ScalarMappable(colors.LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
    #cbax = fig.add_axes([figsize[0]/4,0.05,figsize[0]/2,figsize[0]/16])
    #cbar = plt.colorbar(sm, orientation="horizontal",
    #    pad=pad,label='$\\rho/\\bar{\\rho}$',shrink=shrink,\
    #    cax=cbax)
    #cbar.ax.tick_params(axis='both',labelsize=legendFontSize)
    #cbar.set_label(label = '$\\rho/\\bar{\\rho}$',fontsize = legendFontSize,\
    #    fontfamily = fontname)
    if figOut is not None:
        plt.savefig(figOut,bbox_inches=bbox_inches,dpi=dpi)
    if showFig:
        plt.show()

# Compare catalogues:
def plotCatalogueComparison(mass1,mass1err,name1,mass2,mass2err,name2,\
        highlight = None,highlightLabel = 'BORG clusters',hlScale = 100,\
        scatterColour = seabornColormap[1],fitColour = seabornColormap[2],\
        optColour = seabornColormap[3],\
        hlColour = seabornColormap[9],symbol1 = '$M_{\\mathrm{SZ}}$',\
        symbol2 = '$M_{\\mathrm{X}}$',label1 = "Sunyaev-Zel'dovich mass",\
        label2 = "MCXC X-ray mass",massUnit = "$[10^{14}M_{\\mathrm{\\odot}}]$",\
        top = 0.95,bottom = 0.15,left = 0.1,right=0.965,textwidth=7.1014,\
        fontname = 'serif',fontsize = 10,fontsize2=7):
    # match clusters between the two catalogues:
    inCommon = -np.ones(len(name1),dtype=int)
    for k in range(0,len(name1)):
        location = np.where(name2 == name1[k])[0]
        if len(location) > 0:
            inCommon[k] = location[0]
    list1 = np.where(inCommon > 0)[0]
    list2 = inCommon[list1]
    # Fit for masses:
    opt1 = scipy.optimize.curve_fit(lambda x, b, c: b*x + c,mass1[list1],mass2[list2])
    # Fit without offset
    opt2 = scipy.optimize.curve_fit(lambda x, b: b*x,mass1[list1],mass2[list2])
    # Get the points corresponding to our clusters:
    if highlight is not None:
        intersection = np.intersect1d(list1,highlight,return_indices=True)
    # Plot of masses:
    fig, ax = plt.subplots(figsize=(textwidth,textwidth))
    ax.errorbar(mass1[list1],mass2[list2],xerr = mass1err[:,list1],yerr=mass2err[:,list2],
        linestyle='',zorder=1,color=scatterColour)
    ax.set_xlabel(label1 + ", " + symbol1 + " " + massUnit,
        fontfamily=fontname,fontsize=fontsize)
    ax.set_ylabel(label2 + ", " + symbol2 + " " + massUnit,
        fontfamily=fontname,fontsize=fontsize)
    ax.plot([0,10],[0,10],'k:',label=symbol1 + ' = ' + symbol2,\
        color='k',zorder=2)
    ax.plot([0,10],opt1[0][0]*np.array([0,10]) + opt1[0][1],linestyle='--',\
        color=fitColour,label='Best fit, ' + symbol2 +  '$ = ' +\
        '(' + scientificNotation(opt1[0][0]) + '\\pm' + \
        scientificNotation(np.sqrt(opt1[1][0][0])) + ')$' +  symbol1 + '$ + ' + \
        '(' + scientificNotation(opt1[0][1]) + '\\pm' + \
        scientificNotation(np.sqrt(opt1[1][1][1])) +  ')$',zorder=3)
    ax.plot([0,10],opt2[0][0]*np.array([0,10]),linestyle='--',\
        color = optColour,label = 'Best fit (no offset), ' + symbol2 + '$ = (' + \
        scientificNotation(opt2[0][0]) + '\\pm' + \
        scientificNotation(np.sqrt(opt2[1][0][0])) + ')$' +  symbol1)
    if highlight is not None:
        ax.scatter(mass1[list1][intersection[1]],\
            mass2[list2][intersection[1]],\
            marker='o',facecolors='none',edgecolors=hlColour,\
            s=hlScale,zorder=4,\
            label=highlightLabel)
    plt.subplots_adjust(top = top,bottom = bottom,left = left,right=right)
    plt.legend(prop={"size":fontsize2,"family":"serif"},loc='lower right')
    plt.xlim([0,np.max([np.max(mass2[list2]),np.max(mass1[list1])])])
    plt.ylim([0,np.max([np.max(mass2[list2]),np.max(mass1[list1])])])
    plt.show()


# Plot a density slice from a density field
def plotDensitySlice(ax,den,centre,width,boxsize,N,thickness,pslice,vmin,vmax,\
        cmap,markCentre=False,axesToShow = [0,2],losAxis=1,flip=False,\
        flipud=False,fliplr=False,swapXZ = False,centreMarkerColour='r'):
    [left,right,bottom,top] = [centre[axesToShow[0]] - width/2,\
        centre[axesToShow[0]] + width/2,\
        centre[axesToShow[1]] - width/2,\
        centre[axesToShow[1]] + width/2]
    if ax is None:
        fig, ax = plt.subplots()
    indLow = int((pslice + boxsize/2)*N/boxsize) - int((thickness/2)*N/(boxsize))
    indUpp = int((pslice + boxsize/2)*N/boxsize) + int((thickness/2)*N/(boxsize))
    if losAxis == 0:
        if swapXZ:
            denToPlot = np.mean(den[:,:,indLow:indUpp],2)
        else:
            denToPlot = np.mean(den[indLow:indUpp,:,:],0)
    elif losAxis == 1:
        denToPlot = np.mean(den[:,indLow:indUpp,:],1)
    else:
        if swapXZ:
            denToPlot = np.mean(den[indLow:indUpp,:,:],0)
        else:
            denToPlot = np.mean(den[:,:,indLow:indUpp],2)
    if flip:
        denToPlot = denToPlot.transpose()
    if flipud:
        denToPlot = np.flipud(denToPlot)
    if fliplr:
        denToPlot = np.fliplr(denToPlot)
    ax.imshow(denToPlot,\
        extent=(-boxsize/2,boxsize/2,-boxsize/2,boxsize/2),\
        norm = colors.LogNorm(vmin=vmin, vmax=vmax),cmap=cmap,\
        origin='lower')
    if markCentre:
        ax.scatter([centre[axesToShow[0]]],[centre[axesToShow[1]]],marker='x',\
            c=centreMarkerColour)
    return [left,right,bottom,top,indLow,indUpp]

# Compare two density slices from different density fields.
def plotDensityComparison(denCompareLeft,denCompareRight,clusterNum = 6,N = 256,\
        boxsize = 677.7,showGalaxies = False,savename = None,show=True,\
        showDiff = False,fontsize = 9,titleFontSize = 9,fontname = 'serif',textwidth=7.1014,\
        textheight=9.0971,cmap = 'PuOr_r',textLeft = "Field 1",textRight = "Field 2",\
        width = 200,centre1 = [0,0,0],centre2 = [0,0,0],thickness = 8,\
        vmin = 1/70,vmax = 70,pslice1 = None,pslice2 = None,\
        title = "BORG PM vs GADGET2 simulation",markCentre=False,\
        losAxis = 1,flipLeft=False,flipRight=False,\
        invertAxisLeft=False,invertAxisRight=False,\
        flipCentreLeft=False,flipCentreRight=False,\
        invertCentre = False,galOffset = [0,0,0],\
        swapXZLeft = False,swapXZRight = False,gal_position=None,\
        returnAx = False,flipudLeft=False,fliplrLeft=False,\
        flipudRight=False,fliplrRight=False):
    sort = {0:[1,2],1:[0,2],2:[0,1]}
    if flipCentreLeft:
        centreUse1 = np.flipud(centre1)
    else:
        centreUse1 = centre1
    if flipCentreRight:
        centreUse2 = np.flipud(centre2)
    else:
        centreUse2 = centre2
    axLabels = ['X','Y','Z']
    axesToShow = sort[losAxis]
    if pslice1 is None:
        pslice1 = centreUse1[losAxis]
    if pslice2 is None:
        pslice2 = centreUse2[losAxis]
    if invertAxisLeft:
        pslice1 = - pslice1
    if invertAxisRight:
        pslice2 = - pslice2
    if showDiff:
        fig, ax = plt.subplots(1,3,figsize=(textwidth,0.5*textwidth))
    else:
        fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
    [left1,right1,bottom1,top1,indLow1,indUpp1] = plotDensitySlice(ax[0],
        denCompareLeft,centreUse1,width,boxsize,N,thickness,pslice1,vmin,vmax,\
        cmap,markCentre=markCentre,axesToShow=axesToShow,losAxis=losAxis,\
        flip = flipLeft,flipud=flipudLeft,fliplr=fliplrLeft,swapXZ=swapXZLeft)
    [left2,right2,bottom2,top2,indLow2,indUpp2] = plotDensitySlice(ax[1],
        denCompareRight,centreUse2,width,boxsize,N,thickness,pslice2,vmin,vmax,\
        cmap,markCentre=markCentre,axesToShow=axesToShow,losAxis=losAxis,\
        flip = flipRight,flipud=flipudRight,fliplr=fliplrRight,\
        swapXZ=swapXZRight)
    if showDiff:
        ax[2].imshow(np.mean(denCompareLeft[:,indLow1:indUpp1,:],1)/ \
            np.mean(denCompareRight[:,indLow2:indUpp2,:],1),\
            extent=(-boxsize/2,boxsize/2,-boxsize/2,boxsize/2),\
            norm = colors.LogNorm(vmin=vmin, vmax=vmax),cmap=cmap,origin='lower')
    if showGalaxies:
        if gal_position is None:
            raise Exception("Need to supply galaxy positions.")
        galPos = galOffset + gal_position
        condition1 = np.where((galPos[:,losAxis] > centreUse1[losAxis] - \
            thickness/2) & (galPos[:,losAxis] <= centreUse1[losAxis] + \
            thickness/2))[0]
        condition2 = np.where((galPos[:,losAxis] > centreUse2[losAxis] - \
            thickness/2) & (galPos[:,losAxis] <= centreUse2[losAxis] + \
            thickness/2))[0]
        ax[0].scatter(galPos[condition1,axesToShow[0]],\
            galPos[condition1,axesToShow[1]],marker='.',c='r')
        ax[1].scatter(galPos[condition2,axesToShow[0]],\
            galPos[condition2,axesToShow[1]],marker='.',c='r')
    axData = [[left1,right1,bottom1,top1,indLow1,indUpp1],\
        [left2,right2,bottom2,top2,indLow2,indUpp2],\
        [left1,right1,bottom1,top1,indLow1,indUpp1]]
    for k in range(0,len(ax)):
        ax[k].set_xlim((axData[k][0],axData[k][1]))
        ax[k].set_ylim((axData[k][2],axData[k][3]))
        ax[k].set_xlabel('Equatorial ' + axLabels[axesToShow[0]] + \
            ' $[\\mathrm{Mpc}h^{-1}]$',\
            fontsize=fontsize,fontfamily=fontname)
        ax[k].set_ylabel('Equatorial ' + axLabels[axesToShow[1]] + \
            ' $[\\mathrm{Mpc}h^{-1}]$',\
            fontsize=fontsize,fontfamily=fontname)
        ax[k].tick_params(axis='both',labelsize=fontsize)
    if textLeft is None:
        ax[0].set_title('BORG posterior density, (1000 realisations)',\
            fontsize=titleFontSize,fontfamily=fontname)
    else:
        ax[0].set_title(textLeft,\
            fontsize=titleFontSize,fontfamily=fontname)
    if textRight is None:
        ax[1].set_title('Simulation Density (6 realisations)',\
            fontsize=titleFontSize,fontfamily=fontname)
    else:
        ax[1].set_title(textRight,\
            fontsize=titleFontSize,fontfamily=fontname)
    plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.85,hspace=0.155,wspace=0.315)
    sm = cm.ScalarMappable(colors.LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
    cbax = fig.add_axes([0.87,0.17,0.02,0.64])
    cbar = plt.colorbar(sm, orientation="vertical",label='$\\rho/\\bar{\\rho}$',cax=cbax)
    #plt.savefig(textRight + '_lowres.pdf')
    #plt.suptitle("BORG Posterior vs Simulation Density (low-res)",fontsize=12,fontfamily='serif')
    #plt.suptitle("BORG Posterior vs Simulation Density",fontsize=12,fontfamily='serif')
    plt.suptitle(title,fontsize=12,fontfamily='serif')
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnAx:
        return ax

def plotAalphaSkyDistribution(effAalpha,mBin,nSlice,ns,coordAbell,abell_d,\
        clusterInd,rLimits,snapNumList,figsize = (8,4),\
        showDeviation = False,boundaryOff = True,pad=0.05,showClusters = True,\
        deviationCmap = "PuOr_r",normalCmap='GnBu',cmap=None,\
        margins = (0,0,0,0),xsize=800,guideColor='grey',shrink=0.5,\
        fontsize = 7,titleFontSize=7,fontname='serif',includeZOA = True,\
        nRadialSlices = 10,nside = 4,clusterColour = 'r',\
        ha = ['right','left','left','left','left','center','right',\
        'right','right'],va = ['center','bottom','bottom','bottom','top',\
        'top','center','center','center'],annotationPos = [[-1.2,0.9],\
        [1.3,0.8],[1.8,0.5],[1.5,-1.2],[1.7,-0.7],[-1,0.2],[0.8,0.6],\
        [1.0,0.1],[-1.8,0.5]],nAbsBins = 8,absRange = None,\
        appRange = ['$m \\leq 11.5$','$11.5 < m \\leq 12.5$'],\
        nameList = None,title=None,galacticCentreZOA = [-30,30],\
        bRangeCentre = [-10,10],bRange = [-5,5],nPointsEdgeZOA = 21,\
        nPointsZOA = 201,show=True,savename=None):
    if cmap is None:
        if showDeviation:
            cmap = deviationCmap
        else:    
            cmap= normalCmap
    if showDeviation:
        vmin = -5
        vmax = 5
        norm = colors.Normalize(vmin=vmin,vmax=vmax)
    else:
        vmin = 0
        vmax = 120
        norm = colors.SymLogNorm(vmin=vmin,vmax=vmax,linthresh=1,linscale=1)
    sm = cm.ScalarMappable(norm,cmap=cmap)
    npixels = 12*nside**2
    hpxMaps = np.zeros((nRadialSlices,npixels))
    if absRange is None:
        absRange = ['$' + str(-21 - (k+1)*0.5) + ' \\leq M \\leq ' \
            + str(-21 - k*0.5) + '$' for k in range(0,nAbsBins)]
    if nameList is None:
        clusterNames = np.array([['Perseus-Pisces (A426)'],
           ['Hercules B (A2147)'],
           ['Coma (A1656)'],
           ['Norma (A3627)'],
           ['Shapley (A3571)'],
           ['A548'],
           ['Hercules A (A2199)'],
           ['Hercules C (A2063)'],
           ['Leo (A1367)']], dtype='<U21')
        nameList = [name[0] for name in clusterNames]
    for k in range(0,nRadialSlices):
        if np.isscalar(ns):
            hpxMaps[k,:] = effAalpha[ns,mBin,npixels*k:npixels*(k+1)]
        else:
            hpxMaps[k,:] = np.mean(effAalpha[ns,mBin,npixels*k:npixels*(k+1)],0)
    fig, ax = plt.subplots(1,1,figsize = figsize)
    if showDeviation:
        hpMap = np.zeros(npixels)
        neighbours = healpy.pixelfunc.get_all_neighbours(\
            nside,np.arange(0,npixels))
        for k in range(0,npixels):
            nonZeroPixels = np.where(\
                hpxMaps[nSlice,neighbours[:,k]] != 0.0)
            if len(nonZeroPixels[0]) == 0:
                continue
            meanAlpha = np.mean(hpxMaps[nSlice,neighbours[:,k]][nonZeroPixels])
            stdAlpha = np.std(hpxMaps[nSlice,neighbours[:,k]][nonZeroPixels])
            if stdAlpha != 0.0 and hpxMaps[nSlice,k] != 0.0:
                hpMap[k] = (hpxMaps[nSlice,k] - meanAlpha)/stdAlpha
    else:
        hpMap = hpxMaps[nSlice,:]
    healpy.mollview(hpMap,cmap=cmap,cbar=False,\
        norm = norm,\
        hold=True,fig=fig,margins=margins,xsize=xsize,sub=0,bgcolor='white',\
        badcolor='white',min=vmin,max=vmax)
    ax.set_autoscale_on(True)
    healpy.graticule(color=guideColor)
    ax = plt.gca()
    # Hacky solution to make the boundary grey, since healpy hardcodes this:
    lines = ax.get_lines()
    for l in lines:
        if l.get_color() != guideColor:
            l.set_color(guideColor)
    if boundaryOff:
        # Very hacky, and liable to break if healpy changes, 
        # but not clear how else we would identify which line is the boundary...
        for l in range(20,len(lines)):
            lines[l].set_linestyle('None')
    # Colorbar:
    if showDeviation:
        cbar = plt.colorbar(sm, orientation="horizontal",
            pad=pad,\
            label='$A_{\\alpha}$ standard deviations from local mean',\
            shrink=shrink)
        cbar.ax.tick_params(axis='both',labelsize=fontsize)
        cbar.set_label(\
            label = '$A_{\\alpha}$ standard deviations from local mean',\
            fontsize = fontsize,fontfamily = fontname)
    else:
        cbar = plt.colorbar(sm, orientation="horizontal",
            pad=pad,label='$A_{\\alpha}$',shrink=shrink)
        cbar.ax.tick_params(axis='both',labelsize=fontsize)
        cbar.set_label(label = '$A_{\\alpha}$',fontsize = fontsize,\
            fontfamily = fontname)
    MInd = int(mBin/2)
    mInd = np.mod(mBin,2)
    if title is None:
        if np.isscalar(ns):
            nsName = str(snapNumList[ns])
        else:
            nsName = ",".join([str(snapNumList[ind]) for ind in ns])
        plt.title("$A_{\\alpha}$, Sample " + nsName + \
            ", " + "$" + str(rLimits[nSlice])  + \
            " \\leq r/\\mathrm{Mpc}h^{-1} < " \
            + str(rLimits[nSlice+1]) + "$"+ ", Bin " \
            + str(mBin) + "(" + \
            absRange[MInd] + ', ' + appRange[mInd] + ")")
    else:
        plt.title(title)
    if showClusters:
        anglesToPlotClusters = np.vstack((coordAbell.icrs.ra.value,
            coordAbell.icrs.dec.value)).T
        inRadialBin = np.where((abell_d[clusterInd] >= rLimits[nSlice]) & \
            (abell_d[clusterInd] < rLimits[nSlice+1]))[0]
        mollweideScatter(anglesToPlotClusters[\
            np.array(clusterInd)[inRadialBin],:],\
            color=clusterColour,s=30,
            marker='c',
            text=np.array(nameList)[inRadialBin],fontsize=fontsize,
            horizontalalignment=ha,
            verticalalignment=va,ax=ax,textPos=annotationPos,
            arrowprops= dict(arrowstyle = '->',shrinkA=5,
                color='k',shrinkB = 5,
                connectionstyle="arc3,rad=0."),
            arrowpad = 1)
    if includeZOA:
        # Zone of avoidance?
        lZOA = np.linspace(-np.pi,np.pi,nPointsZOA)
        zoaCentral = np.where((lZOA*180/np.pi > galacticCentreZOA[0]) & \
            (lZOA*180/np.pi < galacticCentreZOA[1]))
        bZOAupp = (np.pi*bRange[1]/180)*np.ones(lZOA.shape)
        bZOAupp[zoaCentral] = (np.pi*bRangeCentre[1]/180)
        bZOAlow = (np.pi*bRange[0]/180)*np.ones(lZOA.shape)
        bZOAlow[zoaCentral] = (np.pi*bRangeCentre[0]/180)
        zoaUppCoord = SkyCoord(l=lZOA*u.rad,b=bZOAupp*u.rad,frame='galactic')
        zoaLowCoord = SkyCoord(l=lZOA*u.rad,b=bZOAlow*u.rad,frame='galactic')
        raZOAUpp = zoaUppCoord.icrs.ra.value
        decZOAUpp = zoaUppCoord.icrs.dec.value
        raZOALow = zoaLowCoord.icrs.ra.value
        decZOALow = zoaLowCoord.icrs.dec.value
        MW = healpy.projector.MollweideProj()
        angleFactor = np.pi/180.0
        XYUpp = MW.ang2xy(theta = np.pi/2 - angleFactor*decZOAUpp,
            phi=angleFactor*raZOAUpp,lonlat=False)
        XYLow = MW.ang2xy(theta = np.pi/2 - angleFactor*decZOALow,
            phi=angleFactor*raZOALow,lonlat=False)
        # Roll around to prevent sudden jumps in the lines:
        rollNumUpp = np.where(XYUpp[0][0:-1]*XYUpp[0][1:] < 0)[0][0] + 1
        rollNumLow = np.where(XYLow[0][0:-1]*XYLow[0][1:] < 0)[0][0] + 1
        # mollweide boundary part of ZOA:
        zoaBoundUpp = MW.xy2ang(XYUpp[0][(rollNumUpp-1):(rollNumUpp+1)],
            XYUpp[1][(rollNumUpp-1):(rollNumUpp+1)])
        zoaBoundLow = MW.xy2ang(XYLow[0][(rollNumLow-1):(rollNumLow+1)],
            XYLow[1][(rollNumLow-1):(rollNumLow+1)])
        zoaBoundLeft = np.linspace(zoaBoundUpp[0,0],zoaBoundLow[0,0],\
            nPointsEdgeZOA)
        zoaBoundRight = np.linspace(zoaBoundLow[0,1],zoaBoundUpp[0,1],\
            nPointsEdgeZOA)
        leftXY = MW.ang2xy(zoaBoundLeft,(-np.pi -1e-2)*np.ones(zoaBoundLeft.shape))
        rightXY = MW.ang2xy(zoaBoundRight,np.pi*np.ones(zoaBoundRight.shape))
        #Polygon defining ZOA:
        polyX = np.hstack((leftXY[0],np.flip(np.roll(XYLow[0],
            -rollNumLow)),rightXY[0],np.roll(XYUpp[0],-rollNumUpp)))
        polyY = np.hstack((leftXY[1],np.flip(np.roll(XYLow[1],
            -rollNumLow)),rightXY[1],np.roll(XYUpp[1],-rollNumUpp)))
        polyXY = np.vstack((polyX,polyY)).T
        polygon = patches.Polygon(polyXY,fc='grey',ec='None',alpha=0.5,
            label='Zone of Avoidance')
        ax.add_patch(polygon)
    handles = []
    if (coordAbell is not None) and showClusters:
        clusterMarkerHandle = mlines.Line2D([],[],linestyle='',marker='o',
            mec=clusterColour,mfc=None,label='Cluster locations')
        handles.append(clusterMarkerHandle)
    if includeZOA:
        handles.append(polygon)
    bbox_to_anchor=(-0.1, -0.2)
    legLoc='lower left'
    ax.legend(handles=handles,frameon=False,
        prop={"size":7,"family":"serif"},
        loc=legLoc,bbox_to_anchor=bbox_to_anchor)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()

def computeAndPlotPPTProfile(ax,expectedLine,realisedLine,rBins,rescale=False,\
        style1 = '-',style2 = ':',title1 = "2M++ Galaxies",\
        title2 = "Posterior prediction",error=None,intervalColour='grey',\
        intervalLabel = "95% Poisson \nInterval",color1='k',color2='k',\
        density=True):
    if rescale:
        factorExp = expectedLine[-1]
        factorReal = realisedLine[-1]
        ylabel = "$n_{\\mathrm{gal}}(<r)/" + \
            "n_{\\mathrm{gal}}(<20\\mathrm{Mpc}h^{-1})$"
    else:
        factorExp = 1
        factorReal = 1
    nz1 = np.where(expectedLine > 0)[0]
    nz2 = np.where(realisedLine > 0)[0]
    h1 = ax.semilogy(binCentres(rBins)[nz1],expectedLine[nz1]\
        /factorExp,linestyle=style1,color=color1,\
        label=title1)
    h2 = ax.semilogy(binCentres(rBins)[nz2],realisedLine[nz2]\
        /factorReal,linestyle=style2,color=color2,\
        label = title2)
    if error is None:
        if density:
            bounds = (scipy.stats.poisson(realisedLine[nz2]*\
                (4*np.pi*rBins[1:][nz2]**3/3)).interval(0.95)/\
                (4*np.pi*rBins[1:][nz2]**3/3))/factorReal
        else:
            bounds = scipy.stats.poisson(realisedLine[nz2]).interval(0.95)
    else:
        # Standard deviation of the mean:
        bounds = [(realisedLine[nz2] - error[nz2])/factorReal,\
            (realisedLine[nz2] + error[nz2])/factorReal]
    h3 = ax.fill_between(binCentres(rBins)[nz2],bounds[0],bounds[1],\
        facecolor=intervalColour,\
        alpha = 0.5,label = intervalLabel)
    return [h1,h2,h3]

# Adjusts the formatting of a grid of plots:
def formatPlotGrid(ax,i,j,ylabelRow,ylabel,xlabelCol,xlabel,nRows,ylim=None,\
        fontsize=8,fontfamily='serif',nCols = 3,xlim=None,\
        logx=False,logy=False):
    if nRows < 2:
        if nCols > 1:
            axij = ax[j]
        else:
            axij = ax
    else:
        axij = ax[i,j]
    axij.set_ylim(ylim)
    axij.set_xlim(xlim)
    if logx:
        axij.set_xscale('log')
    if logy:
        axij.set_yscale('log')
    if j > 0:
        axij.yaxis.label.set_visible(False)
        axij.yaxis.set_major_formatter(NullFormatter())
        axij.yaxis.set_minor_formatter(NullFormatter())
    if ylabelRow is None:
        axij.set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontfamily)
    else:
        if (i == ylabelRow) and (j == 0):
            axij.set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontfamily)
    if i < nRows - 1:
        axij.xaxis.label.set_visible(False)
        axij.xaxis.set_major_formatter(NullFormatter())
        axij.xaxis.set_minor_formatter(NullFormatter())
    if xlabelCol is None:
        axij.set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontfamily)
    else:
        if (j == xlabelCol) and (i == nRows-1):
            axij.set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontfamily)


# Plot the Posterior Predictive Test for a set of clusters. That is, we compare
# the galaxy count from a survey (2M++) to the expected galaxy count given the
# inferred dark matter density field.
def plotPPTProfiles(expectedLine,realisedLine,title1 = "2M++ \nGalaxies",\
        title2 = "Posterior \nprediction",\
        style1 = '-',style2 = ':',intervalLabel = "95% Poisson \nInterval",\
        intervalColour='grey',fontfamily='serif',fontsize=8,\
        ylabel = '$n_{\\mathrm{gal}}(<r)$ $[h^{3}\\mathrm{Mpc}^{-3}]$',\
        xlabel = '$r$ $[\\mathrm{Mpc}h^{-1}]$',ylim=[0.01,2],\
        show=True,savename=None,title = None,showLegend=True,\
        titleSize = 12,nRows=3,nCols=3,legPos = [1,2],ylabelRow = 1,\
        xlabelCol = 1,rBins=None,clusterNames=None,rescale=False,\
        returnHandles=False,error=None,text=None,textPos=[0.1,0.1],\
        splitLegend=True,legLoc=[0.3,0.7],color1='k',color2='k',\
        ax=None,fig=None,density=True,textwidth=7.1014,width=1,height=1,\
        top=0.940,bottom=0.105,left=0.095,right=0.980,hspace=0.215,wspace=0.0,\
        intervalColour2='k',showPoissonRange=True,sigmaFactor=2,\
        showVariance=True):
    if rBins is None:
        rBins = np.linspace(0,20,21)
    if clusterNames is None:
        clusterNames = [str(k+1) for k in range(0,nRows*nCols)]
    if ax is None or fig is None:
        fig, ax = plt.subplots(nRows,nCols,\
            figsize=(width*textwidth,height*textwidth))
    for l in range(0,nCols*nRows):
        i = int(l/nCols)
        j = l - nCols*i
        if error is None:
            errorToPass = error
        else:
            errorToPass = error[:,l]
        if nCols == 1 and nRows == 1:
            axij = ax
        else:
            axij = ax[i,j]
        if len(realisedLine[:,l].shape) == 1:
            [h1,h2,h3] = computeAndPlotPPTProfile(axij,expectedLine[:,l],\
                realisedLine[:,l],rBins,rescale=rescale,style1 = style1,\
                style2 = style2,title1 = title1,title2 = title2,\
                error=errorToPass,intervalColour=intervalColour,\
                intervalLabel = intervalLabel,color1=color1,color2=color2,\
                density=density)
            h4 = None
        else:
            # Have multiple lines to plot. We handle this by plotting all
            # of them, or an average
            if rescale:
                factorExp = expectedLine[-1]
                factorReal = realisedLine[-1]
                ylabel = "$n_{\\mathrm{gal}}(<r)/" + \
                    "n_{\\mathrm{gal}}(<20\\mathrm{Mpc}h^{-1})$"
            else:
                factorExp = 1
                factorReal = 1
            nz1 = np.where(expectedLine[:,l] > 0)[0]
            nz2 = np.where(realisedLine[:,l] > 0)
            h2 = axij.semilogy(binCentres(rBins)[:,None],\
                realisedLine[:,l]/factorReal,\
                linestyle=style2,color=color2,label = title2)
            std = np.std(realisedLine[:,l]/factorReal,1)
            mean = np.mean(realisedLine[:,l]/factorReal,1)
            bounds = [mean + sigmaFactor*std,mean - sigmaFactor*std]
            nz3 = np.where(mean > 0)[0]
            if error is None:
                if density:
                    bounds2 = (scipy.stats.poisson(mean[nz3]*\
                        (4*np.pi*rBins[1:][nz3]**3/3)).interval(0.95)/\
                        (4*np.pi*rBins[1:][nz3]**3/3))/factorReal
                else:
                    bounds2 = scipy.stats.poisson(mean[nz3]).interval(0.95)
            else:
                # Standard deviation of the mean:
                bounds2 = [(mean[nz3] - error[nz3])/factorReal,\
                    (mean[nz3] + error[nz3])/factorReal]
            if showPoissonRange:
                h3 = axij.fill_between(binCentres(rBins)[nz3],bounds2[0],\
                    bounds2[1],facecolor=intervalColour,\
                    alpha = 0.5,label = intervalLabel)
            else:
                h3 = None
            if showVariance:
                h4 = axij.fill_between(binCentres(rBins)[nz3],bounds[0][nz3],\
                    bounds[1][nz3],facecolor=intervalColour2,\
                    alpha = 0.5,label = "Samples variation")
            else:
                h4 = None
            h1 = axij.semilogy(binCentres(rBins)[nz1],expectedLine[:,l][nz1]\
                /factorExp,linestyle=style1,color=color1,\
                label=title1)
        axij.set_title(clusterNames[l][0],fontsize=fontsize,\
            fontfamily=fontfamily)
        formatPlotGrid(ax,i,j,ylabelRow,ylabel,xlabelCol,xlabel,nRows,ylim,\
            nCols = nCols,fontsize=fontsize)
        axij.tick_params(axis='both', which='major', labelsize=fontsize)
        axij.tick_params(axis='both', which='minor', labelsize=fontsize)
        if text is not None and textPos is not None:
            axij.text(textPos[0]*axij.get_xlim()[1],\
                axij.get_ylim()[0]*10**(\
                textPos[1]*np.log10(\
                axij.get_ylim()[1]/axij.get_ylim()[0])),\
                text[l],fontsize=fontsize,fontfamily=fontfamily)
    plt.subplots_adjust(wspace=wspace,hspace=hspace,top=top,bottom=bottom,\
        left=left,right=right)
    if showLegend:
        if splitLegend:
            handleList = [[] for k in range(0,nRows)]
            handleList[np.mod(0,nRows)].append(h1[0])
            handleList[np.mod(1,nRows)].append(h2[0])
            if h3 is not None:
                handleList[np.mod(2,nRows)].append(h3)
            if h4 is not None:
                handleList[np.mod(2,nRows)].append(h4)
            for k in range(0,nRows):
                if len(handleList[k]) > 0:
                    if nCols == 1 and nRows == 1:
                        axij = ax
                    else:
                        axij = ax[k,legPos[1]]
                    axij.legend(loc=legLoc,handles=handleList[k],\
                        prop={"size":fontsize,"family":fontfamily},\
                        frameon=False)
        else:
            if nCols == 1 and nRows == 1:
                axij = ax
            else:
                axij = ax[legPos[0],legPos[1]]
            axij.legend(\
                prop={"size":fontsize,"family":fontfamily},frameon=False)
    if title is not None:
        fig.suptitle(title, fontsize=titleSize,fontfamily=fontfamily)
    plt.subplots_adjust(top=top,bottom=bottom,left=left,right=right)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnHandles:
        return [fig,ax,h1,h2,h3]

def plotPPTProfileProgressive(expectedLine,realisedLine,clusterName=None,\
        nCols = 3,sampleNumList = None,returnAx=False,\
        ylabel = '$n_{\\mathrm{gal}}(<r)$ $[h^{3}\\mathrm{Mpc}^{-3}]$',\
        xlabel = '$r$ $[\\mathrm{Mpc}h^{-1}]$',ylim=[0.01,2],\
        title = "Simulation Density (Eulerian space) " + \
        "vs 2M++ Galaxy number density",\
        title1 = "2M++ Galaxies",fontsize=8,fontfamily='serif',\
        title2 = "Posterior prediction",\
        intervalLabel = "95% Poisson \nInterval",\
        titleSize = 12,legPos = None,ylabelRow = None,\
        xlabelCol = None,error=None,rBins=None,\
        wspace=0,hspace=0.2,show=True,savename=None,rescale=False,\
        style1 = '-',style2 = ':',color1='k',color2='k',intervalColour='grey',\
        bbox_to_anchor = None):
    nsamples = realisedLine.shape[1]
    if rBins is None:
        rBins = np.linspace(0,20,21)
    if clusterName is None:
        clusterName = "Cluster"
    if sampleNumList is None:
        sampleNumList = np.arange(0,nsamples)
    nRows = int(np.ceil(nsamples/3))
    if nRows < 2:
        nCols = nsamples
    if ylabelRow is None:
        ylabelRow = int(nRows/2)
    if xlabelCol is None:
        xlabelCol = int(nCols/2)
    if legPos is None:
        legPos = [int(nRows/2),nCols-1]
    fig, ax = plt.subplots(nRows,nCols)
    for l in range(0,nRows*nCols):
        i = int(l/nCols)
        j = l - nCols*i
        if nRows < 2:
            axij = ax[j]
        else:
            axij = ax[i,j]
        if l < nsamples:
            if error is None:
                errorToPass = error
            else:
                errorToPass = error[:,l]
            [h1,h2,h3] = computeAndPlotPPTProfile(axij,expectedLine,\
                realisedLine[:,l],rBins,rescale=rescale,style1 = style1,\
                style2 = style2,title1 = title1,title2 = title2,\
                error=errorToPass,intervalColour=intervalColour,\
                intervalLabel = intervalLabel,color1=color1,color2=color2)
            axij.set_title("Sample " +str(sampleNumList[l]),\
                fontsize=fontsize,fontfamily=fontfamily)
        formatPlotGrid(ax,i,j,ylabelRow,ylabel,xlabelCol,xlabel,nRows,ylim)
    plt.subplots_adjust(wspace=wspace,hspace=hspace)
    if nRows < 2:
        axij = ax[legPos[1]]
    else:
        axij = ax[legPos[0],legPos[1]]
    axij.legend(handles = [h1[0],h2[0],h3],\
        prop={"size":fontsize,"family":fontfamily},frameon=False,\
        bbox_to_anchor = bbox_to_anchor)
    fig.suptitle("PPT in different samples, " + clusterName,\
        fontsize=titleSize,fontfamily=fontfamily)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnAx:
        return ax

def dereferenceAxis(ax,i,j,nRows,nCols):
    if nRows == 1:
        if nCols == 1:
            axij = ax
        else:
            axij = ax[j]
    else:
        if nCols == 1:
            axij = ax[i]
        else:
            axij = ax[i,j]
    return axij


# Compare 1D dark matter density profiles for a set of clusters.
def compareDensities(rBins,densities,clusterNames,labels = None,styles= None,\
        fontsize = 8, colors = None,fontfamily = "serif",ylabel="$\\rho/\\bar{\\rho}$",\
        xlabel = '$r$ $[\\mathrm{Mpc}h^{-1}]$',nRows=3,nCols=3,ylim=[1,500],\
        title = "BORG vs Simulation Density",wspace=0,hspace=0.2,show=True,\
        savename = None,ylabelRow=None,xlabelCol = None,legPos = [1,2],\
        stdErrorDensities = None):
    if labels is None:
        labels = ["Line " + str(k) for k in range(0,len(densities))]
    if colors is None:
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',\
        'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    if type(colors) == type(""):
        colors = [colors for k in range(0,len(densities))]
    if styles is None:
        if len(densities) <= 4:
            styles = ['-',':','--','-.']
        else:
            styles = ['-' for k in range(0,len(densities))]
    if ylabelRow is None:
        ylabelRow = int(nRows/2)
    if xlabelCol is None:
        xlabelCol = int(nCols/2)
    fig, ax = plt.subplots(nRows,nCols)
    for l in range(0,nRows*nCols):
        i = int(l/nCols)
        j = l - nRows*i
        axij = dereferenceAxis(ax,i,j,nRows,nCols)
        for m in range(0,len(densities)):
            nz = np.where(densities[m][:,l] > 0)[0]
            axij.semilogy(binCentres(rBins)[nz],densities[m][nz,l],\
                color=colors[m],linestyle = styles[m],label=labels[m])
            if stdErrorDensities is not None:
                axij.fill_between(binCentres(rBins)[nz],\
                    densities[m][nz,l] - stdErrorDensities[m][nz,l],\
                    densities[m][nz,l] + stdErrorDensities[m][nz,l],alpha=0.5,\
                    color=colors[m])
        axij.set_title(clusterNames[l],fontsize=fontsize,\
            fontfamily=fontfamily)
        formatPlotGrid(ax,i,j,ylabelRow,ylabel,xlabelCol,xlabel,nRows,ylim,\
            nCols = nCols)
    plt.subplots_adjust(wspace=wspace,hspace=hspace)
    if np.any(legPos >= [nRows,nCols]):
        axij = dereferenceAxis(ax,0,0,nRows,nCols)
    else:
        axij = dereferenceAxis(ax,legPos[0],legPos[1],nRows,nCols)
    axij.legend(prop={"size":fontsize,"family":fontfamily},frameon=False)
    plt.suptitle(title)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()

def computeMeanHMF(haloMasses,massLower=1e12,massUpper = 1e16,nBins=31):
    nSamples = len(haloMasses)
    massBins = 10**np.linspace(np.log10(massLower),np.log10(massUpper),nBins)
    binLists = [plot_utilities.binValues(hnmasses,massBins) \
        for hnmasses in haloMasses]
    sigmaBins = np.std([bins[1] for bins in binLists],0)\
        /np.sqrt(len(haloMasses))
    noInBins = np.mean([bins[1] for bins in binLists],0)
    return [noInBins,sigmaBins]

def plotAverageHMF(haloMasses,boxsize,h=0.6766,omegaM0 = 0.3111,\
        massLower=1e12,massUpper = 1e16,nBins=31,ylim=[1e-1,1e5],\
        ax=None,Delta=200,marker='x',color=None,\
        linestyle='',tmfcolor='r',tmfstyle=':',mass_function="Tinker",\
        label='Halos',labelLine='TMF prediction',\
        poisson_interval=0.95,volSim=None,\
        fill_alpha = 0.5,showLegend=True,legendFontsize=10,font="serif",\
        fontsize=8,labelRight=True,gridcolor='grey',\
        gridstyle=':',gridalpha=0.5,show=True,returnAx=False,\
        xlabel="Mass bin centre [$M_{\odot}h^{-1}$]",\
        ylabel="Number of Halos",title="Halo Mass function",\
        legendLoc='lower left',bbox_to_anchor=None,\
        savename = None,showTheory=True,binError = "poisson",\
        sigma8=0.8102,delta_wrt='SOCritical',\
        errorLabel = None,density=False):
    [dndm,m] = cosmology.TMF_from_hmf(massLower,massUpper,\
        h=h,Om0=omegaM0,Delta=Delta,delta_wrt=delta_wrt,\
        mass_function=mass_function,sigma8=sigma8)
    if volSim is None:
        volSim = boxsize**3
    nSamples = len(haloMasses)
    massBins = 10**np.linspace(np.log10(massLower),np.log10(massUpper),nBins)
    n = cosmology.dndm_to_n(m,dndm,massBins)
    binLists = [plot_utilities.binValues(hnmasses,massBins) \
        for hnmasses in haloMasses]
    noInBins = np.mean([bins[1] for bins in binLists],0)
    if binError == "standard":
        sigmaBins = np.std([bins[1] for bins in binLists],0)\
            /np.sqrt(len(haloMasses))
    elif binError == "poisson":
        #sigmaBins =  np.array(scipy.stats.poisson(np.mean(\
        #    [bins[1] for bins in binLists],0)).interval(\
        #    poisson_interval))
        alphaO2 = (1.0 - poisson_interval)/2.0
        nCount = np.sum([bins[1] for bins in binLists],0)
        sigmaBins = np.abs(np.array([scipy.stats.chi2.ppf(\
            alphaO2,2*nCount)/2,\
            scipy.stats.chi2.ppf(1.0 - alphaO2,2*(nCount+1))/2])/len(haloMasses) - noInBins)
    elif binError == "gaussian":
        zalphaO2 = scipy.stats.norm().interval(poisson_interval)[1]
        sigmaBins = zalphaO2*np.sqrt(np.mean(\
            [bins[1] for bins in binLists],0)/len(haloMasses))
    else:
        raise Exception("Unknown bin error requested.")
    massBinCentres = plot_utilities.binCentres(massBins)
    bounds = scipy.stats.poisson(n*volSim*nSamples).interval(poisson_interval)
    if density:
        noInBins /= volSim
        sigmaBins /= volSim
        n /= volSim
    if ax is None:
        fig, ax = plt.subplots()
    h1 = ax.errorbar(massBinCentres,noInBins,yerr=sigmaBins,\
        marker=marker,linestyle=linestyle,label=label,color=color)
    if showTheory:
        h2 = ax.plot(massBinCentres,n*volSim,tmfstyle,label=labelLine,\
            color=tmfcolor)
        if errorLabel is None:
            errorLabel = ("%.2g" % (100*poisson_interval)) + \
                '% Poisson interval'
        h3 = ax.fill_between(massBinCentres,
            bounds[0]/nSamples,bounds[1]/nSamples,
            facecolor=tmfcolor,alpha=fill_alpha,
            interpolate=True,label=errorLabel)
    else:
        h2 = None
        h3 = None
    if showLegend:
        ax.legend(prop={"size":legendFontsize,"family":font},
        loc=legendLoc,frameon=False,bbox_to_anchor=bbox_to_anchor)
    ax.set_title(title,fontsize=fontsize,fontfamily=font)
    ax.set_xlabel(xlabel,fontsize=fontsize,fontfamily=font)
    ax.set_ylabel(ylabel,fontsize=fontsize,fontfamily=font)
    ax.tick_params(axis='both',labelsize=fontsize,labelright=labelRight,\
        right=labelRight)
    ax.tick_params(axis='both',which='minor',bottom=True,labelsize=fontsize)
    ax.tick_params(axis='y',which='minor')
    ax.yaxis.grid(color=gridcolor,linestyle=gridstyle,alpha=gridalpha)
    ax.set_ylim(ylim)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnAx:
        return [ax,h1,h2,h3]

def plotHMFAMFComparison(constrainedHaloMasses512Old,deltaListMeanOld,\
        deltaListErrorOld,comparableHaloMassesOld,\
        constrainedAntihaloMasses512Old,comparableAntihaloMassesOld,\
        constrainedHaloMasses512New,deltaListMeanNew,deltaListErrorNew,\
        comparableHaloMassesNew,constrainedAntihaloMasses512New,\
        comparableAntihaloMassesNew,referenceSnap,referenceSnapOld,\
        savename = None,ylabelStartOld = 'Old reconstruction',\
        ylabelStartNew = 'New reconstruction',fontsize=8,legendFontsize=8,\
        textwidth=7.1014,nMassBins = 11,mLower = 1e14,\
        mUpper = 3e15,mass_function = 'Tinker',showTheory=False,\
        xlim = (1.5e13,3e15),ylim = (1e-1,1e2),\
        legendLoc = 'upper right',showResLimit = False,density = True,\
        useRandom = True,boxsize=677.7,h=0.6766,omegaM0 = 0.3111,\
        volSim=4*np.pi*135**3/3,show=True,poisson_interval=0.95,\
        haloType = ['SOCritical','SOCritical'],binError="poisson",\
        sigma8List = [0.8288,0.8102],font='serif'):
    if volSim is None:
        volSim = boxsize**3
    if density:
        ylimHMF = np.array(ylim)/volSim
        factor = 1.0/volSim
    else:
        ylimHMF = np.array(ylim)
        factor = 1.0
    massBinCentres = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),\
        nMassBins))
    # Cosmological parameters for the two runs:
    omegaList = [referenceSnapOld.properties['omegaM0'],\
        referenceSnap.properties['omegaM0']]
    hList = [referenceSnapOld.properties['h'],\
        referenceSnap.properties['h']]
    fig, ax = plt.subplots(2,2,figsize=(textwidth,textwidth))
    for i in range(0,2):
        if i == 0:
            # Top row old run (PM10):
            nSamples = len(comparableHaloMassesOld)
            hmfUnderdense = [computeMeanHMF(hmasses,massLower = mLower,\
                massUpper = mUpper,nBins = nMassBins) \
                for hmasses in comparableHaloMassesOld]
            hmfUnderdensePoisson = np.array(scipy.stats.poisson(\
                np.sum(np.array([hmf[0] \
                for hmf in hmfUnderdense]),0)).interval(poisson_interval))/\
                nSamples
            hmfUnderdenseMean = np.mean(np.array([hmf[0] \
                for hmf in hmfUnderdense]),0)
            massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),\
                nMassBins))
            massBinCentres = binCentres(massBins)
            amfUnderdense = [computeMeanHMF(hmasses,massLower = mLower,\
                massUpper = mUpper,nBins = nMassBins) \
                for hmasses in comparableAntihaloMassesOld]
            amfUnderdensePoisson = np.array(scipy.stats.poisson(\
                np.sum(np.array([hmf[0] \
                for hmf in amfUnderdense]),0)).interval(poisson_interval))/\
                nSamples
            amfUnderdenseMean = np.mean(np.array([hmf[0] \
                for hmf in amfUnderdense]),0)
            massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),\
                nMassBins))
            massBinCentres = binCentres(massBins)
            ylabelStart = ylabelStartOld
            constrainedHaloMasses512 = constrainedHaloMasses512Old
            constrainedAntihaloMasses512 = constrainedAntihaloMasses512Old
            deltaListMean = deltaListMeanOld
            deltaListError = deltaListErrorOld
        if i == 1:
            # Bottom row new run (COLA 20):
            nSamples = len(comparableHaloMassesNew)
            hmfUnderdense = [computeMeanHMF(hmasses,massLower = mLower,\
                massUpper = mUpper,nBins = nMassBins) \
                for hmasses in comparableHaloMassesNew]
            hmfUnderdensePoisson = np.array(scipy.stats.poisson(\
                np.sum(np.array([hmf[0] \
                for hmf in hmfUnderdense]),0)).interval(poisson_interval))/\
                nSamples
            hmfUnderdenseMean = np.mean(np.array([hmf[0] \
                for hmf in hmfUnderdense]),0)
            massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),\
                nMassBins))
            massBinCentres = binCentres(massBins)
            amfUnderdense = [computeMeanHMF(hmasses,massLower = mLower,\
                massUpper = mUpper,nBins = nMassBins) \
                for hmasses in comparableAntihaloMassesNew]
            amfUnderdensePoisson = np.array(scipy.stats.poisson(\
                np.sum(np.array([hmf[0] \
                for hmf in amfUnderdense]),0)).interval(poisson_interval))/\
                nSamples
            amfUnderdenseMean = np.mean(np.array([hmf[0] \
                for hmf in amfUnderdense]),0)
            massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),\
                nMassBins))
            massBinCentres = binCentres(massBins)
            ylabelStart = ylabelStartNew
            constrainedHaloMasses512 = constrainedHaloMasses512New
            constrainedAntihaloMasses512 = constrainedAntihaloMasses512New
            deltaListMean = deltaListMeanNew
            deltaListError = deltaListErrorNew
        for j in range(0,2):
            axij = ax[i,j]
            if i == 0 and j == 1:
                theoryLabel = 'Tinker Mass Function \n prediction'
            else:
                theoryLabel = None
            if density:
                ylabel = ylabelStart + \
                    '\nNumber Density ($h^{3}\\mathrm{Mpc}^{-3}$)'
            else:
                ylabel = ylabelStart + '\nNumber of Halos or Antihalos'
            if j == 0:
                [axij,h1,h2,h3] = plotAverageHMF(constrainedHaloMasses512,\
                    boxsize,h=hList[i],omegaM0=omegaList[i],\
                    volSim = volSim,show=False,ax=axij,\
                    labelLine = None,\
                    label = "Constrained",delta_wrt=haloType[i],\
                    showTheory=showTheory,ylim=ylimHMF,\
                    mass_function=mass_function,\
                    showLegend=False,returnAx = True,\
                    tmfcolor=seabornColormap[1],\
                    title = "Halo mass function",fill_alpha=0.25,\
                    errorLabel='Tinker Mass Function',binError=binError,\
                    ylabel = ylabel,xlabel = 'Mass ($M_{\\odot}h^{-1}$)',\
                    fontsize=fontsize,labelRight=False,sigma8=sigma8List[i],\
                    nBins=nMassBins,massLower=mLower,massUpper = mUpper,\
                    density=density,poisson_interval=poisson_interval)
                h4 = axij.fill_between(massBinCentres,\
                    hmfUnderdensePoisson[0]*factor,\
                    hmfUnderdensePoisson[1]*factor,\
                    label = 'Unconstrained, \n' + \
                    '$' + ("%.2g" % (deltaListMean - deltaListError)) + \
                    ' \\leq \\delta < ' + \
                    ("%.2g" % (deltaListMean + deltaListError)) + \
                    '$',color=seabornColormap[0],alpha=0.5)
                if showResLimit:
                    h5 = axij.axvline(mUnitList[i]*100*8,linestyle=':',\
                        color='grey',label = 'Resolution limit ($256^3$)')
                    h6 = axij.axvline(mUnitList[i]*100,linestyle='--',\
                        color='grey',label = 'Resolution limit ($512^3$)')
                else:
                    h5 = None
                    h6 = None
                h7 = axij.plot(massBinCentres,hmfUnderdenseMean*factor,\
                    linestyle=':',color=seabornColormap[0])
                axij.set_xlim(xlim)
                axij.set_ylim(ylimHMF)
                if i == 0:
                    handleList = [h1,h4]
                    if showResLimit and j == 0:
                        handleList.append(h5)
                    if j == 1 and showTheory:
                        handleList.append(h3)
                else:
                    handleList = [h1,h4]
                    if showResLimit and j == 0:
                        handleList.append(h6)
                if j == 1:
                    axij.legend(handles = handleList,\
                        prop={"size":legendFontsize,"family":font},
                                    loc=legendLoc,frameon=False,\
                                    bbox_to_anchor=bbox_to_anchor)
            else:
                [axij,h1,h2,h3] = plotAverageHMF(\
                    constrainedAntihaloMasses512,\
                    boxsize,h=hList[i],omegaM0=omegaList[i],\
                    volSim = volSim,show=False,ax=axij,\
                    labelLine = None,mass_function=mass_function,\
                    label = "Constrained",delta_wrt=haloType[i],\
                    showTheory=showTheory,ylim=ylimHMF,\
                    ylabel=ylabel,xlabel = 'Mass ($M_{\\odot}h^{-1}$)',\
                    showLegend=False,returnAx = True,\
                    tmfcolor=seabornColormap[1],\
                    title = "Antihalo mass function",fill_alpha=0.25,\
                    errorLabel='Tinker Mass Function',binError=binError,\
                    fontsize=fontsize,labelRight=False,sigma8=sigma8List[i],\
                    nBins=nMassBins,massLower=mLower,massUpper = mUpper,\
                    density=density,poisson_interval=poisson_interval)
                h4 = axij.fill_between(massBinCentres,\
                    amfUnderdensePoisson[0]*factor,\
                    amfUnderdensePoisson[1]*factor,\
                    label = 'Unconstrained, \n' + \
                    '($' + ("%.2g" % (deltaListMean - deltaListError)) + \
                    ' \\leq \\delta < ' + \
                    ("%.2g" % (deltaListMean + deltaListError)) + '$)',\
                    color=seabornColormap[0],alpha=0.5)
                if showResLimit:
                    h5 = axij.axvline(mUnitList[i]*100*8,linestyle=':',\
                        color='grey',\
                        label = 'Resolution limit ($256^3$)')
                    h6 = axij.axvline(mUnitList[i]*100,linestyle='--',\
                        color='grey',\
                        label = 'Resolution limit ($512^3$)')
                h7 = axij.plot(massBinCentres,amfUnderdenseMean*factor,\
                    linestyle=':',color=seabornColormap[0])
                axij.set_xlim(xlim)
                axij.set_ylim(ylimHMF)
                if i == 0:
                    handleList = [h1,h4]
                    if showResLimit and j == 0:
                        handleList.append(h5)
                    if j == 1 and showTheory:
                        handleList.append(h3)
                else:
                    handleList = [h1,h4]
                    if showResLimit and j == 0:
                        handleList.append(h6)
                if j == 1:
                    axij.legend(handles = handleList,\
                        prop={"size":legendFontsize,"family":font},
                                    loc=legendLoc,frameon=False,\
                                    bbox_to_anchor=None)
            if i == 0:
                axij.xaxis.label.set_visible(False)
                axij.xaxis.set_major_formatter(NullFormatter())
                axij.xaxis.set_minor_formatter(NullFormatter())
            if j == 0:
                axij.xaxis.get_major_ticks()[-2].set_visible(False)
            if i == 1:
                axij.title.set_visible(False)
                axij.yaxis.get_major_ticks()[-2].set_visible(False)
            if j == 1:
                axij.yaxis.label.set_visible(False)
                axij.yaxis.set_major_formatter(NullFormatter())
                axij.yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()

def plotHMFAMFUnderdenseComparison(\
        constrainedHaloMasses512,deltaListMean,deltaListError,\
        comparableHaloMasses,constrainedAntihaloMasses512,\
        comparableAntihaloMasses,centralHalos,centralAntihalos,\
        centralHaloMasses,centralAntihaloMasses,\
        savename = None,fontsize=8,legendFontsize=8,\
        textwidth=7.1014,nMassBins = 11,mLower = 1e14,\
        mUpper = 3e15,mass_function = 'Tinker',showTheory=False,\
        xlim = (1.5e13,3e15),ylim = (1e-1,1e2),\
        legendLoc = 'upper right',showResLimit = False,density = True,\
        useRandom = True,boxsize=677.7,h=0.6766,omegaM0 = 0.3111,\
        volSim=4*np.pi*135**3/3,show=True,poisson_interval=0.95,\
        haloType = 'SOCritical',binError="poisson",\
        sigma8=0.8102,font='serif',mUnit=None,resolution=512,\
        meanDensityMethod="selection",meanThreshold=0.02):
    fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
    massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),\
        nMassBins))
    massBinCentres = binCentres(massBins)
    if mUnit is None:
        mUnit = omegaM0*2.7754e11*(boxsize/resolution)**3
    if density:
        ylimHMF = np.array(ylim)/volSim
        factor = 1.0/volSim
    else:
        ylimHMF = np.array(ylim)
        factor = 1.0
    nSamples = len(comparableHaloMasses)
    # Bottom row new run (COLA 20):
    hmfUnderdense = [computeMeanHMF(hmasses,massLower = mLower,\
        massUpper = mUpper,nBins = nMassBins) \
        for hmasses in comparableHaloMasses]
    hmfUnderdensePoisson = np.array(scipy.stats.poisson(\
        np.sum(np.array([hmf[0] \
        for hmf in hmfUnderdense]),0)).interval(poisson_interval))/\
        nSamples
    hmfUnderdenseMean = np.mean(np.array([hmf[0] \
        for hmf in hmfUnderdense]),0)
    massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),\
        nMassBins))
    massBinCentres = binCentres(massBins)
    amfUnderdense = [computeMeanHMF(hmasses,massLower = mLower,\
        massUpper = mUpper,nBins = nMassBins) \
        for hmasses in comparableAntihaloMasses]
    amfUnderdensePoisson = np.array(scipy.stats.poisson(\
        np.sum(np.array([hmf[0] \
        for hmf in amfUnderdense]),0)).interval(poisson_interval))/\
        nSamples
    amfUnderdenseMean = np.mean(np.array([hmf[0] \
        for hmf in amfUnderdense]),0)
    massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),\
        nMassBins))
    massBinCentres = binCentres(massBins)
    for j in range(0,2):
        axij = ax[j]
        if j == 1:
            theoryLabel = 'Tinker Mass Function \n prediction'
        else:
            theoryLabel = None
        if j == 0:
            [axij,h1,h2,h3] = plotAverageHMF(constrainedHaloMasses512,\
                boxsize,h=h,omegaM0=omegaM0,\
                volSim = volSim,show=False,ax=axij,\
                labelLine = None,\
                label = "Constrained Simulation",delta_wrt=haloType,\
                showTheory=showTheory,ylim=ylimHMF,mass_function=mass_function,\
                showLegend=False,returnAx = True,tmfcolor=seabornColormap[1],\
                title = "Halo mass function",fill_alpha=0.25,\
                errorLabel='Tinker Mass Function',binError=binError,\
                ylabel = 'Number density ' + \
                '($h^3\\mathrm{Mpc}^{-3}$)',\
                fontsize=fontsize,labelRight=False,sigma8=sigma8,\
                nBins=nMassBins,massLower=mLower,massUpper = mUpper,\
                density = density,xlabel='Mass ($M_{\\odot}h^{-1}$)')
            if useRandom:
                if meanDensityMethod == "central":
                    hmfTheoryMean = computeMeanHMF(centralHaloMasses,\
                        massLower = mLower,massUpper = mUpper,nBins = nMassBins)
                    hmfTheoryPoisson = np.array(scipy.stats.poisson(\
                        hmfTheoryMean[0]).interval(poisson_interval))
                else:
                    hmfTheory = [computeMeanHMF(masses,\
                        massLower = mLower,massUpper = mUpper,\
                        nBins = nMassBins) \
                        for masses in centralHaloMasses]
                    hmfTheoryMean = np.mean(np.array([hmf[0] \
                        for hmf in hmfTheory]),0)
                    hmfTheoryPoisson = np.array(scipy.stats.poisson(\
                        np.sum(np.array([hmf[0] \
                        for hmf in hmfTheory]),0)).interval(poisson_interval))/\
                        len(centralHaloMasses)
                h3 = axij.fill_between(massBinCentres,\
                    hmfTheoryPoisson[0]*factor,\
                    hmfTheoryPoisson[1]*factor,\
                    label = 'Unconstrained, \n (random density)',\
                    color=seabornColormap[1],alpha=0.5)
            h4 = axij.fill_between(massBinCentres,\
                hmfUnderdensePoisson[0]*factor,\
                hmfUnderdensePoisson[1]*factor,\
                label = 'Unconstrained HMFs, \n' + \
                '($' + ("%.2g" % (deltaListMean - deltaListError)) + \
                ' \\leq \\delta < ' + \
                ("%.2g" % (deltaListMean + deltaListError)) + \
                '$)',color=seabornColormap[0],alpha=0.5)
            if showResLimit:
                h5 = axij.axvline(mUnit*100*8,linestyle=':',\
                    color='grey',label = 'Resolution limit ($256^3$)')
                h6 = axij.axvline(mUnit*100,linestyle='--',\
                    color='grey',label = 'Resolution limit ($512^3$)')
            else:
                h5 = None
                h6 = None
            h7 = axij.plot(massBinCentres,hmfUnderdenseMean*factor,\
                linestyle=':',color=seabornColormap[0])
            axij.set_xlim(xlim)
            axij.set_ylim(ylimHMF)
            handleList = [h1,h4]
            if showResLimit and j == 0:
                handleList.append(h5)
            if j == 1 and (showTheory or density):
                handleList.append(h3)
            else:
                handleList = [h1,h4]
                if showResLimit and j == 0:
                    handleList.append(h6)
        else:
            [axij,h1,h2,h3] = plotAverageHMF(constrainedAntihaloMasses512,\
                boxsize,h=h,omegaM0=omegaM0,\
                volSim = volSim,show=False,ax=axij,\
                labelLine = None,mass_function=mass_function,\
                label = "Constrained Simulation",delta_wrt=haloType,\
                showTheory=showTheory,ylim=ylimHMF,\
                ylabel='Number of Antihalos',\
                showLegend=False,returnAx = True,tmfcolor=seabornColormap[1],\
                title = "Antihalo mass function",fill_alpha=0.25,\
                errorLabel='Tinker Mass Function',binError=binError,\
                fontsize=fontsize,labelRight=False,sigma8=sigma8,\
                nBins=nMassBins,massLower=mLower,massUpper = mUpper,\
                density=density,xlabel='Mass ($M_{\\odot}h^{-1}$)')
            if useRandom:
                if meanDensityMethod == "central":
                    amfTheoryMean = computeMeanHMF(centralAntihaloMasses,\
                        massLower = mLower,massUpper = mUpper,nBins = nMassBins)
                    amfTheoryPoisson = np.array(scipy.stats.poisson(\
                        amfTheoryMean[0]).interval(poisson_interval))
                else:
                    amfTheory = [computeMeanHMF(masses,\
                        massLower = mLower,massUpper = mUpper,\
                        nBins = nMassBins) \
                        for masses in centralAntihaloMasses]
                    amfTheoryMean = np.mean(np.array([amf[0] \
                        for amf in amfTheory]),0)
                    amfTheoryPoisson = np.array(scipy.stats.poisson(\
                        np.sum(np.array([amf[0] \
                        for amf in amfTheory]),0)).interval(poisson_interval))/\
                        len(centralAntihaloMasses)
                h3 = axij.fill_between(massBinCentres,\
                    amfTheoryPoisson[0]*factor,\
                    amfTheoryPoisson[1]*factor,\
                    label = 'Unconstrained, \n (random density)',\
                    color=seabornColormap[1],alpha=0.5)
            h4 = axij.fill_between(massBinCentres,\
                amfUnderdensePoisson[0]*factor,\
                amfUnderdensePoisson[1]*factor,\
                label = 'Unconstrained, \n' + \
                '($' + ("%.2g" % (deltaListMean - deltaListError)) + \
                ' \\leq \\delta < ' + \
                ("%.2g" % (deltaListMean + deltaListError)) + '$)',\
                color=seabornColormap[0],alpha=0.5)
            if showResLimit:
                h5 = axij.axvline(mUnit*100*8,linestyle=':',\
                    color='grey',label = 'Resolution limit ($256^3$)')
                h6 = axij.axvline(mUnit*100,linestyle='--',\
                    color='grey',label = 'Resolution limit ($512^3$)')
            h7 = axij.plot(massBinCentres,amfUnderdenseMean*factor,\
                linestyle=':',color=seabornColormap[0])
            axij.set_xlim(xlim)
            axij.set_ylim(ylimHMF)
            handleList = [h1,h4]
            if showResLimit and j == 0:
                handleList.append(h5)
            if j == 1 and (showTheory or density):
                handleList.append(h3)
            else:
                handleList = [h1,h4]
                if showResLimit and j == 0:
                    handleList.append(h6)
            axij.legend(handles = handleList,\
                prop={"size":legendFontsize,"family":font},
                            loc=legendLoc,frameon=False)
        if j == 0:
            axij.xaxis.get_major_ticks()[-2].set_visible(False)
        if j == 1:
            axij.yaxis.label.set_visible(False)
            axij.yaxis.set_major_formatter(NullFormatter())
            axij.yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()


# Void profiles plot for the voids paper:
def plotVoidProfilesPaper(rBinStackCentres,nbarjMean,sigmaMean,nbar,\
        nbarjUnMean,sigmaUnMean,sigmaUn,\
        labelCon='Constrained',labelRand='Unconstrained \nmean',\
        fmtCon = '-',colourCon = 'r',colorRandMean='k',showMean=True,\
        errorAlpha=0.5,meanErrorLabel = 'Unconstrained \nMean',\
        profileErrorLabel = 'Profile \nvariation \n',colourRand = 'grey',
        title=None,showTitle=True,rMin=5,mMin=1e11,mMax=1e16,\
        legendFontSize=8,fontname="serif",fontsize=8,frameon=False,\
        legendLoc = 'upper right',bottom=0.125,left=0.125,includeLegend=True,\
        show = True,hideYLabels = False,ylim = [0,1.4],xlim=None,\
        guideColour = 'grey',guideStyle='--',savename=None,ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(rBinStackCentres,nbarjMean/nbar,
        yerr=sigmaMean/nbar,label=labelCon,fmt=fmtCon,color=colourCon)
    if showMean:
        ax.fill_between(rBinStackCentres,\
            y1 = (nbarjUnMean - sigmaUnMean)/nbar,\
            y2 = (nbarjUnMean + sigmaUnMean)/nbar,alpha=errorAlpha,\
            color = colorRandMean,label=meanErrorLabel)
    ax.fill_between(rBinStackCentres,\
        y1 = (nbarjUnMean - sigmaUn)/nbar,\
        y2 = (nbarjUnMean + sigmaUn)/nbar,alpha=errorAlpha,\
        color = colourRand,\
        label=profileErrorLabel)
    if title is None:
        title = 'Void Profiles, $R_{\\mathrm{eff}} > ' + \
            str(rMin) + '\\mathrm{\\,Mpc}h^{-1}$, $' + \
            plot.scientificNotation(mMin) + ' < M/(M_{\\odot}h^{-1}) < ' + \
            plot.scientificNotation(mMax) + '$'
    if showTitle:
        ax.set_title(title,fontsize=fontsize,fontfamily=fontname)
    ax.set_xlabel('$R/R_{\\mathrm{eff}}$',fontsize=fontsize,\
        fontfamily=fontname)
    if not hideYLabels:
        ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=fontsize,\
            fontfamily=fontname)
    ax.axhline(1.0,linestyle=guideStyle,color=guideColour)
    ax.axvline(1.0,linestyle=guideStyle,color=guideColour)
    if includeLegend:
        ax.legend(prop={"size":legendFontSize,"family":fontname},
            frameon=frameon,loc=legendLoc)
    ax.tick_params(axis='both',labelsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if hideYLabels:
        ax.set_yticklabels([])
    plt.subplots_adjust(bottom=bottom,left=left)
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()



def plotMassTypeComparison(massList1,massListFull1,massList2,massListFull2,\
        stepsListGADGET,stepsList,logstepsList,stepsList1024,\
        stepsListEPS_0p662,resStepsList,clusterNames,\
        name1 = "$M_{200\\mathrm{c}}$",name2 = "$M_{100\\mathrm{m}}$",\
        show=False,save = True,colorLinear = seabornColormap[0],\
        colorLog=seabornColormap[1],colorGadget='k',colorAdaptive='grey',\
        showGadgetAdaptive = True,showGadget1024=False,\
        showGadgetEps0p662=False,savename = "mass_convergence_comparison.pdf",\
        massName = "M",extraMasses = None,extraMassLabel = 'Extra mass scale',\
        xlabel='Number of Steps',ylim1=[0,2.5e15],ylim2=[0,3.5e15],\
        xlim=[0.9,256],returnHandles=False,showLegend=True,nCols=3,alpha=0.5,\
        markerGadget=None,markerCOLA=None,markerPM=None,markerOther=None,\
        wspace=0,hspace=0.0,top=0.9,bottom=0.1,left=0.1,right=0.93,scale=1e15,\
        fontsize=8,fontfamily='serif',colorRes = seabornColormap[2],\
        showGADGET=True,showPM=True,showCOLA=True,markerRes='x',\
        showResMasses = True,resList = [256,512,1024,2048],\
        resToShow=None,figsize=None,title=None,logy=False,\
        xticks=None,yticks1=None,yticks2=None,massLabelPos=0.98,\
        colaColour=seabornColormap[0],pmColour=seabornColormap[1],\
        logStyle=':',linStyle='-',legendMethod="auto",\
        linText=[0.3,0.7],logText=[0.3,0.6],colaText=[0.4,0.8],\
        pmText=[0.6,0.8],legLoc=[0.3,0.8],secondYAxis=False,\
        massLabelsOnRight=False,selectModels=True,selectCOLA=4,\
        selectPM = 2,selectSize=5,selectMarker='o'):
    if resToShow is None:
        resToShow = np.arange(0,len(resList))
    nsamples = len(massList1)
    # Compute means over all samples:
    meanMassFull1 = np.mean(np.vstack(massListFull1),0)/scale
    stdMassFull1 = np.std(np.vstack(massListFull1),0)\
        /np.sqrt(nsamples)/scale
    massArray1 = np.array([np.vstack(massList1[k]) \
        for k in range(0,nsamples)])/scale
    meanMass1 = np.mean(massArray1,0)
    stdMass1 = np.std(massArray1,0)/np.sqrt(nsamples)
    meanMassFull2 = np.mean(np.vstack(massListFull2),0)/scale
    stdMassFull2 = np.std(np.vstack(massListFull2),0)\
        /np.sqrt(nsamples)/scale
    massArray2 = np.array([np.vstack(massList2[k]) \
        for k in range(0,nsamples)])/scale
    meanMass2 = np.mean(massArray2,0)
    stdMass2 = np.std(massArray2,0)/np.sqrt(nsamples)
    if extraMasses is not None:
        meanExtraMass = np.mean(np.vstack(extraMasses),0)/scale
        stdExtraMass = np.std(np.vstack(extraMasses),0)\
            /np.sqrt(len(extraMasses))/scale
    # Plot the masses:
    fig, ax = plt.subplots(2,nCols,figsize=figsize)
    handles = [[] for l in range(0,2*nCols)]
    for l in range(0,nCols):
        i = int(l/nCols)
        j = l - nCols*i
        iterator = 0
        if nCols > 1:
            ax0 = ax[0,l]
            ax1 = ax[1,l]
        else:
            ax0 = ax[0]
            ax1 = ax[1]
        # GADGET masses:
        if showGADGET:
            handles[l].append(ax0.errorbar(stepsListGADGET,\
                    meanMass1[iterator:(iterator + len(stepsListGADGET)),l],\
                    yerr=stdMass1[iterator:(iterator + \
                    len(stepsListGADGET)),l],\
                    marker = markerGadget,linestyle=linStyle,\
                    label='GADGET2 ($128^3$ PM-grid)',color = colorGadget))
            handles[l + nCols].append(ax1.errorbar(stepsListGADGET,\
                    meanMass2[iterator:(iterator + len(stepsListGADGET)),l],\
                    yerr=stdMass2[iterator:(iterator + \
                    len(stepsListGADGET)),l],\
                    marker = markerGadget,linestyle=linStyle,\
                    label='GADGET2 ($128^3$ PM-grid)',color = colorGadget))
        iterator += len(stepsListGADGET)
        # COLA masses:
        if showCOLA:
            handles[l].append(ax0.errorbar(stepsList,\
                    meanMass1[iterator:(iterator + len(stepsList)),l]\
                    ,yerr=stdMass1[iterator:(iterator + len(stepsList)),l]\
                    ,marker = markerCOLA,linestyle=linStyle,\
                    label='COLA (linear steps)',color=colaColour))
            handles[l + nCols].append(ax1.errorbar(stepsList,\
                    meanMass2[iterator:(iterator + len(stepsList)),l]\
                    ,yerr=stdMass2[iterator:(iterator + len(stepsList)),l]\
                    ,marker = markerCOLA,linestyle=linStyle,\
                    label='COLA (linear steps)', color=colaColour))
            if selectModels:
                ax0.scatter(stepsList[selectCOLA],\
                    meanMass1[iterator + selectCOLA,l],marker=selectMarker,\
                    s=selectSize,edgecolors=colaColour,facecolors='none')
                ax1.scatter(stepsList[selectCOLA],\
                    meanMass2[iterator + selectCOLA,l],marker=selectMarker,\
                    s=selectSize,edgecolors=colaColour,facecolors='none')
        iterator += len(stepsList)
        # PM masses:
        if showPM:
            handles[l].append(ax0.errorbar(stepsList,\
                    meanMass1[iterator:(iterator + len(stepsList)),l],\
                    yerr=stdMass1[iterator:(iterator + len(stepsList)),l],\
                    marker = markerPM,linestyle=linStyle,\
                    label='PM (linear steps)',color=pmColour))
            handles[l+nCols].append(ax1.errorbar(stepsList,\
                    meanMass2[iterator:(iterator + len(stepsList)),l],\
                    yerr=stdMass2[iterator:(iterator + len(stepsList)),l],\
                    marker = markerPM,linestyle=linStyle,\
                    label='PM (linear steps)',color=pmColour))
        iterator += len(stepsList)
        # Include some extra masses if provided, otherwise skip them:
        if showGadget1024:
            handles[l].append(ax0.errorbar(stepsList1024,\
                meanMass1[iterator:(iterator + len(stepsList1024)),l],\
                yerr=stdMass1[iterator:(iterator + len(stepsList1024)),l],\
                marker = None,linestyle='-.',\
                label='GADGET2 ($1024^3$ PM-grid)',color=colorGadget))
            handles[l+nCols].append(ax0.errorbar(stepsList1024,\
                meanMass2[iterator:(iterator + len(stepsList1024)),l],\
                yerr=stdMass2[iterator:(iterator + len(stepsList1024)),l],\
                marker = None,linestyle='-.',\
                label='GADGET2 ($1024^3$ PM-grid)',color=colorGadget))
        iterator += len(stepsList1024)
        if showGadgetEps0p662:
            handles[l].append(ax0.errorbar(stepsListEPS_0p662,\
                meanMass1[iterator:(iterator + len(stepsListEPS_0p662)),l],\
                yerr=stdMass1[iterator:(iterator + len(stepsListEPS_0p662)),l],\
                marker = markerGadget,linestyle='-.',\
                label='GADGET2 ($\\epsilon = 0.662$)',color=colorGadget))
            handles[l + nCols].append(ax1.errorbar(stepsListEPS_0p662,\
                meanMass2[iterator:(iterator + len(stepsListEPS_0p662)),l],\
                yerr=stdMass2[iterator:(iterator + len(stepsListEPS_0p662)),l],\
                marker = markerGadget,linestyle='-.',\
                label='GADGET2 ($\\epsilon = 0.662$)',color=colorGadget))
        iterator += len(stepsListEPS_0p662)
        # Skip over resolution masses:
        if showResMasses:
            if showCOLA:
                for m in resToShow:
                    handles[l].append(ax0.errorbar(resStepsList,\
                        meanMass1[iterator:(iterator + len(resStepsList)),l],\
                        yerr=stdMass1[iterator:(iterator + len(resStepsList)),l],\
                        marker = markerRes,linestyle='-',\
                        label='COLA ($' + str(resList[m]) + '$)',color=colorRes))
                    handles[l + nCols].append(ax1.errorbar(resStepsList,\
                        meanMass2[iterator:(iterator + len(resStepsList)),l],\
                        yerr=stdMass2[iterator:(iterator + len(resStepsList)),l],\
                        marker = markerRes,linestyle='-',\
                        label='COLA ($' + str(resList[m]) + '$)',color=colorRes))
            iterator += len(resStepsList)*len(resList)
            if showPM:
                for m in resToShow:
                    handles[l].append(ax0.errorbar(resStepsList,\
                        meanMass1[iterator:(iterator + len(resStepsList)),l],\
                        yerr=stdMass1[iterator:(iterator + len(resStepsList)),l],\
                        marker = markerGadget,linestyle='-.',\
                        label='PM ($' + str(resList[m]) + '$)',color=colorGadget))
                    handles[l + nCols].append(ax1.errorbar(resStepsList,\
                        meanMass2[iterator:(iterator + len(resStepsList)),l],\
                        yerr=stdMass2[iterator:(iterator + len(resStepsList)),l],\
                        marker = markerGadget,linestyle='-.',\
                        label='PM ($' + str(resList[m]) + '$)',color=colorGadget))
            iterator += len(resStepsList)*len(resList)
        else:
            iterator += 2*len(resStepsList)*len(resList)
        # Log COLA masses:
        if showCOLA:
            handles[l].append(ax0.errorbar(logstepsList,\
                    meanMass1[iterator:(iterator + len(logstepsList)),l]\
                    ,yerr=stdMass1[iterator:(iterator + len(logstepsList)),l]\
                    ,marker = markerCOLA,linestyle=logStyle,\
                    label='COLA (log steps)',color=colaColour))
            handles[l + nCols].append(ax1.errorbar(logstepsList,\
                    meanMass2[iterator:(iterator + len(logstepsList)),l]\
                    ,yerr=stdMass2[iterator:(iterator + len(logstepsList)),l]\
                    ,marker = markerCOLA,linestyle=logStyle,\
                    label='COLA (log steps)',color=colaColour))
        iterator += len(logstepsList)
        # Log PM masses:
        if showPM:
            handles[l].append(ax0.errorbar(logstepsList,\
                    meanMass1[iterator:(iterator + len(logstepsList)),l],\
                    yerr=stdMass1[iterator:(iterator + len(logstepsList)),l],\
                    marker = markerPM,linestyle=logStyle,\
                    label='PM (log steps)',color=pmColour))
            handles[l + nCols].append(ax1.errorbar(logstepsList,\
                    meanMass2[iterator:(iterator + len(logstepsList)),l],\
                    yerr=stdMass2[iterator:(iterator + len(logstepsList)),l],\
                    marker = markerPM,linestyle=logStyle,\
                    label='PM (log steps)',color=pmColour))
            if selectModels:
                ax0.scatter(logstepsList[selectPM],\
                    meanMass1[iterator + selectPM,l],marker=selectMarker,\
                    s=selectSize,edgecolors=pmColour,facecolors='none')
                ax1.scatter(logstepsList[selectPM],\
                    meanMass2[iterator + selectPM,l],marker=selectMarker,\
                    s=selectSize,edgecolors=pmColour,facecolors='none')
        iterator += len(logstepsList)
        # Adaptive gadget masses:
        if showGadgetAdaptive:
            handles[l].append(ax0.fill_between([0,256],\
                [meanMassFull1[l] - stdMassFull1[l],meanMassFull1[l] - \
                    stdMassFull1[l]],\
                [meanMassFull1[l] + stdMassFull1[l],meanMassFull1[l] + \
                    stdMassFull1[l]],\
                color=colorAdaptive,alpha=alpha,\
                label='Gadget, adaptive steps'))
            handles[l + nCols].append(ax1.fill_between([0,256],\
                [meanMassFull2[l] - stdMassFull2[l],meanMassFull2[l] - \
                    stdMassFull2[l]],\
                [meanMassFull2[l] + stdMassFull2[l],meanMassFull2[l] + \
                    stdMassFull2[l]],\
                color=colorAdaptive,alpha=alpha,\
                label='Gadget, adaptive steps'))
        if extraMasses is not None:
            handles[l].append(ax0.fill_between([0,256],\
                [meanExtraMass[l] - stdExtraMass[l],meanExtraMass[l] - \
                    stdExtraMass[l]],\
                [meanExtraMass[l] + stdExtraMass[l],meanExtraMass[l] + \
                    stdExtraMass[l]],\
                color='r',alpha=alpha,label=extraMassLabel))
            handles[l + nCols].append(ax0.fill_between([0,256],\
                [meanExtraMass[l] - stdExtraMass[l],meanExtraMass[l] - \
                    stdExtraMass[l]],\
                [meanExtraMass[l] + stdExtraMass[l],meanExtraMass[l] + \
                    stdExtraMass[l]],\
                color='r',alpha=alpha,label=extraMassLabel))
        # Axis formatting:
        if clusterNames is not None:
            ax0.set_title(clusterNames[l][0],\
                fontsize=fontsize,fontfamily=fontfamily)
        #ax1.set_title(clusterNames[l][0] + " (" + name2 + ")",\
        #    fontsize=fontsize,fontfamily=fontfamily)
        if logy:
            ax0.set_yscale('log')
            ax1.set_yscale('log')
        ax0.xaxis.label.set_visible(False)
        ax0.xaxis.set_major_formatter(NullFormatter())
        ax0.xaxis.set_minor_formatter(NullFormatter())
        ax1.set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontfamily)
        ax0.set_ylim(np.array(ylim1)/scale)
        ax1.set_ylim(np.array(ylim2)/scale)
        ax0.set_xscale('log')
        ax1.set_xscale('log')
        ax0.set_xlim(xlim)
        ax1.set_xlim(xlim)
        ax0.tick_params(axis='both', which='major', labelsize=fontsize)
        ax0.tick_params(axis='both', which='minor', labelsize=fontsize)
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)
        ax1.tick_params(axis='both', which='minor', labelsize=fontsize)
        if xticks is not None:
            ax1.set_xticks(xticks)
            xlabels = ["$" + scientificNotation(tick,powerRange=2) + "$"\
                for tick in xticks]
            ax1.xaxis.set_ticklabels(xlabels,fontsize=fontsize)
        else:
            xlabels = ["$" + scientificNotation(tick,powerRange=2) + "$"\
                for tick in ax1.get_xticks()]
            ax1.xaxis.set_ticklabels(xlabels,fontsize=fontsize)
        if yticks1 is not None:
            rescaledTicks = np.array(yticks1)/scale
            ax0.set_yticks(rescaledTicks)
            ylabels = ["$" + scientificNotation(tick,powerRange=2) + "$"\
                for tick in rescaledTicks]
            #print(ylabels)
            ax0.yaxis.set_ticklabels(ylabels,fontsize=fontsize)
            ax0.yaxis.set_minor_formatter(NullFormatter())
        if yticks2 is not None:
            rescaledTicks = np.array(yticks2)/scale 
            ax1.set_yticks(rescaledTicks)
            ylabels = ["$" + scientificNotation(tick,powerRange=2) + "$"\
                for tick in rescaledTicks]
            #print(ylabels)
            ax1.yaxis.set_ticklabels(ylabels,fontsize=fontsize)
            ax1.yaxis.set_minor_formatter(NullFormatter())
        if l > 0:
            ax0.yaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax0.yaxis.label.set_visible(False)
            ax0.yaxis.set_major_formatter(NullFormatter())
            ax0.yaxis.set_minor_formatter(NullFormatter())
            ax1.yaxis.label.set_visible(False)
            ax1.yaxis.set_major_formatter(NullFormatter())
            ax1.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax0.set_ylabel(name1 + " [$10^{15}M_{\\odot}h^{-1}$]",\
                fontsize=fontsize,fontfamily=fontfamily)
            ax1.set_ylabel(name2 + " [$10^{15}M_{\\odot}h^{-1}$]",\
                fontsize=fontsize,fontfamily=fontfamily)
        ax0.set_xticks([])
    plt.subplots_adjust(wspace=wspace,hspace=hspace,top=top,bottom=bottom,\
        left=left,right=right)
    # Mass labels:
    if massLabelsOnRight:
        fig.text(massLabelPos,bottom + 0.75*(top-bottom),name1,\
            va='center',rotation='vertical',fontsize=fontsize,\
            fontfamily=fontfamily)
        fig.text(massLabelPos,bottom + 0.25*(top-bottom),name2,\
            va='center',rotation='vertical',fontsize=fontsize,\
            fontfamily=fontfamily)
    # Add a second y axis:
    if secondYAxis:
        if nCols > 1:
            twinAx0 = ax[0,nCols-1].twinx()
            twinAx1 = ax[1,nCols-1].twinx()
        else:
            twinAx0 = ax[0].twinx()
            twinAx1 = ax[1].twinx()
        twinAx0.set_ylim(np.array(ylim1)/scale)
        twinAx1.set_ylim(np.array(ylim2)/scale)
        # Set ticks:
        rescaledTicks = np.array(yticks1)/scale
        twinAx0.set_yticks(rescaledTicks)
        ylabels = ["$" + scientificNotation(tick,powerRange=2) + "$"\
            for tick in rescaledTicks]
        #print(ylabels)
        twinAx0.yaxis.set_ticklabels(ylabels,fontsize=fontsize)
        twinAx0.yaxis.set_minor_formatter(NullFormatter())
        rescaledTicks = np.array(yticks2)/scale 
        twinAx1.set_yticks(rescaledTicks)
        ylabels = ["$" + scientificNotation(tick,powerRange=2) + "$"\
            for tick in rescaledTicks]
        #print(ylabels)
        twinAx1.yaxis.set_ticklabels(ylabels,fontsize=fontsize)
        twinAx1.yaxis.set_minor_formatter(NullFormatter())
        # Axis labels:
        twinAx0.set_ylabel(name1 + " [$10^{15}M_{\\odot}h^{-1}$]",\
                fontsize=fontsize,fontfamily=fontfamily)
        twinAx1.set_ylabel(name2 + " [$10^{15}M_{\\odot}h^{-1}$]",\
            fontsize=fontsize,fontfamily=fontfamily)
    # Legend:
    if showLegend:
        if legendMethod == "auto":
            # Try to split legend entries automatically between panels:
            if nCols > 1:
                if len(handles[0]) < 3:
                    ax[0,0].legend(prop={"size":8,"family":"serif"},\
                        frameon=False)
                elif len(handles[0]) < 5:    
                    ax[0,0].legend(handles = handles[2][0:2],\
                        prop={"size":8,"family":"serif"},frameon=False)
                    ax[0,1].legend(handles = handles[5][2:],\
                        prop={"size":8,"family":"serif"},frameon=False)
                else:
                    ax[0,0].legend(handles = handles[2][0:2],\
                        prop={"size":8,"family":"serif"},frameon=False)
                    ax[0,1].legend(handles = handles[5][2:4],\
                        prop={"size":8,"family":"serif"},frameon=False)
                    ax[0,2].legend(handles = handles[5][4:],\
                        prop={"size":8,"family":"serif"},frameon=False)
            else:
                # Single column version.
                if len(handles[0]) < 4:
                    ax[0].legend(prop={"size":8,"family":"serif"},\
                        frameon=False)
                else:    
                    ax[0].legend(handles = handles[0][0:3],\
                        prop={"size":8,"family":"serif"},frameon=False)
                    ax[1].legend(handles = handles[1][3:],\
                        prop={"size":8,"family":"serif"},frameon=False)
        elif legendMethod == "grid":
            # Implemented a gridded legeng
            #print("Not yet implemented")
            # Fake legend entries:
            if nCols > 1:
                legAx = ax[0,1]
            else:
                legAx = ax[0]
            colaLog = matplotlib.lines.Line2D([0],[0],linestyle=logStyle,\
                color=colaColour,label="")
            colaLin = matplotlib.lines.Line2D([0],[0],linestyle=linStyle,\
                color=colaColour,label="")
            pmLog = matplotlib.lines.Line2D([0],[0],linestyle=logStyle,\
                color=pmColour,label="")
            pmLin = matplotlib.lines.Line2D([0],[0],linestyle=linStyle,\
                color=pmColour,label="")
            # Create a gridded legend:
            legAx.legend(handles = [colaLin,colaLog,pmLin,pmLog],\
                prop={"size":8,"family":"serif"},\
                frameon=False,ncol=2,loc=legLoc)
            # Text:
            legAx.text(linText[0],linText[1],"Linear steps:",\
                fontsize=fontsize,fontfamily=fontfamily,\
                transform=legAx.transAxes)
            legAx.text(logText[0],logText[1],"Log steps:",\
                fontsize=fontsize,fontfamily=fontfamily,\
                transform=legAx.transAxes)
            legAx.text(colaText[0],colaText[1],"COLA:",\
                fontsize=fontsize,fontfamily=fontfamily,\
                transform=legAx.transAxes)
            legAx.text(pmText[0],pmText[1],"PM:",\
                fontsize=fontsize,fontfamily=fontfamily,\
                transform=legAx.transAxes)
        else:
            # Manual positioning of legend entries:
            if nCols > 1:
                ax[0,1].legend(prop={"size":8,"family":"serif"},\
                        frameon=False)
            else:
                ax[0].legend(prop={"size":8,"family":"serif"},\
                        frameon=False)
    if title is not None:
        plt.suptitle()
    #plt.tight_layout()
    if save:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnHandles:
        return [fig, ax, handles]



def plotMassFunction(masses,volSim,ax=None,Om0=0.3,h=0.8,ns=1.0,\
        Delta=200,sigma8=0.8,fontsize=8,legendFontsize=8,font="serif",\
        Ob0=0.049,mass_function='Tinker',delta_wrt='SOCritical',massLower=5e13,\
        massUpper=1e15,figsize=(4,4),marker='x',linestyle='--',\
        color=None,colorTheory = None,\
        nBins=21,poisson_interval = 0.95,legendLoc='lower left',\
        label="Gadget Simulation",transfer_model='EH',fname=None,\
        xlabel="Mass [$M_{\odot}h^{-1}$]",ylabel="Number of halos",\
        ylim=[1e1,2e4],title="Gadget Simulation",showLegend=True,\
        tickRight=False,tickLeft=True,savename=None,\
        linking_length=0.2,showTheory=True,returnHandles=False):
    [dndm,m] = cosmology.TMF_from_hmf(massLower,massUpper,\
        h=h,Om0=Om0,Delta=Delta,delta_wrt=delta_wrt,\
        mass_function=mass_function,sigma8=sigma8,Ob0 = Ob0,\
        transfer_model=transfer_model,fname=fname,ns=ns,\
        linking_length=linking_length)
    massBins = 10**np.linspace(np.log10(massLower),np.log10(massUpper),nBins)
    if showTheory:
        n = cosmology.dndm_to_n(m,dndm,massBins)
        bounds = np.array(scipy.stats.poisson(n*volSim).interval(\
            poisson_interval))
    alphaO2 = (1.0 - poisson_interval)/2.0
    massBinCentres = plot_utilities.binCentres(massBins)
    if type(masses) == list:
        [noInBins,sigmaBins] = computeMeanHMF(masses,\
            massLower=massLower,massUpper=massUpper,nBins = nBins)
    else:
        noInBins = plot_utilities.binValues(masses,massBins)[1]
        sigmaBins = np.abs(np.array([scipy.stats.chi2.ppf(\
            alphaO2,2*noInBins)/2,\
            scipy.stats.chi2.ppf(1.0 - alphaO2,2*(noInBins+1))/2]) - \
            noInBins)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,4))
    if color is None:
        color = seabornColormap[0]
    handles = []
    handles.append(ax.errorbar(massBinCentres,noInBins,sigmaBins,marker=marker,\
        linestyle=linestyle,label=label,color=color))
    if showTheory:
        if colorTheory is None:
            colorTheory = colorTheory
        handles.append(ax.plot(massBinCentres,n*volSim,":",\
            label=mass_function + ' prediction',color=colorTheory))
        handles.append(ax.fill_between(massBinCentres,
                        bounds[0],bounds[1],
                        facecolor=colorTheory,alpha=0.5,interpolate=True,\
                        label='$' + str(100*poisson_interval) + \
                        '\%$ Confidence \nInterval'))
    if showLegend:
        ax.legend(prop={"size":legendFontsize,"family":font},
            loc=legendLoc,frameon=False)
    ax.set_title(title,fontsize=fontsize,fontfamily=font)
    ax.set_xlabel(xlabel,fontsize=fontsize,fontfamily=font)
    ax.set_ylabel(ylabel,fontsize=fontsize,fontfamily=font)
    ax.tick_params(axis='both',labelsize=fontsize,\
        labelright=tickRight,right=tickRight)
    ax.tick_params(axis='both',which='minor',bottom=True,labelsize=fontsize)
    ax.tick_params(axis='y',which='minor')
    ax.yaxis.grid(color='grey',linestyle=':',alpha=0.5)
    ax.set_ylim(ylim)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if savename is not None:
        plt.tight_layout()
        plt.savefig(savename)
    if returnHandles:
        return handles

def massFunctionComparison(massesLeft,massesRight,volSim,Om0=0.3,h=0.8,\
        Delta=200,sigma8=0.8,fontsize=8,legendFontsize=8,font="serif",\
        Ob0=0.049,mass_function='Tinker',delta_wrt='SOCritical',massLower=5e13,\
        massUpper=1e15,figsize=(8,4),marker='x',linestyle='--',ax=None,\
        color=seabornColormap[0],colorTheory = seabornColormap[1],\
        nBins=21,poisson_interval = 0.95,legendLoc='lower left',\
        labelLeft = 'Gadget Simulation',labelRight='ML Simulation',\
        xlabel="Mass [$M_{\odot}h^{-1}$]",ylabel="Number of halos",\
        ylim=[1e1,2e4],savename=None,show=True,transfer_model='EH',fname=None,\
        returnAx = False,ns=1.0,rows=1,cols=2,titleLeft = "Gadget Simulation",\
        titleRight = "Gadget Simulation",saveLeft=None,saveRight=None,\
        ylimRight=None,volSimRight=None):
    if ax is None:
        fig, ax = plt.subplots(rows,cols,figsize=(8,4))
    if volSimRight is None:
        volSimRight = volSim
    if ylimRight is None:
        ylimRight = ylim
    plotMassFunction(massesLeft,volSim,ax=ax[0],Om0=Om0,h=h,ns=ns,\
        Delta=Delta,sigma8=sigma8,fontsize=fontsize,\
        legendFontsize=legendFontsize,font="serif",\
        Ob0=Ob0,mass_function=mass_function,delta_wrt=delta_wrt,\
        massLower=massLower,title=titleLeft,\
        massUpper=massUpper,marker=marker,linestyle=linestyle,\
        color=color,colorTheory = colorTheory,\
        nBins=nBins,poisson_interval = poisson_interval,legendLoc=legendLoc,\
        label=labelLeft,transfer_model=transfer_model,ylim=ylim,\
        savename=saveLeft)
    plotMassFunction(massesRight,volSimRight,ax=ax[1],Om0=Om0,h=h,ns=ns,\
        Delta=Delta,sigma8=sigma8,fontsize=fontsize,\
        legendFontsize=legendFontsize,font="serif",\
        Ob0=Ob0,mass_function=mass_function,delta_wrt=delta_wrt,\
        massLower=massLower,title = titleRight,\
        massUpper=massUpper,marker=marker,linestyle=linestyle,\
        color=color,colorTheory = colorTheory,\
        nBins=nBins,poisson_interval = poisson_interval,legendLoc=legendLoc,\
        label=labelRight,transfer_model=transfer_model,ylim=ylimRight,\
        savename=saveRight)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnAx:
        return ax




def singleMassFunctionPlot(masses,mlow,mupp,nMassBins,textwidth=7.1014,\
        mLimLower=None,comparableHaloMasses=None,\
        rSphere=135,density=False,poisson_interval=0.95,nmax=200,\
        ns=0.9611,Om0=0.3,sigma8=0.8,Ob0=0.04825,h=0.7,\
        mass_function='Tinker',delta_wrt='SOCritical',\
        marker='x',linestyle='--',plotColour=seabornColormap[0],\
        colorTheory = seabornColormap[1],\
        legendLoc='lower left',label="PM10",transfer_model='EH',\
        fname=None,xlabel="Mass [$M_{\odot}h^{-1}$]",ylabel="Number of halos",\
        title=None,showLegend=True,tickRight=False,tickLeft=True,\
        savename=None,compColour=seabornColormap[0],volSim=None,\
        deltaListMean=-0.06,deltaListError=0.003,showTheory=True,ax=None,\
        returnHandles=False,xticks=None,fontsize=8,legendFontsize=11):
    if volSim is None:
        volSim = 4*np.pi*rSphere**3/3
    ylim=[1,nmax]
    poisson_interval = 0.95
    if density:
        ylimHMF = np.array(ylim)/volSim
        factor = 1.0/volSim
    else:
        ylimHMF = np.array(ylim)
        factor = 1.0
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(0.45*textwidth,0.45*textwidth))
    handles = plotMassFunction(masses,volSim,ax=ax,\
            Om0=Om0,h=h,ns=ns,Delta=200,sigma8=sigma8,fontsize=fontsize,\
            legendFontsize=legendFontsize,font="serif",\
            Ob0=Ob0,mass_function=mass_function,delta_wrt=delta_wrt,\
            massLower=mlow,massUpper=mupp,figsize=(4,4),\
            marker=marker,linestyle=linestyle,\
            color=plotColour,colorTheory = colorTheory,\
            nBins=nMassBins,poisson_interval = poisson_interval,\
            legendLoc=legendLoc,\
            label=label,transfer_model=transfer_model,fname=fname,\
            xlabel=xlabel,ylabel=ylabel,\
            ylim=ylim,title=title,showLegend=showLegend,\
            tickRight=tickRight,tickLeft=tickLeft,savename=savename,\
            showTheory=showTheory,returnHandles=True)
    if mLimLower is None:
        mLimLower = mlow
    ax.set_xlim((mLimLower,mupp))
    # Add in the comparison with the unconstrained data:
    if comparableHaloMasses is not None:
        nsamples = len(comparableHaloMasses)
        hmfUnderdense = [computeMeanHMF(hmasses,massLower = mlow,\
                        massUpper = mupp,nBins = nMassBins) \
                        for hmasses in comparableHaloMasses]
        #hmfUnderdensePoisson = np.array(scipy.stats.poisson(\
        #    np.sum(np.array([hmf[0] \
        #    for hmf in hmfUnderdense]),0)).interval(poisson_interval))/\
        #    nsamples
        hmfUnderdenseMean = np.mean(np.array([hmf[0] \
            for hmf in hmfUnderdense]),0)
        hmfUnderdensePoisson = np.array(scipy.stats.poisson(\
            np.mean(np.array([hmf[0] \
            for hmf in hmfUnderdense]),0)).interval(poisson_interval))
        massBins = 10**(np.linspace(np.log10(mlow),np.log10(mupp),\
            nMassBins))
        massBinCentres = binCentres(massBins)
        handles.append(ax.fill_between(massBinCentres,\
            hmfUnderdensePoisson[0]*factor,\
            hmfUnderdensePoisson[1]*factor,\
            label = '$\\Lambda$-CDM, \n' + \
            '$' + ("%.2g" % (deltaListMean - deltaListError)) + \
            ' \\leq \\delta < ' + \
            ("%.2g" % (deltaListMean + deltaListError)) + \
            '$',color=compColour,alpha=0.5))
    plt.tight_layout()
    if xticks is not None:
        ax.get_xaxis().set_major_formatter(\
            matplotlib.ticker.ScalarFormatter())
        #ax.xaxis.get_major_formatter().set_scientific(True)
        ax.set_xticks(xticks)
        ax.xaxis.set_ticklabels([scientificNotation(i,latex=True) \
            for i in xticks])
    if savename is not None:
        plt.savefig(savename)
    if returnHandles:
        return handles


















