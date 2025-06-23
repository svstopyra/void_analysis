# Tools for computing properties of galaxy surveys
import scipy
import numpy as np
import numexpr as ne
from .cosmology import comovingToLuminosity
import scipy.special as special
import healpy
import astropy
from .simulation_tools import getGriddedGalCount
from .cosmology import comovingToLuminosity
from .import snapedit

# Incomplete gamma function, needed for doing survey completeness integrals.
def incompleteGamma(a,x):
    return special.gamma(a)*special.gammaincc(a,x)

# (DEPRECATED) - Old method for computing completeness
def gammafunc(m,r10,Mstar,alpha,Mmin,Mmax,corr=0.0,mmin = None):
    Mr10 = m - 5*np.log10(r10) + corr
    if mmin is None:
        absolute_ml0 = Mmax
    else:
        absolute_ml0 = np.maximum(mmin - 5*np.log10(r10) + corr,Mmax)
    Madj = np.minimum(np.maximum(Mr10,Mmin),absolute_ml0)
    return incompleteGamma(1 + alpha,10**(0.4*(Mstar - Madj)))

# Compute the radial completeness for a given luminosity selection of galaxies.
def getCompletenessInSelection(rl,mlow,mupp,Mlow,Mupp,cosmo,nstar,keCorr,\
        numericalIntegration,interpolateMask,Ninterp,alpha,Mstar):
    if interpolateMask:
        rRange = np.linspace(np.min(rl),np.max(rl),Ninterp)
        cRange = radialCompleteness(rRange,alpha,Mstar,Mlow,Mupp,mupp,cosmo,\
            nstar=nstar,keCorr=keCorr,mmin=mlow,\
            numericalIntegration=numericalIntegration)
        c = scipy.interpolate.interp1d(rRange,cRange,kind='cubic')(rl)
    else:
        c = radialCompleteness(rl,alpha,Mstar,Mlow,Mupp,mupp,cosmo,\
            nstar=nstar,keCorr=keCorr,mmin=mlow,\
            numericalIntegration=numericalIntegration)
    return np.maximum(0.0,c)

def keCorr(z,fit = [-1.456552772320231,-0.7687913554110967]):
    """
    Apply ke correction. Originally we used a fitted version, but for now
    just using a fixed value.
    
    Parameters:
        z (float or array): Redshift
        fit (list): Fitting parameters. Currently unused.
        
    Returns:
        float or array (same size as z): ke correction
    
    Tests:
        Tested in test_survey.py
        Regression test: test_keCorr
    """
    #return -fit[0]*z# - fit[1]
    return 2.9*z

# Apply the survey mask for the 2M++ galaxy survey
def surveyMask(points,mask11,mask12,cosmo,alpha,Mstar,\
        centre = np.array([0,0,0]),boxsize=None,nside=512,Mmin = -25,\
        Mmax = -21,mmax=12.5,mmin=None,mcut = 11.5,Ninterp=1000,\
        nstar=1.14e-2*(0.705**3),\
        keCorr = None,numericalIntegration=False,interpolateMask = True,\
        splitApparent = True,splitAbsolute=True,nLumBins=8,nAppMagBins=2,\
        returnComponents = False):
    equatorial = pointsToEquatorial(points,boxsize=boxsize,centre=centre)
    theta = np.pi/2 - equatorial[:,2]
    # Now need to get luminosity distance:
    [rl, z] = comovingToLuminosity(equatorial[:,0],cosmo,Ninterp=Ninterp,\
        return_z = True)
    # Convert to healpix co-ordinates:
    npix = healpy.nside2npix(nside)
    ind = healpy.ang2pix(nside,theta,equatorial[:,1])
    Mr = Mmin + 5*np.log10(rl/(1e-5))
    # Assuming point-like:
    #c = radialCompleteness(rl,alpha,Mstar,Mmin,Mmax)
    MabsRange = np.linspace(Mmax,Mmin,nLumBins + 1)
    if mmin is None:
        mmin = -100
    if splitApparent:
        if splitAbsolute:
            c = np.zeros((nAppMagBins*nLumBins,len(points)))
            for k in range(0,nLumBins):
                c[2*k,:] = getCompletenessInSelection(rl,mmin,mcut,\
                    MabsRange[k+1],MabsRange[k],cosmo,nstar,keCorr,\
                    numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
                c[2*k+1,:] = getCompletenessInSelection(rl,mcut,mmax,\
                    MabsRange[k+1],MabsRange[k],cosmo,nstar,keCorr,\
                    numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
        else:
            c = np.zeros((nAppMagBins*nLumBins,len(points)))
            c[0,:] = getCompletenessInSelection(rl,mmin,mcut,\
                Mmin,Mmax,cosmo,nstar,keCorr,\
                numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
            c[1,:] = getCompletenessInSelection(rl,mcut,mmax,\
                Mmin,Mmax,cosmo,nstar,keCorr,\
                numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
            for k in range(1,nLumBins):
                c[2*k,:] = c[0,:]
                c[2*k+1,:] = c[1,:]
    else:
        if splitAbsolute:
            c = np.zeros((nAppMagBins*nLumBins,len(points)))
            for k in range(0,nLumBins):
                c[2*k,:] = getCompletenessInSelection(rl,mmin,mmax,\
                    MabsRange[k+1],MabsRange[k],cosmo,nstar,keCorr,\
                    numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
                c[2*k+1,:] = c[2*k,:]
        else:
            c = getCompletenessInSelection(rl,mmin,mmax,\
                    Mmin,Mmax,cosmo,nstar,keCorr,\
                    numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
    #angularMask = np.zeros(len(rl),dtype=np.float32)
    # Mask per bin:
    angularMask = np.zeros(c.shape)
    if splitApparent:
        for k in range(0,nLumBins):
            angularMask[2*k,:] = mask11[ind]
            angularMask[2*k+1,:] = mask12[ind]
    else:
        if splitAbsolute:
            for k in range(0,nLumBins):
                angularMask[2*k,:] = mask11[ind]
                angularMask[2*k+1,:] = mask12[ind]
        else:
            condition_highM = (Mr < mmax) & (Mr >= mcut)
            condition_lowM = (Mr < mcut)
            maskHigh = np.zeros(len(rl),dtype=np.float32)
            maskLow = np.zeros(len(rl),dtype=np.float32)
            angularMask[condition_highM] = mask12[ind[condition_highM]]
            maskHigh[condition_highM] = mask12[ind[condition_highM]]
            angularMask[condition_lowM] = mask11[ind[condition_lowM]]
            maskLow[condition_lowM] = mask11[ind[condition_lowM]]
    condition_highM = (Mr < mmax) & (Mr >= mcut)
    condition_lowM = (Mr < mcut)
    maskHigh = np.zeros(len(rl),dtype=np.float32)
    maskLow = np.zeros(len(rl),dtype=np.float32)
    #angularMask[condition_highM] = mask12[ind[condition_highM]]
    maskHigh[condition_highM] = mask12[ind[condition_highM]]
    #angularMask[condition_lowM] = mask11[ind[condition_lowM]]
    maskLow[condition_lowM] = mask11[ind[condition_lowM]]
    if returnComponents:
        return [angularMask*c,angularMask,c,maskHigh,maskLow]
    else:
        return angularMask*c

# Convert a set of cartesian points to comoving distance, and equatorial
# (right ascension, declanation) co-ordinates.
def pointsToEquatorial(points,boxsize=None,centre=np.array([0,0,0])):
    if boxsize is not None:
        dispXYZ = snapedit.unwrap(points - centre,boxsize)
    else:
        dispXYZ = points - centre
    # Angular coordinates
    r = np.sqrt(np.sum(dispXYZ**2,1)) # Assumed to be Mpc
    alpha = np.arctan2(dispXYZ[:,1],dispXYZ[:,0])
    delta = np.arcsin(dispXYZ[:,2]/r)
    return np.vstack((r,alpha,delta)).T

# Radial completeness computed using luminosity functions.
def radialCompleteness(rl,alpha,Mstar,Mmin,Mmax,mmax,cosmo,nstar=1.14e-2,\
        keCorr = None,mmin = None,numericalIntegration=False,Ninterp=1000):
    if mmin is None:
        mmin = 0
    if keCorr is None:
        corr = 0.0
    else:
        # Obtain redshift:
        distL = rl*astropy.units.Mpc
        zqMin = astropy.cosmology.z_at_value(\
            cosmo.luminosity_distance,np.min(distL))
        zqMax = astropy.cosmology.z_at_value(\
            cosmo.luminosity_distance,np.max(distL))
        zgrid = np.linspace(zqMin,zqMax,Ninterp)
        Dgrid = cosmo.luminosity_distance(zgrid)
        z = np.interp(distL.value,Dgrid.value,zgrid)
        if callable(keCorr):
            corr = keCorr(z)
        elif keCorr == "ke":
            corr = 2.9*z
        else:
            raise Exception("Undefined correction function.")
    absolute_mu0 = mmax - 5 * np.log10(rl) - 25 + corr
    absolute_ml0 = mmin - 5 * np.log10(rl) - 25 + corr
    abmu = np.minimum(absolute_mu0, Mmax)
    abml = np.maximum(absolute_ml0, Mmin)
    abmu = np.maximum(abmu, abml)
    xl0 = 10.0**(0.4 * (Mstar - abmu))
    xu0 = 10.0**(0.4 * (Mstar - abml))
    xl1 = 10.0**(0.4 * (Mstar - Mmax))
    xu1 = 10.0**(0.4 * (Mstar - Mmin))
    if numericalIntegration:
        Phi0 = np.zeros(rl.shape)
        for k in range(0,len(rl)):
            Phi0[k] = scipy.integrate.quad(lambda x: (x**alpha)*np.exp(-x),\
                xl0[k], xu0[k])[0]
        Phi1 = scipy.integrate.quad(lambda x: (x**alpha)*np.exp(-x),xl1, xu1)[0]
    else:
        Phi0 = incompleteGamma(1.0 + alpha,xl0) - incompleteGamma(1.0 + alpha,xu0)
        Phi1 = incompleteGamma(1.0 + alpha,xl1) - incompleteGamma(1.0 + alpha,xu1)
    return np.maximum(0.0,Phi0/Phi1)


# Load 2M++ catalogue data and return the galaxy counts in
# each magnitude bin and voxel:
def griddedGalCountFromCatalogue(cosmo,tmppFile="2mpp_data/2MPP.txt",\
        nAbsBins=8,N=256,nAppBins=2,MMin=-25,MMax=-21,boxsize=677.7,\
        Kcorrection = True,recomputeAbsolute = True):
    tmpp = np.loadtxt(tmppFile)
    # Comoving distance in Mpc/h
    h = cosmo.h
    d = cosmo.comoving_distance(tmpp[:,3]).value*h
    dL = comovingToLuminosity(d[np.where(d > 0)],cosmo)
    posD = np.where(d > 0)[0]
    # Angular co-ordinates:
    theta = tmpp[:,2]
    phi = tmpp[:,1]
    # Cartesian positions:
    Z = d*np.sin(theta)
    X = d*np.cos(theta)*np.cos(phi)
    Y = d*np.cos(theta)*np.sin(phi)
    pos2mpp = np.vstack((X,Y,Z)).T[posD]
    # Density in cells of an N^3 grid:
    ng2mpp = getGriddedGalCount(pos2mpp,N,boxsize)
    M2mpp = tmpp[:,5] # Absolute magnitude
    m2mpp = tmpp[:,4] # Apparent magnitude
    K2mpp = tmpp[posD,4] # filtered apparent magnitude for useable
        # galaxies
    K = tmpp[posD,5] # filtered absolute magnitude
    # Compute number of galaxies in each of 16 luminosity bins:
    nMagBins = nAbsBins*nAppBins
    ng2mppK = np.zeros((nMagBins,N,N,N))
    # Effective absolute magnitudes, with K-corrections:
    if recomputeAbsolute:
        MabsEff = m2mpp[posD] - 5*np.log10(dL) - 25
        if Kcorrection:
            MabsEff += 2.9*tmpp[posD,3]
    else:
        MabsEff = M2mpp[posD]
    Kbins = np.linspace(-MMax,-MMin,nAbsBins+ 1)
    # Filter to the apparent magnitude limits:
    indices11p5 = [np.where((-MabsEff > Kbins[l]) & (-MabsEff <= Kbins[l+1]) \
        & (K2mpp <= 11.5))[0] for l in range(0,len(Kbins)-1)]
    indices12p5 = [np.where((-MabsEff > Kbins[l]) & (-MabsEff <= Kbins[l+1]) \
        & (K2mpp > 11.5) & (K2mpp <= 12.5))[0] for l in range(0,len(Kbins)-1)]
    # Compute galaxy counts in each bin:
    for k in range(0,nAbsBins):
        posKlow = pos2mpp[indices11p5[k]]
        posKupp = pos2mpp[indices12p5[k]]
        ng2mppK[2*k,:] = getGriddedGalCount(posKlow,N,boxsize)
        ng2mppK[2*k+1,:] = getGriddedGalCount(posKupp,N,boxsize)
    return ng2mppK

