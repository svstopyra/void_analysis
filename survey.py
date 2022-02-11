# Tools for computing properties of galaxy surveys
import scipy
import numpy as np
from .cosmology import comovingToLuminosity
import scipy.special as special
import healpy
import astropy

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


# Apply the survey mask for the 2M++ galaxy survey
def surveyMask(points,mask11,mask12,cosmo,alpha,Mstar,\
        centre = np.array([0,0,0]),boxsize=None,nside=512,Mmin = -25,\
        Mmax = -21,mmax=12.5,mmin=None,Mcut = 11.5,Ninterp=1000,\
        nstar=1.14e-2*(0.705**3),\
        keCorr = None,numericalIntegration=False,interpolateMask = True,\
        splitApparent = True,splitAbsolute=True,nLumBins=8,nAppMagBins=2):
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
                c[2*k,:] = getCompletenessInSelection(rl,mmin,Mcut,\
                    MabsRange[k+1],MabsRange[k],cosmo,nstar,keCorr,\
                    numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
                c[2*k+1,:] = getCompletenessInSelection(rl,Mcut,mmax,\
                    MabsRange[k+1],MabsRange[k],cosmo,nstar,keCorr,\
                    numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
        else:
            c = np.zeros((nAppMagBins*nLumBins,len(points)))
            c[0,:] = getCompletenessInSelection(rl,mmin,Mcut,\
                Mmin,Mmax,cosmo,nstar,keCorr,\
                numericalIntegration,interpolateMask,Ninterp,alpha,Mstar)
            c[1,:] = getCompletenessInSelection(rl,Mcut,mmax,\
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
    angularMask = np.zeros(len(rl),dtype=np.float32)
    condition_highM = (Mr < Mmax) & (Mr >= Mcut)
    condition_lowM = (Mr < Mcut)
    angularMask[condition_highM] = mask12[ind[condition_highM]]
    angularMask[condition_lowM] = mask11[ind[condition_lowM]]
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

