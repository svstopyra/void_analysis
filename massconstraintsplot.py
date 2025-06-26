import healpy
import pynbody
import numpy as np
from scipy import io
from void_analysis import snapedit, context, stacking, cosmology,tools, \
    plot_utilities
from void_analysis import plot
import pynbody.plot.sph as sph
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
import pickle
import os
import astropy
import matplotlib.cm as cm
import matplotlib.colors as colors
import astropy.units as u
import scipy
import seaborn as sns
# Use seaborn colours:
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
# Set maths font to serif:
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
from void_analysis.tools import MassConstraint
from astropy.io import fits
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# MASS CONSTRAINTS PLOT
def showClusterMassConstraints(meanCriticalMass,stdErrorCriticalMass,\
        figOut = '.',catFolder = "../../../catalogues/",h=0.705,Om0 = 0.3111,\
        savename = "mass_constraints_plot.pdf",savePlotData=False,show=True):
    # Record mass constraints from different sources:
    # Norma mass from Woudt et al.:
    Mvt = 10.4e14
    Mrvt = 11e14
    Mpme = 14.7e14
    Mmid = (Mpme + Mvt)/2
    Mupp = Mpme - Mmid
    Mlow = Mmid - Mvt
    abell_folder = "./VII_110A"
    # Load properties of the Abell catalogue:
    [abell_l,abell_b,abell_n,abell_z,abell_d] = \
        tools.loadAbellCatalogue(abell_folder)
    # SZ data:
    PLANCK = fits.open(catFolder + "PSZ2v1.fits") # Uses Msol
    ACT = fits.open(catFolder + "ACT2020/" + \
        "DR5_cluster-catalog_v1.1_forSZDB.fits") # Uses Msol
    SPT2500 = fits.open(catFolder + \
        "SPT2020/2500d_cluster_sample_Bocquet19_forSZDB.fits") # Uses Msol/h
    SPT100 = fits.open(catFolder + \
        "SPT2020/sptpol100d_catalog_huang19_forSZDB.fits")
    SPTECS = fits.open(catFolder + \
        "SPT2020/sptecs_catalog_oct919_forSZDB.fits")
    ACO = [426,2147,1656,3627,3571,548,2199,2063,1367]
    PLANCK_NAMES = {426:'',2147:'PSZ2 G029.06+44.55',1656:'PSZ2 G057.80+88.00',\
        3627:'PSZ2 G325.17-07.05',3571:'PSZ2 G316.31+28.53',\
        548:'PSZ2 G230.28-24.42',2199:'PSZ2 G062.94+43.69',\
        2063:'PSZ2 G012.81+49.68',1367:'PSZ2 G234.59+73.01'}
    ACT_NAMES = {426:'',2147:'',1656:'',3627:'ACT-CL J1614.3-6052',\
        3571:'ACT-CL J1347.4-3250',548:'',2199:'',2063:'',1367:''}
    # PLANCK DATA:
    # Extract indices corresponding to the clusters of interest:
    PLANCK_INDICES = []
    for k in range(0,len(ACO)):
        location = np.where(PLANCK[1].data['NAME'] == PLANCK_NAMES[ACO[k]])[0]
        if len(location) > 0:
            PLANCK_INDICES.append(location[0])
        else:
            PLANCK_INDICES.append(-1)
    # Convert the M500c masses from Planck into M200c masses.
    # First, extract M500c mean and error data:
    M500c_Planck = np.zeros(len(ACO))
    M500c_Planck_upp = np.zeros(len(ACO))
    M500c_Planck_low = np.zeros(len(ACO))
    Z_PLANCK = PLANCK[1].data['REDSHIFT']
    bias_param = 0.8
    errBiasRat_upp = 0.2
    errBiasRat_low = 0.1
    for k in range(0,len(ACO)):
        if PLANCK_INDICES[k] != -1:
            M500c_Planck[k] = PLANCK[1].data['MSZ'][PLANCK_INDICES[k]]*1e14
            # Errors from scaling relationships:
            errScale_upp = PLANCK[1].data['MSZ_ERR_UP'][PLANCK_INDICES[k]]*1e14
            errScale_low = PLANCK[1].data['MSZ_ERR_LOW'][PLANCK_INDICES[k]]*1e14
            # Errors from uncertainty on mass bias, (1-b)
            errBias_upp = M500c_Planck[k]*bias_param/\
                (bias_param - errBiasRat_low) \
                -M500c_Planck[k]
            errBias_low = M500c_Planck[k] - \
                M500c_Planck[k]*bias_param/(bias_param + errBiasRat_upp)
            # Combined errors in quadrature:
            M500c_Planck_upp[k] = np.sqrt(errScale_upp**2 + errBias_upp**2)
            M500c_Planck_low[k] = np.sqrt(errScale_low**2 + errBias_low**2)
        else:
            M500c_Planck[k] = -1
            M500c_Planck_upp[k] = -1
            M500c_Planck_low[k] = -1
    # Now convert these to M200c:
    M200c_Planck = np.zeros(len(ACO))
    M200c_Planck_upp = np.zeros(len(ACO))
    M200c_Planck_low = np.zeros(len(ACO))
    D1 = 500
    D2 = 200
    for k in range(0,len(ACO)):
        if M500c_Planck[k] < 0:
            M200c_Planck[k] = -1
            M200c_Planck_upp[k] = -1
            M200c_Planck_low[k] = -1
        else:
            # Convert masses to M200c:
            mass200c = cosmology.convertCriticalMass(M500c_Planck[k],D1,D2=D2,\
                Om=Om0,z = Z_PLANCK[PLANCK_INDICES[k]],\
                Ol = 1 - Om0,h=h,returnError=True)
            M200c_Planck[k] = mass200c[0]
            # Errors on M500c estimate converted to M200c errors:
            errorM500c_upp = cosmology.convertCriticalMass(M500c_Planck[k] + \
                M500c_Planck_upp[k],D1,D2) - M200c_Planck[k]
            errorM500c_low = M200c_Planck[k] - \
                cosmology.convertCriticalMass(M500c_Planck[k] - \
                    M500c_Planck_low[k],D1,D2)
            # Errors from conversion to M200c 
            # (due to scatter in assumed concentration-mass relationship):
            errorCon_upp = mass200c[2]
            errorCon_low = mass200c[1]
            # Combine errors in quadrature to estimate the new error:
            M200c_Planck_low[k] = np.sqrt(errorCon_low**2 + errorM500c_low**2)
            M200c_Planck_upp[k] = np.sqrt(errorCon_upp**2 + errorM500c_upp**2)
    rhocrit = 2.7754e11*h**2
    # R200c and R500c virial radii:
    R200c_Planck = np.cbrt(3*M200c_Planck/(4*np.pi*200*rhocrit))
    R200c_Planck_upp = np.cbrt(3*(M200c_Planck + M200c_Planck_upp)/\
        (4*np.pi*200*rhocrit)) - R200c_Planck
    R200c_Planck_low = R200c_Planck - \
        np.cbrt(3*(M200c_Planck - M200c_Planck_low)/(4*np.pi*200*rhocrit))
    R500c_Planck = np.cbrt(3*M500c_Planck/(4*np.pi*500*rhocrit))
    R500c_Planck_upp = np.cbrt(3*(M500c_Planck + M500c_Planck_upp)/\
        (4*np.pi*200*rhocrit)) - R500c_Planck
    R500c_Planck_low = R500c_Planck - \
        np.cbrt(3*(M500c_Planck - M500c_Planck_low)/(4*np.pi*200*rhocrit))
    #ACO = [426,2147,1656,3627,3571,548,2199,2063,1367]
    ACT = fits.open(catFolder + "/ACT2020/" + \
        "DR5_cluster-catalog_v1.1_forSZDB.fits")
    PLANCK = fits.open(catFolder + "/PSZ2v1.fits")
    # Planck 2015 M200c masses:
    Planck2015 = {\
        'A1656':MassConstraint(M200c_Planck[2],M200c_Planck_low[2],\
            M200c_Planck_upp[2],R200c_Planck[2],0,0,method='SZ'),\
            'A2147':MassConstraint(M200c_Planck[1],M200c_Planck_low[1],\
            M200c_Planck_upp[1],R200c_Planck[1],0,0,method='SZ'),\
            'A3627':MassConstraint(M200c_Planck[3],M200c_Planck_low[3],\
            M200c_Planck_upp[3],R200c_Planck[3],0,0,method='SZ'),\
            'A3571':MassConstraint(M200c_Planck[4],M200c_Planck_low[4],\
            M200c_Planck_upp[4],R200c_Planck[4],0,0,method='SZ'),\
            'A548':MassConstraint(M200c_Planck[5],M200c_Planck_low[5],\
            M200c_Planck_upp[5],R200c_Planck[5],0,0,method='SZ'),\
            'A2199':MassConstraint(M200c_Planck[6],M200c_Planck_low[6],\
            M200c_Planck_upp[6],R200c_Planck[6],0,0,method='SZ'),\
            'A2063':MassConstraint(M200c_Planck[7],M200c_Planck_low[7],\
            M200c_Planck_upp[7],R200c_Planck[7],0,0,method='SZ'),\
            'A1367':MassConstraint(M200c_Planck[8],M200c_Planck_low[8],\
            M200c_Planck_upp[8],R200c_Planck[8],0,0,method='SZ')}
    # Planck 2015 M500c masses:
    Planck2015M500c = {\
        'A1656':MassConstraint(M500c_Planck[2],M500c_Planck_low[2],\
        M500c_Planck_upp[2],R500c_Planck[3],0,0,method='SZ'),\
        'A2147':MassConstraint(M500c_Planck[1],M500c_Planck_low[1],\
        M500c_Planck_upp[1],R500c_Planck[1],0,0,method='SZ'),\
        'A3627':MassConstraint(M500c_Planck[3],M500c_Planck_low[3],\
        M500c_Planck_upp[3],R500c_Planck[3],0,0,method='SZ'),\
        'A3571':MassConstraint(M500c_Planck[4],M500c_Planck_low[4],\
        M500c_Planck_upp[4],R500c_Planck[4],0,0,method='SZ'),\
        'A548':MassConstraint(M500c_Planck[5],M500c_Planck_low[5],\
        M500c_Planck_upp[5],R500c_Planck[5],0,0,method='SZ'),\
        'A2199':MassConstraint(M500c_Planck[6],M500c_Planck_low[6],\
        M500c_Planck_upp[6],R500c_Planck[6],0,0,method='SZ'),\
        'A2063':MassConstraint(M500c_Planck[7],M500c_Planck_low[7],\
        M500c_Planck_upp[7],R500c_Planck[7],0,0,method='SZ'),\
        'A1367':MassConstraint(M500c_Planck[8],M500c_Planck_low[8],\
        M500c_Planck_upp[8],R500c_Planck[8],0,0,method='SZ')}
    # Piffaretti et al. (2011) don't give errors on their X-ray masses for 
    # M500c, so we will assume that the fractional errors are the same as the 
    # mean of the M500c masses from Planck:
    errorFrac = np.array([(Planck2015M500c[x].massLow + \
        Planck2015M500c[x].massHigh)/\
        (2*Planck2015M500c[x].mass) for x in Planck2015M500c])
    meanErrorFrac = np.mean(errorFrac)
    meanErrorFrac_upp = 0.08 # Assume 20% errors, because MCXC don't give any???
    meanErrorFrac_low = 0.08
    # Piffaretti (2011), M500c masses:
    Piffaretti2011 = {\
        'A426':MassConstraint(6.1508e14,meanErrorFrac_low*6.1508e14,\
        meanErrorFrac_upp*6.1508e14,1.2856,0,0,method='X-Ray'),\
        'A1656':MassConstraint(4.2846e14,meanErrorFrac_low*4.2846e14,\
        meanErrorFrac_upp*4.2846,1.1378,0,0,method='X-Ray'),\
        'A1367':MassConstraint(2.1398e14,meanErrorFrac_low*2.1398e14,\
        meanErrorFrac_upp*2.1398e14,0.9032,0,0,method='X-Ray'),\
        'A2147':MassConstraint(2.4052e14,meanErrorFrac_low*2.4052e14,\
        meanErrorFrac_upp*2.4052e14,0.9351,0,0,method='X-Ray'),\
        'A3627':MassConstraint(2.1360e14,meanErrorFrac_low*2.1360e14,\
        meanErrorFrac_upp*2.1360e14,0.9042,0,0,method='X-Ray'),\
        'A3571':MassConstraint(4.5067e14,meanErrorFrac_low*4.5067e14,\
        meanErrorFrac_upp*4.5067e14,1.1514,0,0,method='X-Ray'),\
        'A548':MassConstraint(1.3823e14,meanErrorFrac_low*1.3823e14,\
        meanErrorFrac_upp*1.3823e14,0.7758,0,0,method='X-Ray'),\
        'A2199':MassConstraint(2.9626e14,meanErrorFrac_low*2.9626,\
        meanErrorFrac_upp*2.9626e14,1.0040,0,0,method='X-Ray'),\
        'A2063':MassConstraint(2.1598e14,meanErrorFrac_low*2.1598,\
        meanErrorFrac_upp*2.1598e14,0.9020,0,0,method='X-Ray')}
    # Piffaretti (2011) M200c masses, converted from M500c:
    mPiff = np.array([6.1508e14,2.4052e14,4.2846e14,2.1360e14,4.5067e14,\
        1.3823e14,2.9626e14,2.1598e14,2.1398e14],dtype=np.double)
    zPiff = np.array([0.0179,0.0353,0.0231,0.0157,0.0391,0.0420,\
        0.0299,0.0355,0.0214],dtype=np.double)
    mPiffLow = meanErrorFrac_low*mPiff
    mPiffHigh = meanErrorFrac_upp*mPiff
    mPiff200c = -np.ones(mPiff.shape)
    mPiff200cLow = -np.ones(mPiff.shape)
    mPiff200cHigh = -np.ones(mPiff.shape)
    for k in range(0,len(mPiff)):
        if mPiff[k] > 0:
            # Convert masses to M200c:
            mass200c = cosmology.convertCriticalMass(mPiff[k],D1,D2=D2,\
                Om=Om0,z = zPiff[k],Ol = 1 - Om0,h=h,returnError=True)
            mPiff200c[k] = mass200c[0]
            # Errors on M500c estimate converted to M200c errors:
            errorM500c_upp = cosmology.convertCriticalMass(mPiff[k] + \
                mPiffHigh[k],D1,D2) - mPiff200c[k]
            errorM500c_low = mPiff200c[k] - \
                cosmology.convertCriticalMass(mPiff[k] - mPiffLow[k],\
                D1,D2)
            # Errors from conversion to M200c 
            # (due to scatter in assumed concentration-mass relationship):
            errorCon_upp = mass200c[2]
            errorCon_low = mass200c[1]
            # Combine errors in quadrature to estimate the new error:
            mPiff200cLow[k] = np.sqrt(errorCon_low**2 + errorM500c_low**2)
            mPiff200cHigh[k] = np.sqrt(errorCon_upp**2 + errorM500c_upp**2)
    rPiff200c = -np.ones(mPiff.shape)
    rPiff200c[np.where(mPiff200c > 0)] = np.cbrt(\
        3*mPiff200c[np.where(mPiff200c > 0)]/(4*np.pi*200*rhocrit))
    Piffaretti2011_200c = {\
        'A426':MassConstraint(mPiff200c[0],mPiff200cLow[0],mPiff200cHigh[0],\
            rPiff200c[0],0,0,method='X-Ray'),\
        'A1656':MassConstraint(mPiff200c[2],mPiff200cLow[2],mPiff200cHigh[2],\
            rPiff200c[2],0,0,method='X-Ray'),\
        'A1367':MassConstraint(mPiff200c[8],mPiff200cLow[8],mPiff200cHigh[8],\
            rPiff200c[8],0,0,method='X-Ray'),\
        'A2147':MassConstraint(mPiff200c[1],mPiff200cLow[1],mPiff200cHigh[1],\
            rPiff200c[1],0,0,method='X-Ray'),\
        'A3627':MassConstraint(mPiff200c[3],mPiff200cLow[3],mPiff200cHigh[3],\
            rPiff200c[3],0,0,method='X-Ray'),\
        'A3571':MassConstraint(mPiff200c[4],mPiff200cLow[4],mPiff200cHigh[4],\
            rPiff200c[4],0,0,method='X-Ray'),\
        'A548':MassConstraint(mPiff200c[5],mPiff200cLow[5],mPiff200cHigh[5],\
            rPiff200c[5],0,0,method='X-Ray'),\
        'A2199':MassConstraint(mPiff200c[6],mPiff200cLow[6],mPiff200cHigh[6],\
            rPiff200c[6],0,0,method='X-Ray'),\
        'A2063':MassConstraint(mPiff200c[7],mPiff200cLow[7],mPiff200cHigh[7],\
            rPiff200c[7],0,0,method='X-Ray')}
    # Virial masses:
    Girardi1998 = {\
        'A1656':MassConstraint(0.497e15/h,0.057e15/h,0.068e15/h,1.64/h,0,0,\
        method='Dynamical'),\
        'A3571':MassConstraint(0.817e15/h,0.219e15/h,0.240e15/h,2.09/h,0,0,\
        method='Dynamical'),\
        'A2199':MassConstraint(0.571e15/h,0.121e15/h,0.156e15/h,1.60/h,0,0,\
        method='Dynamical'),\
        'A1367':MassConstraint(0.494e15/h,0.094e15/h,0.102e15/h,1.60/h,0,0,\
        method='Dynamical')}
    Rines2003 = {\
        'A1656':MassConstraint(85.5e13/h,5.3e13/h,5.3e13/h,1.50/h,0,0,\
        method='Dynamical'),\
        'A2199':MassConstraint(36.8e13/h,4.1e13/h,4.1e13/h,1.12/h,0,0,\
        method='Dynamical'),\
        'A1367':MassConstraint(40.9e13/h,6.4e13/h,6.4e13/h,1.18/h,0,0,\
        method='Dynamical')}
    Hanski1999 = {\
        'A426':MassConstraint(5.5e15/h,1.5e15/h,1.5e15/h,15/h,0,0,\
        method='Dynamical')}
    Escalera1994 = {\
        'A426':MassConstraint(2.556e15,0.468e15,0.468e15,2.50,0.24,0.24,\
        method='Dynamical'),\
        'A548':MassConstraint(1.105e15,0.115e15,0.115e15,2.11,0.10,0.10,\
        method='Dynamical')}
    # Converting virial masses to M200c:
    Escalera1994_200c = {}
    for constraint in Escalera1994:
        m200c = cosmology.convertCriticalMass(Escalera1994[constraint].mass,\
            D1,D2,type1 = 'virial',\
            z=abell_z[np.where(abell_n == int(constraint[1:]))][0],\
            Om=Om0,Ol = 1 - Om0,h=h,returnError=True)
        mcentre = m200c[0]
        mlow = np.sqrt(m200c[1]**2 + Escalera1994[constraint].massLow**2)
        mupp = np.sqrt(m200c[2]**2 + Escalera1994[constraint].massHigh**2)
        r200c = np.cbrt(3*mcentre/(4*np.pi*200*rhocrit))
        Escalera1994_200c[constraint] = MassConstraint(mcentre,mlow,mupp,\
            r200c,0,0,method=Escalera1994[constraint].method)
    Kopylova2013 = {\
        'A2147':MassConstraint(1.057e15,0.171e15,0.171e15,2.08/h,0,0,\
        method='Dynamical'),\
        'A2199':MassConstraint(7.09e14,1.25e14,1.25e14,1.82,0,0,\
        method='Dynamical'),\
        'A2197':MassConstraint(2.80e14,0.64e14,0.64e14,1.34,0,0,\
        method='Dynamical'),\
        'A2063':MassConstraint(0.728e15,0.18e15,0.13e15,1.83,0,0,\
        method='Dynamical'),\
        'A2151':MassConstraint(6.74e14,1.27e14,1.27e14,1.78,0,0,\
        method='Dynamical'),\
        'A2152':MassConstraint(0.75e14,0.30e14,0.30e14,0.86,0,0,\
        method='Dynamical'),\
        'A2052':MassConstraint(4.12e14,1.15e14,1.15e14,1.52,0,0,\
        method='Dynamical')}
    # Kopylova masses:
    sumMass2199 = Kopylova2013['A2199'].mass + Kopylova2013['A2197'].mass
    sumMassErrorLow2199 = np.sqrt(Kopylova2013['A2199'].massLow**2 + \
        Kopylova2013['A2197'].massLow**2)
    sumMassErrorHigh2199 = np.sqrt(Kopylova2013['A2199'].massHigh**2 + \
        Kopylova2013['A2197'].massHigh**2)
    sumMass2147 = Kopylova2013['A2147'].mass + Kopylova2013['A2151'].mass + \
        Kopylova2013['A2152'].mass
    sumMassErrorLow2147 = np.sqrt(Kopylova2013['A2147'].massLow**2 + \
        Kopylova2013['A2151'].massLow**2 + Kopylova2013['A2152'].massLow**2)
    sumMassErrorHigh2147 = np.sqrt(Kopylova2013['A2147'].massHigh**2 + \
        Kopylova2013['A2151'].massHigh**2 + Kopylova2013['A2152'].massHigh**2)
    sumMass2063 = Kopylova2013['A2063'].mass + Kopylova2013['A2052'].mass
    sumMassErrorLow2063 = np.sqrt(Kopylova2013['A2063'].massLow**2 + \
        Kopylova2013['A2052'].massLow**2)
    sumMassErrorHigh2063 = np.sqrt(Kopylova2013['A2063'].massHigh**2 + \
        Kopylova2013['A2052'].massHigh**2)
    Kopylova2013['SumA2199'] = MassConstraint(sumMass2199,\
        sumMassErrorLow2199,sumMassErrorHigh2199,
        Kopylova2013['A2199'].virial,0,0)
    Kopylova2013['SumA2147'] = MassConstraint(sumMass2147,sumMassErrorLow2147,\
        sumMassErrorHigh2147,Kopylova2013['A2199'].virial,0,0)
    Kopylova2013['SumA2063'] = MassConstraint(sumMass2063,sumMassErrorLow2063,\
        sumMassErrorHigh2063,Kopylova2013['A2199'].virial,0,0)
    # Weak Lensing masses:
    Kubo2007 = {\
        'A1656':MassConstraint(1.88e15/h,0.56e15/h,0.65e15/h,\
            1.99/h,0.22/h,0.21/h,'Weak Lensing')}
    M200c_2199 = 10**(14.66)
    M200c_2199_upp = 10**(14.66 + 0.22) - M200c_2199 
    M200c_2199_low = M200c_2199 - 10**(14.66 - 0.32)
    R200c_2199 = np.cbrt(3*(M200c_2199/h)/(4*np.pi*200*rhocrit))*h
    Kubo2009 = {\
        'A2199':MassConstraint(M200c_2199/h,M200c_2199_upp/h,M200c_2199_low/h,\
            R200c_2199/h,0,0,'Weak Lensing')}
    Gavazzi2009 = {\
        'A1656':MassConstraint(9.7e14/h,6.1e14/h,3.5e14/h,2.2/h,0,0,\
            method='Weak Lensing')}
    Sereno2017 = {\
        'A2063':MassConstraint(4.1e14,2.1e14,2.1e14,1.5,0,0,\
        method='Weak Lensing')}
    Okabe2014 = {\
        'A1656':MassConstraint(6.23e14/h,1.58e14/h,2.53e14/h,1.75,0,0,\
        method='Weak Lensing')}
    # Other Dynamical masses:
    Hughes1989 = {\
        'A1656':MassConstraint(1.85e15/0.5,0.24e15/0.5,0.24e15/0.5,5/0.5,0,0,\
        'Caustic')}
    Woudt2007 = {\
        'A3627':MassConstraint(Mmid/h,Mlow/h,Mupp/h,2.02/h,0,0,\
        method='Dynamical')}
    Woudt2007_200c = {}
    deltaA3627 = Woudt2007['A3627'].mass/\
        ((4*np.pi*(Woudt2007['A3627'].virial)**3/3)*rhocrit)
    for constraint in Woudt2007:
        m200c = cosmology.convertCriticalMass(Woudt2007[constraint].mass,\
            deltaA3627,D2,\
            type1 = 'critical',\
            z=abell_z[np.where(abell_n == int(constraint[1:]))][0],\
            Om=Om0,Ol = 1 - Om0,h=h,returnError=True)
        mcentre = m200c[0]
        mlow = np.sqrt(m200c[1]**2 + Woudt2007[constraint].massLow**2)
        mupp = np.sqrt(m200c[2]**2 + Woudt2007[constraint].massHigh**2)
        r200c = np.cbrt(3*mcentre/(4*np.pi*200*rhocrit))
        Woudt2007_200c[constraint] = MassConstraint(mcentre,mlow,mupp,\
            r200c,0,0,method=Woudt2007[constraint].method)
    Lopes2018 = {\
        'A2147':MassConstraint(27.05e14,6.07e14,10.67e14,2.84,0.21,0.37,\
        method='Dynamical'),\
        'A3571':MassConstraint(7.74e14,0.78e14,0.9e14,1.87,0.06,0.07,\
        method='Dynamical'),\
        'A2199':MassConstraint(9.73e14,2.30e14,4.19e14,2.02,0.16,0.29,\
        method='Dynamical'),\
        'A2151':MassConstraint(15.13e14,3.80e14,7.22e14,2.34,0.20,0.37,\
        method='Dynamical'),\
        'A2063':MassConstraint(5.60e14,0.53e14,0.63e14,1.68,0.05,0.06,\
        method='Dynamical'),\
        'A2052':MassConstraint(3.56e14,0.40e14,0.54e14,1.45,0.05,0.07,\
        method='Dynamical')}
    Lopes2018['SumA2199'] = MassConstraint(sumMass2199,sumMassErrorLow2199,\
        sumMassErrorHigh2199,Lopes2018['A2199'].virial,0,0)
    Lopes2018['SumA2147'] = MassConstraint(sumMass2147,sumMassErrorLow2147,\
        sumMassErrorHigh2147,Lopes2018['A2199'].virial,0,0)
    Lopes2018['SumA2063'] = MassConstraint(sumMass2063,sumMassErrorLow2063,\
        sumMassErrorHigh2063,Lopes2018['A2199'].virial,0,0)
    Babyk2013 = {\
        'A1656':MassConstraint(37.98e14,1.09e14,1.09e14,1.64,0,0,\
        method='Dynamical'),\
        'A3571':MassConstraint(18.69e14,2.55e14,2.55e14,2.09,0,0,\
        method='Dynamical'),\
        'A2199':MassConstraint(6.76e14,1.85e14,1.85e14,1.60,0,0,\
        method='Dynamical')}
    Lokas2003 = {\
        'A1656':MassConstraint(1.4e15/h,0.3*1.4e15/h,0.3*1.4e15/h,2.9/h,0,0,\
        method='Dynamical')}
    Aguerri2020 = {\
        'A426':MassConstraint(1.23e15,0.25e15,0.25e15,2.2,0,0,\
        method='Dynamical')}
    # X-Ray masses:
    Babyk2013Xray = {\
        'A1656':MassConstraint(41.94e14,1.09e14,1.09e14,1.64,0,0,\
        method='X-Ray'),\
        'A3571':MassConstraint(30.41e14,2.55e14,2.55e14,2.09,0,0,\
        method='X-Ray'),\
        'A2199':MassConstraint(12.38e14,1.85e14,1.85e14,1.60,0,0,\
        method='X-Ray')}
    Meusinger2020 = {\
        'A426':MassConstraint(2.6e15,0.3e15,0.3e15,2.8,0.1,0.1,\
        method='Dynamical')}
    Simionescu2011 = {\
        'A426':MassConstraint(6.65e14,0.46e14,0.43e14,1.79,0.04,0.04,\
        method='X-Ray')}
    Bohringer1996 = {\
        'A3627':MassConstraint(1.3e15,0.9e15,0.9e15,3/0.5,0,0,method='X-Ray')}
    Simionescu2011_200c = {}
    for constraint in Simionescu2011:
        m200c = cosmology.convertCriticalMass(Simionescu2011[constraint].mass,\
            D1,D2,type1 = 'critical',\
            z=abell_z[np.where(abell_n == int(constraint[1:]))][0],\
            Om=Om0,Ol = 1 - Om0,h=h,returnError=True)
        mcentre = m200c[0]
        mlow = np.sqrt(m200c[1]**2 + Simionescu2011[constraint].massLow**2)
        mupp = np.sqrt(m200c[2]**2 + Simionescu2011[constraint].massHigh**2)
        r200c = np.cbrt(3*mcentre/(4*np.pi*200*rhocrit))
        Simionescu2011_200c[constraint] = MassConstraint(mcentre,mlow,mupp,\
            r200c,0,0,method=Simionescu2011[constraint].method)
    # Dictionary to map references to constraints:
    refToConstraints = {'Girardi1998':Girardi1998,'Rines2003':Rines2003,\
        'Hanski1999':Hanski1999,\
        'Escalera1994':Escalera1994,'Kopylova2013':Kopylova2013,\
        'Kubo2007':Kubo2007,'Hughes1989':Hughes1989,\
        'Woudt2007':Woudt2007,'Lopes2018':Lopes2018,\
        'Babyk2013':Babyk2013,'Meusinger2020':Meusinger2020,\
        'Simionescu2011':Simionescu2011,\
        'Lokas2003':Lokas2003,'Bohringer1996':Bohringer1996,\
        'Aguerri2020':Aguerri2020,\
        'Planck2015':Planck2015,'Planck2015M500c':Planck2015M500c,\
        'Piffaretti2011':Piffaretti2011,'Babyk2013Xray':Babyk2013Xray,\
        'Gavazzi2009':Gavazzi2009,'Sereno2017':Sereno2017,\
        'Piffaretti2011_200c':Piffaretti2011_200c,'Kubo2009':Kubo2009,\
        'Escalera1994_200c':Escalera1994_200c,'Woudt2007_200c':Woudt2007_200c,\
        'Simionescu2011_200c':Simionescu2011_200c,'Okabe2014':Okabe2014}
    # Dictionary mapping reference to publication year
    refYear = {'Girardi1998':1998,'Rines2003':2003,'Hanski1999':1999,\
        'Escalera1994':1994,'Kopylova2013':2013,'Kubo2007':2007,\
        'Hughes1989':1989,'Woudt2007':2007,'Lopes2018':2018,\
        'Babyk2013':2013,'Meusinger2020':2020,'Simionescu2011':2011,\
        'Lokas2003':2003,'Bohringer1996':1996,'Aguerri2020':2020,\
        'Planck2015':2016,'Planck2015M500c':2016,'Piffaretti2011':2011,\
        'Babyk2013Xray':2013,'Gavazzi2009':2009,'Sereno2017':2017,\
        'Piffaretti2011_200c':2011,'Kubo2009':2009,\
        'Escalera1994_200c':1994,'Woudt2007_200c':2007,\
        'Simionescu2011_200c':2011,'Okabe2014':2014}
    # List of Abell clusters identified as centre of halos:
    clusterIndex = ['A426','A2147','A1656','A3627',\
        'A3571','A548','A2199','A2063','A1367']
    #[426,2147,1656,3627,3571,548,2197,2052,1367]
    # Full reference used in legend:
    refNames = {\
        'Hanski1999':'Hanski et al. (1999)',\
        'Lopes2018':'Lopes et al. (2018)',\
        'Kopylova2013':'Kopylova & Kopylov (2013)',\
        'Kubo2007':'Kubo et al. (2007)',\
        'Woudt2007':'Woudt et al. (2007)',\
        'Escalera1994':'Escalera et al. (1994)',\
        'Babyk2013':'Babyk and Vavilova (2013)',\
        'Hughes1989':'Hughes (1989)',\
        'Sum':'Sum of nearby clusters',\
        'Stopyra2021':'Simulation mass with \n ' + \
            'virial radius ($R_{200m}$)\n from halo finder',\
        'Girardi1998':'Girardi et al. (1998)',\
        'Rines2003':'Rines et al. (2003)',\
        'Meusinger2020':'Meusinger et al. (2020)',\
        'Simionescu2011':'Simionescu et al. (2011)',\
        'Lokas2003':'\u0141okas and Mamon (2003)',\
        'Bohringer1996':'B\u00F6hringer et al. (1996)',\
        'Aguerri2020':'Aguerri et al. (2020)',\
        'Planck2015':'Planck Collaboration (2016b)',\
        'Planck2015M500c':'Planck Collaboration (2016b)',\
        'Piffaretti2011':'Piffaretti et al. (2011)',\
        'Babyk2013Xray':'Babyk and Vavilova (2013)',\
        'Gavazzi2009':'Gavazzi et al. (2009)',\
        'Sereno2017':'Sereno et al. (2017)',\
        'Piffaretti2011_200c':'Piffaretti et al. (2011)',\
        'Kubo2009':'Kubo et al. (2009)',\
        'Escalera1994_200c':'Escalera et al. (1994)',\
        'Woudt2007_200c':'Woudt et al. (2007)',\
        'Simionescu2011_200c':'Simionescu et al. (2011)',\
        'Okabe2014':'Okabe et al. (2014)'}
    # Method to colour:
    methodToColour = {\
        'Dynamical':seabornColormap[0],'X-Ray':seabornColormap[1],\
        'SZ':seabornColormap[2],'Weak Lensing':seabornColormap[6],\
        'Caustic':seabornColormap[4]}
    # Colour associated to each reference:
    colorRef = {\
        'Hanski1999':seabornColormap[0],'Lopes2018':seabornColormap[1],\
        'Girardi1998':seabornColormap[1],'Kopylova2013':seabornColormap[2],\
        'Kubo2007':seabornColormap[3],'Woudt2007':seabornColormap[4],\
        'Escalera1994':seabornColormap[5],'Rines2003':seabornColormap[7],\
        'Babyk2013':seabornColormap[8],'Hughes1989':seabornColormap[9],\
        'Meusinger2020':seabornColormap[6],'Simionescu2011':seabornColormap[0],\
        'Lokas2003':seabornColormap[9],'Bohringer1996':seabornColormap[0],\
        'Aguerri2020':seabornColormap[0],\
        'Sum':'lightgrey','Stopyra2021':'lightgrey','Planck2015':'g',\
        'Planck2015M500c':seabornColormap[9],\
        'Piffaretti2011':seabornColormap[8],\
        'Babyk2013Xray':seabornColormap[8],'Gavazzi2009':seabornColormap[8],\
        'Sereno2017':seabornColormap[8],\
        'Piffaretti2011_200c':seabornColormap[8],\
        'Kubo2009':seabornColormap[3],'Escalera1994_200c':seabornColormap[5],\
        'Woudt2007_200c':seabornColormap[4],\
        'Simionescu2011_200c':seabornColormap[0],\
        'Okabe2014':seabornColormap[6]}
    # Specify the references to be used in the plot:
    reference = [['Meusinger2020','Aguerri2020','Escalera1994_200c',\
        'Simionescu2011_200c','Piffaretti2011_200c'],\
        ['Kopylova2013','Lopes2018','Planck2015','Piffaretti2011_200c'],\
        ['Kubo2007','Gavazzi2009','Planck2015','Piffaretti2011_200c',\
        'Rines2003','Babyk2013','Okabe2014','Babyk2013Xray'],\
        ['Woudt2007_200c','Planck2015','Piffaretti2011_200c'],\
        ['Lopes2018','Planck2015','Piffaretti2011_200c',\
        'Babyk2013Xray','Babyk2013'],\
        ['Escalera1994_200c','Planck2015','Piffaretti2011_200c'],\
        ['Kubo2009','Kopylova2013','Planck2015',\
                'Piffaretti2011_200c','Babyk2013Xray','Lopes2018'],\
        ['Sereno2017','Kopylova2013','Planck2015','Piffaretti2011_200c'],\
        ['Rines2003','Planck2015','Piffaretti2011_200c']]
    # Specify alignments for reference number:
    text_align = [['left' for l in k] for k in reference]
    text_align[7][0] = 'centre'
    text_align[7][1] = 'right'
    text_align[7][3] = 'right'
    text_align[5][1] = 'right'
    text_align[8][2] = 'right'
    text_align[4][3] = 'right'
    text_align[4][0] = 'right'
    text_align[3][1] = 'right'
    text_align[3][0] = 'right'
    text_align[6][3] = 'right'
    #text_align[6][1] = 'right'
    text_align[0][0] = 'right'
    text_align[0][1] = 0.0
    text_align[2][0] = 'right'
    text_align[2][1] = 'right'
    text_align[2][4] = 'right'
    text_align[1][2] = 'right'
    text_align[6][5] = 'right'
    # Fraction of each axis taken up by the vertical shading, 
    # measured from the bottom:
    bfrac = [[1.0 for l in k] for k in reference]
    bfrac[2][0] = 0.7
    bfrac[2][6] = 0.7
    bfrac[7][0] = 0.7
    #bfrac[6][1] = 0.7
    bfrac[4][0] = 0.7
    bfrac[4][1] = 0.7
    bfrac[3][1] = 0.7
    bfrac[0][0] = 0.7
    bfrac[0][4] = 0.7
    bfrac[6][0] = 0.7
    bfrac[6][5] = 0.7
    bfrac[7][2] = 0.7
    bfrac[2][7] = 0.7
    # Fraction measured from the top:
    tfrac = [[1.0 for l in k] for k in reference]
    tfrac[2][1] = 0.7
    # Name lists:
    clusterOrder = [426,2147,1656,3627,3545,548,2197,2052,1367]
    nameList = {426:'Perseus-Pisces (A426)',2147:'Hercules B (A2147)',\
        1656:'Coma (A1656)',3627:'Norma (A3627)',3545:'Shapley (A3571)',\
        548:'A548',2199:'Hercules A (A2199)',2063:'Hercules C (A2063)',\
        1367:'Leo (A1367)'}
    clusterSelection = [426,2147,1656,3627,3545,548,2199,2063,1367]
    nameListLargeClusters = [nameList[clusterSelection[k]] \
        for k in range(0,len(clusterOrder))]
    # Construct the plot showing all the constraints together:
    fontsize1 = 7
    fontsize2 = 7
    textwidth=7.1014
    textheight=9.0971
    fontname = 'serif'
    alpha=0.5
    fig, ax = plt.subplots(9,1,figsize=(textwidth,0.5*textwidth))
    refsToUse = []
    methodsToUse = []
    refYearsList = []
    colorsList = [[] for k in range(0,9)]
    for k in range(0,9):
        for l in range(0,len(reference[k])):
            colorsList[k].append(\
                methodToColour[refToConstraints[\
                reference[k][l]][clusterIndex[k]].method])
    sortedClusters = np.argsort(meanCriticalMass)
    textOffset = 0
    leftOffset = 0
    rightOffset = -0.02
    useMh = True
    showBorg = True
    if useMh:
        scaleMassh = 1
        scaleMass = h
    else:
        scaleMassh = h
        scaleMass = 1
    tick_list = np.array([1e14,3e14,5e14,1e15,2e15,3e15])
    for k in range(0,9):
        ks = sortedClusters[k]
        for l in range(0,len(reference[ks])):
            if refNames[reference[ks][l]] not in refsToUse:
                refsToUse.append(refNames[reference[ks][l]])
                refYearsList.append(refYear[reference[ks][l]])
    sortedRefs = np.argsort(refYearsList)
    plotDataMass = [[] for k in range(0,9)]
    plotDataRange = [[] for k in range(0,9)]
    plotDataNames = [[] for k in range(0,9)]
    plotDataClusters = []
    for k in range(0,9):
        ks = sortedClusters[k]
        if savePlotData:
            plotDataClusters.append(nameListLargeClusters[ks])
        ax[k].set_frame_on(False)
        ax[k].set_xscale('log')
        ax[k].set_ylabel(nameListLargeClusters[ks],rotation=0,
            fontfamily=fontname,fontsize=fontsize1,ha='right',va='center')
        ax[k].xaxis.set_visible(False)
        plt.setp(ax[k].spines.values(),visible=False)
        ax[k].tick_params(left=False,labelleft=False)
        ax[k].tick_params(axis='both',labelsize=7)
        # This mass is in Msol/h, so divide by h to get Msol:
        errHandle = ax[k].errorbar(meanCriticalMass[ks]/scaleMassh,0.5,
            xerr=stdErrorCriticalMass[ks]/scaleMassh,marker='o',color='k')
        if not showBorg:
            errHandle.remove()
        for l in range(0,len(reference[ks])):
            constraint = refToConstraints[reference[ks][l]][clusterIndex[ks]]
            if refToConstraints[reference[ks][l]][clusterIndex[ks]].method \
                not in methodsToUse:
                methodsToUse.append(\
                    refToConstraints[reference[ks][l]][clusterIndex[ks]].method)
            ax[k].axvspan(\
                constraint.mass*scaleMass - constraint.massLow*scaleMass,
                constraint.mass*scaleMass + constraint.massHigh*scaleMass,\
                alpha=alpha,
                color=colorsList[ks][l],ec=colorsList[ks][l],ymax=bfrac[ks][l],\
                ymin = 1.0 - tfrac[ks][l])
            if savePlotData:
                plotDataMass[k].append(constraint.mass*scaleMass)
                plotDataRange[k].append(np.array([\
                    constraint.mass*scaleMass - constraint.massLow*scaleMass,\
                    constraint.mass*scaleMass + constraint.massHigh*scaleMass]))
                plotDataNames[k].append(refNames[reference[ks][l]])
            # Select alignment for the reference label:
            refID = np.where(np.array(refsToUse)[sortedRefs]==\
                refNames[reference[ks][l]])[0][0] +1
            if text_align[ks][l] == 'left':
                refTextPosAxes = ax[k].transAxes.inverted().transform(\
                    ax[k].transData.transform((\
                    constraint.mass*scaleMass - \
                    constraint.massLow*scaleMass,0.5)))[0] + textOffset + \
                    leftOffset
            elif text_align[ks][l] == 'right':
                refTextPosAxes = ax[k].transAxes.inverted().transform(\
                    ax[k].transData.transform((\
                    constraint.mass*scaleMass + \
                    constraint.massHigh*scaleMass,0.5)))[0] + \
                    textOffset + rightOffset
                if refID > 9:
                    refTextPosAxes += rightOffset
            elif text_align[ks][l] == 'centre':
                refTextPosAxes = ax[k].transAxes.inverted().transform(\
                    ax[k].transData.transform((\
                    constraint.mass*scaleMass,0.5)))[0] + textOffset
            elif type(text_align[ks][l]) == type(1.0):
                refTextPosAxes = ax[k].transAxes.inverted().transform(\
                    ax[k].transData.transform((\
                    constraint.mass*scaleMass,0.5)))[0] + \
                    text_align[ks][l] + textOffset
            else:
                raise Exception('Unrecognised type alignment.')
            refTextPosData = ax[k].transData.inverted().transform(\
                    ax[k].transAxes.transform((refTextPosAxes,0.5)))[0]
            ax[k].annotate(str(refID),\
                (refTextPosData,0.5 + (5*bfrac[ks][l] - 4)/100),\
                fontfamily=fontname,fontsize=fontsize1)
        ax[k].set_xlim([1e14*scaleMass,3e15*scaleMass])
        ax[k].set_xticks([])
        ax[k].set_xticklabels([])
        ax[k].set_xticklabels([],minor=True)
    ax[8].axes.get_xaxis().set_visible(True)
    if useMh:
        ax[8].set_xlabel('Mass, $M_{200c}$ [$M_{\odot}h^{-1}$]',\
            fontfamily=fontname,fontsize=fontsize1)
    else:
        ax[8].set_xlabel('Mass, $M_{200c}$ [$M_{\odot}$]',fontfamily=fontname,\
            fontsize=fontsize1)
    plt.subplots_adjust(bottom=0.165,right=0.75,left=0.170)
    ax[8].tick_params(axis='both',labelsize=7)
    for axk in ax:
        axk.set_xticks(tick_list)
    ax[8].set_xticklabels(['$' + plot.scientificNotation(k) + '$' \
        for k in tick_list])
    # Fake legend:
    patchList = [mpatches.Patch(color=methodToColour[k],label=k,
        alpha=alpha,ec=methodToColour[k]) for k in methodsToUse]
    simLine = mlines.Line2D([],[],linestyle='-',marker='x',color='k',
        label='Simulation masses\n(colour = ref. for radius)')
    simLine2 = mlines.Line2D([],[],linestyle='-',marker='x',color='k',
        label='Simulation mass \nat observed radius' + \
        ' \n(color matches reference)')
    simLine3 = mlines.Line2D([],[],linestyle='-',marker='o',color='grey',
        label='Simulation virial\n masses ($M_{200m}$)')
    simLine4 = mlines.Line2D([],[],linestyle='-',marker='o',color='k',
        label='Simulation virial\n masses ($M_{200c}$)')
    simLine5 = mlines.Line2D([],[],linestyle='-',marker='o',color='lightgrey',
        label='Simulation virial\n masses ($M_{500c}$)')
    simLine6 = mlines.Line2D([],[],linestyle='-',marker='x',color='grey',
        label='BORG Particle virial\n masses ($M_{200c}$)')
    simLineGuide = mlines.Line2D([],[],linestyle=':',color='grey',\
        label='$10^{15}M_{\\odot}h^{-1}$')
    if showBorg:
        handles = [simLineGuide,simLine4]
    else:
        handles = [simLineGuide]
    # Sort references by date:
    refYearsListSorted = np.argsort(refYearsList)
    for k in range(0,len(patchList)):
        #handles.append(patchList[refYearsListSorted[k]])
        handles.append(patchList[k])
    refText1 = ''
    for k in range(0,np.min([17,len(refsToUse)])):
        refText1 = refText1 + str(k+1) + ': ' + np.array(refsToUse)[sortedRefs][k] +\
        '\n'
    ax[0].text(1.0,1.5,refText1,transform=ax[0].transAxes,\
        fontfamily=fontname,fontsize=fontsize1,va='top')
    if len(refsToUse) > 17:
        refText2 = ''
        for k in range(8,len(refsToUse)):
            refText2 = refText2 + str(k+1) + ': ' + refsToUse[k] + '\n'
        ax[0].text(1.1,1.5,refText2,transform=ax[0].transAxes,\
            fontfamily=fontname,fontsize=fontsize1,va='top')
    # Verticle line at mass threshold:
    for k in range(0,len(ax)):
        ax[k].axvline(x=1e15,ymin=-0.17,ymax=1,c="grey",linestyle=':',\
                    zorder=0, clip_on=False)
    #handles.append(simLine2)
    vertOffset = 0
    if showBorg:
        vertOffset = -0.2
    plt.legend(handles=handles,loc=(1.05,-0.3 + vertOffset),frameon=True,
        prop={"size":fontsize2,"family":"serif"})
    ax[0].set_title('Mass estimates of clusters in the local super-volume ' + \
        '($\\mathbf{r < 135 \\mathrm{\\bf\\,Mpc}}\\bf{h^{-1}}$)',\
        fontfamily=fontname,fontsize=fontsize1,fontweight='bold')
    plt.savefig(savename,dpi=500)
    if show:
        plt.show()
    if savePlotData:
        plotData = [plotDataMass,plotDataRange,plotDataNames,plotDataClusters]
        tools.savePickle(plotData,savename + ".data.p")
        return plotData












