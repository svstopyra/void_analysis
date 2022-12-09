# CONFIGURATION
from void_analysis import plot, snapedit, tools, simulation_tools, halos
from void_analysis import stacking, real_clusters, survey, cosmology
from void_analysis.tools import loadOrRecompute
from void_analysis.simulation_tools import ngPerLBin
from void_analysis.simulation_tools import biasNew, biasOld
from void_analysis.survey import radialCompleteness, surveyMask
from void_analysis import plot_utilities
import seaborn as sns
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
import numpy as np
import scipy
import pynbody
import astropy
import pickle
import gc
import copy
import h5py
import healpy
import matplotlib.pylab as plt
import os
import sys



# KE correction used to compute magnitudes. Used by PPTs:
def keCorr(z,fit = [-1.456552772320231,-0.7687913554110967]):
    #return -fit[0]*z# - fit[1]
    return 2.9*z

# Convert per-voxel to per-healpix patch average galaxy counts:
def getAllNgsToHealpix(ngList,hpIndices,sampleList,sampleFolder,nside,\
        recomputeData=False,nres=256):
    return [loadOrRecompute(sampleFolder + "sample" + str(sampleList[k]) + \
        "/ngHP.p",tools.getCountsInHealpixSlices,ngList[k],hpIndices,\
        nside=nside,nres=nres,_recomputeData=recomputeData) \
        for k in range(0,len(sampleList))]

# Generate data for the PPT plots:
def getPPTPlotData(nBins = 31,nClust=9,nMagBins = 16,N=256,\
        restartFile = 'new_chain_restart/merged_restart.h5',\
        snapNumList = [7000, 7200, 7400],samplesFolder = 'new_chain/',\
        surveyMaskPath = "./2mpp_data/",\
        Om0 = 0.3111,Ode0 = 0.6889,boxsize = 677.7,h=0.6766,Mstarh = -23.28,\
        mmin = 0.0,mmax = 12.5,recomputeData = False,rBinMin = 0.1,\
        rBinMax = 20,abell_nums = [426,2147,1656,3627,3571,548,2197,2063,1367],\
        nside = 4,nRadialSlices=10,rmax=600,tmppFile = "2mpp_data/2MPP.txt",\
        reductions = 4,iterations = 20,verbose=True,hpIndices=None,\
        centreMethod="density",catFolder="",\
        snapname="/gadget_full_forward_512/snapshot_001"):
    # Parameters:
        # nBins - Number of radial bins for galaxy counts in PPT
        # nClust - Number of clusters to do PPTs for.
        # nMagBins - Total number of absolute and apparent magnitude cuts used
        #            by BORG.
        # N - resolution of BORG constraints
        # restartFile - restart file for the BORG MCMC run. Used to extract the 
        #               healpix map.
        # snapNumList - MCMC samples used in the PPT analysis.
        # samplesFolder - folder where MCMC samples are found. 
        # surveyMaskPath - folder where 2MPP survey mask data is stored.
        # Om0 - Matter density fraction LCDM parameter
        # Ode0 - Dark energy density fraction LCDM parameter
        # boxsize - size of periodic domain in constrained simulations.
        # h - Hubble rate in units of 100 kms^{-1}Mpc^{-1}
        # Mstarh - Parameter of the Schecter function used by the survey mask
        # mmin - minimum apparent magnitude of objects used by BORG
        # mmax - maximum apparent magnitude of objects used by BORG
        # recomputeData - If true, recompute all data from scratch, rather than
        #               using cached data.
        # rBinMin - Minimum radius for PPTs
        # rBinMax - Maximum radius for PPTs.
        # abell_nums - Abell numbers of the clusters to perform PPTs for.
        # nside - nside parameter for healpix patches on the sky.
        # nRadialSlices - number of radial slices over each healpix patch
        # rmax - maximum radius for healpix slices
        # tmppFile - 2M++ catalogue file.
        # reductions - Number of times to reduce in half the sphere radius
        #               when computing the centres of mass.
        # iterations - Number of iterations per reduction when computing the
        #               centre of mass.
        # verbose - print indications of progress
    # Cosmology:
    cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
    # 2M++ catalogue data, and cluster locations:
    [combinedAbellN,combinedAbellPos,abell_nums] = \
        real_clusters.getCombinedAbellCatalogue(Om0 = 0.3111,Ode0 = 0.6889,\
            h=0.6766,catFolder=catFolder)
    clusterInd = [np.where(combinedAbellN == n)[0] for n in abell_nums]
    clusterLoc = np.zeros((len(clusterInd),3))
    for k in range(0,len(clusterInd)):
        if len(clusterInd[k]) == 1:
            clusterLoc[k,:] = combinedAbellPos[clusterInd[k][0],:]
        else:
            # Average positions:
            clusterLoc[k,:] = np.mean(combinedAbellPos[clusterInd[k],:],0)
    # Bias model data for each MCMC sample, and density fields:
    if verbose:
        print("Loading bias data...")
    biasData = [h5py.File(samplesFolder + "/sample" + str(k) + "/mcmc_" + \
        str(k) + ".h5",'r') for k in snapNumList]
    mcmcDen = [1.0 + sample['scalars']['BORG_final_density'][()] \
        for sample in biasData]
    if N < mcmcDen[0].shape[0]:
        print("Warning!!: requested resolution is lower than " + 
        "the MCMC file. Density field will be downgraded.")
        mcmcDen = [tools.downsample(den,int(den.shape[0]/N)) \
            for den in mcmcDen]
    elif N > mcmcDen[0].shape[0]:
        print("Warning!!: requested resolution is higher than " + 
        "the MCMC file. Density field will be interpolated.")
        mcmcDen = [scipy.ndimage.zoom(den,N/den.shape[0]) \
            for den in mcmcDen]
    mcmcDenLin = [np.reshape(den,N**3) for den in mcmcDen]
    mcmcDen_r = [np.reshape(den,(N,N,N),order='F') for den in mcmcDenLin]
    mcmcDenLin_r = [np.reshape(den,N**3) for den in mcmcDen_r]
    biasParam = [np.array([[sample['scalars']['galaxy_bias_' + str(k)][()] \
        for k in range(0,nMagBins)]]) for sample in  biasData]
    # Positions of density voxels, and KDtree for rapid access:
    if verbose:
        print("Computing KD-tree...")
    grid = snapedit.gridListPermutation(N,perm=(2,1,0))
    centroids = grid*boxsize/N + boxsize/(2*N)
    positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
    tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
        boxsize=boxsize)
    # Survey mask data:
    if verbose:
        print("Computing survey mask...")
    surveyMask11 = healpy.read_map(surveyMaskPath + "completeness_11_5.fits")
    surveyMask12 = healpy.read_map(surveyMaskPath + "completeness_12_5.fits")
    [mask,angularMask,radialMas,mask12,mask11] = tools.loadOrRecompute(\
        "surveyMask.p",surveyMask,\
        positions,surveyMask11,surveyMask12,cosmo,-0.94,\
        Mstarh,keCorr = keCorr,mmin=mmin,numericalIntegration=True,\
        mmax=mmax,splitApparent=True,splitAbsolute=True,returnComponents=True,\
        _recomputeData=recomputeData)
    # Obtain galaxy counts:
    nsamples = len(snapNumList)
    galaxyCountExp = np.zeros((nBins,nClust,nMagBins))
    galaxyCountsRobustAll = np.zeros((nBins,nClust,nsamples,nMagBins))
    # Obtain properties:
    rBins = np.linspace(rBinMin,rBinMax,nBins+1)
    wrappedPos = snapedit.wrap(clusterLoc + boxsize/2,boxsize)
    # Healpix indices for each voxel:
    if verbose:
        print("Loading healpix data...")
    if hpIndices is None:
        restart = h5py.File(restartFile)
        hpIndices = restart['scalars']['colormap3d'][()]
    hpIndicesLinear = hpIndices.reshape(N**3)
    # Get predicted galaxy counts in each voxel, for each MCMC sample:
    if verbose:
        print("Computing posterior predicted galaxy counts...")
    ngMCMC = np.vstack([tools.loadOrRecompute(samplesFolder + "sample" + \
            str(snapNumList[k]) + "/ngMCMC.p",ngPerLBin,\
            biasParam,return_samples=True,mask=mask,\
            accelerate=True,N=N,\
            delta = [mcmcDenLin_r[k]],contrast=False,sampleList=[0],\
            beta=biasParam[k][:,:,1],rhog = biasParam[k][:,:,3],\
            epsg=biasParam[k][:,:,2],\
            nmean=biasParam[k][:,:,0],biasModel = biasNew,\
            _recomputeData = recomputeData) \
            for k in range(0,nsamples)])
    # Convert MCMC galaxy counts in voxels to counts in each healpix patch:
    if verbose:
        print("Converting voxel counts to healpix patch counts...")
    ngHPMCMC = tools.loadOrRecompute("ngHPMCMC.p",getAllNgsToHealpix,ngMCMC,\
        hpIndices,snapNumList,samplesFolder,nside,nres=N,\
        _recomputeData=recomputeData)
    # Compute counts in each healpix pixel for 2M++ survey:
    if verbose:
        print("Computing 2M++ galaxy counts...")
    ng2MPP = np.reshape(tools.loadOrRecompute("mg2mppK3.p",\
        survey.griddedGalCountFromCatalogue,\
        cosmo,tmppFile=tmppFile,Kcorrection = True,N=N,\
        _recomputeData=recomputeData),(nMagBins,N**3))
    ngHP = tools.loadOrRecompute("ngHP3.p",tools.getCountsInHealpixSlices,\
        ng2MPP,hpIndices,nside=nside,nres=N,_recomputeData=recomputeData)
    # Amplitudes of the bias model in each healpix patch:
    if verbose:
        print("Computing bias patch amplitudes...")
    npixels = 12*(nside**2)
    Aalpha = np.zeros((nsamples,nMagBins,npixels*nRadialSlices))
    for k in range(0,nsamples):
        nz = np.where(ngHPMCMC[k] != 0.0)
        Aalpha[k][nz] = ngHP[nz]/ngHPMCMC[k][nz]
    # Get centres of clusters in each MCMC sample:
    if verbose:
        print("Computing cluster centres in each sample...")
    clusterCentresSim = []
    densityList = [np.reshape(den,N**3) for den in mcmcDen_r]
    for k in range(0,nsamples): 
        snapPath = samplesFolder + "sample" + str(snapNumList[k]) + snapname
        clusterCentresSim.append(\
            simulation_tools.getClusterCentres(clusterLoc,\
            snapPath = snapPath,fileSuffix = "clusters1",\
            recompute=recomputeData,density=densityList[k],boxsize=boxsize,\
            positions=positions,positionTree=tree,\
            method="density",reductions=reductions,\
            iterations=iterations))
    if verbose:
        print("Computing radial galaxy counts...")
    # Loop over all bins:
    for k in range(0,nBins):
        indices = tree.query_ball_point(wrappedPos,rBins[k+1])
        if centreMethod == "snapshot":
            indicesGad = [tree.query_ball_point(\
                -np.fliplr(snapedit.wrap(centres,boxsize)),\
                rBins[k+1]) for centres in clusterCentresSim]
        elif centreMethod == "density":
            indicesGad = [tree.query_ball_point(\
                snapedit.wrap(centres,boxsize),rBins[k+1]) \
                for centres in clusterCentresSim]
        if np.any(np.array(indices,dtype=bool)):
            for l in range(0,nClust):
                for m in range(0,nMagBins):
                    galaxyCountExp[k,l,m] = np.sum(ng2MPP[m][indices[l]])/\
                        (4*np.pi*rBins[k+1]**3/3)
                    for n in range(0,nsamples):
                        galaxyCountsRobustAll[k,l,n,m] += np.sum(\
                            Aalpha[n,m,hpIndicesLinear[indicesGad[n][l]]]*\
                            ngMCMC[n,m][indicesGad[n][l]])/\
                            (4*np.pi*rBins[k+1]**3/3)
    galaxyCountsRobust = np.mean(galaxyCountsRobustAll,2)
    # Convert density to galaxy counts:
    galaxyNumberCountsRobust = (4*np.pi*rBins[1:,None,None]**3/3)*\
        galaxyCountsRobust
    galaxyNumberCountExp = np.array((4*np.pi*rBins[1:,None,None]**3/3)*\
        galaxyCountExp,dtype=int)
    if verbose:
        print("Done.")
    return [galaxyNumberCountExp,galaxyNumberCountsRobust]


def getPPTForPoints(points,nBins = 31,nClust=9,nMagBins = 16,N=256,\
        restartFile = 'new_chain_restart/merged_restart.h5',\
        snapNumList = [7000, 7200, 7400],samplesFolder = 'new_chain/',\
        surveyMaskPath = "./2mpp_data/",\
        Om0 = 0.3111,Ode0 = 0.6889,boxsize = 677.7,h=0.6766,Mstarh = -23.28,\
        mmin = 0.0,mmax = 12.5,recomputeData = False,rBinMin = 0.1,\
        rBinMax = 20,abell_nums = [426,2147,1656,3627,3571,548,2197,2063,1367],\
        nside = 4,nRadialSlices=10,rmax=600,tmppFile = "2mpp_data/2MPP.txt",\
        reductions = 4,iterations = 20,verbose=True,hpIndices=None,\
        centreMethod="density",catFolder="",\
        snapname="/gadget_full_forward_512/snapshot_001",\
        returnAll=False):
    # Get tree of density voxel positions:
    grid = snapedit.gridListPermutation(N,perm=(2,1,0))
    centroids = grid*boxsize/N + boxsize/(2*N)
    positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
    tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
        boxsize=boxsize)
    # Cosmology:
    cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
    nsamples = len(snapNumList)
    # Heapix indices:
    if hpIndices is None:
        restart = h5py.File(restartFile)
        hpIndices = restart['scalars']['colormap3d'][()]
    hpIndicesLinear = hpIndices.reshape(N**3)
    # Get bias data:
    biasData = [h5py.File(samplesFolder + "/sample" + str(k) + "/mcmc_" + \
        str(k) + ".h5",'r') for k in snapNumList]
    mcmcDen = [1.0 + sample['scalars']['BORG_final_density'][()] \
        for sample in biasData]
    if N < mcmcDen[0].shape[0]:
        print("Warning!!: requested resolution is lower than " + 
        "the MCMC file. Density field will be downgraded.")
        mcmcDen = [tools.downsample(den,int(den.shape[0]/N)) \
            for den in mcmcDen]
    elif N > mcmcDen[0].shape[0]:
        print("Warning!!: requested resolution is higher than " + 
        "the MCMC file. Density field will be interpolated.")
        mcmcDen = [scipy.ndimage.zoom(den,N/den.shape[0]) \
            for den in mcmcDen]
    mcmcDenLin = [np.reshape(den,N**3) for den in mcmcDen]
    mcmcDen_r = [np.reshape(den,(N,N,N),order='F') for den in mcmcDenLin]
    mcmcDenLin_r = [np.reshape(den,N**3) for den in mcmcDen_r]
    biasParam = [np.array([[sample['scalars']['galaxy_bias_' + str(k)][()] \
        for k in range(0,nMagBins)]]) for sample in  biasData]
    # Survey mask:
    surveyMask11 = healpy.read_map(surveyMaskPath + "completeness_11_5.fits")
    surveyMask12 = healpy.read_map(surveyMaskPath + "completeness_12_5.fits")
    [mask,angularMask,radialMas,mask12,mask11] = tools.loadOrRecompute(\
        "surveyMask.p",surveyMask,\
        positions,surveyMask11,surveyMask12,cosmo,-0.94,\
        Mstarh,keCorr = keCorr,mmin=mmin,numericalIntegration=True,\
        mmax=mmax,splitApparent=True,splitAbsolute=True,returnComponents=True,\
        _recomputeData=recomputeData)
    # Get the posterior galaxy counts:
    ngMCMC = np.vstack([tools.loadOrRecompute(samplesFolder + "sample" + \
            str(snapNumList[k]) + "/ngMCMC.p",ngPerLBin,\
            biasParam,return_samples=True,mask=mask,\
            accelerate=True,N=N,\
            delta = [mcmcDenLin_r[k]],contrast=False,sampleList=[0],\
            beta=biasParam[k][:,:,1],rhog = biasParam[k][:,:,3],\
            epsg=biasParam[k][:,:,2],\
            nmean=biasParam[k][:,:,0],biasModel = biasNew,\
            _recomputeData = recomputeData) \
            for k in range(0,nsamples)])
    ngHPMCMC = tools.loadOrRecompute("ngHPMCMC.p",getAllNgsToHealpix,ngMCMC,\
        hpIndices,snapNumList,samplesFolder,nside,nres=N,\
        _recomputeData=recomputeData)
    # Compute counts in each healpix pixel for 2M++ survey:
    ng2MPP = np.reshape(tools.loadOrRecompute("mg2mppK3.p",\
        survey.griddedGalCountFromCatalogue,\
        cosmo,tmppFile=tmppFile,Kcorrection = True,N=N,\
        _recomputeData=recomputeData),(nMagBins,N**3))
    ngHP = tools.loadOrRecompute("ngHP3.p",tools.getCountsInHealpixSlices,\
        ng2MPP,hpIndices,nside=nside,nres=N,_recomputeData=recomputeData)
    # Aalpha:
    npixels = 12*(nside**2)
    Aalpha = np.zeros((nsamples,nMagBins,npixels*nRadialSlices))
    for k in range(0,nsamples):
        nz = np.where(ngHPMCMC[k] != 0.0)
        Aalpha[k][nz] = ngHP[nz]/ngHPMCMC[k][nz]
    # Perform the PPT:
    galaxyCountExp = np.zeros((nBins,nClust,nMagBins))
    galaxyCountsRobustAll = np.zeros((nBins,nClust,nsamples,nMagBins))
    rBins = np.linspace(rBinMin,rBinMax,nBins+1)
    wrappedPos = snapedit.wrap(points + boxsize/2,boxsize)
    for k in range(0,nBins):
        indices = tree.query_ball_point(wrappedPos,rBins[k+1])
        if np.any(np.array(indices,dtype=bool)):
            for l in range(0,nClust):
                for m in range(0,nMagBins):
                    galaxyCountExp[k,l,m] = np.sum(ng2MPP[m][indices[l]])/\
                        (4*np.pi*rBins[k+1]**3/3)
                    for n in range(0,nsamples):
                        galaxyCountsRobustAll[k,l,n,m] += np.sum(\
                            Aalpha[n,m,hpIndicesLinear[indices[l]]]*\
                            ngMCMC[n,m][indices[l]])/\
                            (4*np.pi*rBins[k+1]**3/3)
    galaxyCountsRobust = np.mean(galaxyCountsRobustAll,2)
    # Convert density to galaxy counts:
    galaxyNumberCountsRobust = (4*np.pi*rBins[1:,None,None]**3/3)*\
        galaxyCountsRobust
    galaxyNumberCountExp = np.array((4*np.pi*rBins[1:,None,None]**3/3)*\
        galaxyCountExp,dtype=int)
    if not returnAll:
        return [galaxyNumberCountExp,galaxyNumberCountsRobust]
    else:
        galaxyNumberCountsAllRobust = (4*np.pi*rBins[1:,None,None,None]**3/3)*\
            galaxyCountsRobustAll
        return [galaxyNumberCountExp,galaxyNumberCountsRobust,\
            galaxyNumberCountsAllRobust]

def getHMFAMFDataFromSnapshots(snapNumList,snapname,snapnameRev,samplesFolder,\
        fileSuffix = '',recomputeData = False,reCentreSnap=True,rSphere=135,\
        Om0 = 0.3111,boxsize=677.7,verbose=True,recomputeCentres=False):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" + \
        snapname) for snapNum in snapNumList]
    snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + snapnameRev) for snapNum in snapNumList]
    if reCentreSnap:
        for snap in snapList:
            tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
            snap.recentred = True
    else:
        for snap in snaplist:
            snap.recentred = False
    # Load or recompute mass and centre data:
    if recomputeCentres:
        if verbose:
            print("Computing halo centres...")
        massesAndCentres512 = loadOrRecompute(samplesFolder + \
            "all_halo_properties_512" + fileSuffix + ".p",\
            halos.getAllHaloCentresAndMasses,\
            snapList,boxsize,recompute=recomputeData,\
            _recomputeData=recomputeData)
        if verbose:
            print("Computing antihalo centres...")
        antihaloMassesAndCentres512 = loadOrRecompute(samplesFolder + \
            "all_antihalo_properties_512" + fileSuffix + ".p",\
            halos.getAllAntihaloCentresAndMassesFromSnaplists,\
            snapList,snapListRev,\
            recompute=recomputeData,_recomputeData=recomputeData)
    else:
        # Load from the pre-computed list:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapList]
        massesAndCentres512 = [[\
            tools.remapAntiHaloCentre(props[0],boxsize),props[1]] \
            for props in ahProps]
        antihaloMassesAndCentres512 = [[\
            tools.remapAntiHaloCentre(props[5],boxsize),props[3]] \
            for props in ahProps]
    haloCentres512 = [massesAndCentres512[k][0] \
        for k in range(0,len(snapNumList))]
    antihaloCentres512 = [antihaloMassesAndCentres512[k][0] \
        for k in range(0,len(snapNumList))]
    haloMasses512 = [massesAndCentres512[k][1] \
        for k in range(0,len(snapNumList))]
    antihaloMasses512 = [antihaloMassesAndCentres512[k][1] \
        for k in range(0,len(snapNumList))]
    # Get the halos and antihalos in the centre of the simulations:
    centralHalos512 = [tools.getAntiHalosInSphere(hcentres,135) \
        for hcentres in haloCentres512]
    constrainedHaloMasses512 = [haloMasses512[k][centralHalos512[k][0]] \
        for k in range(0,len(centralHalos512))]
    centralAntihalos512 = [tools.getAntiHalosInSphere(hcentres,135) \
        for hcentres in antihaloCentres512]
    constrainedAntihaloMasses512 = [antihaloMasses512[k][\
        centralAntihalos512[k][0]] \
        for k in range(0,len(centralAntihalos512))]
    # Get Mean density in local supervolume:
    if verbose:
        print("Computing local supervolume density...")
    mUnit = snapList[0]['mass'][0]*1e10
    volSphere = 4*np.pi*rSphere**3/3
    rhoCrit = 2.7754e11
    rhoMean = rhoCrit*Om0
    deltaList = []
    for k in range(0,len(snapList)):
        snap = pynbody.load(samplesFolder + "sample" \
            + str(snapNumList[k]) + "/" + snapname)
        gc.collect() # Clear memory of the previous snapshot
        tree = tools.getKDTree(snap)
        gc.collect()
        deltaList.append(mUnit*tree.query_ball_point(\
            [boxsize/2,boxsize/2,boxsize/2],135,\
            workers=-1,return_length=True)/(volSphere*rhoMean) - 1.0)
    deltaListMean = np.mean(deltaList)
    deltaListError = np.std(deltaList)/np.sqrt(len(snapList))
    if verbose:
        print("Done.")
    return [constrainedHaloMasses512,constrainedAntihaloMasses512,\
        deltaListMean,deltaListError]

def getUnconstrainedHMFAMFData(snapNumListUncon,snapName,snapNameRev,\
        unconstrainedFolder,deltaListMean,deltaListError,\
        fileSuffix = '',recomputeData=False,boxsize=677.7,verbose=True,\
        reCentreSnaps = True,Om0=0.3111,rSphere=135,nRandCentres = 10000,\
        recomputeCentres = False,randomSeed=1000,meanDensityMethod="selection",\
        meanThreshold=0.02,nRandMax=100):
    # Load unconstrained snaps:
    if verbose:
        print("Loading snapshots...")
    snapListUnconstrained = [pynbody.load(unconstrainedFolder + "sample" \
        + str(snapNum) + "/" + snapName) for snapNum in snapNumListUncon]
    snapListUnconstrainedRev = [pynbody.load(unconstrainedFolder + \
            "sample" + str(snapNum) + "/" +  snapNameRev) \
            for snapNum in snapNumListUncon]
    # recentre snaps:
    # Get halo and antihalo centres:
    if recomputeCentres:
        if reCentreSnaps:
            for snap in snapListUnconstrained:
                tools.remapBORGSimulation(snap)
                snap.recentred = True
        else:
            for snap in snapListUnconstrained:
                snap.recentred = False
        if verbose:
            print("Computing halo centres...")
        massesAndCentres_512uncon = tools.loadOrRecompute(\
            unconstrainedFolder + "all_halo_properties_512" + fileSuffix + \
            ".p",halos.getAllHaloCentresAndMasses,snapListUnconstrained,\
            boxsize,_recomputeData=recomputeData)
        if verbose:
            print("Computing antihalo centres...")
        antihaloMassesAndCentresUn512 = loadOrRecompute(unconstrainedFolder + \
                "all_antihalo_properties_512" + fileSuffix + ".p",\
                halos.getAllAntihaloCentresAndMassesFromSnaplists,\
                snapListUnconstrained,snapListUnconstrainedRev,\
                recompute=recomputeData,_recomputeData=recomputeData)
    else:
        # Load from the pre-computed list:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapListUnconstrained]
        massesAndCentres_512uncon = [[\
            tools.remapAntiHaloCentre(props[0],boxsize),props[1]] \
            for props in ahProps]
        antihaloMassesAndCentresUn512 = [[\
            tools.remapAntiHaloCentre(props[5],boxsize),props[3]] \
            for props in ahProps]
    haloMasses512uncon = [massesAndCentres_512uncon[k][1] \
        for k in range(0,len(snapNumListUncon))]
    antihaloMasses512uncon = [antihaloMassesAndCentresUn512[k][1] \
        for k in range(0,len(snapNumListUncon))]
    haloCentres512uncon = [massesAndCentres_512uncon[k][0] \
        for k in range(0,len(snapNumListUncon))]
    antihaloCentres512uncon = [antihaloMassesAndCentresUn512[k][0] \
        for k in range(0,len(snapNumListUncon))]
    # Find regions with comparable underdensity:
    if verbose:
        print("Computing random centres and overdensities...")
    rhoCrit = 2.7754e11
    rhoMean = rhoCrit*Om0
    volSphere = 4*np.pi*rSphere**3/3
    # Seed RNG:
    np.random.seed(randomSeed)
    # Get random selection of centres:
    randCentres = np.random.random((nRandCentres,3))*boxsize
    mUnit = snapListUnconstrained[0]['mass'][0]*1e10
    randOverDen = []
    for k in range(0,len(snapListUnconstrained)):
        snap = pynbody.load(unconstrainedFolder + "sample" \
            + str(snapNumListUncon[k]) + "/" + snapName)
        gc.collect() # Clear memory of the previous snapshot
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
        gc.collect()
        randOverDen.append(mUnit*tree.query_ball_point(randCentres,rSphere,\
            workers=-1,return_length=True)/(volSphere*rhoMean) - 1.0)
    # Halos/antihalos in regions with similar underdensity:
    if verbose:
        print("Finding halos in similar density regions...")
    comparableUnfilt = [np.where((delta >= deltaListMean - deltaListError) & \
        (delta < deltaListMean + deltaListError))[0] \
        for delta in randOverDen]
    comparable = [comp[0:np.min([nRandMax,len(comp)])] \
        for comp in comparableUnfilt]
    comparableCentresList = [randCentres[indices,:] for indices in comparable]
    comparableHalos = [[tools.getAntiHalosInSphere(\
        snapedit.wrap(haloCentres512uncon[k] + boxsize/2,boxsize),rSphere,\
        origin = comparableCentresList[k][l]) \
        for l in range(0,len(comparableCentresList[k]))]
        for k in range(0,len(snapListUnconstrained))]
    comparableHaloMasses = [[haloMasses512uncon[k][comparableHalos[k][l][0]] \
        for l in range(0,len(comparableHalos[k]))] \
        for k in range(0,len(snapListUnconstrained))]
    comparableAntihalos = [[tools.getAntiHalosInSphere(\
        snapedit.wrap(antihaloCentres512uncon[k] + boxsize/2,boxsize),rSphere,\
        origin = comparableCentresList[k][l]) \
        for l in range(0,len(comparableCentresList[k]))]
        for k in range(0,len(snapListUnconstrained))]
    comparableAntihaloMasses = [[\
        antihaloMasses512uncon[k][comparableAntihalos[k][l][0]] \
        for l in range(0,len(comparableAntihalos[k]))] \
        for k in range(0,len(snapListUnconstrained))]
    if meanDensityMethod == "central":
        centralHalos = [tools.getAntiHalosInSphere(hcentres,rSphere) \
                for hcentres in haloCentres512uncon]
        centralAntihalos = [tools.getAntiHalosInSphere(hcentres,rSphere) \
                for hcentres in antihaloCentres512uncon]
        centralHaloMasses = [\
                haloMasses512uncon[k][centralHalos[k][0]] \
                for k in range(0,len(centralHalos))]
        centralAntihaloMasses = [\
                antihaloMasses512uncon[k][centralAntihalos[k][0]] \
                for k in range(0,len(centralAntihalos))]
    elif meanDensityMethod == "selection":
        meanDensityRegionsUnfilt = [\
            np.where((delta >= - meanThreshold) & \
            (delta < meanThreshold))[0] \
            for delta in randOverDen]
        meanDensityRegions = [comp[0:np.min([nRandMax,len(comp)])] \
        for comp in meanDensityRegionsUnfilt]
        meanDensityCentresList = [randCentres[indices,:] \
            for indices in meanDensityRegions]
        centralHalos = [[tools.getAntiHalosInSphere(\
            snapedit.wrap(haloCentres512uncon[k] + boxsize/2,boxsize),rSphere,\
            origin = meanDensityCentresList[k][l]) \
            for l in range(0,len(meanDensityCentresList[k]))]
            for k in range(0,len(snapListUnconstrained))]
        centralHaloMasses = [[haloMasses512uncon[k][\
            centralHalos[k][l][0]] \
            for l in range(0,len(centralHalos[k]))] \
            for k in range(0,len(snapListUnconstrained))]
        centralAntihalos = [[tools.getAntiHalosInSphere(\
            snapedit.wrap(antihaloCentres512uncon[k] + boxsize/2,boxsize),\
            rSphere,origin = meanDensityCentresList[k][l]) \
            for l in range(0,len(meanDensityCentresList[k]))]
            for k in range(0,len(snapListUnconstrained))]
        centralAntihaloMasses = [[\
            antihaloMasses512uncon[k][centralAntihalos[k][l][0]] \
            for l in range(0,len(centralAntihalos[k]))] \
            for k in range(0,len(snapListUnconstrained))]
    else:
        raise Exception("Unrecognise mean density method.")
    if verbose:
        print("Done.")
    return [comparableHalos,comparableHaloMasses,\
            comparableAntihalos,comparableAntihaloMasses,\
            centralHalos,centralAntihalos,\
            centralHaloMasses,centralAntihaloMasses]


# Generate HMF/AMF data:
def getHMFAMFData(snapNumList,snapNumListOld,snapNumListUncon,\
        snapNumListUnconOld,samplesFolder="new_chain/",\
        samplesFolderOld = "./",recomputeData=False,\
        boxsize = 677.7,snapnameNew = "gadget_full_forward_512/snapshot_001",\
        snapnameNewRev = "gadget_full_reverse_512/snapshot_001",\
        snapnameOld = "forward_output/snapshot_006",\
        snapnameOldRev = "reverse_output/snapshot_006",\
        unconstrainedFolderNew = "new_chain/unconstrained_samples/",\
        unconstrainedFolderOld = "unconstrainedSamples/",verbose=True,\
        reCentreSnaps = True,Om0=0.3111,rSphere=135,nRandCentres = 10000,\
        meanDensityMethod = "selection",meanThreshold=0.02):
    if type(recomputeData) == bool:
        recomputeDataList = [recomputeData for k in range(0,4)]
    elif type(recomputeData) == list:
        if len(recomputeData) != 4:
            raise Exception("Invalid number of elements in recomputeData.")
        recomputeDataList = recomputeData
    else:
        raise Exception("Invalid recomputeData argument")
    # New snapshots, constrained:
    [constrainedHaloMasses512New,constrainedAntihaloMasses512New,\
        deltaListMeanNew,deltaListErrorNew] = tools.loadOrRecompute(\
        "constrained_new.p",\
        getHMFAMFDataFromSnapshots,\
        snapNumList,snapnameNew,snapnameNewRev,samplesFolder,\
        recomputeData = recomputeDataList[0],reCentreSnap=reCentreSnaps,\
        rSphere=rSphere,Om0 = Om0,boxsize=boxsize,verbose=verbose,\
        _recomputeData=recomputeDataList[0])
    gc.collect()
    # Old snapshots, constrained:
    [constrainedHaloMasses512Old,constrainedAntihaloMasses512Old,\
        deltaListMeanOld,deltaListErrorOld] = tools.loadOrRecompute(\
        "constrained_old.p",\
        getHMFAMFDataFromSnapshots,\
        snapNumListOld,snapnameOld,snapnameOldRev,samplesFolderOld,\
        recomputeData=recomputeDataList[1],reCentreSnap=reCentreSnaps,\
        rSphere=rSphere,Om0 = Om0,boxsize=boxsize,verbose=verbose,\
        fileSuffix = '_old',_recomputeData=recomputeDataList[1])
    gc.collect()
    # Unconstrained halos/antihalos with similar underdensity:
    [comparableHalosNew,comparableHaloMassesNew,\
            comparableAntihalosNew,comparableAntihaloMassesNew,\
            centralHalosNew,centralAntihalosNew,\
            centralHaloMassesNew,centralAntihaloMassesNew] = \
                tools.loadOrRecompute("unconstrained_new.p",\
                    getUnconstrainedHMFAMFData,\
                    snapNumListUncon,snapnameNew,\
                    snapnameNewRev,unconstrainedFolderNew,deltaListMeanNew,\
                    deltaListErrorNew,boxsize=boxsize,\
                    reCentreSnaps = reCentreSnaps,Om0=Om0,rSphere=rSphere,\
                    nRandCentres=nRandCentres,verbose=verbose,\
                    recomputeData=recomputeDataList[2],\
                    meanThreshold=meanThreshold,\
                    meanDensityMethod=meanDensityMethod,\
                    _recomputeData=recomputeDataList[2])
    gc.collect()
    [comparableHalosOld,comparableHaloMassesOld,\
            comparableAntihalosOld,comparableAntihaloMassesOld,\
            centralHalosOld,centralAntihalosOld,\
            centralHaloMassesOld,centralAntihaloMassesOld] = \
                tools.loadOrRecompute("unconstrained_old.p",\
                    getUnconstrainedHMFAMFData,\
                    snapNumListUnconOld,snapnameOld,\
                    snapnameOldRev,unconstrainedFolderOld,deltaListMeanOld,\
                    deltaListErrorOld,boxsize=boxsize,\
                    reCentreSnaps = reCentreSnaps,Om0=Om0,rSphere=rSphere,\
                    nRandCentres=nRandCentres,verbose=verbose,\
                    fileSuffix = '_old',recomputeData=recomputeDataList[3],\
                    meanThreshold=meanThreshold,\
                    meanDensityMethod = meanDensityMethod,\
                    _recomputeData=recomputeDataList[3])
    gc.collect()
    return [constrainedHaloMasses512New,constrainedAntihaloMasses512New,\
        deltaListMeanNew,deltaListErrorNew,\
        constrainedHaloMasses512Old,constrainedAntihaloMasses512Old,\
        deltaListMeanOld,deltaListErrorOld,\
        comparableHalosNew,comparableHaloMassesNew,\
        comparableAntihalosNew,comparableAntihaloMassesNew,\
        centralHalosNew,centralAntihalosNew,\
        centralHaloMassesNew,centralAntihaloMassesNew,\
        comparableHalosOld,comparableHaloMassesOld,\
        comparableAntihalosOld,comparableAntihaloMassesOld,\
        centralHalosOld,centralAntihalosOld,\
        centralHaloMassesOld,centralAntihaloMassesOld]



def getPartialPairCountsAndVols(snapNameList,antihaloRadii,antihaloMassesList,\
        ahCentresList,vorVols,rBins,method,\
        rMin,rMax,mMin,mMax,boxsize,filterListToApply=None):
    newPairCounts = []
    newVolumesList = []
    filtersList = []
    for ns in range(0,len(snapNameList)):
        if filterListToApply is None:
            filterToApply = np.where((antihaloRadii[ns] > rMin) & \
                (antihaloRadii[ns] < rMax) & \
                (antihaloMassesList[ns] > mMin) & \
                (antihaloMassesList[ns] <= mMax))[0]
        else:
            filterToApply = filterListToApply[ns]
        snap = tools.getPynbodySnap(snapNameList[ns])
        gc.collect()
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
        gc.collect()
        [pairs,vols] = stacking.getPairCounts(\
            ahCentresList[ns][filterToApply,:],\
            antihaloRadii[ns][filterToApply],snap,rBins,\
            nThreads=-1,tree=tree,\
            method=method,vorVolumes=vorVols[ns])
        newPairCounts.append(pairs)
        newVolumesList.append(vols)
        filtersList.append(filterToApply)
    return [newPairCounts,newVolumesList,filtersList]


def getCentreListUnconstrained(snapListUnconstrained,
        randomSeed = 1000,numDenSamples = 1000,rSphere = 135,\
        densityRange = [-0.051,-0.049]):
    np.random.seed(randomSeed)
    boxsize = snapListUnconstrained[0].properties['boxsize'].ratio(\
        "Mpc a h**-1")
    sampleCentresUnmapped = np.random.random((numDenSamples,3))*boxsize
    sampleCentresMapped = tools.remapAntiHaloCentre(sampleCentresUnmapped,\
        boxsize)
    N = int(np.cbrt(len(snapListUnconstrained[0])))
    Om = snapListUnconstrained[0].properties['omegaM0']
    rhoM = 2.7754e11*Om
    mUnit = rhoM*(boxsize/N)**3
    densitiesInCentres = []
    centreListUn = []
    denListUn = []
    for ns in range(0,len(snapListUnconstrained)):
        snap = pynbody.load(snapListUnconstrained[ns].filename)
        gc.collect()
        tree = tools.getKDTree(snap)
        gc.collect()
        den = mUnit*tree.query_ball_point(\
            sampleCentresUnmapped,\
            rSphere,return_length=True,workers=-1)/\
            (4*np.pi*rhoM*rSphere**3/3) - 1.0
        densitiesInCentres.append(den)
        centreListUn.append(tools.remapAntiHaloCentre(\
            sampleCentresMapped[(den > densityRange[0]) & \
            (den <= densityRange[1]),:],boxsize))
        denListUn.append(den[(den > densityRange[0]) & \
            (den <= densityRange[1])])
        gc.collect()
    return [centreListUn,densitiesInCentres,denListUn]

def getVoidProfilesData(snapNumList,snapNumListUncon,\
        snapList = None,snapListRev = None,\
        samplesFolder="new_chain/",\
        unconstrainedFolder="new_chain/unconstrained_samples/",\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        reCentreSnaps = False,N=512,boxsize=677.7,mMin = 1e14,mMax = 1e15,\
        rMin=5,rMax=25,verbose=True,combineSims=False,\
        method="poisson",errorType = "Weighted",\
        unconstrainedCentreList = np.array([[0,0,0]]),rSphere=135,\
        additionalConditions = None,densityRange=None,numDenSamples = 1000,\
        randomSeed = 1000,redoPairCounts = False,\
        rEffMax=3.0,rEffMin=0.0,nBins=31,pairCountsListUn=None,\
        volumesListUn=None,pairCountsList=None,volumesList=None,\
        ahPropsConstrained = None,ahPropsUnconstrained = None,\
        snapListUnconstrained=None,snapListUnconstrainedRev=None,\
        data_folder = "./",recomputeData=True):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    if snapList is None:
        snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapname) for snapNum in snapNumList]
    if snapListRev is None:
        snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapnameRev) for snapNum in snapNumList]
    # Load unconstrained snaps:
    if verbose:
        print("Loading snapshots...")
    if snapListUnconstrained is None:
        snapListUnconstrained = [pynbody.load(unconstrainedFolder + "sample" \
            + str(snapNum) + "/" + snapname) for snapNum in snapNumListUncon]
    if snapListUnconstrainedRev is None:
        snapListUnconstrainedRev = [pynbody.load(unconstrainedFolder + \
                "sample" + str(snapNum) + "/" + snapnameRev) \
                for snapNum in snapNumListUncon]
    if ahPropsConstrained is None:
        ahPropsConstrained = [tools.loadPickle(snap.filename + \
            ".AHproperties.p") \
            for snap in snapList]
    if ahPropsUnconstrained is None:
        ahPropsUnconstrained = [tools.loadPickle(snap.filename + \
            ".AHproperties.p")\
            for snap in snapListUnconstrained]
    snapNameList = [snap.filename for snap in snapList]
    snapNameListUn = [snap.filename for snap in snapListUnconstrained]
    nbar = (N/boxsize)**3
    # Constrained antihalo properties:
    ahCentresList = [props[5] \
        for props in ahPropsConstrained]
    ahCentresListRemap = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahPropsConstrained]
    antihaloMassesList = [props[3] for props in ahPropsConstrained]
    antihaloRadii = [props[7] for props in ahPropsConstrained]
    deltaCentralList = [props[11] for props in ahPropsConstrained]
    deltaMeanList = [props[12] for props in ahPropsConstrained]
    rBins = np.linspace(rEffMin,rEffMax,nBins)
    rBinStackCentres = plot.binCentres(rBins)
    centralAntihalosCon = [tools.getAntiHalosInSphere(hcentres,rSphere) \
        for hcentres in ahCentresListRemap]
    # Unconstrained antihalo properties:
    ahCentresListUn = [props[5] \
        for props in ahPropsUnconstrained]
    ahCentresListRemapUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahPropsUnconstrained]
    antihaloMassesListUn = [props[3] for props in ahPropsUnconstrained]
    antihaloRadiiUn = [props[7] for props in ahPropsUnconstrained]
    deltaCentralListUn = [props[11] for props in ahPropsUnconstrained]
    deltaMeanListUn = [props[12] for props in ahPropsUnconstrained]
    vorVols = [props[4] for props in ahPropsConstrained]
    vorVolsUn = [props[4] for props in ahPropsUnconstrained]
    if not redoPairCounts:
        if pairCountsListUn is None:
            pairCountsListUn = [props[9] for props in ahPropsUnconstrained]
        if volumesListUn is None:
            volumesListUn = [props[10] for props in ahPropsUnconstrained]
        if pairCountsList is None:
            pairCountsList = [props[9] for props in ahPropsConstrained]
        if volumesList is None:
            volumesList = [props[10] for props in ahPropsConstrained]
    # Density sampling, if specified:
    if densityRange is None:
        centreListUn = unconstrainedCentreList
    else:
        # Sample densities in each sphere to get densities in a particular
        # range:
        np.random.seed(randomSeed)
        if len(densityRange) != 2:
            raise Exception("Invalid densityRange")
        sampleCentresUnmapped = np.random.random((numDenSamples,3))*boxsize
        sampleCentresMapped = tools.remapAntiHaloCentre(sampleCentresUnmapped,\
            boxsize)
        unconstrainedTrees = [tools.getKDTree(snap) \
            for snap in snapListUnconstrained]
        Om = snapListUnconstrained[0].properties['omegaM0']
        rhoM = 2.7754e11*Om
        mUnit = rhoM*(boxsize/N)**3
        densitiesInCentres = [mUnit*tree.query_ball_point(\
            sampleCentresUnmapped,\
            rSphere,return_length=True,workers=-1)/\
            (4*np.pi*rhoM*rSphere**3/3) - 1.0 \
            for tree in unconstrainedTrees]
        centreListUn = [tools.remapAntiHaloCentre(\
            sampleCentresMapped[(den > densityRange[0]) & \
            (den <= densityRange[1]),:],boxsize) for den in densitiesInCentres]
        denListUn = [den[(den > densityRange[0]) & \
            (den <= densityRange[1])] for den in densitiesInCentres]
        lengths = np.array([len(cen) for cen in centreListUn])
        if np.any(lengths < 1):
            print(lengths)
            raise Exception("Did not find random centres with appropriate " + \
                "density. Expand density range, or increase number of samples.")
    if type(centreListUn) == list:
        centralAntihalosUn = [[tools.getAntiHalosInSphere(ahCentresListRemapUn[ns],\
            rSphere,origin=centre) for centre in centreListUn[ns]] \
            for ns in range(0,len(snapListUnconstrained))]
    elif type(centreListUn) == np.ndarray:
        centralAntihalosUn = [[tools.getAntiHalosInSphere(ahCentresListRemapUn[ns],\
            rSphere,origin=centre) for centre in centreListUn] \
            for ns in range(0,len(snapListUnconstrained))]
    else:
        raise Exception("Unconstrained centre list format not understood.")
    if additionalConditions is None:
        additionalConditions = [np.ones(len(centres),dtype=bool) \
            for centres in ahCentresListRemap]
    # Select antihalos in the central region:
    conditionList = [(deltaCentralList[ns] < 0) & \
        (centralAntihalosCon[ns][1]) & additionalConditions[ns] \
        for ns in range(0,len(snapNumList))]
    conditionListUn = [[(deltaCentralListUn[ns] < 0) & \
        (centralAHs[1]) for centralAHs in centralAntihalosUn[ns]] \
        for ns in range(0,len(snapNumListUncon))]
    conditionListMrange = [(deltaCentralList[ns] < 0) & \
        (centralAntihalosCon[ns][1]) & (antihaloMassesList[ns] > mMin) & \
        (antihaloMassesList[ns] <= mMax) & additionalConditions[ns] & \
        (antihaloRadii[ns] > rMin) & \
        (antihaloRadii[ns] <= rMax)
        for ns in range(0,len(snapNumList))]
    conditionListMrangeUn = [[(deltaCentralListUn[ns] < 0) & \
        (centralAHs[1]) & (antihaloMassesListUn[ns] > mMin) & \
        (antihaloMassesListUn[ns] <= mMax) & \
        (antihaloRadiiUn[ns] > rMin) & \
        (antihaloRadiiUn[ns] <= rMax)
        for centralAHs in centralAntihalosUn[ns]] \
        for ns in range(0,len(snapNumListUncon))]
    # Stacked profile data:
    stackedRadii = np.hstack(antihaloRadii)
    stackedMasses = np.hstack(antihaloMassesList)
    stackedConditions = np.hstack(conditionListMrange)
    if (pairCountsList is not None) and (volumesList is not None):
        stackedPairCounts = np.vstack(pairCountsList)
        stackedVolumesList = np.vstack(volumesList)
    else:
        # Recompute only those which we need:
        [newPairCounts,newVolumesList,filtersList] = \
            tools.loadOrRecompute(data_folder + "pair_counts_data.p",\
            getPartialPairCountsAndVols,\
            snapNameList,antihaloRadii,antihaloMassesList,\
            ahCentresList,vorVols,rBins,method,\
            rMin,rMax,mMin,mMax,boxsize,\
            filterListToApply=conditionListMrange,\
            _recomputeData=recomputeData)
        if len(newPairCounts) != len(snapList):
            raise Exception("Pair counts list does not match sample list.")
        stackedPairCounts = np.vstack(newPairCounts)
        stackedVolumesList = np.vstack(newVolumesList)
    [nbarjAllStacked,sigmaAllStacked] = stacking.stackScaledVoids(\
        np.vstack(ahCentresList)[stackedConditions,:],\
        stackedRadii[stackedConditions],\
        snapList[0],rBins,\
        nPairsList = stackedPairCounts,\
        volumesList = stackedVolumesList,\
        method=method,errorType=errorType)
    allPairCountsUn = [[] for ns in range(0,len(snapListUnconstrained))]
    allVolumesListsUn = [[] for ns in range(0,len(snapListUnconstrained))]
    if (pairCountsListUn is not None) and (volumesListUn is not None):
        # Combined stacked withe known pair lists:
        stackedPairCountsUn = np.vstack(pairCountsListUn)
        stackedVolumesListUn = np.vstack(volumesListUn)
        stackedRadiiUn = np.hstack(antihaloRadiiUn)
        stackedMassesUn = np.hstack(antihaloMassesListUn)
        stackedCentresUn = np.vstack(ahCentresListUn)
        combinedUnAll = np.array([],dtype=bool)
    else:
        # Pre-define here. Will be filled below.
        stackedPairCountsUn = np.zeros((0,nBins-1))
        stackedVolumesListUn = np.zeros((0,nBins-1))
        stackedRadiiUn= np.array([])
        stackedMassesUn= np.array([])
        stackedCentresUn = np.zeros((0,3))
    for ns in range(0,len(snapListUnconstrained)):
        for l in range(0,len(centreListUn[ns])):
            if (pairCountsListUn is not None) and (volumesListUn is not None):
                # Use pre-generated pair counts:
                newPairCountsUn = pairCountsListUn[\
                    conditionListMrangeUn[ns][l],:]
                newVolumesListUn = volumesListUn[\
                    conditionListMrangeUn[ns][l],:]
            else:
                # Regenerate them from scratch:
                [newPairCountsUn,newVolumesListUn,filtersListUn] = \
                    tools.loadOrRecompute(data_folder + \
                        "pair_counts_data_unconstrained_sample_" + \
                        str(ns) + "_region_" + str(l) + ".p",\
                        getPartialPairCountsAndVols,\
                        [snapNameListUn[ns]],[antihaloRadiiUn[ns]],\
                        [antihaloMassesListUn[ns]],\
                        [ahCentresListUn[ns]],[vorVolsUn[ns]],rBins,method,\
                        rMin,rMax,mMin,mMax,boxsize,\
                        filterListToApply=[conditionListMrangeUn[ns][l]],\
                        _recomputeData=recomputeData)
                stackedPairCountsUn = np.vstack(\
                    [stackedPairCountsUn,newPairCountsUn[0]])
                stackedVolumesListUn = np.vstack(\
                    [stackedVolumesListUn,newVolumesListUn[0]])
                stackedRadiiUn = np.hstack([stackedRadiiUn,\
                    antihaloRadiiUn[ns][conditionListMrangeUn[ns][l]]])
                stackedMassesUn = np.hstack([stackedMassesUn,\
                    antihaloMassesListUn[ns][conditionListMrangeUn[ns][l]]])
                stackedCentresUn = np.vstack([stackedCentresUn,\
                    ahCentresListUn[ns][conditionListMrangeUn[ns][l],:]])
            allPairCountsUn[ns].append(newPairCountsUn[0])
            allVolumesListsUn[ns].append(newVolumesListUn[0])
    # Stack voids from all bins for the combined unconstrained profile:
    if (pairCountsListUn is not None) and (volumesListUn is not None):
        # Combined stacked withe known pair lists:
        combinedUnAll = np.array([],dtype=bool)
        for ns in range(0,len(snapListUnconstrained)):
            union = np.zeros(len(antihaloRadii[ns]),dtype=bool)
            for l in range(0,len(conditionListMrangeUn[ns])):
                union = np.logical_or(union,conditionListMrangeUn[ns][l])
            combinedUnAll = np.hstack([combinedUnAll,union])
        [nbarjAllStackedUn,sigmaAllStackedUn] = stacking.stackVoidsWithFilter(\
            stackedCentresUn,stackedRadiiUn,combinedUnAll,\
            snapListUnconstrained[0],rBins,\
            nPairsList = stackedPairCountsUn,\
            volumesList = stackedVolumesListUn,\
            method=method,errorType=errorType)
    else:
        # Recompute only those antihalo profiles we need:
        [nbarjAllStackedUn,sigmaAllStackedUn] = stacking.stackScaledVoids(\
            stackedCentresUn,stackedRadiiUn,snapListUnconstrained[0],rBins,\
            nPairsList = stackedPairCountsUn,volumesList = stackedVolumesListUn,\
            method=method,errorType=errorType)
    # Profiles in each constrained region separately:
    if (pairCountsList is not None) and (volumesList is not None):
        pCounts = pairCountsList
        vList = volumesList
        centres = ahCentresList
        radii = antihaloRadii
        masses = antihaloMassesList
        cond = conditionListMrange
    else:
        pCounts = newPairCounts
        vList = newVolumesList
        centres = [ahCentresList[ns][conditionListMrange[ns],:] \
            for ns in range(0,len(snapList))]
        radii = [antihaloRadii[ns][conditionListMrange[ns]] \
            for ns in range(0,len(snapList))]
        masses = [antihaloMassesList[ns][conditionListMrange[ns]] \
            for ns in range(0,len(snapList))]
        cond = None
    [nbarjSepStack,sigmaSepStack] = stacking.computeMeanStacks(\
        centres,radii,masses,cond,pCounts,\
        vList,snapNameList,nbar,rBins,rMin,rMax,mMin,mMax)
    # Profiles in selected unconstrained regions:
    indStack = []
    pCounts = allPairCountsUn
    vList = allVolumesListsUn
    if type(centreListUn) == np.ndarray:
        centreListUn = [centreListUn \
            for ns in range(0,len(snapListUnconstrained))]
    for ns in range(0,len(snapNumListUncon)):
        for l in range(0,len(centreListUn[ns])):
            condition = conditionListMrangeUn[ns][l]
            indStack.append(stacking.stackScaledVoids(
                ahCentresListUn[ns][condition,:],\
                antihaloRadiiUn[ns][condition],\
                snapListUnconstrained[ns],rBins,\
                nPairsList = pCounts[ns][l],\
                volumesList=vList[ns][l]))
    nbarjSepStackUn = np.vstack([sim[0] for sim in indStack])
    sigmaSepStackUn = np.vstack([sim[1] for sim in indStack])
    return [rBinStackCentres,nbarjSepStack,\
        sigmaSepStack,nbarjSepStackUn,sigmaSepStackUn,\
        nbarjAllStacked,sigmaAllStacked,nbarjAllStackedUn,sigmaAllStackedUn,\
        nbar,rMin,mMin,mMax]

def getAntihaloSkyPlotData(snapNumList,nToPlot=20,verbose=True,\
        samplesFolder = "new_chain/",recomputeData = False,\
        snapForFolder = "gadget_full_forward_512",\
        snapRevFolder = "gadget_full_reverse_512",\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        reCentreSnaps = True,figuresFolder='',\
        snapList = None,snapListRev = None,antihaloCatalogueList=None,\
        snapsortList = None,ahProps = None,massRange = None,rRange = None,\
        additionalFilters = None,rSphere=135):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    if snapList is None:
        snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapname) for snapNum in snapNumList]
    if snapListRev is None:
        snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapnameRev) for snapNum in snapNumList]
    if reCentreSnaps:
        for snap in snapList:
            tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
            snap.recentred = True
    if verbose:
        print("Computing halo/antihalo centres...")
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    if antihaloCatalogueList is None:
        antihaloCatalogueList = [snap.halos() for snap in snapListRev]
    if ahProps is None:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapList]
    antihaloRadii = [props[7] for props in ahProps]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    if snapsortList is None:
        snapsortList = [np.argsort(snap['iord']) \
            for snap in snapList]
    if rRange is None:
        rRangeCond = [np.ones(len(antihaloRadii[k]),dtype=bool) \
            for k in range(0,len(snapNumList))]
    else:
        rRangeCond = [(antihaloRadii[k] > rRange[0]) & \
            (antihaloRadii[k] <= rRange[1]) for k in range(0,len(snapNumList))]
    if massRange is None:
        mRangeCond = [np.ones(len(antihaloRadii[k]),dtype=bool) \
            for k in range(0,len(snapNumList))]
    else:
        mRangeCond = [(antihaloMasses[k] > massRange[0]) & \
            (antihaloMasses[k] <= massRange[1]) \
            for k in range(0,len(snapNumList))]
    filterCond = [rRangeCond[k] & mRangeCond[k] \
        for k in range(0,len(snapNumList))]
    if additionalFilters is not None:
        filterCond = [filterCond[k] & additionalFilters[k] \
            for k in range(0,len(snapNumList))]
    centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],\
            rSphere,filterCondition = filterCond[k]) \
            for k in range(0,len(snapNumList))]
    massSortCentral = [\
        np.flip(np.argsort(antihaloMasses[k][centralAntihalos[k][0]])) \
        for k in range(0,len(centralAntihalos))]
    largeAntihalos = [np.array(centralAntihalos[ns][0],dtype=int)[\
        massSortCentral[ns][0:nToPlot]] \
        for ns in range(0,len(snapList))]
    if verbose:
        print("Computing alpha shapes...")
    alpha_shape_list = [tools.computeMollweideAlphaShapes(\
        snapList[ns],largeAntihalos[ns],antihaloCatalogueList[ns]) \
        for ns in range(0,len(snapList))]
    return [alpha_shape_list,largeAntihalos,\
        snapsortList]

def zobovVolumesToPhysical(zobovVolumes,snap,dtype=np.double):
    N = np.round(np.cbrt(len(snap))).astype(int)
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if type(zobovVolumes) == type(""):
        vols = np.fromfile(zobovVolumes,dtype=dtype,offset=4)
    else:
        vols = zobovVolumes
    return vols*(boxsize/N)**3


def getMatchPynbody(snap1,snap2,cat1,cat2,quantity1,quantity2,\
        max_index = 200,threshold = 0.5,\
        quantityThresh=0.5,fractionType='normal'):
    bridge = pynbody.bridge.OrderBridge(snap1,snap2,monotonic=False)
    # Ensure ratio is above the threshold:
    match = [-2]
    candidatesList = []
    # get shared particle fractions:
    catTransfer = bridge.catalog_transfer_matrix(groups_1 = cat1,\
        groups_2 = cat2,max_index=max_index)
    cat2Lengths = np.array([len(halo) for halo in cat2])
    for k in range(0,np.min([len(cat1),max_index])):
        # get quantity (mass or radius) ratio, defined as lower/highest
        # so that it is always <= 1:
        bigger = np.where(quantity2 > quantity1[k])[0]
        quantRatio = quantity2/quantity1[k]
        quantRatio[bigger] = quantity1[k]/quantity2[bigger]
        if fractionType == 'normal':
            fraction = catTransfer[k]/len(cat1[k+1])
        elif fractionType == 'symmetric':
            fraction = catTransfer[k]/np.sqrt(len(cat1[k+1]*cat2Lengths))
        else:
            raise Exception("Unrecognised fraction type requested.")
        # Find candidates that satisfy the threshold requirements:
        candidates = np.where((quantRatio >= quantityThresh) & \
            (fraction > threshold))[0]
        candidatesList.append(candidates)
        if len(candidates) < 1:
            match.append(-1)
        else:
            # Select the match with the highest shared particle fraction
            # as the most probable:
            mostProbable = candidates[np.argmax(fraction[candidates])]
            match.append(mostProbable + 1)
        return [np.array(match,dtype=int),candidatesList]

def getMatchDistance(snap1,snap2,centres1,centres2,\
        quantity1,quantity2,tree1=None,tree2=None,distMax = 20.0,\
        max_index=200,quantityThresh=0.5,sortMethod='distance',\
        mode="fractional",sortQuantity = 0,cat1=None,cat2=None,volumes1=None,\
        volumes2=None,overlap = None):
    boxsize = snap1.properties['boxsize'].ratio("Mpc a h**-1")
    if tree1 is None:
        tree1 = scipy.spatial.cKDTree(snapedit.wrap(centres1,boxsize),\
            boxsize=boxsize)
    if tree2 is None:
        tree2 = scipy.spatial.cKDTree(snapedit.wrap(centres2,boxsize),\
            boxsize=boxsize)
    # Our procedure here is to get the closest anti-halo that lies within the 
    # threshold:
    match = [-2]
    candidatesList = []
    ratioList = []
    distList = []
    if mode == "fractional":
        # Interpret distMax as a fraction of the void radius, not the 
        # distance in Mpc/h.
        # Choose a search radius that is no greater than the void radius divided
        # by the radius ratio. If the other anti-halo is further away than this
        # then it wouldn't match to us anyway, so we don't need to consider it.
        radii1 = quantity1/quantityThresh
        radii2 = quantity2/quantityThresh
        if len(radii1.shape) > 1:
            searchRadii = radii1[:,0]
        else:
            searchRadii = radii1
        searchOther = tree2.query_ball_point(snapedit.wrap(centres1,boxsize),\
            searchRadii,workers=-1)
    else:
        searchOther = tree1.query_ball_tree(tree2,distMax)
    if overlap is None and sortMethod == "volumes":
        if cat1 is None or cat2 is None or \
        volumes1 is None or volumes2 is None:
            raise Exception("Anti-halo catalogue required for " + \
                "volumes based matching.")
        overlap = overlapMap(cat1,cat2,volumes1,volumes2)
    for k in range(0,np.min([len(centres1),max_index])):
        if len(searchOther[k]) > 0:
            # Sort indices:
            distances = np.sqrt(np.sum((\
                    centres2[searchOther[k],:] - centres1[k,:])**2,1))
            if len(quantity1.shape) > 1:
                quantRatio = np.zeros((len(searchOther[k]),\
                    quantity1.shape[1]))
                for l in range(0,quantity1.shape[1]):
                    bigger = np.where(\
                        quantity2[searchOther[k],l] > quantity1[k,l])[0]
                    quantRatio[:,l] = quantity2[searchOther[k],l]/\
                        quantity1[k,l]
                    quantRatio[bigger,l] = quantity1[k,l]/\
                        quantity2[searchOther[k],l][bigger]
            else:
                bigger = np.where(\
                    quantity2[searchOther[k]] > quantity1[k])[0]
                quantRatio = quantity2[searchOther[k]]/quantity1[k]
                quantRatio[bigger] = quantity1[k]/\
                    quantity2[searchOther[k]][bigger]
            if sortMethod == 'distance':
                # Sort the antihalos by distance. Candidate is the closest
                # halo which satisfies the threshold criterion:
                indSort = np.argsort(distances)
                sortedCandidates = np.array(searchOther[k],dtype=int)[indSort]
                bigger = np.where(quantity2[sortedCandidates] > quantity1[k])[0]
                quantRatio = quantity2[sortedCandidates]/quantity1[k]
                quantRatio[bigger] = quantity1[k]/\
                    quantity2[sortedCandidates][bigger]
            elif sortMethod == 'ratio':
                # sort the quantRatio from biggest to smallest, so we find
                # the most similar anti-halo within the search distance:
                if len(quantity1.shape) > 1:
                    indSort = np.flip(np.argsort(quantRatio[:,sortQuantity]))
                else:
                    indSort = np.flip(np.argsort(quantRatio))
                quantRatio = quantRatio[indSort]
                sortedCandidates = np.array(searchOther[k],dtype=int)[indSort]
            elif sortMethod == "volumes":
                volOverlapFrac = overlap[k,searchOther[k]]
                indSort = np.flip(np.argsort(volOverlapFrac))
                quantRatio = quantRatio[indSort]
                sortedCandidates = np.array(searchOther[k],dtype=int)[indSort]
            else:
                raise Exception("Unrecognised sorting method")
            # Get thresholds for these candidates:
            if mode == "fractional":
                if len(quantity1.shape) > 1:
                    candRadii = quantity2[sortedCandidates,0]
                    # Geometric mean of radii, to ensure symmetry.
                    geometricRadii = np.sqrt(quantity1[k,0]*candRadii)
                    condition = (quantRatio[:,0] >= quantityThresh[0]) & \
                        (distances[indSort] <= geometricRadii*distMax)
                    for l in range(1,quantity1.shape[1]):
                        condition = condition & \
                            (quantRatio[:,l] >= quantityThresh[l])
                    candidates = np.where(condition)[0]
                else:
                    candRadii = quantity2[sortedCandidates]
                    # Geometric mean of radii, to ensure symmetry.
                    geometricRadii = np.sqrt(quantity1[k]*candRadii)
                    candidates = np.where((quantRatio >= quantityThresh) & \
                        (distances[indSort] <= geometricRadii*distMax))[0]
            else:
                if len(quantity1.shape) > 1:
                    condition = np.ones(quantityRatio.shape[0],dtype=bool)
                    for l in range(0,quantity1.shape[1]):
                        condition = condition & \
                            (quantRatio[:,l] >= quantityThresh[l])
                    candidates = np.where(condition)[0]
                else:
                    candidates = np.where((quantRatio >= quantityThresh))[0]
            candidatesList.append(np.array(searchOther[k])[candidates])
            ratioList.append(quantRatio[candidates])
            distList.append(distances[candidates])
            if len(candidates) > 0:
                # Add the most probable - remembering the +1 offset for 
                # pynbody halo catalogue IDs:
                match.append(sortedCandidates[candidates[0]] + 1)
            else:
                match.append(-1)
        else:
            match.append(-1)
            candidatesList.append(np.array([]))
            ratioList.append([])
            distList.append([])
    return [np.array(match,dtype=int),candidatesList,ratioList,distList]

def getMatchVolumes(snap1,snap2,cat1,cat2,max_index=200,threshold=0.5):
        vols1 = zobovVolumesToPhysical(snap1.filename + ".vols",snap1,\
            dtype=np.double)
        vols2 = zobovVolumesToPhysical(snap2.filename + ".vols",snap2,\
            dtype=np.double)
        order1 = np.argsort(snap1['iord'])
        order2 = np.argsort(snap2['iord'])
        overlapMatrix = np.zeros((max_index,max_index))
        for k in range(0,max_index):
            for l in range(0,max_index):
                iordOverlap = np.intersect1d(\
                    cat1[k+1]['iord'],cat2[l+1]['iord'])
                if len(iordOverlap) > 0:
                    overlapMatrix = np.sum(vols1[order1[iordOverlap]])/\
                        np.sum(vols1[order1[cat1[k+1]['iord']]])

# Check if an antihalo centre lies in the ZoA:
def isInZoA(centre,inUnits="equatorial",galacticCentreZOA = [-30,30],\
        bRangeCentre = [-10,10],bRange = [-5,5]):
    if len(centre.shape) < 2:
        R = np.sqrt(np.sum(centre**2))
        ra = np.arctan2(centre[1],centre[0])
        dec = np.arcsin(centre[2]/R)
    else:
        R = np.sqrt(np.sum(centre**2,1))
        ra = np.arctan2(centre[:,1],centre[:,0])
        dec = np.arcsin(centre[:,2]/R)
    coords = astropy.coordinates.SkyCoord(ra=ra*astropy.units.rad,
        dec=dec*astropy.units.rad)
    l = coords.galactic.l.value
    b = coords.galactic.b.value
    if len(centre.shape) < 2:
        if (l > galacticCentreZOA[0]) and (l < galacticCentreZOA[1]):
            result = (b > bRangeCentre[0]) and (b < bRangeCentre[1])
        else:
            result = (b > bRange[0]) and (b < bRange[1])
    else:
        result = np.zeros(len(centre),dtype=bool)
        result[(l > galacticCentreZOA[0]) & (l < galacticCentreZOA[1]) \
            & (b > bRangeCentre[0]) & (b < bRangeCentre[1])] = True
        result[((l < galacticCentreZOA[0]) | (l > galacticCentreZOA[1])) \
            & ((b > bRange[0]) & (b < bRange[1]))] = True
    return result

def overlapMap(cat1,cat2,volumes1,volumes2,checkFirst = False,verbose=False):
    overlap = np.zeros((len(cat1),len(cat2)))
    vol1 = np.array([np.sum(volumes1[halo['iord']]) for halo in cat1])
    vol2 = np.array([np.sum(volumes2[halo['iord']]) for halo in cat2])
    if checkFirst:
        for k in range(0,len(cat1)):
            for l in range(0,len(cat2)):
                if checkOverlap(cat1[k+1]['iord'],cat2[l+1]['iord']):
                    intersection = np.intersect1d(cat1[k+1]['iord'],\
                        cat2[l+1]['iord'])
                    overlap[k,l] = np.sum(\
                        volumes1[intersection])/np.sqrt(vol1[k]*vol2[l])
    else:
        for k in range(0,len(cat1)):
            for l in range(0,len(cat2)):
                intersection = np.intersect1d(cat1[k+1]['iord'],\
                    cat2[l+1]['iord'])
                if len(intersection) > 0:
                    overlap[k,l] = np.sum(\
                        volumes1[intersection])/np.sqrt(vol1[k]*vol2[l])
            if verbose:
                print(("%.3g" % (100*(k*len(cat1) + l + 1)/\
                    (len(cat1)*len(cat2)))) + "% complete")
    return overlap

# Check whether two halos have any overlap:
def checkOverlap(list1,list2):
    [min1,max1] = minmax(list1)
    [min2,max2] = minmax(list2)
    if max1 < min2:
        return False
    elif min1 > max2:
        return False
    else:
        if len(list1) < len(list2):
            listMin = list1
            listMax = list2
        else:
            listMin = list2
            listMax = list1
        for k in range(0,len(listMin)):
            if np.isin(listMin[k],listMax):
                return True
        return False

def linearFromIJ(i,j,N):
    if i >= N or i < 0:
        raise Exception("i out of range.")
    if j >= N or j < 0:
        raise Exception("j out of range.")
    if i < j:
        return int(i*N - i*(i+1)/2 + j - i - 1)
    if j < i:
        return int(j*N - j*(j+1)/2 + i - j - 1)
    else:
        raise Exception("i = j not valid.")



# Construct an anti-halo catalogue from reversed snapshots
def constructAntihaloCatalogue(snapNumList,samplesFolder="new_chain/",\
        verbose=True,rSphere=135,max_index=None,thresh=0.5,\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        fileSuffix= '',matchType='distance',crossMatchQuantity='radius',\
        crossMatchThreshold = 0.5,distMax=20.0,sortMethod='ratio',\
        blockDuplicates=True,twoWayOnly = True,\
        snapList=None,snapListRev=None,ahProps=None,hrList=None,\
        rMin = 5,rMax = 30,mode="fractional",massRange = None,\
        snapSortList = None,overlapList = None,NWayMatch = False,\
        additionalFilters = None):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    if snapList is None:
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
    if snapListRev is None:
        snapListRev =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapnameRev) for snapNum in snapNumList]
    # Load centres so that we can filter to the constrained region:
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    if verbose:
        print("Extracting anti-halo properties...")
    if ahProps is None:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    antihaloRadii = [props[7] for props in ahProps]
    if sortMethod == "volumes":
        if snapSortList is None:
            snapSortList = [np.argsort(snap['iord']) for snap in snapList]
        volumesList = [ahProps[k][4][snapSortList[k]] \
            for k in range(0,len(ahProps))]
    else:
        volumesList = [None for k in range(0,len(ahProps))]
    # Build up the filter for which anti-halos we include, first applying
    # a radius cut:
    filterCond = [(antihaloRadii[k] > rMin) & (antihaloRadii[k] <= rMax) \
            for k in range(0,len(snapNumList))]
    # Apply a mass cut if provided:
    if massRange is not None:
        if len(massRange) < 2:
            raise Exception("Mass range must have an upper and a lower " + \
                "limit.")
        for k in range(0,len(snapNumList)):
            filterCond[k] = filterCond[k] & \
                (antihaloMasses[k] > massRange[0]) & \
                (antihaloMasses[k] <= massRange[1])
    # Apply and additional filters that have been provided:
    if additionalFilters is not None:
        for k in range(0,len(snapNumList)):
            filterCond[k] = filterCond[k] & additionalFilters[k]
    # Construct filtered anti-halo lists:
    centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],\
        rSphere,filterCondition = filterCond[k]) \
        for k in range(0,len(snapNumList))]
    centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
    ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
    if max_index is None:
        max_index = np.max(ahCounts)
    # Construct new antihalo catalogues from the filtered list:
    if verbose:
        print("Constructing constrained region catalogues...")
    if hrList is None:
        hrList = [snap.halos() for snap in snapListRev]
    # Sort new catalogues in descending order of mass:
    sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
        for k in range(0,len(snapNumList))]
    if matchType == "pynbody" or sortMethod == "volumes":
        hrListCentral = [copy.deepcopy(halos) for halos in hrList]
    else:
        hrListCentral = [None for halos in hrList]
    shortHaloList = []
    for l in range(0,len(snapNumList)):
        # For the "volumes" sort method, we define a candidate to be
        # the anti-halo with the greatest shared volume. This requires
        # a list of Voronoi volumes.
        if matchType == "pynbody" or sortMethod == "volumes":
            hrListCentral[l]._halos = dict([(k+1,\
                hrList[l][centralAntihalos[l][0][sortedList[l][k]]+1]) \
                for k in range(0,len(centralAntihalos[l][0]))])
            hrListCentral[l]._nhalos = len(centralAntihalos[l][0])
        shortHaloList.append(\
            np.array(centralAntihalos[l][0])[sortedList[l]] + 1)
    if overlapList is None and sortMethod == "volumes":
        # Construct overlap list for all possible pairs:
        overlapList = []
        for k in range(0,len(snapNumList)):
            for l in range(0,len(snapNumList)):
                if k >= l:
                    continue
                overlapList.append(overlapMap(hrListCentral[k],\
                    hrListCentral[l],volumesList[k],volumesList[l],\
                    verbose=False))
    if sortMethod == "volumes":
        # Check that the supplied overlap list is the correct size. If not,
        # this probably means that a bad overlap list was given:
        if len(overlapList) != int(len(snapNumList)*(len(snapNumList) - 1)/2):
            raise Exception("Invalid overlapList!")
    # Construct matches:
    matchArrayList = [[] for k in range(0,len(snapNumList))]
    allCandidates = []
    allRatios = []
    allDistances = []
    if crossMatchQuantity == 'radius':
        #quantityList = np.array([[antihaloRadii[l][\
        #    centralAntihalos[l][0][sortedList[l][k]]] \
        #    for k in range(0,max_index)] \
        #    for l in range(0,len(snapNumList))]).transpose()
        quantityList = [np.array([antihaloRadii[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    elif crossMatchQuantity == 'mass':
        #quantityList = np.array([[antihaloMasses[l][\
        #    centralAntihalos[l][0][sortedList[l][k]]] \
        #    for k in range(0,max_index)] \
        #    for l in range(0,len(snapNumList))]).transpose()
        quantityList = [np.array([antihaloMasses[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    elif crossMatchQuantity == "both":
        quantityListRad = [np.array([antihaloRadii[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
        quantityListMass = [np.array([antihaloMasses[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    else:
        raise Exception('Unrecognised cross-match quantity.')
    #centresListShort = [np.array([antihaloCentres[l][\
    #    centralAntihalos[l][0][sortedList[l][k]]] \
    #    for k in range(0,max_index)]) \
    #    for l in range(0,len(snapNumList))]
    centresListShort = [np.array([antihaloCentres[l][\
        centralAntihalos[l][0][sortedList[l][k]],:] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]
    treeList = [scipy.spatial.cKDTree(\
        snapedit.wrap(centres,boxsize),boxsize=boxsize) \
        for centres in centresListShort]
    if verbose:
        print("Computing matches...")
    for k in range(0,len(snapNumList)):
        matchArrayListNew = matchArrayList[k]
        allCandidatesNew = []
        allRatiosNew = []
        allDistancesNew = []
        for l in range(0,len(snapNumList)):
            if l != k:
                if matchType == 'pynbody':
                    [match, candidatesList] = getMatchPynbody(snapListRev[k],\
                            snapListRev[l],hrListCentral[k],hrListCentral[l],\
                            quantityList[k],quantityList[l],\
                            max_index=max_index,threshold=thresh,\
                            quantityThresh = crossMatchThreshold)
                    matchArrayListNew.append(match)
                elif matchType == 'distance':
                    if crossMatchQuantity == "both":
                        if sortMethod == "volumes":
                            linearIndex = linearFromIJ(k,l,len(snapNumList))
                            if k < l:
                                overlap = overlapList[linearIndex]
                            else:
                                overlap = overlapList[linearIndex].transpose()
                        else:
                            overlap = None
                        [match, candidatesList,ratioList,distList] = \
                            getMatchDistance(snapListRev[k],\
                                snapListRev[l],centresListShort[k],\
                                centresListShort[l],\
                                np.array([quantityListRad[k],\
                                    quantityListMass[k]]).transpose(),\
                                np.array([quantityListRad[l],\
                                    quantityListMass[l]]).transpose(),\
                                tree1=treeList[k],\
                                tree2=treeList[l],distMax = distMax,\
                                max_index=max_index,\
                                quantityThresh=crossMatchThreshold,\
                                sortMethod=sortMethod,mode=mode,\
                                cat1 = hrListCentral[k],\
                                cat2 = hrListCentral[l],\
                                volumes1 = volumesList[k],\
                                volumes2 = volumesList[l],\
                                overlap = overlap)
                    else:
                        [match, candidatesList,ratioList,distList] = \
                            getMatchDistance(snapListRev[k],\
                                snapListRev[l],centresListShort[k],\
                                centresListShort[l],quantityList[k],\
                                quantityList[l],tree1=treeList[k],\
                                tree2=treeList[l],distMax = distMax,\
                                max_index=max_index,\
                                quantityThresh=crossMatchThreshold,\
                                sortMethod=sortMethod,mode=mode)
                    matchArrayListNew.append(match)
                else:
                    raise Exception("Unrecognised matching type requested.")
            elif l == k:
                matchArrayListNew.append([-2] + \
                    list(np.arange(1,np.min([ahCounts[l],max_index])+1)))
                candidatesList = np.array([[m] \
                    for m in np.arange(1,np.min([ahCounts[l],max_index])+1)])
                ratioList = np.ones(len(candidatesList))
                distList = np.zeros(len(candidatesList))
            allCandidatesNew.append(candidatesList)
            allRatiosNew.append(ratioList)
            allDistancesNew.append(distList)
        matchArrayList[k] = list(matchArrayListNew)
        allCandidates.append(allCandidatesNew)
        allRatios.append(allRatiosNew)
        allDistances.append(allDistancesNew)
    # Construct a single catalogue:
    if verbose:
        print("Combining to a single catalogue...")
    matchedCats = [[] for k in range(0,len(snapNumList))]
    twoWayMatchLists = [[] for k in range(0,len(snapNumList))]
    finalCat = []
    finalCandidates = []
    finalRatios = []
    finalDistances = []
    finalCombinatoricFrac = []
    finalCatFrac = []
    candidateCounts = [np.zeros((len(snapNumList),ahCounts[l]),dtype=int) \
        for l in range(0,len(snapNumList))]
    alreadyMatched = np.zeros((len(snapNumList),max_index),dtype=bool)
    matrixFullList = [np.array(matchArrayList[k]).transpose()[1:,:] \
        for k in range(0,len(snapNumList))]
    diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
        for k in range(0,len(snapNumList))]
    for k in range(0,len(snapNumList)):
        matrixFull = matrixFullList[k]
        otherColumns = np.setdiff1d(np.arange(0,len(snapNumList)),[k])
        matrix = matrixFull[:,otherColumns]
        for l in range(0,np.min([ahCounts[k],max_index])):
            # Check if a 2-way match:
            twoWayMatch = np.zeros(matrix[l].shape,dtype=bool)
            for m in range(0,len(snapNumList)):
                candidateCounts[k][m,l] = len(allCandidates[k][m][l])
            for m in range(0,len(snapNumList)-1):
                if matrixFull[l,otherColumns[m]] < 0:
                    # Fails if we don't match to anything
                    twoWayMatch[m] = False
                else:
                    # 2-way only if the other matches back to this:
                    twoWayMatch[m] = (matrixFullList[otherColumns[m]][\
                        matrixFull[l,otherColumns[m]] - 1,k] == l+1)
            twoWayMatchLists[k].append(twoWayMatch)
            if alreadyMatched[k,l] or \
                    ((not np.any(twoWayMatch)) and twoWayOnly):
                continue # No need to include if it already has a partner
            if np.any(candidateCounts[k][otherColumns,l] == 1):
                # Succeed in finding a match ONLY if we found a single
                # candidate for at least one other catalogue
                # If we found a successful match, 
                # check if these matches have already been found:
                isNewMatch = np.zeros(\
                    matrixFull[l].shape,dtype=bool)
                for m in range(0,len(isNewMatch)):
                    if (matrixFull[l][m] > 0) and (m != k):
                        if blockDuplicates:
                            isNewMatch[m] = \
                            not alreadyMatched[m,matrixFull[l][m]-1]
                        else:
                            isNewMatch[m] = True
                    if (matrixFull[l][m] < 0):
                        isNewMatch[m] = False
                if np.any(isNewMatch[otherColumns]):
                    # Add antihalo to the global catalogue:
                    if not NWayMatch:
                        finalCat.append(matrixFull[l])
                    candm = []
                    ratiosm = []
                    distancesm = []
                    # Mark companions as already included:
                    for m in range(0,len(snapNumList)):
                        if (m != k) and (matrixFull[l][m] > 0):
                            # Only deem something to be already matched
                            # if it maps back to this with a single unique 
                            # candidate
                            if not NWayMatch:
                                alreadyMatched[m][matrixFull[l][m] - 1] = \
                                    (matrixFullList[m]\
                                    [matrixFull[l,m] - 1,k] == l+1) \
                                    and (len(allCandidates[m][k]\
                                    [matrixFull[l][m] - 1]) == 1)
                        if (m != k):
                            candm.append(allCandidates[k][m][l])
                            ratiosm.append(allRatios[k][m][l])
                            distancesm.append(allDistances[k][m][l])
                        if m == k and not NWayMatch:
                            alreadyMatched[m][l] = True
                    if NWayMatch:
                        # Track down all possible matches that are connected
                        # to this one:
                        allCands = [[] for m in range(0,len(snapNumList))]
                        lengthsList = np.zeros(len(snapNumList),dtype=int)
                        for m in range(0,len(snapNumList)):
                            twoWayCand = matrixFull[l][m]
                            haveMatch = twoWayCand > -1
                            alreadyIncluded = alreadyMatched[m][twoWayCand - 1]
                            if haveMatch and (not alreadyIncluded):
                                allCands[m].append(twoWayCand-1)
                        lengthsListNew = np.array(\
                            [len(cand) for cand in allCands],dtype=int)
                        # Keep iterating until we stop finding new matches:
                        while not np.all(lengthsListNew == lengthsList):
                            lengthsList = lengthsListNew
                            # Loop over all catalogues:
                            for n in range(0,len(snapNumList)):
                                if len(allCands[n]) > 0:
                                    # If we have candidates, follow their 
                                    # 2-way matches to the other catalogues.
                                    # Loop over all other catalogues, d, and 
                                    # the candidate anti-halos in catalogue n:
                                    for d in diffMap[n]:
                                        for m in range(0,len(allCands[n])):
                                            # For each antihalo in catalogue n, 
                                            # get the candidates in catalogue d
                                            # to which it has two-way matches:
                                            otherCatCandidates = \
                                                allCandidates[n][d][\
                                                allCands[n][m]]
                                            if len(otherCatCandidates) > 0:
                                                # The first candidate is the
                                                # two way match:
                                                bestCand = otherCatCandidates[0]
                                                # Add this iff we haven't
                                                # already found it, and it
                                                # hasn't already been marked
                                                # as belonging to another
                                                # void:
                                                inOtherList = np.isin(\
                                                        bestCand,allCands[d])
                                                #if inOtherList:
                                                #    print("Found " + \
                                                #        str(bestCand) + " in " + \
                                                #        "catalogue " + str(d))
                                                #    print(allCands[d])
                                                alreadyIncluded = \
                                                    alreadyMatched[d,bestCand]
                                                #if alreadyIncluded:
                                                #    print("Already matched void " + \
                                                #        str(bestCand) + " in catalogue " + \
                                                #        str(n))
                                                if (not inOtherList) and \
                                                        (not alreadyIncluded):
                                                    allCands[d].append(bestCand)
                            lengthsListNew = np.array([len(cand) \
                                for cand in allCands],dtype=int)
                        # Count two way matches:
                        twoWayMatchesAllCands = [[] \
                            for m in range(0,len(snapNumList))]
                        for m in range(0,len(snapNumList)):
                            for n in range(0,len(allCands[m])):
                                nTW = np.sum(np.array(\
                                    [len(allCandidates[m][d][allCands[m][n]]) \
                                    for d in diffMap[m]]) > 0)
                                twoWayMatchesAllCands[m].append(nTW)
                        # Compute the average fractions:
                        ratioAverages = [[] for k in range(0,len(snapNumList))]
                        for m in range(0,len(snapNumList)):
                            for n in range(0,len(allCands[m])):
                                ratios = np.zeros(len(snapNumList)-1)
                                for d in range(0,len(snapNumList)-1):
                                    if len(allRatios[m][diffMap[m][d]][\
                                            allCands[m][n]]) > 0:
                                        ratios[d] = allRatios[m][\
                                            diffMap[m][d]][allCands[m][n]][0]
                                qR = np.mean(ratios)
                                ratioAverages[m].append(qR)
                        # Compute the distances:
                        distAverages = [[] for k in range(0,len(snapNumList))]
                        for m in range(0,len(snapNumList)):
                            for n in range(0,len(allCands[m])):
                                distances = np.zeros(len(snapNumList)-1)
                                for d in range(0,len(snapNumList)-1):
                                    if len(allDistances[m][diffMap[m][d]][\
                                            allCands[m][n]]) > 0:
                                        distances[d] = allDistances[m][\
                                            diffMap[m][d]][allCands[m][n]][0]
                                qR = np.mean(distances)
                                distAverages[m].append(qR)
                        # Now figure out the best candidates to include:
                        bestCandidates = -np.ones(len(snapNumList),dtype=int)
                        bestRatios = np.zeros(len(snapNumList))
                        bestDistances = np.zeros(len(snapNumList))
                        numberOfLinks = 0
                        for m in range(0,len(snapNumList)):
                            if len(allCands[m]) == 1:
                                bestCandidates[m] = allCands[m][0] + 1
                                bestRatios[m] = ratioAverages[m][0]
                                bestDistances[m] = distAverages[m][0]
                                numberOfLinks += twoWayMatchesAllCands[m][0]
                            elif len(allCands[m]) > 1:
                                maxTW = np.max(allCands[m])
                                haveMaxTW = np.where(\
                                    np.array(allCands[m]) == maxTW)[0]
                                if len(haveMaxTW) > 1:
                                    # Need to use the ratio criteria to choose
                                    # instead
                                    maxRat = np.max(ratioAverages[m])
                                    haveMaxRat = np.where(\
                                        np.array(ratioAverages[m]) == maxRat)[0]
                                    bestCandidates[m] = allCands[m][\
                                        haveMaxRat[0]] + 1
                                    bestRatios[m] = ratioAverages[m][\
                                        haveMaxRat[0]]
                                    bestDistances[m] = distAverages[m][\
                                        haveMaxRat[0]]
                                    numberOfLinks += twoWayMatchesAllCands[m][\
                                        haveMaxRat[0]]
                                else:
                                    bestCandidates[m] = allCands[m][\
                                        haveMaxTW[0]] + 1
                                    bestRatios[m] = ratioAverages[m][\
                                        haveMaxTW[0]]
                                    bestDistances[m] = distAverages[m][\
                                        haveMaxTW[0]]
                                    numberOfLinks += twoWayMatchesAllCands[m][\
                                        haveMaxTW[0]]
                            # If no candidates, just leave it as -1
                        # Now we mark the other voids as already included:
                        for m in range(0,len(snapNumList)):
                            alreadyMatched[m,bestCandidates[m] - 1] = True
                        finalCat.append(bestCandidates)
                        finalCandidates.append(candm)
                        finalRatios.append(bestRatios)
                        finalDistances.append(bestDistances)
                        finalCombinatoricFrac.append(float(numberOfLinks/\
                            (len(snapNumList)*(len(snapNumList)-1))))
                        finalCatFrac.append(float(\
                            np.sum(bestCandidates > 0)/len(snapNumList)))
                    else:
                        finalCandidates.append(candm)
                        finalRatios.append(ratiosm)
                        finalDistances.append(distancesm)
                        finalCatFrac.append(\
                            float(len(np.where(matrixFull[l] > 0)[0])/\
                            len(snapNumList)))
                        # Compute the combinatoric fraction:
                        Ncats = len(snapNumList)
                        twoWayMatchCounts = 0
                        for m in range(0,Ncats):
                            for d in diffMap[m]:
                                allCands = allCandidates[m][d][matrixFull[l][m]-1]
                                if len(allCands) > 0:
                                    if allCands[0] == matrixFull[l][d]-1:
                                        twoWayMatchCounts += 1
                        finalCombinatoricFrac.append(twoWayMatchCounts/\
                            (Ncats*(Ncats-1)))
    return [np.array(finalCat),shortHaloList,np.array(twoWayMatchLists),\
        finalCandidates,finalRatios,finalDistances,allCandidates,\
        candidateCounts,allRatios,np.array(finalCombinatoricFrac),\
        np.array(finalCatFrac),alreadyMatched]

def getMatchRatios(matchList,quantityList):
    ratioArray = np.zeros(matchList.shape)
    for k in range(0,len(matchList)):
        values = np.zeros(len(matchList[k]))
        for l in range(0,len(matchList[k])):
            if matchList[k,l] < 0.0:
                values[l] = -1
            else:
                values[l] = quantityList[:,l][matchList[k,l]-1]
        maxVal = np.max(values)
        haveValue = np.where(values > 0)
        ratioArray[k][haveValue] = values[haveValue]/maxVal
    return ratioArray



def plotMassFunction(masses,volSim,ax=None,Om0=0.3,h=0.8,ns=1.0,\
        Delta=200,sigma8=0.8,fontsize=12,legendFontsize=10,font="serif",\
        Ob0=0.049,mass_function='Tinker',delta_wrt='SOCritical',massLower=5e13,\
        massUpper=1e15,figsize=(4,4),marker='x',linestyle='--',\
        color=seabornColormap[0],colorTheory = seabornColormap[1],\
        nBins=21,poisson_interval = 0.95,legendLoc='lower left',\
        label="Gadget Simulation",transfer_model='EH',fname=None,\
        xlabel="Mass [$M_{\odot}h^{-1}$]",ylabel="Number of halos",\
        ylim=[1e1,2e4],title="Gadget Simulation",showLegend=True,\
        tickRight=False,tickLeft=True,savename=None):
    [dndm,m] = cosmology.TMF_from_hmf(massLower,massUpper,\
        h=h,Om0=Om0,Delta=Delta,delta_wrt=delta_wrt,\
        mass_function=mass_function,sigma8=sigma8,Ob0 = Ob0,\
        transfer_model=transfer_model,fname=fname,ns=ns)
    massBins = 10**np.linspace(np.log10(massLower),np.log10(massUpper),nBins)
    n = cosmology.dndm_to_n(m,dndm,massBins)
    bounds = np.array(scipy.stats.poisson(n*volSim).interval(poisson_interval))
    massBinCentres = plot_utilities.binCentres(massBins)
    alphaO2 = (1.0 - poisson_interval)/2.0
    if type(masses) == list:
        [noInBins,sigmaBins] = plot.computeMeanHMF(masses,\
            massLower=massLower,massUpper=massUpper,nBins = nBins)
    else:
        noInBins = plot_utilities.binValues(masses,massBins)[1]
        sigmaBins = np.abs(np.array([scipy.stats.chi2.ppf(\
            alphaO2,2*noInBins)/2,\
            scipy.stats.chi2.ppf(1.0 - alphaO2,2*(noInBins+1))/2]) - \
            noInBins)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,4))
    ax.errorbar(massBinCentres,noInBins,sigmaBins,marker=marker,\
        linestyle=linestyle,label=label,color=color)
    ax.plot(massBinCentres,n*volSim,":",label=mass_function + ' prediction',\
                    color=seabornColormap[1])
    ax.fill_between(massBinCentres,
                    bounds[0],bounds[1],
                    facecolor=colorTheory,alpha=0.5,interpolate=True,\
                    label='$' + str(100*poisson_interval) + \
                    '\%$ Confidence \nInterval')
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

def massFunctionComparison(massesLeft,massesRight,volSim,Om0=0.3,h=0.8,\
        Delta=200,sigma8=0.8,fontsize=12,legendFontsize=10,font="serif",\
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


def getPoissonSamples(lam,nSamples):
    if np.isscalar(lam):
        samples = np.random.poisson(lam,nSamples)
    else:
        samples = np.zeros((len(lam),nSamples),dtype=int)
        for k in range(0,len(lam)):
            samples[k,:] = np.random.poisson(lam[k],nSamples)
    return samples

def estimatePoissonRatioErrorbarMonteCarlo(n1,n2,errorVal = 0.67,seed = None,\
        nSamples=1000,returnSamples=False):
    interval1 = scipy.stats.poisson(n1).interval(errorVal)
    interval2 = scipy.stats.poisson(n2).interval(errorVal)
    np.random.seed(seed)
    # Generate random samples
    n1Samples = getPoissonSamples(n1,nSamples)
    n2Samples = getPoissonSamples(n2,nSamples)
    nonZero = np.where(n2Samples != 0)
    ratioValues = np.zeros(n1Samples.shape)
    ratioValues[nonZero] = n1Samples[nonZero]/n2Samples[nonZero]
    if np.isscalar(n1):
        lower = np.percentile(ratioValues,100*(0.5 - errorVal/2))
        upper = np.percentile(ratioValues,100*(0.5 + errorVal/2))
    else:
        lower = np.zeros(n1.shape)
        upper = np.zeros(n1.shape)
        for k in range(0,len(n1)):
            lower[k] = np.percentile(ratioValues[k,:],0.5 - errorVal/2)
            upper[k] = np.percentile(ratioValues[k,:],100*(0.5 + errorVal/2))
    returnList = [n1/n2,lower,upper]
    if returnSamples:
        returnList.append(ratioValues)
    return returnList


def getRadialUniqueFractions(rMax,rWidth,centresListShort,catalogue,\
        candidateCounts,errorInterval = 0.67,errorType="Monte Carlo",\
        nSamples = 10000,seed=None):
    numCats = catalogue.shape[1]
    diffMap = [np.setdiff1d(np.arange(0,numCats),[k]) \
        for k in range(0,numCats)]
    candCountStripped = [candidateCounts[k][diffMap[k],:] \
        for k in range(0,numCats)]
    radialBins = np.arange(0,rMax,rWidth)
    radialBinCentres = plot.binCentres(radialBins)
    distances = [np.sqrt(np.sum(centres**2,1)) \
        for centres in centresListShort]
    uniqueMatchFracDistAll = np.zeros(len(radialBinCentres))
    if errorType == "gaussian":
        uniqueMatchFracDistAllErr = np.zeros(len(radialBinCentres))
    else:
        uniqueMatchFracDistAllErr = np.zeros((2,len(radialBinCentres)))
    for k in range(0,len(radialBins)-1):
        inRangeDistAll = [np.where(\
            (distances[ns] > radialBins[k]) & \
            (distances[ns] <= radialBins[k+1]))[0] \
            for ns in range(0,numCats)]
        if errorType == "gaussian":
            uniqueMatchFracDistAll[k] = np.mean(\
                [[len(np.where(candCountStripped[m][\
                l,inRangeDistAll[m]] == 1)[0])/\
                np.max([1,len(inRangeDistAll[diffMap[m][l]])]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])
            uniqueMatchFracDistAllErr[k] = np.std(\
                [[len(np.where(candCountStripped[m][\
                l,inRangeDistAll[m]] == 1)[0])/\
                np.max([1,len(inRangeDistAll[diffMap[m][l]])]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])/\
                np.sqrt(numCats*(numCats-1))
        elif errorType == "Poisson":
            numerator = np.sum(\
                [[len(np.where(candCountStripped[m][\
                l,inRangeDistAll[m]] == 1)[0]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])
            denominator = np.max([1,np.sum([[\
                len(inRangeDistAll[m]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])])
            interval = scipy.stats.poisson(numerator).interval(errorInterval)
            uniqueMatchFracDistAll[k] = numerator/denominator
            uniqueMatchFracDistAllErr[0,k] = (numerator - interval[0])/\
                denominator
            uniqueMatchFracDistAllErr[1,k] = (interval[1] - numerator)/\
                denominator
        elif errorType == "Monte Carlo":
            #numerators = np.array([[len(np.where(candCountStripped[m][\
            #    l,inRangeDistAll[m]] == 1)[0]) \
            #    for l in range(0,numCats-1)] \
            #    for m in range(0,numCats)]).reshape(numCats*(numCats-1))
            #denominators = np.array(\
            #    [[len(inRangeDistAll[diffMap[m][l]]) \
            #    for l in range(0,numCats-1)] \
            #    for m in range(0,numCats)]).reshape(numCats*(numCats-1))
            #[rat,lower,upper] = estimatePoissonRatioErrorbarMonteCarlo(\
            #    numerators,denominators,errorVal = errorInterval,\
            #    nSamples=nSamples,seed=seed)
            #finite = np.isfinite(rat) & (rat != 0.0)
            #meanError = (lower[finite] + upper[finite])/2
            #uniqueMatchFracDistAll[k] = stacking.weightedMean(rat[finite],\
            #    1.0/meanError**2)
            #uniqueMatchFracDistAllErr[k] = np.sqrt(\
            #    np.sum((rat[finite] - uniqueMatchFracDistAll[k])**2\
            #        /meanError**4)/\
            #        np.sum(1.0/meanError**2)**2)
            numerator = np.sum(\
                [[len(np.where(candCountStripped[m][\
                l,inRangeDistAll[m]] == 1)[0]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])
            denominator = np.max([1,np.sum([[\
                len(inRangeDistAll[m]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])])
            [rat,lower,upper] = estimatePoissonRatioErrorbarMonteCarlo(\
                numerator,denominator,errorVal = errorInterval,\
                nSamples=nSamples,seed=seed)
            uniqueMatchFracDistAll[k] = numerator/denominator
            uniqueMatchFracDistAllErr[0,k] = rat - lower
            uniqueMatchFracDistAllErr[1,k] = upper - rat
        else:
            raise Exception("Unrecognised error type")
    return [radialBinCentres,uniqueMatchFracDistAll,uniqueMatchFracDistAllErr]

def getRadialCatalogueFractions(rMax,rWidth,centresListShort,catalogue,\
        candidateCounts):
    numCats = catalogue.shape[1]
    diffMap = [np.setdiff1d(np.arange(0,numCats),[k]) \
        for k in range(0,numCats)]
    candCountStripped = [candidateCounts[k][diffMap[k],:] \
        for k in range(0,numCats)]
    radialBins = np.arange(0,rMax,rWidth)
    radialBinCentres = plot.binCentres(radialBins)
    distances = [np.sqrt(np.sum(centres**2,1)) \
        for centres in centresListShort]
    [distancesOpt,distErrOpt] = getMeanProperty(\
        getPropertyFromCat(catalogue,distances))
    catFractionFinalDistanceAll = np.zeros(len(radialBinCentres))
    catFractionFinalDistanceAllErr = np.zeros(len(radialBinCentres))
    for k in range(0,len(radialBins)-1):
        inRangeDist = np.where(\
            (distancesOpt > radialBins[k]) & \
            (distancesOpt <= radialBins[k+1]))[0]
        catFractionFinalDistanceAll[k] = np.mean(\
            np.sum(catalogue[inRangeDist,:] > 0,1)/numCats)
        catFractionFinalDistanceAllErr[k] = np.std(\
            np.sum(catalogue[inRangeDist,:] > 0,1)/numCats)/\
            np.sqrt(len(inRangeDist))
    return [radialBinCentres,catFractionFinalDistanceAll,\
        catFractionFinalDistanceAllErr]


def getMeanCentresFromCombinedCatalogue(combinedCat,centresList,\
        returnError=False,boxsize=None):
    meanCentresArray = np.zeros((len(combinedCat),3))
    if returnError:
        stdCentresArray = np.zeros((len(combinedCat),3))
    for nV in range(0,len(combinedCat)):
        centresArray = []
        for l in range(0,combinedCat.shape[1]):
            if combinedCat[nV][l] > -1:
                centresArray.append(centresList[l][combinedCat[nV,l] - 1])
        meanCentresArray[nV,:] = np.mean(centresArray,0)
        if returnError:
            stdCentresArray[nV,:] = np.std(centresArray,0)/\
                np.sqrt(len(centresArray))
    if returnError:
        return [meanCentresArray,stdCentresArray]
    else:
        return meanCentresArray


def getMeanCentreDistance(combinedCat,centresList,returnError=False,\
        boxsize=None):
    meanDistanceArray = np.zeros(len(combinedCat))
    if returnError:
        stdDistanceArray = np.zeros(len(combinedCat))
    for nV in range(0,len(combinedCat)):
        centresArray = []
        for l in range(0,combinedCat.shape[1]):
            if combinedCat[nV][l] > -1:
                centresArray.append(centresList[l][combinedCat[nV,l] - 1])
        meanCentre = context.computePeriodicCentreWeighted(\
            np.array(centresArray),periodicity=boxsize)
        distances = np.array([tools.getPeriodicDistance(\
            meanCentre,centre,boxsize=boxsize) \
            for centre in centresArray])
        meanDistanceArray[nV] = np.mean(distances)
        if returnError:
            stdDistanceArray[nV,:] = np.std(distances)/\
                np.sqrt(len(centresArray))
    if returnError:
        return [meanDistanceArray,stdDistanceArray]
    else:
        return meanDistanceArray

# Search radii around centre that have some threshold density relative
# to the mean density.
def getThresholdRadius(centres,snap,thresh=0.2,rSearchMax = 100,\
        rSearchMin=5.0):
    tree = tools.getKDTree(snap)
    radii = np.array(len(centres))
    Om = snap.properties['omegaM0']
    N = np.cbrt(len(snap))
    rhoM = 2.7754e11*Om
    mUnit = rhoM*(boxsize/N)**3
    for k in range(0,len(centres)):
        func = lambda r: mUnit*tree.query_ball_point(centres[k,:],r,\
            return_length=True,workers=-1)/(4*np.pi*rhoM*r**3/3) - thresh
        valMin = func(rSearchMin)
        valMax = func(rSearchMax)
        if valMax*valMin > 0:
            radii[k] = -1
        else:
            radii[k] = scipy.optimize.brentq(func,rSearchMin,rSearchMax)[0]
    return radii




def getDistanceMatrix(ahIndices,centresList,lowerTriangleOnly=False):
    centres = [centresList[k][ahIndices[k]-1] \
        for k in range(0,len(ahIndices))]
    matrix = np.zeros((len(ahIndices),len(ahIndices)))
    for k in range(0,len(ahIndices)):
        for l in range(0,len(ahIndices)):
            if l > k and lowerTriangleOnly:
                continue
            if l == k:
                continue
            if ahIndices[k] > 0 and ahIndices[l] > 0:
                matrix[k,l] = np.sqrt(np.sum((centres[k] - centres[l])**2))
            else:
                matrix[k,l] = np.nan
    return matrix

def getBestCandidateMeanDistance(catalogue,centresList):
    meanDistances = np.zeros(len(catalogue))
    if len(catalogue) == 0:
        return np.array([])
    lowerTriInd = np.tril_indices(len(catalogue[0]),k=-1)
    for k in range(0,len(catalogue)):
        matrix = getDistanceMatrix(catalogue[k],centresList,\
            lowerTriangleOnly=True)
        meanDistances[k] = np.nanmean(matrix[lowerTriInd])
    return meanDistances

# Ratio of two radii, smallest/biggest
def getRadiusRatio(radii1,radii2):
    if np.isscalar(radii1):
        if np.isscalar(radii2):
            if radii2 > radii1:
                return radii1/radii2
            else:
                return radii2/radii1
        else:
            returnArr = radii2/radii1
            bigger = np.where(radii2 > radii1)[0]
            returnArr[bigger] = radii1/radii2[bigger]
            return returnArr
    else:
        if np.isscalar(radii2):
            returnArr = radii1/radii2
            smaller = np.where(radii2 < radii1)[0]
            returnArr[smaller] = radii2/radii1[smaller]
            return returnArr
        else:
            returnArr = radii2/radii1
            bigger = np.where(radii2 > radii1)[0]
            returnArr[bigger] = radii1[bigger]/radii2[bigger]
            return returnArr

def getRatioMatrix(ahIndices,radiiList,lowerTriangleOnly = False):
    radii = [radiiList[:,k][ahIndices[k]-1] \
        for k in range(0,len(ahIndices))]
    matrix = np.ones((len(ahIndices),len(ahIndices)))
    for k in range(0,len(ahIndices)):
        for l in range(0,len(ahIndices)):
            if l > k and lowerTriangleOnly:
                continue
            if l == k:
                continue
            if ahIndices[k] > 0 and ahIndices[l] > 0:
                matrix[k,l] = getRadiusRatio(radii[k],radii[l])
            else:
                matrix[k,l] = np.nan
    return matrix

def getBestCandidateMeanRatio(catalogue,radiiList):
    meanRatios = np.zeros(len(catalogue))
    if len(catalogue) == 0:
        return np.array([])
    lowerTriInd = np.tril_indices(len(catalogue[0]),k=-1)
    for k in range(0,len(catalogue)):
        matrix = getRatioMatrix(catalogue[k],radiiList,lowerTriangleOnly=True)
        # Exclude the diagonals:
        meanRatios[k] = np.nanmean(matrix[lowerTriInd])
    return meanRatios

# Get statistics on performance for different parameters:
def specialNanMin(x):
    if len(x) == 0:
        return np.nan
    else:
        return np.nanmin(x)

def specialNanMax(x):
    if len(x) == 0:
        return np.nan
    else:
        return np.nanmax(x)


def getPropertyFromCat(catList,propertyList):
    propertyListOut = -np.ones(catList.shape,dtype=float)
    for k in range(0,len(catList)):
        for l in range(0,len(catList[0])):
            if catList[k,l] > 0:
                propertyListOut[k,l] = propertyList[l][catList[k,l]-1]
    return propertyListOut

def getCentresFromCat(catList,centresList,ns):
    centresListOut = np.zeros((len(catList),3),dtype=float)
    for k in range(0,len(catList)):
        if catList[k,ns] > 0:
            centresListOut[k,:] = centresList[ns][catList[k,ns]-1]
        else:
            centresListOut[k,:] = np.nan
    return centresListOut

def getMeanProperty(propertyList,stripNaN=True,lowerLimit=0,stdError=True):
    meanProperty = np.zeros(len(propertyList))
    sigmaProperty = np.zeros(len(propertyList))
    for k in range(0,len(propertyList)):
        condition = propertyList[k,:] > lowerLimit
        if stripNaN:
            condition = condition & (np.isfinite(propertyList[k,:] > 0))
        haveProperty = np.where(condition)[0]
        meanProperty[k] = np.mean(propertyList[k,haveProperty])
        sigmaProperty[k] = np.std(propertyList[k,haveProperty])
        if stdError:
            sigmaProperty[k] /= np.sqrt(len(haveProperty))
    return [meanProperty,sigmaProperty]



def getFinalCatalogue(snapNumList,snapNumListUncon,snrThresh = 10,\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        samplesFolder="new_chain/",snapList = None,snapListRev = None,\
        snapListUnconstrained = None,snapListUnconstrainedRev=None,\
        mLower = "auto",mUpper = 2e15,nBins = 8,muOpt = 0.9,rSearchOpt = 1,\
        rSphere = 300,rSphereInner = 135,NWayMatch = True,rMin=5,rMax=30,\
        mMin=1e11,mMax = 1e16,percThresh=99,chainFile="chain_properties.p",\
        Nden=256,recomputeUnconstrained = False,data_folder="./",\
        unconstrainedFolderNew = "new_chain/unconstrained_samples/",\
        recomputeData=True,verbose=True):
    # Load snapshots:
    if verbose:
        print("Catalogue construction. \nLoading snapshots..")
        sys.stdout.flush()
    if snapList is None:
        snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapname) for snapNum in snapNumList]
    if snapListRev is None:
        snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + \
            "/" + snapnameRev) for snapNum in snapNumList]
    # Parameters:
    Om = snapList[0].properties['omegaM0']
    rhoc = 2.7754e11
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    N = int(np.cbrt(len(snapList[0])))
    mUnit = 8*Om*rhoc*(boxsize/N)**3
    if mLower == "auto":
        mLower = 100*mUnit
    if mUpper == "auto":
        mUpper = 10*mLower
    # Load properties of the anti-halo catalogues:
    hrList = [snap.halos() for snap in snapListRev]
    ahProps = [pickle.load(\
                open(snap.filename + ".AHproperties.p","rb")) \
                for snap in snapList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    ahCentresUnmapped = [props[5] for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    vorVols = [props[4] for props in ahProps]
    antihaloRadii = [props[7] for props in ahProps]
    # SNR data:
    if verbose:
        print("Loading chain data...")
        sys.stdout.flush()
    [mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
        std_field,hmc_Elh,hmc_Eprior,hades_accept_count,\
        hades_attempt_count] = pickle.load(open(chainFile,"rb"))
    snrField = mean_field**2/std_field**2
    snrFieldLin = np.reshape(snrField,Nden**3)
    # Centres about which to compute SNR:
    if verbose:
        print("Computing SNR...")
        sys.stdout.flush()
    grid = snapedit.gridListPermutation(Nden,perm=(2,1,0))
    centroids = grid*boxsize/Nden + boxsize/(2*Nden)
    positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
    tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
        boxsize=boxsize)
    # SNR for each catalogue:
    nearestPointsList = [tree.query_ball_point(\
        snapedit.wrap(antihaloCentres[k] + boxsize/2,boxsize),\
        antihaloRadii[k],workers=-1) \
        for k in range(0,len(antihaloCentres))]
    snrAllCatsList = [np.array([np.mean(snrFieldLin[points]) \
        for points in nearestPointsList[k]]) for k in range(0,len(snapNumList))]
    snrFilter = [snr > snrThresh for snr in snrAllCatsList]
    # Central anti-halos with appropriate filteR:
    if verbose:
        print("Filtering voids...")
        sys.stdout.flush()
    centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
            (antihaloMasses[k] <= mMax) & snrFilter[k]) \
            for k in range(0,len(snapNumList))]
    centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
    sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
            for k in range(0,len(snapNumList))]
    # Gather these into shortened property lists for the anti-halos
    # in the central region, including only those that match the filter:
    ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
    max_index = np.max(ahCounts)
    massListShort = [np.array([antihaloMasses[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    centresListShort = [np.array([antihaloCentres[l][\
            centralAntihalos[l][0][sortedList[l][k]],:] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    radiiListShort = [np.array([antihaloRadii[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
    # Maps from each catalogue to all the other catalogues:
    diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
        for k in range(0,len(snapNumList))]
    # Catalogue construction:
    if verbose:
        print("Constructing catalogue...")
        sys.stdout.flush()
    [finalCatOpt,shortHaloListOpt,twoWayMatchListOpt,finalCandidatesOpt,\
        finalRatiosOpt,finalDistancesOpt,allCandidatesOpt,candidateCountsOpt,\
        allRatiosOpt,finalCombinatoricFracOpt,finalCatFracOpt,\
        alreadyMatched] = \
        tools.loadOrRecompute(data_folder + "catalogue_all_data.p",\
        constructAntihaloCatalogue,snapNumList,snapList=snapList,\
        snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=rSphere,\
        massRange = [mMin,mMax],NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
        additionalFilters = snrFilter)
    # Mean radius, centre, and mass:
    if verbose:
        print("Computing catalogue properties...")
        sys.stdout.flush()
    radiiListOpt = getPropertyFromCat(finalCatOpt,radiiListShort)
    massListOpt = getPropertyFromCat(finalCatOpt,massListShort)
    [radiiMeanOpt, radiiSigmaOpt]  = getMeanProperty(radiiListOpt)
    [massMeanOpt, massSigmaOpt]  = getMeanProperty(massListOpt)
    finalCentresOptList = np.array([getCentresFromCat(\
        finalCatOpt,centresListShort,ns) for ns in range(0,len(snapNumList))])
    meanCentreOpt = np.nanmean(finalCentresOptList,0)
    [meanCentresArray,stdCentresArray] = getMeanCentresFromCombinedCatalogue(\
        finalCatOpt,centresListShort,returnError=True)
    # SNR about void centres:
    nearestPoints = tree.query_ball_point(\
        snapedit.wrap(meanCentreOpt + boxsize/2,boxsize),radiiMeanOpt,\
        workers=-1)
    snrList = np.array([np.mean(snrFieldLin[points]) \
        for points in nearestPoints])
    # Save catalogue details:
    np.savez(data_folder + "catalogue_data.npz",catalogue=finalCatOpt,\
        radii=radiiMeanOpt,sigma_radii=radiiSigmaOpt,mass=massMeanOpt,\
        sigma_mass=massSigmaOpt,centres=meanCentresArray,\
        sigma_centres=stdCentresArray,snr=snrList,\
        cat_frac=finalCatFracOpt,comb_frac=finalCombinatoricFracOpt)
    # Generate an unconstrained catalogue to determine threshold for
    # spurious voids:
    # Unconstrained catalogue:
    if verbose:
        print("Getting random catalogue...")
        sys.stdout.flush()
    unconstrainedCatFile = data_folder + "unconstrained_catalogue.p"
    if (not os.path.isfile(unconstrainedCatFile)) or recomputeUnconstrained:
        snapListUn = [pynbody.load("new_chain/unconstrained_samples/sample" + \
            str(num) + "/gadget_full_forward_512/snapshot_001") \
            for num in snapNumListUncon]
        snapListRevUn = [pynbody.load(\
            "new_chain/unconstrained_samples/sample" + \
            str(num) + "/gadget_full_reverse_512/snapshot_001") \
            for num in snapNumListUncon]
        hrListUn = [snap.halos() for snap in snapListRevUn]
        ahPropsUn = [pickle.load(\
                    open(snap.filename + ".AHproperties.p","rb")) \
                    for snap in snapListUn]
        [finalCatUn,shortHaloListUn,twoWayMatchListUn,finalCandidatesOUn,\
            finalRatiosUn,finalDistancesUn,allCandidatesUn,candidateCountsUn,\
            allRatiosUn,finalCombinatoricFracUn,finalCatFracUn,\
            alreadyMatched] = \
            constructAntihaloCatalogue(snapNumListUncon,snapList=snapListUn,\
            snapListRev=snapListRevUn,ahProps=ahPropsUn,hrList=hrListUn,\
            max_index=None,twoWayOnly=True,blockDuplicates=True,\
            crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=rSphere,\
            massRange = [mMin,mMax],NWayMatch = NWayMatch,rMin=rMin,rMax=rMax)
        antihaloCentresUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in ahPropsUn]
        antihaloMassesUn = [props[3] for props in ahPropsUn]
        antihaloRadiiUn = [props[7] for props in ahPropsUn]
        centralAntihalosUn = [tools.getAntiHalosInSphere(antihaloCentresUn[k],\
            rSphere,filterCondition = (antihaloRadiiUn[k] > rMin) & \
            (antihaloRadiiUn[k] <= rMax) & (antihaloMassesUn[k] > mMin) & \
            (antihaloMassesUn[k] <= mMax)) \
            for k in range(0,len(snapNumListUncon))]
        centralAntihaloMassesUn = [\
            antihaloMassesUn[k][centralAntihalosUn[k][0]] \
            for k in range(0,len(centralAntihalosUn))]
        sortedListUn = [np.flip(np.argsort(centralAntihaloMassesUn[k])) \
            for k in range(0,len(snapNumListUncon))]
        ahCountsUn = np.array([len(cahs[0]) for cahs in centralAntihalosUn])
        max_indexUn = np.max(ahCountsUn)
        radiiListShortUn = [np.array([antihaloRadiiUn[l][\
            centralAntihalosUn[l][0][sortedListUn[l][k]]] \
            for k in range(0,np.min([ahCountsUn[l],max_indexUn]))]) \
            for l in range(0,len(snapNumListUncon))]
        massListShortUn = [np.array([antihaloMassesUn[l][\
                centralAntihalosUn[l][0][sortedListUn[l][k]]] \
                for k in range(0,np.min([ahCountsUn[l],max_indexUn]))]) \
                for l in range(0,len(snapNumListUncon))]
        pickle.dump([finalCatUn,shortHaloListUn,twoWayMatchListUn,\
            finalCandidatesOUn,\
            finalRatiosUn,finalDistancesUn,allCandidatesUn,candidateCountsUn,\
            allRatiosUn,finalCombinatoricFracUn,finalCatFracUn,\
            antihaloCentresUn,antihaloMassesUn,antihaloRadiiUn,\
            centralAntihalosUn,centralAntihaloMassesUn,sortedListUn,\
            ahCountsUn,radiiListShortUn,massListShortUn],\
            open(unconstrainedCatFile,"wb"))
    else:
        [finalCatUn,shortHaloListUn,twoWayMatchListUn,\
            finalCandidatesOUn,\
            finalRatiosUn,finalDistancesUn,allCandidatesUn,candidateCountsUn,\
            allRatiosUn,finalCombinatoricFracUn,finalCatFracUn,\
            antihaloCentresUn,antihaloMassesUn,antihaloRadiiUn,\
            centralAntihalosUn,centralAntihaloMassesUn,sortedListUn,\
            ahCountsUn,radiiListShortUn,massListShortUn] = pickle.load(open(\
                unconstrainedCatFile,"rb"))
    # Unconstrained radii and masses:
    radiiListCombUn = getPropertyFromCat(finalCatUn,radiiListShortUn)
    massListCombUn = getPropertyFromCat(finalCatUn,massListShortUn)
    [radiiListMeanUn,radiiListSigmaUn] = getMeanProperty(radiiListCombUn)
    [massListMeanUn,massListSigmaUn] = getMeanProperty(massListCombUn)
    # Determine percentiles from the unconstrained catalogues:
    massBins = 10**(np.linspace(np.log10(mLower),np.log10(mUpper),nBins))
    [inMassBins,noInMassBins] = plot.binValues(massListMeanUn,massBins)
    percentilesComb = []
    percentilesCat = []
    for k in range(0,len(inMassBins)):
        if len(inMassBins[k]) > 0:
            #percentiles.append(np.percentile(finalCatFracUn[inMassBins[k]],\
            #    percThresh))
            percentilesComb.append(np.percentile(\
                finalCombinatoricFracUn[inMassBins[k]],percThresh))
            percentilesCat.append(np.percentile(\
                finalCatFracUn[inMassBins[k]],percThresh))
        else:
            percentilesComb.append(0.0)
            percentilesCat.append(0.0)
    massListComb = getPropertyFromCat(finalCatOpt,massListShort)
    [massListMean,massListSigma] = getMeanProperty(massListComb)
    # Construct the filter by mass bin:
    if verbose:
        print("Filtering voids by mass bin...")
        sys.stdout.flush()
    catFracThresholds = percentilesCat
    massFilter = [(massListMean > massBins[k]) & \
        (massListMean <= massBins[k+1]) \
        for k in range(0,len(massBins) - 1)]
    catFracFilter = [finalCatFracOpt > thresh for thresh in percentilesCat]
    combFracFilter = [finalCombinatoricFracOpt > thresh \
        for thresh in percentilesComb]
    combinedFilter = np.zeros(massListMean.shape,dtype=bool)
    for k in range(0,len(catFracThresholds)):
        combinedFilter = combinedFilter | (massFilter[k] & catFracFilter[k] & \
            combFracFilter[k])
    combinedFilter = combinedFilter & (snrList > snrThresh)
    distanceArray = np.sqrt(np.sum(meanCentresArray**2,1))
    combinedFilter135 = combinedFilter135 = combinedFilter & \
        (distanceArray < rSphereInner)
    # Conditions to supply to the void profile code:
    additionalConditions = [np.isin(np.arange(0,len(antihaloMasses[k])),\
        np.array(centralAntihalos[k][0])[sortedList[k][finalCatOpt[\
            (finalCatOpt[:,k] >= 0) & \
            combinedFilter135,k] - 1]]) \
        for k in range(0,len(snapList))]
    # Data for unconstrained sims:
    if verbose:
        print("Getting unconstrained regions with similar density...")
        sys.stdout.flush()
    snapListUnconstrained = [pynbody.load(unconstrainedFolderNew + "sample" \
            + str(snapNum) + "/" + snapname) for snapNum in snapNumListUncon]
    snapListUnconstrainedRev = [pynbody.load(unconstrainedFolderNew + \
            "sample" + str(snapNum) + "/" + snapnameRev) \
            for snapNum in snapNumListUncon]
    ahPropsUnconstrained = [tools.loadPickle(snap.filename + ".AHproperties.p")\
            for snap in snapListUnconstrained]
    ahCentresList = [props[5] for props in ahProps]
    ahCentresListUn = [props[5] for props in ahPropsUnconstrained]
    antihaloRadiiUn = [props[7] for props in ahPropsUnconstrained]
    rEffMin = 0.0
    rEffMax = 10.0
    nBins = 101
    rBinStack = np.linspace(rEffMin,rEffMax,nBins)
    vorVols = [props[4] for props in ahProps]
    vorVolsUn = [props[4] for props in ahPropsUnconstrained]
    pairCountsList = []
    volumesList = []
    pairCountsListUn = []
    volumesListUn = []
    # Centres of regions with similar density contrast to the BORG region:
    [centreListUn,densitiesInCentres,denListUn] = tools.loadOrRecompute(\
        data_folder + "centre_list_unconstrained_data.p",\
        getCentreListUnconstrained,\
        snapListUnconstrained,
        randomSeed = 1000,numDenSamples = 1000,rSphere = 135,\
        densityRange = [-0.051,-0.049],_recomputeData=recomputeData)
    gc.collect()
    if verbose:
        print("Computing combined catalogue profiles...")
        sys.stdout.flush()
    [rBinStackCentresCombined,nbarjSepStackCombined,\
            sigmaSepStackCombined,nbarjSepStackUnCombined,\
            sigmaSepStackUnCombined,\
            nbarjAllStackedCombined,sigmaAllStackedCombined,\
            nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,\
            nbar,rMin2,mMin2,mMax2] = tools.loadOrRecompute(\
                data_folder + "void_profiles_data_combined.p",\
                getVoidProfilesData,\
                snapNumList,snapNumListUncon,\
                snapList = snapList,snapListRev = snapListRev,\
                samplesFolder="new_chain/",\
                unconstrainedFolder="new_chain/unconstrained_samples/",\
                snapname = "gadget_full_forward_512/snapshot_001",\
                snapnameRev = "gadget_full_reverse_512/snapshot_001",\
                reCentreSnaps = False,N=512,boxsize=677.7,mMin = mLower,\
                mMax = mUpper,rMin=5,rMax=30,verbose=True,combineSims=False,\
                method="poisson",errorType = "Weighted",\
                unconstrainedCentreList = centreListUn,\
                additionalConditions = additionalConditions,\
                redoPairCounts=True,rEffMax=10.0,rEffMin=0.0,nBins=101,\
                pairCountsListUn=None,\
                volumesListUn=None,pairCountsList=None,\
                volumesList=None,\
                ahPropsConstrained = ahProps,\
                ahPropsUnconstrained = ahPropsUnconstrained,\
                snapListUnconstrained=snapListUnconstrained,\
                snapListUnconstrainedRev=snapListUnconstrainedRev,\
                _recomputeData=recomputeData,data_folder=data_folder)
    gc.collect()
    if verbose:
        print("Computing all voids profiles...")
    [rBinStackCentres,nbarjSepStack,\
            sigmaSepStack,nbarjSepStackUn,sigmaSepStackUn,\
            nbarjAllStacked,sigmaAllStacked,nbarjAllStackedUn,\
            sigmaAllStackedUn,\
            nbar,rMin2,mMin2,mMax2] = tools.loadOrRecompute(\
                data_folder + "void_profiles_data_all.p",\
                getVoidProfilesData,\
                snapNumList,snapNumListUncon,\
                snapList = snapList,snapListRev = snapListRev,\
                samplesFolder="new_chain/",\
                unconstrainedFolder="new_chain/unconstrained_samples/",\
                snapname = "gadget_full_forward_512/snapshot_001",\
                snapnameRev = "gadget_full_reverse_512/snapshot_001",\
                reCentreSnaps = False,N=512,boxsize=677.7,mMin = mLower,\
                mMax = mUpper,rMin=5,rMax=30,verbose=True,combineSims=False,\
                method="poisson",errorType = "Weighted",\
                unconstrainedCentreList = centreListUn,\
                additionalConditions = None,\
                redoPairCounts=True,rEffMax=10.0,rEffMin=0.0,nBins=101,\
                pairCountsListUn=None,\
                volumesListUn=None,pairCountsList=None,\
                volumesList=None,\
                ahPropsConstrained = ahProps,\
                ahPropsUnconstrained = ahPropsUnconstrained,\
                snapListUnconstrained=snapListUnconstrained,\
                snapListUnconstrainedRev=snapListUnconstrainedRev,\
                _recomputeData=recomputeData,data_folder=data_folder)
    gc.collect()
    return [massListMean,combinedFilter135,combinedFilter,rBinStackCentresCombined,\
    nbarjSepStackCombined,sigmaSepStackCombined,\
    nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,nbar,rMin2,\
    mMin2,mMax2,nbarjSepStackUn,sigmaSepStackUn,\
    rBinStackCentres,nbarjSepStack,\
    sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,\
    nbarjSepStackUn,sigmaSepStackUn]









