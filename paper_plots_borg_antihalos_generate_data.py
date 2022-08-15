# CONFIGURATION
from void_analysis import plot, snapedit, tools, simulation_tools, halos
from void_analysis import stacking, real_clusters, survey
from void_analysis.tools import loadOrRecompute
from void_analysis.simulation_tools import ngPerLBin, ngBias
from void_analysis.simulation_tools import biasNew, biasOld
from void_analysis.survey import radialCompleteness, surveyMask
import numpy as np
import scipy
import pynbody
import astropy
import pickle
import gc
import copy
import h5py
import healpy


# KE correction used to compute magnitudes. Used by PPTs:
def keCorr(z,fit = [-1.456552772320231,-0.7687913554110967]):
    #return -fit[0]*z# - fit[1]
    return 2.9*z

# Convert per-voxel to per-healpix patch average galaxy counts:
def getAllNgsToHealpix(ngList,hpIndices,sampleList,sampleFolder,nside,\
        recomputeData=False):
    return [loadOrRecompute(sampleFolder + "sample" + str(sampleList[k]) + \
        "/ngHP.p",tools.getCountsInHealpixSlices,ngList[k],hpIndices,\
        nside=nside,_recomputeData=recomputeData) \
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
        centreMethod="density"):
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
            h=0.6766,catFolder="")
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
            accelerate=True,\
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
        hpIndices,snapNumList,samplesFolder,nside,\
        _recomputeData=recomputeData)
    # Compute counts in each healpix pixel for 2M++ survey:
    if verbose:
        print("Computing 2M++ galaxy counts...")
    ng2MPP = np.reshape(tools.loadOrRecompute("mg2mppK3.p",\
        survey.griddedGalCountFromCatalogue,\
        cosmo,tmppFile=tmppFile,Kcorrection = True,\
        _recomputeData=recomputeData),(nMagBins,N**3))
    ngHP = tools.loadOrRecompute("ngHP3.p",tools.getCountsInHealpixSlices,\
        ng2MPP,hpIndices,nside=nside,_recomputeData=recomputeData)
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
        snapPath = samplesFolder + "sample" + str(snapNumList[k]) + \
                "/gadget_full_forward_512/snapshot_001"
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
        ahProps = [pickle.load(\
            open(snap.filename + ".AHproperties.p","rb")) \
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
        ahProps = [pickle.load(\
            open(snap.filename + ".AHproperties.p","rb")) \
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


def getVoidProfilesData(snapNumList,snapNumListUncon,\
        samplesFolder="new_chain/",\
        unconstrainedFolder="new_chain/unconstrained_samples/",\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        reCentreSnaps = False,N=512,boxsize=677.7,mMin = 1e14,mMax = 1e15,\
        rMin=5,rMax=25,verbose=True,combineSims=False,\
        method="poisson",errorType = "Weighted",\
        unconstrainedCentreList = np.array([[0,0,0]])):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" + \
        snapname) for snapNum in snapNumList]
    snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + snapnameRev) for snapNum in snapNumList]
    # Load unconstrained snaps:
    if verbose:
        print("Loading snapshots...")
    snapListUnconstrained = [pynbody.load(unconstrainedFolder + "sample" \
        + str(snapNum) + "/" + snapname) for snapNum in snapNumListUncon]
    snapListUnconstrainedRev = [pynbody.load(unconstrainedFolder + \
            "sample" + str(snapNum) + "/" + snapnameRev) \
            for snapNum in snapNumListUncon]
    ahPropsConstrained = [pickle.load(\
        open(snap.filename + ".AHproperties.p","rb")) \
        for snap in snapList]
    ahPropsUnconstrained = [pickle.load(\
        open(snap.filename + ".AHproperties.p","rb")) \
        for snap in snapListUnconstrained]
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
    pairCountsList = [props[9] for props in ahPropsConstrained]
    volumesList = [props[10] for props in ahPropsConstrained]
    rBins = ahPropsConstrained[0][8]
    rBinStackCentres = plot.binCentres(rBins)
    centralAntihalosCon = [tools.getAntiHalosInSphere(hcentres,135) \
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
    pairCountsListUn = [props[9] for props in ahPropsUnconstrained]
    volumesListUn = [props[10] for props in ahPropsUnconstrained]
    centralAntihalosUn = [[tools.getAntiHalosInSphere(hcentres,135,\
        origin=centre) for centre in unconstrainedCentreList] \
        for hcentres in ahCentresListRemapUn]
    # Select antihalos in the central region:
    conditionList = [(deltaCentralList[ns] < 0) & \
        (centralAntihalosCon[ns][1]) for ns in range(0,len(snapNumList))]
    conditionListUn = [[(deltaCentralListUn[ns] < 0) & \
        (centralAHs[1]) for centralAHs in centralAntihalosUn[ns]] \
        for ns in range(0,len(snapNumListUncon))]
    conditionListMrange = [(deltaCentralList[ns] < 0) & \
        (centralAntihalosCon[ns][1]) & (antihaloMassesList[ns] > mMin) & \
        (antihaloMassesList[ns] <= mMax) \
        for ns in range(0,len(snapNumList))]
    conditionListMrangeUn = [[(deltaCentralListUn[ns] < 0) & \
        (centralAHs[1]) & (antihaloMassesListUn[ns] > mMin) & \
        (antihaloMassesListUn[ns] <= mMax) \
        for centralAHs in centralAntihalosUn[ns]] \
        for ns in range(0,len(snapNumListUncon))]
    # Stacked profile data:
    stackedRadii = np.hstack(antihaloRadii)
    stackedMasses = np.hstack(antihaloMassesList)
    stackedConditions = np.hstack(conditionList)
    [nbarjAllStacked,sigmaAllStacked] = stacking.stackVoidsWithFilter(\
        np.vstack(ahCentresList),stackedRadii,\
        np.where((stackedRadii > rMin) & (stackedRadii < rMax) & \
        stackedConditions & (stackedMasses > mMin) & \
        (stackedMasses <= mMax))[0],snapList[0],rBins,\
        nPairsList = np.vstack(pairCountsList),\
        volumesList = np.vstack(volumesList),\
        method=method,errorType=errorType)
    stackedRadiiUn = np.hstack(antihaloRadiiUn)
    stackedMassesUn = np.hstack(antihaloMassesListUn)
    #stackedConditionsUn = np.hstack(conditionListUn)
    # Stack voids from all bins for the combined unconstrained profile:
    [nbarjAllStackedUn,sigmaAllStackedUn] = stacking.stackVoidsWithFilter(\
        np.vstack(ahCentresListUn),stackedRadiiUn,\
        np.where((stackedRadiiUn > rMin) & (stackedRadiiUn < rMax) & \
        (stackedMassesUn > mMin) & (stackedMassesUn <= mMax))[0],\
        snapListUnconstrained[0],rBins,\
        nPairsList = np.vstack(pairCountsListUn),\
        volumesList = np.vstack(volumesListUn),\
        method=method,errorType=errorType)
    # Profiles in each constrained region separately:
    [nbarjSepStack,sigmaSepStack] = stacking.computeMeanStacks(ahCentresList,\
        antihaloRadii,antihaloMassesList,conditionList,pairCountsList,\
        volumesList,snapList,nbar,rBins,rMin,rMax,mMin,mMax)
    # Profiles in selected unconstrained regions:
    indStack = []
    for l in range(0,len(unconstrainedCentreList)):
        condition = [conditionListMrangeUn[k][l] \
            for k in range(0,len(snapNumListUncon))]
        indStack.append(stacking.computeMeanStacks(\
            ahCentresListUn,antihaloRadiiUn,antihaloMassesListUn,\
            condition,pairCountsListUn,volumesListUn,snapListUnconstrained,\
            nbar,rBins,rMin,rMax,mMin,mMax))
    nbarjSepStackUn = np.vstack([sim[0] for sim in indStack])
    sigmaSepStackUn = np.vstack([sim[1] for sim in indStack])
    #[nbarjSepStackUn,sigmaSepStackUn] = stacking.computeMeanStacks(\
    #    ahCentresListUn,antihaloRadiiUn,antihaloMassesListUn,\
    #    conditionListUn,pairCountsListUn,volumesListUn,snapListUnconstrained,\
    #    nbar,rBins,rMin,rMax,mMin,mMax)
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
        reCentreSnaps = True,rCentre=135,figuresFolder=''):
    # Load snapshots:
    if verbose:
        print("Loading snapshots...")
    snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" + \
        snapname) for snapNum in snapNumList]
    snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + snapnameRev) for snapNum in snapNumList]
    if reCentreSnaps:
        for snap in snapList:
            tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
            snap.recentred = True
    else:
        for snap in snaplist:
            snap.recentred = False
    if verbose:
        print("Computing halo/antihalo centres...")
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    antihaloCatalogueList = [snap.halos() for snap in snapListRev]
    snapsortList = [np.argsort(snap['iord']) for snap in snapList]
    ahProps = [pickle.load(\
        open(snap.filename + ".AHproperties.p","rb")) \
        for snap in snapList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    centralAntihalos = [tools.getAntiHalosInSphere(hcentres,rCentre) \
        for hcentres in antihaloCentres]
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
        mode="fractional"):
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
        searchOther = tree2.query_ball_point(snapedit.wrap(centres1,boxsize),\
            radii1,workers=-1)
    else:
        searchOther = tree1.query_ball_tree(tree2,distMax)
    for k in range(0,np.min([len(centres1),max_index])):
        if len(searchOther[k]) > 0:
            # Sort indices:
            distances = np.sqrt(np.sum((\
                    centres2[searchOther[k],:] - centres1[k,:])**2,1))
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
                bigger = np.where(quantity2[searchOther[k]] > quantity1[k])[0]
                quantRatio = quantity2[searchOther[k]]/quantity1[k]
                quantRatio[bigger] = quantity1[k]/\
                    quantity2[searchOther[k]][bigger]
                indSort = np.flip(np.argsort(quantRatio))
                quantRatio = quantRatio[indSort]
                sortedCandidates = np.array(searchOther[k],dtype=int)[indSort]
            else:
                raise Exception("Unrecognised sorting method")
            # Get thresholds for these candidates:
            if mode == "fractional":
                candRadii = quantity2[sortedCandidates]
                # Geometric mean of radii, to ensure symmetry.
                geometricRadii = np.sqrt(quantity1[k]*candRadii)
                candidates = np.where((quantRatio >= quantityThresh) & \
                    (distances[indSort] <= geometricRadii*distMax))[0]
            else:
                candidates = np.where((quantRatio >= quantityThresh))[0]
            candidatesList.append(candidates)
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


# Construct an anti-halo catalogue from reversed snapshots
def constructAntihaloCatalogue(snapNumList,samplesFolder="new_chain/",\
        verbose=True,rSphere=135,max_index=None,thresh=0.5,\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        fileSuffix= '',matchType='distance',crossMatchQuantity='radius',\
        crossMatchThreshold = 0.5,distMax=20.0,sortMethod='ratio',\
        blockDuplicates=True,twoWayOnly = True,\
        snapList=None,snapListRev=None,ahProps=None,hrList=None,\
        rMin = 5,rMax = 100,mode="fractional"):
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
        ahProps = [pickle.load(\
            open(snap.filename + ".AHproperties.p","rb")) \
            for snap in snapList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    antihaloRadii = [props[7] for props in ahProps]
    centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
        filterCondition = (antihaloRadii[k] > rMin) & \
        (antihaloRadii[k] <= rMax)) for k in range(0,len(snapNumList))]
    centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
    ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
    if max_index is None:
        max_index = np.max(ahCounts)
    # Construct constrained-only antihalo catalogues:
    if verbose:
        print("Constructing constrained region catalogues...")
    if hrList is None:
        hrList = [snap.halos() for snap in snapListRev]
    sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
        for k in range(0,len(snapNumList))]
    if matchType == "pynbody":
        hrListCentral = [copy.deepcopy(halos) for halos in hrList]
    shortHaloList = []
    for l in range(0,len(snapNumList)):
        if matchType == "pynbody":
            hrListCentral[l]._halos = dict([(k+1,\
                hrList[l][centralAntihalos[l][0][sortedList[l][k]]+1]) \
                for k in range(0,len(centralAntihalos[l][0]))])
            hrListCentral[l]._nhalos = len(centralAntihalos[l][0])
        shortHaloList.append(\
            np.array(centralAntihalos[l][0])[sortedList[l]] + 1)
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
    candidateCounts = [np.zeros((len(snapNumList),ahCounts[l]),dtype=int) \
        for l in range(0,len(snapNumList))]
    alreadyMatched = np.zeros((len(snapNumList),max_index),dtype=bool)
    matrixFullList = [np.array(matchArrayList[k]).transpose()[1:,:] \
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
                    finalCat.append(matrixFull[l])
                    candm = []
                    ratiosm = []
                    distancesm = []
                    # Mark companions as already included:
                    for m in range(0,len(snapNumList)):
                        if (m != k) and (matrixFull[l][m] > 0):
                            # Only deem something to be already matched if it
                            # maps back to this with a single unique candidate
                            alreadyMatched[m][matrixFull[l][m] - 1] = \
                                (matrixFullList[m]\
                                [matrixFull[l,m] - 1,k] == l+1) \
                                and (len(allCandidates[m][k]\
                                [matrixFull[l][m] - 1]) == 1)
                        if (m != k):
                            candm.append(allCandidates[k][m][l])
                            ratiosm.append(allRatios[k][m][l])
                            distancesm.append(allDistances[k][m][l])
                        if m == k:
                            alreadyMatched[m][l] = True
                    finalCandidates.append(candm)
                    finalRatios.append(ratiosm)
                    finalDistances.append(distancesm)
    return [np.array(finalCat),shortHaloList,np.array(twoWayMatchLists),\
        finalCandidates,finalRatios,finalDistances,allCandidates,\
        candidateCounts]

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















